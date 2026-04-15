const std = @import("std");
const llama_gpu = @import("runtime/gpu/session.zig");
const runtime_types = @import("runtime/types.zig");

pub const SampleCandidate = struct {
    token_id: u32 = 0,
    logit: f32 = 0,
    weight: f64 = 0,
};

pub fn sampleToken(
    logits: []const f32,
    recent_tokens: []const u32,
    options: runtime_types.GenerationOptions,
    random: std.Random,
    candidates: []SampleCandidate,
) u32 {
    if (options.temperature <= 0) return argmax(logits);

    const top_k = if (options.top_k == 0) logits.len else @min(options.top_k, logits.len);
    const candidate_count = if (top_k < logits.len)
        collectTopKCandidates(logits, top_k, candidates[0..top_k])
    else
        collectAllCandidates(logits, candidates[0..logits.len]);

    const active = candidates[0..candidate_count];
    if (active.len == 0) return argmax(logits);
    return sampleCollectedCandidates(active, recent_tokens, options, random);
}

pub fn sampleShortlist(
    shortlist: []const llama_gpu.ShortlistEntry,
    recent_tokens: []const u32,
    options: runtime_types.GenerationOptions,
    random: std.Random,
    candidates: []SampleCandidate,
) u32 {
    if (shortlist.len == 0) return 0;
    for (shortlist, 0..) |entry, index| {
        candidates[index] = .{
            .token_id = entry.token_id,
            .logit = entry.logit,
        };
    }
    return sampleCollectedCandidates(candidates[0..shortlist.len], recent_tokens, options, random);
}

pub fn shortlistLenFor(options: runtime_types.GenerationOptions, vocab_size: usize) usize {
    if (vocab_size == 0) return 0;
    if (options.temperature <= 0) return 0;
    const configured_top_k = if (options.top_k == 0) llama_gpu.max_shortlist_len else options.top_k;
    return @min(vocab_size, @min(configured_top_k, llama_gpu.max_shortlist_len));
}

fn sampleCollectedCandidates(
    base_candidates: []SampleCandidate,
    recent_tokens: []const u32,
    options: runtime_types.GenerationOptions,
    random: std.Random,
) u32 {
    if (base_candidates.len == 0) return 0;

    var active = base_candidates;
    var max_logit = -std.math.inf(f32);
    var total_weight: f64 = 0;
    var fallback_token = active[0].token_id;
    var fallback_logit = active[0].logit;

    for (active) |*candidate| {
        const adjusted_logit = applyRepeatPenalty(candidate.logit, candidate.token_id, recent_tokens, options.repeat_penalty);
        candidate.logit = adjusted_logit;
        if (adjusted_logit > fallback_logit) {
            fallback_logit = adjusted_logit;
            fallback_token = candidate.token_id;
        }
        max_logit = @max(max_logit, adjusted_logit);
    }

    for (active) |*candidate| {
        const shifted = (@as(f64, candidate.logit) - @as(f64, max_logit)) / @as(f64, options.temperature);
        candidate.weight = @exp(shifted);
        total_weight += candidate.weight;
    }
    if (total_weight <= 0 or !std.math.isFinite(total_weight)) return fallback_token;

    if (options.min_p > 0) {
        const kept = truncateByMinP(active, options.min_p);
        active = active[0..kept];
        total_weight = 0;
        for (active) |candidate| total_weight += candidate.weight;
        if (total_weight <= 0 or !std.math.isFinite(total_weight)) return fallbackToken(active);
    }

    if (options.top_p < 1.0) {
        std.mem.sort(SampleCandidate, active, {}, lessSampleCandidate);
        const kept = truncateByTopP(active, total_weight, options.top_p);
        active = active[0..kept];
        total_weight = 0;
        for (active) |candidate| total_weight += candidate.weight;
        if (total_weight <= 0 or !std.math.isFinite(total_weight)) return fallbackToken(active);
    }

    const target = random.float(f64) * total_weight;
    var cumulative: f64 = 0;
    for (active) |candidate| {
        cumulative += candidate.weight;
        if (target <= cumulative) return candidate.token_id;
    }
    return active[active.len - 1].token_id;
}

fn fallbackToken(candidates: []const SampleCandidate) u32 {
    var best = candidates[0];
    for (candidates[1..]) |candidate| {
        if (candidate.logit > best.logit) best = candidate;
    }
    return best.token_id;
}

fn applyRepeatPenalty(logit: f32, token_id: u32, recent_tokens: []const u32, repeat_penalty: f32) f32 {
    if (repeat_penalty <= 1.0) return logit;
    for (recent_tokens) |recent| {
        if (recent != token_id) continue;
        return if (logit < 0) logit * repeat_penalty else logit / repeat_penalty;
    }
    return logit;
}

fn collectAllCandidates(logits: []const f32, out: []SampleCandidate) usize {
    for (logits, 0..) |logit, index| {
        out[index] = .{
            .token_id = @intCast(index),
            .logit = logit,
        };
    }
    return logits.len;
}

fn collectTopKCandidates(logits: []const f32, top_k: usize, out: []SampleCandidate) usize {
    var count: usize = 0;
    for (logits, 0..) |logit, index| {
        const candidate = SampleCandidate{
            .token_id = @intCast(index),
            .logit = logit,
        };

        if (count < top_k) {
            out[count] = candidate;
            count += 1;
            var pos = count - 1;
            while (pos > 0 and out[pos].logit > out[pos - 1].logit) : (pos -= 1) {
                std.mem.swap(SampleCandidate, &out[pos], &out[pos - 1]);
            }
            continue;
        }

        if (candidate.logit <= out[top_k - 1].logit) continue;
        out[top_k - 1] = candidate;
        var pos = top_k - 1;
        while (pos > 0 and out[pos].logit > out[pos - 1].logit) : (pos -= 1) {
            std.mem.swap(SampleCandidate, &out[pos], &out[pos - 1]);
        }
    }
    return count;
}

fn truncateByTopP(candidates: []SampleCandidate, total_weight: f64, top_p: f32) usize {
    var cumulative: f64 = 0;
    for (candidates, 0..) |candidate, index| {
        cumulative += candidate.weight;
        if (cumulative / total_weight >= @as(f64, top_p)) return index + 1;
    }
    return candidates.len;
}

fn truncateByMinP(candidates: []SampleCandidate, min_p: f32) usize {
    var max_weight: f64 = 0;
    for (candidates) |candidate| {
        max_weight = @max(max_weight, candidate.weight);
    }
    const threshold = max_weight * @as(f64, min_p);
    var kept: usize = 0;
    for (candidates) |candidate| {
        if (candidate.weight >= threshold) {
            candidates[kept] = candidate;
            kept += 1;
        }
    }
    return @max(@as(usize, 1), kept);
}

fn lessSampleCandidate(_: void, lhs: SampleCandidate, rhs: SampleCandidate) bool {
    return lhs.weight > rhs.weight;
}

fn argmax(values: []const f32) u32 {
    var best_index: usize = 0;
    var best_value = values[0];
    for (values[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return @intCast(best_index);
}

test "sampleToken uses seed-driven randomness when temperature is enabled" {
    const logits = [_]f32{ 2.0, 1.8, 1.7, 1.6 };
    var candidates_a: [logits.len]SampleCandidate = undefined;
    var candidates_b: [logits.len]SampleCandidate = undefined;
    var candidates_c: [logits.len]SampleCandidate = undefined;

    var prng_a = std.Random.DefaultPrng.init(42);
    var prng_b = std.Random.DefaultPrng.init(42);
    var prng_c = std.Random.DefaultPrng.init(99);

    const sample_a = sampleToken(&logits, &.{}, .{ .temperature = 0.8 }, prng_a.random(), &candidates_a);
    const sample_b = sampleToken(&logits, &.{}, .{ .temperature = 0.8 }, prng_b.random(), &candidates_b);
    const sample_c = sampleToken(&logits, &.{}, .{ .temperature = 0.8 }, prng_c.random(), &candidates_c);

    try std.testing.expectEqual(sample_a, sample_b);
    try std.testing.expect(sample_a < logits.len);
    try std.testing.expect(sample_c < logits.len);
}

test "sampleToken falls back to argmax for non-positive temperature" {
    const logits = [_]f32{ -1.0, 3.0, 0.5 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    try std.testing.expectEqual(@as(u32, 1), sampleToken(&logits, &.{}, .{ .temperature = 0 }, prng.random(), &candidates));
}

test "sampleToken top-k restricts sampling set" {
    const logits = [_]f32{ 10.0, 9.0, 1.0, 0.5 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(123);

    for (0..32) |_| {
        const sample = sampleToken(&logits, &.{}, .{ .temperature = 1.0, .top_k = 2 }, prng.random(), &candidates);
        try std.testing.expect(sample == 0 or sample == 1);
    }
}

test "sampleToken top-p trims low-mass tail" {
    const logits = [_]f32{ 8.0, 7.5, 1.0, 0.2 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(321);

    for (0..32) |_| {
        const sample = sampleToken(&logits, &.{}, .{ .temperature = 1.0, .top_p = 0.9 }, prng.random(), &candidates);
        try std.testing.expect(sample == 0 or sample == 1);
    }
}

test "sampleToken repeat penalty demotes repeated tokens" {
    const logits = [_]f32{ 4.0, 3.9, 1.0 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(5);

    for (0..32) |_| {
        const sample = sampleToken(&logits, &.{0}, .{ .temperature = 0.7, .repeat_penalty = 1.5 }, prng.random(), &candidates);
        try std.testing.expect(sample != 2);
    }
}

test "sampleToken min-p trims low-probability tail" {
    const logits = [_]f32{ 6.0, 5.5, 0.5, 0.2 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(17);

    for (0..32) |_| {
        const sample = sampleToken(&logits, &.{}, .{ .temperature = 1.0, .min_p = 0.5 }, prng.random(), &candidates);
        try std.testing.expect(sample == 0 or sample == 1);
    }
}

test "sampleShortlist reuses cpu sampler policy for exact top-k shortlist" {
    const logits = [_]f32{ 4.0, 3.5, 2.0, 1.0, -1.0 };
    const shortlist = [_]llama_gpu.ShortlistEntry{
        .{ .token_id = 0, .logit = 4.0 },
        .{ .token_id = 1, .logit = 3.5 },
        .{ .token_id = 2, .logit = 2.0 },
    };
    const options = runtime_types.GenerationOptions{
        .temperature = 0.9,
        .top_k = 3,
        .top_p = 0.85,
        .min_p = 0.0,
        .repeat_penalty = 1.2,
    };

    var full_candidates: [logits.len]SampleCandidate = undefined;
    var shortlist_candidates: [shortlist.len]SampleCandidate = undefined;
    var prng_a = std.Random.DefaultPrng.init(1234);
    var prng_b = std.Random.DefaultPrng.init(1234);

    const full_sample = sampleToken(&logits, &.{1}, options, prng_a.random(), &full_candidates);
    const shortlist_sample = sampleShortlist(&shortlist, &.{1}, options, prng_b.random(), &shortlist_candidates);
    try std.testing.expectEqual(full_sample, shortlist_sample);
}

test "shortlistLenFor uses fixed gpu shortlist cap when top-k is disabled" {
    try std.testing.expectEqual(@as(usize, 64), shortlistLenFor(.{ .temperature = 0.7 }, 32000));
    try std.testing.expectEqual(@as(usize, 12), shortlistLenFor(.{ .temperature = 0.7, .top_k = 12 }, 32000));
    try std.testing.expectEqual(@as(usize, 0), shortlistLenFor(.{ .temperature = 0 }, 32000));
}
