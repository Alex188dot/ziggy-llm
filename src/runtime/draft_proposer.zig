const std = @import("std");

const max_candidates: usize = 32;
const shortlist_capacity: usize = 16;

pub fn proposeDraftTokens(
    current_token: u32,
    history: []const u32,
    logits: []const f32,
    repeat_penalty: f32,
    max_draft: usize,
    out: []u32,
) usize {
    if (max_draft == 0 or out.len == 0 or logits.len == 0) return 0;

    const draft_cap = @min(max_draft, out.len);
    var shortlist_ids: [shortlist_capacity]u32 = undefined;
    const shortlist_len = buildTopShortlist(logits, &shortlist_ids);
    if (shortlist_len == 0) return 0;

    var prev = current_token;
    var produced: usize = 0;

    while (produced < draft_cap) : (produced += 1) {
        const next = chooseSuccessor(
            prev,
            history,
            logits,
            shortlist_ids[0..shortlist_len],
            out[0..produced],
            repeat_penalty,
        ) orelse break;
        out[produced] = next;
        prev = next;
    }

    return produced;
}

fn chooseSuccessor(
    prev: u32,
    history: []const u32,
    logits: []const f32,
    shortlist: []const u32,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
) ?u32 {
    const fallback = topShortlistByPenalizedLogit(shortlist, logits, history, drafted_prefix, repeat_penalty) orelse return null;

    if (history.len < 2) return fallback;

    var candidate_ids: [max_candidates]u32 = undefined;
    var candidate_counts: [max_candidates]u16 = undefined;
    var candidate_len: usize = 0;

    for (0..history.len - 1) |i| {
        if (history[i] != prev) continue;
        const next = history[i + 1];
        var slot: usize = 0;
        while (slot < candidate_len and candidate_ids[slot] != next) : (slot += 1) {}
        if (slot < candidate_len) {
            candidate_counts[slot] +|= 1;
            continue;
        }
        if (candidate_len >= max_candidates) continue;
        candidate_ids[candidate_len] = next;
        candidate_counts[candidate_len] = 1;
        candidate_len += 1;
    }

    if (candidate_len == 0) return fallback;

    var best_idx: usize = 0;
    var i: usize = 1;
    while (i < candidate_len) : (i += 1) {
        if (isBetter(
            candidate_ids[i],
            candidate_counts[i],
            candidate_ids[best_idx],
            candidate_counts[best_idx],
            logits,
            shortlist,
            history,
            drafted_prefix,
            repeat_penalty,
        )) {
            best_idx = i;
        }
    }

    const history_best = candidate_ids[best_idx];
    const history_best_score = scoreCandidate(history_best, candidate_counts[best_idx], logits, history, drafted_prefix, repeat_penalty);
    const fallback_score = scoreCandidate(fallback, 0, logits, history, drafted_prefix, repeat_penalty);
    return if (fallback_score > history_best_score + 0.1) fallback else history_best;
}

fn isBetter(
    lhs_id: u32,
    lhs_count: u16,
    rhs_id: u32,
    rhs_count: u16,
    logits: []const f32,
    shortlist: []const u32,
    history: []const u32,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
) bool {
    const lhs_score = scoreCandidate(lhs_id, lhs_count, logits, history, drafted_prefix, repeat_penalty);
    const rhs_score = scoreCandidate(rhs_id, rhs_count, logits, history, drafted_prefix, repeat_penalty);
    if (lhs_score != rhs_score) return lhs_score > rhs_score;

    const lhs_short = inShortlist(lhs_id, shortlist);
    const rhs_short = inShortlist(rhs_id, shortlist);
    if (lhs_short != rhs_short) return lhs_short;

    return lhs_id < rhs_id;
}

fn scoreCandidate(
    token_id: u32,
    count: u16,
    logits: []const f32,
    history: []const u32,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
) f64 {
    const raw_logit = if (token_id < logits.len) logits[token_id] else -std.math.inf(f32);
    const penalized = applyRepeatPenalty(raw_logit, token_id, history, drafted_prefix, repeat_penalty);
    // Favor local transition evidence, but let logit dominate if history is weak.
    return @as(f64, penalized) + @as(f64, @floatFromInt(count)) * 0.25;
}

fn topShortlistByPenalizedLogit(
    shortlist: []const u32,
    logits: []const f32,
    history: []const u32,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
) ?u32 {
    if (shortlist.len == 0) return null;

    var best_id = shortlist[0];
    var best_score = applyRepeatPenalty(logits[best_id], best_id, history, drafted_prefix, repeat_penalty);

    for (shortlist[1..]) |token_id| {
        const score = applyRepeatPenalty(logits[token_id], token_id, history, drafted_prefix, repeat_penalty);
        if (score > best_score or (score == best_score and token_id < best_id)) {
            best_id = token_id;
            best_score = score;
        }
    }

    return best_id;
}

fn applyRepeatPenalty(
    logit: f32,
    token_id: u32,
    history: []const u32,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
) f32 {
    if (repeat_penalty <= 1.0) return logit;
    if (containsToken(history, token_id) or containsToken(drafted_prefix, token_id)) {
        return if (logit < 0) logit * repeat_penalty else logit / repeat_penalty;
    }
    return logit;
}

fn containsToken(tokens: []const u32, token_id: u32) bool {
    for (tokens) |token| {
        if (token == token_id) return true;
    }
    return false;
}

fn inShortlist(token_id: u32, shortlist: []const u32) bool {
    for (shortlist) |candidate| {
        if (candidate == token_id) return true;
    }
    return false;
}

fn buildTopShortlist(logits: []const f32, out: []u32) usize {
    if (logits.len == 0 or out.len == 0) return 0;

    var out_len: usize = 0;
    var out_scores: [shortlist_capacity]f32 = undefined;

    for (logits, 0..) |score, idx| {
        if (out_len < out.len) {
            out[out_len] = @intCast(idx);
            out_scores[out_len] = score;
            out_len += 1;
            continue;
        }

        var worst_i: usize = 0;
        for (1..out_len) |j| {
            if (out_scores[j] < out_scores[worst_i]) worst_i = j;
        }
        if (score > out_scores[worst_i]) {
            out[worst_i] = @intCast(idx);
            out_scores[worst_i] = score;
        }
    }

    var i: usize = 0;
    while (i < out_len) : (i += 1) {
        var best = i;
        var j = i + 1;
        while (j < out_len) : (j += 1) {
            if (out_scores[j] > out_scores[best]) best = j;
        }
        if (best != i) {
            const score_tmp = out_scores[i];
            out_scores[i] = out_scores[best];
            out_scores[best] = score_tmp;
            const id_tmp = out[i];
            out[i] = out[best];
            out[best] = id_tmp;
        }
    }

    return out_len;
}

test "proposeDraftTokens favors logits fallback when no transition exists" {
    const history = [_]u32{ 1, 2, 3, 4 };
    const logits = [_]f32{ 0.1, 0.2, 1.7, 0.3, 0.5 };
    var out: [3]u32 = undefined;
    const drafted = proposeDraftTokens(9, &history, &logits, 1.0, 3, &out);
    try std.testing.expect(drafted >= 1);
    try std.testing.expectEqual(@as(u32, 2), out[0]);
}

test "proposeDraftTokens uses transition evidence but respects repeat penalty" {
    const history = [_]u32{ 4, 7, 4, 7, 4, 7, 9 };
    const logits = [_]f32{ 0, 0, 0, 0, 0, 0, 0.2, 3.0, 0.1, 2.5 };
    var out: [2]u32 = undefined;

    const drafted_no_penalty = proposeDraftTokens(4, &history, &logits, 1.0, 2, &out);
    try std.testing.expect(drafted_no_penalty >= 1);
    try std.testing.expectEqual(@as(u32, 7), out[0]);

    const drafted_with_penalty = proposeDraftTokens(4, &history, &logits, 1.5, 2, &out);
    try std.testing.expect(drafted_with_penalty >= 1);
    try std.testing.expect(out[0] == 7 or out[0] == 9);
}
