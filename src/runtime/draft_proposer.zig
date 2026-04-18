const std = @import("std");

const max_candidates: usize = 32;
const shortlist_capacity: usize = 8;

pub fn proposeDraftTokens(
    current_token: u32,
    history: []const u32,
    logits: []const f32,
    max_draft: usize,
    out: []u32,
) usize {
    if (max_draft == 0 or out.len == 0 or history.len < 2) return 0;

    const draft_cap = @min(max_draft, out.len);
    var shortlist_ids: [shortlist_capacity]u32 = undefined;
    const shortlist_len = buildTopShortlist(logits, &shortlist_ids);
    var prev = current_token;
    var produced: usize = 0;

    while (produced < draft_cap) : (produced += 1) {
        const next = chooseSuccessor(prev, history, logits, shortlist_ids[0..shortlist_len]) orelse break;
        out[produced] = next;
        prev = next;
    }

    return produced;
}

fn chooseSuccessor(prev: u32, history: []const u32, logits: []const f32, shortlist: []const u32) ?u32 {
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

    if (candidate_len == 0) return null;

    var best_idx: usize = 0;
    for (1..candidate_len) |i| {
        if (isBetter(candidate_ids[i], candidate_counts[i], candidate_ids[best_idx], candidate_counts[best_idx], logits, shortlist)) {
            best_idx = i;
        }
    }
    return candidate_ids[best_idx];
}

fn isBetter(
    lhs_id: u32,
    lhs_count: u16,
    rhs_id: u32,
    rhs_count: u16,
    logits: []const f32,
    shortlist: []const u32,
) bool {
    if (lhs_count != rhs_count) return lhs_count > rhs_count;

    const lhs_short = inShortlist(lhs_id, shortlist);
    const rhs_short = inShortlist(rhs_id, shortlist);
    if (lhs_short != rhs_short) return lhs_short;

    const lhs_logit = if (lhs_id < logits.len) logits[lhs_id] else -std.math.inf(f32);
    const rhs_logit = if (rhs_id < logits.len) logits[rhs_id] else -std.math.inf(f32);
    if (lhs_logit != rhs_logit) return lhs_logit > rhs_logit;

    return lhs_id < rhs_id;
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

    // Keep shortlist deterministic for tie-break scans.
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

test "proposeDraftTokens favors frequent successors" {
    const history = [_]u32{ 9, 4, 7, 4, 7, 4, 8 };
    const logits = [_]f32{ 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.9, 0 };
    var out: [4]u32 = undefined;
    const drafted = proposeDraftTokens(4, &history, &logits, 4, &out);
    try std.testing.expect(drafted >= 2);
    try std.testing.expectEqual(@as(u32, 7), out[0]);
    try std.testing.expectEqual(@as(u32, 4), out[1]);
}

test "proposeDraftTokens uses shortlist/logits on tie" {
    const history = [_]u32{ 5, 10, 5, 11, 1 };
    const logits = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1.8 };
    var out: [4]u32 = undefined;
    const drafted = proposeDraftTokens(5, &history, &logits, 4, &out);
    try std.testing.expect(drafted >= 1);
    try std.testing.expectEqual(@as(u32, 11), out[0]);
}
