const std = @import("std");
const gpu = @import("gpu/session.zig");

const max_candidates: usize = 32;
const shortlist_capacity: usize = 16;

pub const ProposalSettings = struct {
    trace: bool = false,
    trace_top_n: usize = 5,
    prefer_conservative_first_token: bool = true,
    enable_first_token_sanity_guard: bool = true,
    // If the drafted first token's penalized-logit is far below shortlist top-1,
    // skip speculation for this step.
    first_token_penalized_gap_limit: f32 = 1.25,
    first_token_transition_weight: f64 = 0.08,
    tail_transition_weight: f64 = 0.20,
    fallback_override_margin: f64 = 0.05,
    min_first_token_raw_gap: f32 = 1e-4,
};

pub const ProposeResult = struct {
    drafted_len: usize = 0,
    first_token_guard_reject: bool = false,
    first_token_gap: f32 = 0.0,
    top1_token: u32 = 0,
};

pub const TailProposalSource = enum {
    exact_suffix,
    suffix_backoff,
    logits_chain,
};

pub const TailProposalTrace = struct {
    source: TailProposalSource = .exact_suffix,
    matched_context_tokens: usize = 0,
    match_count: u16 = 0,
    weighted_score: f32 = 0,

    pub fn label(self: TailProposalTrace) []const u8 {
        return switch (self.source) {
            .exact_suffix => "exact_suffix",
            .suffix_backoff => "suffix_backoff",
            .logits_chain => "logits_chain",
        };
    }
};

const ScoredToken = struct {
    token_id: u32,
    raw_logit: f32,
    penalized_logit: f32,
    transition_count: u16,
    transition_bonus: f64,
    total_score: f64,
    in_shortlist: bool,
};

pub fn proposeDraftTokens(
    current_token: u32,
    history: []const u32,
    logits: []const f32,
    repeat_penalty: f32,
    max_draft: usize,
    out: []u32,
) usize {
    const result = proposeDraftTokensDetailed(
        current_token,
        history,
        logits,
        repeat_penalty,
        max_draft,
        out,
        .{},
    );
    return result.drafted_len;
}

pub fn proposeDraftTokensFromShortlistDetailed(
    current_token: u32,
    history: []const u32,
    shortlist: []const gpu.ShortlistEntry,
    repeat_penalty: f32,
    max_draft: usize,
    out: []u32,
    settings: ProposalSettings,
) ProposeResult {
    var result = ProposeResult{};
    if (max_draft == 0 or out.len == 0 or shortlist.len == 0) return result;

    const draft_cap = @min(max_draft, out.len);
    result.top1_token = shortlist[0].token_id;
    if (shortlist.len >= 2) {
        const gap = shortlist[0].logit - shortlist[1].logit;
        if (settings.enable_first_token_sanity_guard and gap <= settings.min_first_token_raw_gap) {
            result.first_token_guard_reject = true;
            result.first_token_gap = gap;
            if (settings.trace) {
                std.debug.print(
                    "BLOCK_PROPOSER_GUARD decision=REJECT reason=flat_shortlist raw_gap={d:.6} limit={d:.6} top1={d} top2={d}\n",
                    .{
                        gap,
                        settings.min_first_token_raw_gap,
                        shortlist[0].token_id,
                        shortlist[1].token_id,
                    },
                );
            }
            return result;
        }
    }

    var prev = current_token;
    var produced: usize = 0;
    while (produced < draft_cap) : (produced += 1) {
        const choose = chooseSuccessorFromShortlistDetailed(
            prev,
            history,
            shortlist,
            out[0..produced],
            repeat_penalty,
            settings,
        ) orelse break;

        if (produced == 0) {
            result.top1_token = choose.shortlist_top1.token_id;
            result.first_token_gap = choose.first_token_penalized_gap;
            if (settings.enable_first_token_sanity_guard and choose.first_token_penalized_gap > settings.first_token_penalized_gap_limit) {
                result.first_token_guard_reject = true;
                if (settings.trace) {
                    std.debug.print(
                        "BLOCK_PROPOSER_GUARD decision=REJECT reason=first_token_gap gap={d:.3} limit={d:.3} chosen={d} shortlist_top1={d}\n",
                        .{
                            choose.first_token_penalized_gap,
                            settings.first_token_penalized_gap_limit,
                            choose.chosen.token_id,
                            choose.shortlist_top1.token_id,
                        },
                    );
                }
                return result;
            }
        }

        if (settings.trace) traceChoice(produced, choose, settings.trace_top_n);
        out[produced] = choose.chosen.token_id;
        prev = choose.chosen.token_id;
    }

    result.drafted_len = produced;
    return result;
}

pub fn proposeDraftTokensFromHistoryChain(
    current_token: u32,
    history: []const u32,
    max_draft: usize,
    out: []u32,
) usize {
    if (max_draft == 0 or out.len == 0 or history.len == 0) return 0;

    const draft_cap = @min(max_draft, out.len);
    var produced: usize = 0;

    while (produced < draft_cap) : (produced += 1) {
        const prefix = out[0..produced];
        const next = mostFrequentHistoryContinuation(current_token, prefix, history) orelse break;
        out[produced] = next;
    }

    return produced;
}

pub fn proposeTailTokensFromAcceptedContext(
    history: []const u32,
    max_draft: usize,
    out: []u32,
    trace_out: ?[]TailProposalTrace,
) usize {
    if (max_draft == 0 or out.len == 0 or history.len == 0) return 0;

    const draft_cap = @min(max_draft, out.len);
    var produced: usize = 0;
    while (produced < draft_cap) : (produced += 1) {
        const continuation = bestContextContinuation(history, out[0..produced]) orelse break;
        out[produced] = continuation.token_id;
        if (trace_out) |trace| {
            if (produced < trace.len) {
                trace[produced] = .{
                    .source = continuation.trace.source,
                    .matched_context_tokens = continuation.trace.matched_context_tokens,
                    .match_count = continuation.trace.match_count,
                    .weighted_score = continuation.trace.weighted_score,
                };
            }
        }
    }
    return produced;
}

pub fn proposeDraftTokensDetailed(
    current_token: u32,
    history: []const u32,
    logits: []const f32,
    repeat_penalty: f32,
    max_draft: usize,
    out: []u32,
    settings: ProposalSettings,
) ProposeResult {
    var result = ProposeResult{};
    if (max_draft == 0 or out.len == 0 or logits.len == 0) return result;

    const draft_cap = @min(max_draft, out.len);
    var shortlist_ids: [shortlist_capacity]u32 = undefined;
    const shortlist_len = buildTopShortlist(logits, &shortlist_ids);
    if (shortlist_len == 0) return result;
    const top1_id = shortlist_ids[0];
    result.top1_token = top1_id;
    if (shortlist_len >= 2) {
        const gap = logits[shortlist_ids[0]] - logits[shortlist_ids[1]];
        if (settings.enable_first_token_sanity_guard and gap <= settings.min_first_token_raw_gap) {
            result.first_token_guard_reject = true;
            result.first_token_gap = gap;
            if (settings.trace) {
                std.debug.print(
                    "BLOCK_PROPOSER_GUARD decision=REJECT reason=flat_logits raw_gap={d:.6} limit={d:.6} top1={d} top2={d}\n",
                    .{
                        gap,
                        settings.min_first_token_raw_gap,
                        shortlist_ids[0],
                        shortlist_ids[1],
                    },
                );
            }
            return result;
        }
    }

    var prev = current_token;
    var produced: usize = 0;

    while (produced < draft_cap) : (produced += 1) {
        const choose = chooseSuccessorDetailed(
            prev,
            history,
            logits,
            shortlist_ids[0..shortlist_len],
            out[0..produced],
            repeat_penalty,
            settings,
        ) orelse break;

        if (produced == 0) {
            result.top1_token = choose.shortlist_top1.token_id;
            result.first_token_gap = choose.first_token_penalized_gap;
            if (settings.enable_first_token_sanity_guard and choose.first_token_penalized_gap > settings.first_token_penalized_gap_limit) {
                result.first_token_guard_reject = true;
                if (settings.trace) {
                    std.debug.print(
                        "BLOCK_PROPOSER_GUARD decision=REJECT reason=first_token_gap gap={d:.3} limit={d:.3} chosen={d} shortlist_top1={d}\n",
                        .{
                            choose.first_token_penalized_gap,
                            settings.first_token_penalized_gap_limit,
                            choose.chosen.token_id,
                            choose.shortlist_top1.token_id,
                        },
                    );
                }
                return result;
            }
        }

        if (settings.trace) {
            traceChoice(produced, choose, settings.trace_top_n);
        }

        out[produced] = choose.chosen.token_id;
        prev = choose.chosen.token_id;
    }

    result.drafted_len = produced;
    return result;
}

const ChoiceResult = struct {
    chosen: ScoredToken,
    shortlist_top1: ScoredToken,
    first_token_penalized_gap: f32,
    shortlist_ranked: [shortlist_capacity]ScoredToken,
    shortlist_ranked_len: usize,
};

fn chooseSuccessorDetailed(
    prev: u32,
    history: []const u32,
    logits: []const f32,
    shortlist: []const u32,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
    settings: ProposalSettings,
) ?ChoiceResult {
    const is_first = drafted_prefix.len == 0;
    const transition_weight: f64 = if (is_first) settings.first_token_transition_weight else settings.tail_transition_weight;

    var shortlist_ranked: [shortlist_capacity]ScoredToken = undefined;
    var shortlist_ranked_len: usize = 0;
    for (shortlist) |token_id| {
        if (shortlist_ranked_len >= shortlist_ranked.len) break;
        shortlist_ranked[shortlist_ranked_len] = scoreCandidateDetailed(
            token_id,
            transitionCount(prev, token_id, history),
            true,
            logits,
            history,
            drafted_prefix,
            repeat_penalty,
            transition_weight,
        );
        shortlist_ranked_len += 1;
    }
    sortScoredDescending(shortlist_ranked[0..shortlist_ranked_len]);
    if (shortlist_ranked_len == 0) return null;

    const shortlist_top1 = shortlist_ranked[0];
    const fallback = shortlist_top1;

    if (history.len < 2) {
        return .{
            .chosen = fallback,
            .shortlist_top1 = shortlist_top1,
            .first_token_penalized_gap = 0,
            .shortlist_ranked = shortlist_ranked,
            .shortlist_ranked_len = shortlist_ranked_len,
        };
    }

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

    if (candidate_len == 0) {
        return .{
            .chosen = fallback,
            .shortlist_top1 = shortlist_top1,
            .first_token_penalized_gap = 0,
            .shortlist_ranked = shortlist_ranked,
            .shortlist_ranked_len = shortlist_ranked_len,
        };
    }

    var history_best = scoreCandidateDetailed(
        candidate_ids[0],
        candidate_counts[0],
        inShortlist(candidate_ids[0], shortlist),
        logits,
        history,
        drafted_prefix,
        repeat_penalty,
        transition_weight,
    );
    for (1..candidate_len) |i| {
        const scored = scoreCandidateDetailed(
            candidate_ids[i],
            candidate_counts[i],
            inShortlist(candidate_ids[i], shortlist),
            logits,
            history,
            drafted_prefix,
            repeat_penalty,
            transition_weight,
        );
        if (isBetterScored(scored, history_best)) history_best = scored;
    }

    if (is_first and settings.prefer_conservative_first_token) {
        // First drafted token should remain close to current logits top-1.
        const history_gap = shortlist_top1.penalized_logit - history_best.penalized_logit;
        if (!history_best.in_shortlist or history_gap > settings.first_token_penalized_gap_limit) {
            return .{
                .chosen = fallback,
                .shortlist_top1 = shortlist_top1,
                .first_token_penalized_gap = 0,
                .shortlist_ranked = shortlist_ranked,
                .shortlist_ranked_len = shortlist_ranked_len,
            };
        }
    }

    const chosen = if (fallback.total_score > history_best.total_score + settings.fallback_override_margin)
        fallback
    else
        history_best;

    const first_gap = if (is_first) shortlist_top1.penalized_logit - chosen.penalized_logit else @as(f32, 0);
    return .{
        .chosen = chosen,
        .shortlist_top1 = shortlist_top1,
        .first_token_penalized_gap = first_gap,
        .shortlist_ranked = shortlist_ranked,
        .shortlist_ranked_len = shortlist_ranked_len,
    };
}

fn chooseSuccessorFromShortlistDetailed(
    prev: u32,
    history: []const u32,
    shortlist: []const gpu.ShortlistEntry,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
    settings: ProposalSettings,
) ?ChoiceResult {
    const is_first = drafted_prefix.len == 0;
    const transition_weight: f64 = if (is_first) settings.first_token_transition_weight else settings.tail_transition_weight;

    var shortlist_ranked: [shortlist_capacity]ScoredToken = undefined;
    var shortlist_ranked_len: usize = 0;
    for (shortlist) |entry| {
        if (shortlist_ranked_len >= shortlist_ranked.len) break;
        shortlist_ranked[shortlist_ranked_len] = scoreCandidateFromShortlistDetailed(
            entry.token_id,
            transitionCount(prev, entry.token_id, history),
            true,
            shortlist,
            history,
            drafted_prefix,
            repeat_penalty,
            transition_weight,
        );
        shortlist_ranked_len += 1;
    }
    sortScoredDescending(shortlist_ranked[0..shortlist_ranked_len]);
    if (shortlist_ranked_len == 0) return null;

    const shortlist_top1 = shortlist_ranked[0];
    const fallback = shortlist_top1;
    if (history.len < 2) {
        return .{
            .chosen = fallback,
            .shortlist_top1 = shortlist_top1,
            .first_token_penalized_gap = 0,
            .shortlist_ranked = shortlist_ranked,
            .shortlist_ranked_len = shortlist_ranked_len,
        };
    }

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

    if (candidate_len == 0) {
        return .{
            .chosen = fallback,
            .shortlist_top1 = shortlist_top1,
            .first_token_penalized_gap = 0,
            .shortlist_ranked = shortlist_ranked,
            .shortlist_ranked_len = shortlist_ranked_len,
        };
    }

    var history_best = scoreCandidateFromShortlistDetailed(
        candidate_ids[0],
        candidate_counts[0],
        inShortlistEntry(candidate_ids[0], shortlist),
        shortlist,
        history,
        drafted_prefix,
        repeat_penalty,
        transition_weight,
    );
    for (1..candidate_len) |i| {
        const scored = scoreCandidateFromShortlistDetailed(
            candidate_ids[i],
            candidate_counts[i],
            inShortlistEntry(candidate_ids[i], shortlist),
            shortlist,
            history,
            drafted_prefix,
            repeat_penalty,
            transition_weight,
        );
        if (isBetterScored(scored, history_best)) history_best = scored;
    }

    if (is_first and settings.prefer_conservative_first_token) {
        const history_gap = shortlist_top1.penalized_logit - history_best.penalized_logit;
        if (!history_best.in_shortlist or history_gap > settings.first_token_penalized_gap_limit) {
            return .{
                .chosen = fallback,
                .shortlist_top1 = shortlist_top1,
                .first_token_penalized_gap = 0,
                .shortlist_ranked = shortlist_ranked,
                .shortlist_ranked_len = shortlist_ranked_len,
            };
        }
    }

    const chosen = if (fallback.total_score > history_best.total_score + settings.fallback_override_margin) fallback else history_best;
    const first_gap = if (is_first) shortlist_top1.penalized_logit - chosen.penalized_logit else @as(f32, 0);
    return .{
        .chosen = chosen,
        .shortlist_top1 = shortlist_top1,
        .first_token_penalized_gap = first_gap,
        .shortlist_ranked = shortlist_ranked,
        .shortlist_ranked_len = shortlist_ranked_len,
    };
}

fn traceChoice(step_idx: usize, choice: ChoiceResult, top_n: usize) void {
    std.debug.print(
        "BLOCK_PROPOSER step_idx={d} chosen={d} raw={d:.4} penalized={d:.4} trans_count={d} trans_bonus={d:.4} total={d:.4} top1={d} top1_penalized={d:.4} gap={d:.4}\n",
        .{
            step_idx,
            choice.chosen.token_id,
            choice.chosen.raw_logit,
            choice.chosen.penalized_logit,
            choice.chosen.transition_count,
            choice.chosen.transition_bonus,
            choice.chosen.total_score,
            choice.shortlist_top1.token_id,
            choice.shortlist_top1.penalized_logit,
            choice.first_token_penalized_gap,
        },
    );

    const n = @min(top_n, choice.shortlist_ranked_len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const s = choice.shortlist_ranked[i];
        std.debug.print(
            "BLOCK_PROPOSER_CAND step_idx={d} rank={d} token={d} raw={d:.4} penalized={d:.4} trans_count={d} trans_bonus={d:.4} total={d:.4}\n",
            .{ step_idx, i + 1, s.token_id, s.raw_logit, s.penalized_logit, s.transition_count, s.transition_bonus, s.total_score },
        );
    }
}

fn scoreCandidateDetailed(
    token_id: u32,
    count: u16,
    shortlist_hit: bool,
    logits: []const f32,
    history: []const u32,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
    transition_weight: f64,
) ScoredToken {
    const raw_logit = if (token_id < logits.len) logits[token_id] else -std.math.inf(f32);
    const penalized = applyRepeatPenalty(raw_logit, token_id, history, drafted_prefix, repeat_penalty);
    const transition_bonus = @as(f64, @floatFromInt(count)) * transition_weight;
    return .{
        .token_id = token_id,
        .raw_logit = raw_logit,
        .penalized_logit = penalized,
        .transition_count = count,
        .transition_bonus = transition_bonus,
        .total_score = @as(f64, penalized) + transition_bonus,
        .in_shortlist = shortlist_hit,
    };
}

fn scoreCandidateFromShortlistDetailed(
    token_id: u32,
    count: u16,
    shortlist_hit: bool,
    shortlist: []const gpu.ShortlistEntry,
    history: []const u32,
    drafted_prefix: []const u32,
    repeat_penalty: f32,
    transition_weight: f64,
) ScoredToken {
    const raw_logit = shortlistLogitOf(token_id, shortlist);
    const penalized = applyRepeatPenalty(raw_logit, token_id, history, drafted_prefix, repeat_penalty);
    const transition_bonus = @as(f64, @floatFromInt(count)) * transition_weight;
    return .{
        .token_id = token_id,
        .raw_logit = raw_logit,
        .penalized_logit = penalized,
        .transition_count = count,
        .transition_bonus = transition_bonus,
        .total_score = @as(f64, penalized) + transition_bonus,
        .in_shortlist = shortlist_hit,
    };
}

fn transitionCount(prev: u32, next: u32, history: []const u32) u16 {
    if (history.len < 2) return 0;
    var count: u16 = 0;
    for (0..history.len - 1) |i| {
        if (history[i] == prev and history[i + 1] == next) count +|= 1;
    }
    return count;
}

fn isBetterScored(lhs: ScoredToken, rhs: ScoredToken) bool {
    if (lhs.total_score != rhs.total_score) return lhs.total_score > rhs.total_score;
    if (lhs.in_shortlist != rhs.in_shortlist) return lhs.in_shortlist;
    return lhs.token_id < rhs.token_id;
}

fn sortScoredDescending(tokens: []ScoredToken) void {
    if (tokens.len < 2) return;
    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        var best = i;
        var j = i + 1;
        while (j < tokens.len) : (j += 1) {
            if (tokens[j].total_score > tokens[best].total_score) best = j;
        }
        if (best != i) {
            const tmp = tokens[i];
            tokens[i] = tokens[best];
            tokens[best] = tmp;
        }
    }
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

fn inShortlistEntry(token_id: u32, shortlist: []const gpu.ShortlistEntry) bool {
    for (shortlist) |entry| {
        if (entry.token_id == token_id) return true;
    }
    return false;
}

fn shortlistLogitOf(token_id: u32, shortlist: []const gpu.ShortlistEntry) f32 {
    for (shortlist) |entry| {
        if (entry.token_id == token_id) return entry.logit;
    }
    return -std.math.inf(f32);
}

const ContextContinuation = struct {
    token_id: u32,
    trace: TailProposalTrace,
};

fn bestContextContinuation(history: []const u32, drafted_prefix: []const u32) ?ContextContinuation {
    if (history.len == 0) return null;

    const total_context_len = history.len + drafted_prefix.len;
    const max_context_len = @min(total_context_len, @as(usize, 8));
    if (max_context_len == 0) return null;

    var best_token_id: u32 = 0;
    var best_context_len: usize = 0;
    var best_match_count: u16 = 0;
    var best_score: f32 = 0;
    var found = false;

    var context_len = max_context_len;
    while (context_len >= 1) : (context_len -= 1) {
        var candidate_ids: [max_candidates]u32 = undefined;
        var candidate_counts: [max_candidates]u16 = undefined;
        var candidate_scores: [max_candidates]f32 = [_]f32{0} ** max_candidates;
        var candidate_len: usize = 0;

        var i: usize = 0;
        while (i + context_len < history.len) : (i += 1) {
            if (!historyContextMatches(history, drafted_prefix, i, context_len)) continue;
            const next = history[i + context_len];
            const score = contextMatchScore(history.len, i, context_len);
            var slot: usize = 0;
            while (slot < candidate_len and candidate_ids[slot] != next) : (slot += 1) {}
            if (slot < candidate_len) {
                candidate_counts[slot] +|= 1;
                candidate_scores[slot] += score;
                continue;
            }
            if (candidate_len >= max_candidates) continue;
            candidate_ids[candidate_len] = next;
            candidate_counts[candidate_len] = 1;
            candidate_scores[candidate_len] = score;
            candidate_len += 1;
        }

        if (candidate_len == 0) {
            if (context_len == 1) break;
            continue;
        }

        var best_idx: usize = 0;
        var k: usize = 1;
        while (k < candidate_len) : (k += 1) {
            if (candidate_scores[k] > candidate_scores[best_idx] or
                (candidate_scores[k] == candidate_scores[best_idx] and candidate_counts[k] > candidate_counts[best_idx]) or
                (candidate_scores[k] == candidate_scores[best_idx] and candidate_counts[k] == candidate_counts[best_idx] and candidate_ids[k] < candidate_ids[best_idx]))
            {
                best_idx = k;
            }
        }

        const best_count = candidate_counts[best_idx];
        const best_candidate_score = candidate_scores[best_idx];
        if (!passesContextSupportThreshold(context_len, best_count, best_candidate_score)) {
            if (context_len == 1) break;
            continue;
        }

        if (!found or
            best_candidate_score > best_score or
            (best_candidate_score == best_score and context_len > best_context_len) or
            (best_candidate_score == best_score and context_len == best_context_len and best_count > best_match_count) or
            (best_candidate_score == best_score and context_len == best_context_len and best_count == best_match_count and candidate_ids[best_idx] < best_token_id))
        {
            found = true;
            best_token_id = candidate_ids[best_idx];
            best_context_len = context_len;
            best_match_count = best_count;
            best_score = best_candidate_score;
        }
    }

    if (!found) return null;
    return .{
        .token_id = best_token_id,
        .trace = .{
            .source = if (best_context_len == max_context_len) .exact_suffix else .suffix_backoff,
            .matched_context_tokens = best_context_len,
            .match_count = best_match_count,
            .weighted_score = best_score,
        },
    };
}

fn contextMatchScore(history_len: usize, history_start: usize, context_len: usize) f32 {
    const recency_ratio = @as(f32, @floatFromInt(history_start + context_len)) / @as(f32, @floatFromInt(@max(@as(usize, 1), history_len)));
    const recency_bonus = 1.0 + 2.0 * recency_ratio;
    const context_bonus = 1.0 + 0.75 * @as(f32, @floatFromInt(context_len));
    const recent_window_bonus: f32 = if (history_start + context_len + 16 >= history_len) 1.5 else 0.0;
    return recency_bonus * context_bonus + recent_window_bonus;
}

fn passesContextSupportThreshold(context_len: usize, match_count: u16, weighted_score: f32) bool {
    if (context_len >= 4) return match_count >= 1;
    if (context_len == 3) return match_count >= 1 and weighted_score >= 4.5;
    if (context_len == 2) return match_count >= 1 and weighted_score >= 5.0;
    return match_count >= 2 and weighted_score >= 6.0;
}

fn historyContextMatches(history: []const u32, drafted_prefix: []const u32, history_start: usize, context_len: usize) bool {
    const total_context_len = history.len + drafted_prefix.len;
    if (context_len > total_context_len) return false;
    const context_start = total_context_len - context_len;

    var offset: usize = 0;
    while (offset < context_len) : (offset += 1) {
        if (history[history_start + offset] != contextTokenAt(history, drafted_prefix, context_start + offset)) {
            return false;
        }
    }
    return true;
}

fn contextTokenAt(history: []const u32, drafted_prefix: []const u32, idx: usize) u32 {
    if (idx < history.len) return history[idx];
    return drafted_prefix[idx - history.len];
}

fn mostFrequentHistoryContinuation(current_token: u32, drafted_prefix: []const u32, history: []const u32) ?u32 {
    var candidate_ids: [max_candidates]u32 = undefined;
    var candidate_counts: [max_candidates]u16 = undefined;
    var candidate_len: usize = 0;

    if (history.len <= drafted_prefix.len + 1) return null;

    var i: usize = 0;
    while (i + drafted_prefix.len + 1 < history.len) : (i += 1) {
        if (history[i] != current_token) continue;

        var matches = true;
        var j: usize = 0;
        while (j < drafted_prefix.len) : (j += 1) {
            if (history[i + 1 + j] != drafted_prefix[j]) {
                matches = false;
                break;
            }
        }
        if (!matches) continue;

        const next = history[i + 1 + drafted_prefix.len];
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
    var k: usize = 1;
    while (k < candidate_len) : (k += 1) {
        if (candidate_counts[k] > candidate_counts[best_idx] or
            (candidate_counts[k] == candidate_counts[best_idx] and candidate_ids[k] < candidate_ids[best_idx]))
        {
            best_idx = k;
        }
    }

    return candidate_ids[best_idx];
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

test "conservative first token resists off-shortlist history pick" {
    const history = [_]u32{ 5, 0, 5, 0, 5, 0 };
    const logits = [_]f32{ 1.0, 0.2, 0.1, 0.0, -0.5, -0.8, 1.9, 1.7 };
    var out: [2]u32 = undefined;
    const res = proposeDraftTokensDetailed(5, &history, &logits, 1.0, 2, &out, .{
        .enable_first_token_sanity_guard = true,
        .prefer_conservative_first_token = true,
    });
    try std.testing.expect(res.drafted_len >= 1);
    try std.testing.expect(out[0] == 6 or out[0] == 7);
}

test "first token sanity guard can reject speculation" {
    const history = [_]u32{ 9, 0, 9, 0, 9, 0, 9, 0 };
    var logits = [_]f32{0} ** 32;
    logits[0] = -1.0;
    logits[9] = 0.2;
    logits[31] = 2.0;
    var out: [2]u32 = undefined;

    const res = proposeDraftTokensDetailed(9, &history, &logits, 1.0, 2, &out, .{
        .enable_first_token_sanity_guard = true,
        .prefer_conservative_first_token = false,
        .first_token_transition_weight = 2.0,
        .first_token_penalized_gap_limit = 0.25,
    });
    try std.testing.expect(res.first_token_guard_reject);
    try std.testing.expectEqual(@as(usize, 0), res.drafted_len);
}

test "flat logits guard rejects draft step" {
    const history = [_]u32{ 1, 2, 3, 4 };
    const logits = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    var out: [2]u32 = undefined;
    const res = proposeDraftTokensDetailed(4, &history, &logits, 1.0, 2, &out, .{
        .enable_first_token_sanity_guard = true,
        .min_first_token_raw_gap = 1e-6,
    });
    try std.testing.expect(res.first_token_guard_reject);
    try std.testing.expectEqual(@as(usize, 0), res.drafted_len);
}

test "history chain proposer follows repeated continuation" {
    const history = [_]u32{ 10, 20, 30, 40, 10, 20, 30, 50, 10, 20, 30, 40 };
    var out: [3]u32 = undefined;
    const drafted = proposeDraftTokensFromHistoryChain(10, &history, 3, &out);
    try std.testing.expectEqual(@as(usize, 3), drafted);
    try std.testing.expectEqual(@as(u32, 20), out[0]);
    try std.testing.expectEqual(@as(u32, 30), out[1]);
    try std.testing.expectEqual(@as(u32, 40), out[2]);
}

test "accepted-context tail proposer uses longest suffix and backoff" {
    const history = [_]u32{ 7, 8, 9, 10, 11, 13, 7, 8, 9, 10, 11, 14, 7, 8, 9, 10 };
    var out: [2]u32 = undefined;
    var trace: [2]TailProposalTrace = undefined;
    const drafted = proposeTailTokensFromAcceptedContext(&history, 2, &out, &trace);
    try std.testing.expectEqual(@as(usize, 2), drafted);
    try std.testing.expectEqual(@as(u32, 11), out[0]);
    try std.testing.expectEqual(@as(u32, 13), out[1]);
    try std.testing.expectEqual(@as(usize, 4), trace[0].matched_context_tokens);
    try std.testing.expectEqualStrings("suffix_backoff", trace[0].label());
    try std.testing.expect(trace[0].weighted_score > 0);
}

test "accepted-context proposer prefers recent stronger continuation" {
    const history = [_]u32{
        1, 2, 9,
        1, 2, 9,
        1, 2, 9,
        7, 8, 1,
        2, 5, 7,
        8, 1, 2,
    };
    var out: [1]u32 = undefined;
    var trace: [1]TailProposalTrace = undefined;
    const drafted = proposeTailTokensFromAcceptedContext(&history, 1, &out, &trace);
    try std.testing.expectEqual(@as(usize, 1), drafted);
    try std.testing.expectEqual(@as(u32, 5), out[0]);
    try std.testing.expect(trace[0].matched_context_tokens >= 2);
}
