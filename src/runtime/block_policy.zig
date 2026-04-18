const std = @import("std");

// Conservative default: only speculate when the top-1 token is clearly separated.
pub const default_confidence_margin_threshold: f32 = 0.75;
pub const cooldown_draft_cap: usize = 1;
const rollback_ema_unstable: f64 = 0.35;
const rollback_ema_disable: f64 = 0.55;
const accepted_ema_min_for_medium: f64 = 0.9;
const accepted_ema_high: f64 = 1.8;
const high_confidence_multiplier: f32 = 2.0;
const medium_confidence_multiplier: f32 = 1.35;

pub fn top1Top2Margin(logits: []const f32) f32 {
    std.debug.assert(logits.len > 0);
    var best = logits[0];
    var second = -std.math.inf(f32);
    for (logits[1..]) |value| {
        if (value > best) {
            second = best;
            best = value;
        } else if (value > second) {
            second = value;
        }
    }
    if (!std.math.isFinite(second)) {
        // Only one logit: treat as maximally confident for gating.
        return std.math.inf(f32);
    }
    return best - second;
}

pub fn shouldSpeculateFromLogits(logits: []const f32, margin_threshold: f32) bool {
    if (logits.len <= 1) return false;
    return top1Top2Margin(logits) >= margin_threshold;
}

pub fn applyCooldownDraftLimit(draft_limit: usize, cooldown_remaining: *usize) usize {
    if (cooldown_remaining.* == 0) return draft_limit;
    cooldown_remaining.* -= 1;
    return @min(draft_limit, cooldown_draft_cap);
}

pub fn selectAdaptiveDraftLimit(
    exp_block_cap: usize,
    confidence_margin: f32,
    confidence_threshold: f32,
    recent_accepted_ema: f64,
    recent_rollback_ema: f64,
    observed_steps: usize,
) usize {
    if (exp_block_cap == 0) return 0;
    if (confidence_margin < confidence_threshold) return 0;
    if (exp_block_cap == 1) return 1;

    // After warmup, avoid speculative work in clearly unstable regions.
    if (observed_steps >= 8 and recent_rollback_ema >= rollback_ema_disable and recent_accepted_ema < 0.5) {
        return 0;
    }

    if (recent_rollback_ema >= rollback_ema_unstable) {
        return @min(exp_block_cap, @as(usize, 1));
    }

    const high_confidence_threshold = confidence_threshold * high_confidence_multiplier;
    if (exp_block_cap >= 4 and confidence_margin >= high_confidence_threshold and recent_accepted_ema >= accepted_ema_high and recent_rollback_ema <= 0.15) {
        return @min(exp_block_cap, @as(usize, 4));
    }

    const medium_confidence_threshold = confidence_threshold * medium_confidence_multiplier;
    if (confidence_margin >= medium_confidence_threshold and recent_accepted_ema >= accepted_ema_min_for_medium) {
        return @min(exp_block_cap, @as(usize, 2));
    }

    return 1;
}

pub fn acceptedPrefixInvariantHolds(draft_tokens: []const u32, verified_tokens: []const u32, accepted_count: usize) bool {
    if (accepted_count == 0) return false;
    if (accepted_count > draft_tokens.len + 1) return false;
    if (verified_tokens.len < accepted_count) return false;

    const accepted_prefix_len = accepted_count - 1;
    for (0..accepted_prefix_len) |i| {
        if (verified_tokens[i] != draft_tokens[i]) return false;
    }
    if (accepted_prefix_len < draft_tokens.len and verified_tokens[accepted_prefix_len] == draft_tokens[accepted_prefix_len]) {
        return false;
    }
    return true;
}

test "top1Top2Margin returns best-second gap" {
    const logits = [_]f32{ -2.0, 0.5, 1.75, 1.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), top1Top2Margin(&logits), 1e-6);
}

test "shouldSpeculateFromLogits gates low margin" {
    const logits = [_]f32{ 2.0, 1.4, 0.1, -0.3 };
    try std.testing.expect(shouldSpeculateFromLogits(&logits, 0.5));
    try std.testing.expect(!shouldSpeculateFromLogits(&logits, 0.7));
}

test "applyCooldownDraftLimit clamps while countdown is active" {
    var remaining: usize = 3;
    try std.testing.expectEqual(@as(usize, 1), applyCooldownDraftLimit(4, &remaining));
    try std.testing.expectEqual(@as(usize, 2), remaining);
    try std.testing.expectEqual(@as(usize, 1), applyCooldownDraftLimit(2, &remaining));
    try std.testing.expectEqual(@as(usize, 1), remaining);
    try std.testing.expectEqual(@as(usize, 1), applyCooldownDraftLimit(8, &remaining));
    try std.testing.expectEqual(@as(usize, 0), remaining);
    try std.testing.expectEqual(@as(usize, 5), applyCooldownDraftLimit(5, &remaining));
}

test "selectAdaptiveDraftLimit gates low confidence and scales on stable high confidence" {
    try std.testing.expectEqual(@as(usize, 0), selectAdaptiveDraftLimit(4, 0.6, 0.75, 2.0, 0.1, 10));
    try std.testing.expectEqual(@as(usize, 2), selectAdaptiveDraftLimit(4, 1.1, 0.75, 1.6, 0.1, 10));
    try std.testing.expectEqual(@as(usize, 4), selectAdaptiveDraftLimit(4, 1.8, 0.75, 2.2, 0.1, 10));
    try std.testing.expectEqual(@as(usize, 1), selectAdaptiveDraftLimit(4, 1.8, 0.75, 2.2, 0.5, 10));
    try std.testing.expectEqual(@as(usize, 0), selectAdaptiveDraftLimit(4, 1.2, 0.75, 0.3, 0.7, 12));
}

test "acceptedPrefixInvariantHolds enforces prefix semantics" {
    const draft = [_]u32{ 10, 11, 12 };
    const verified_match_then_mismatch = [_]u32{ 10, 11, 9 };
    const verified_full_accept = [_]u32{ 10, 11, 12, 13 };
    const bad_prefix = [_]u32{ 10, 11 };

    try std.testing.expect(acceptedPrefixInvariantHolds(&draft, &verified_match_then_mismatch, 3));
    try std.testing.expect(acceptedPrefixInvariantHolds(&draft, &verified_full_accept, 4));
    try std.testing.expect(!acceptedPrefixInvariantHolds(&draft, &bad_prefix, 2));
}
