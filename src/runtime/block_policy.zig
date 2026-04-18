const std = @import("std");

// Conservative default: only speculate when the top-1 token is clearly separated.
pub const default_confidence_margin_threshold: f32 = 0.75;
pub const cooldown_draft_cap: usize = 1;
const rollback_ema_unstable: f64 = 0.45;
const accepted_ema_high: f64 = 1.8;
const high_confidence_multiplier: f32 = 2.0;
const medium_confidence_multiplier: f32 = 1.25;

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
) usize {
    if (exp_block_cap == 0) return 0;
    if (confidence_margin < confidence_threshold) return 0;
    if (exp_block_cap == 1) return 1;

    if (recent_rollback_ema >= rollback_ema_unstable) {
        return @min(exp_block_cap, @as(usize, 1));
    }

    const high_confidence_threshold = confidence_threshold * high_confidence_multiplier;
    if (exp_block_cap >= 4 and confidence_margin >= high_confidence_threshold and recent_accepted_ema >= accepted_ema_high) {
        return @min(exp_block_cap, @as(usize, 4));
    }

    const medium_confidence_threshold = confidence_threshold * medium_confidence_multiplier;
    if (confidence_margin >= medium_confidence_threshold) {
        return @min(exp_block_cap, @as(usize, 2));
    }

    return 1;
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
    try std.testing.expectEqual(@as(usize, 0), selectAdaptiveDraftLimit(4, 0.6, 0.75, 2.0, 0.1));
    try std.testing.expectEqual(@as(usize, 2), selectAdaptiveDraftLimit(4, 1.0, 0.75, 1.6, 0.1));
    try std.testing.expectEqual(@as(usize, 4), selectAdaptiveDraftLimit(4, 1.8, 0.75, 2.2, 0.1));
    try std.testing.expectEqual(@as(usize, 1), selectAdaptiveDraftLimit(4, 1.8, 0.75, 2.2, 0.5));
}
