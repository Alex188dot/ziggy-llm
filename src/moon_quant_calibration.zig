const std = @import("std");
const moon_quant = @import("moon_quant.zig");

pub const Role = enum {
    attn_q,
    attn_k,
    attn_v,
    attn_output,
    ffn_gate,
    ffn_up,
    ffn_down,
    output,
};

pub const Observation = struct {
    name: []const u8,
    role: Role,
    rows: usize,
    cols: usize,
    current_format: moon_quant.TargetFormat,
    values: []const f32,
};

pub const gated_ffn_threshold_candidates = [_]f32{ 0.015, 0.025, 0.04, 0.06 };

pub const GatedFfnPolicy = struct {
    layer_index: u32,
    tensor_name: []u8,
    threshold: f32,
    active_block_ratio: f32,
    avg_active_blocks: f32,
    avg_total_blocks: f32,
    selected: bool,

    pub fn deinit(self: *GatedFfnPolicy, allocator: std.mem.Allocator) void {
        allocator.free(self.tensor_name);
        self.* = undefined;
    }
};

const Accumulator = struct {
    name: []u8,
    role: Role,
    rows: usize,
    cols: usize,
    current_format: moon_quant.TargetFormat,
    samples: usize = 0,
    total_mean_square: f64 = 0,
    peak_abs: f32 = 0,
    gate_total_blocks: u64 = 0,
    gate_active_blocks_by_threshold: [gated_ffn_threshold_candidates.len]u64 = .{0} ** gated_ffn_threshold_candidates.len,
    gate_peak_active_ratio_by_threshold: [gated_ffn_threshold_candidates.len]f32 = .{0} ** gated_ffn_threshold_candidates.len,

    fn meanSquare(self: Accumulator) f64 {
        if (self.samples == 0) return 0;
        return self.total_mean_square / @as(f64, @floatFromInt(self.samples));
    }
};

pub const Entry = struct {
    name: []u8,
    role: Role,
    rows: usize,
    cols: usize,
    current_format: moon_quant.TargetFormat,
    target_format: moon_quant.TargetFormat,
    samples: usize,
    mean_square: f64,
    peak_abs: f32,
    importance_score: f64,

    pub fn deinit(self: *Entry, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.* = undefined;
    }
};

pub const Plan = struct {
    entries: []Entry,
    gated_ffn_policies: []GatedFfnPolicy = &.{},
    q4_k_m_count: usize = 0,
    q5_k_m_count: usize = 0,
    q6_k_count: usize = 0,
    q8_0_count: usize = 0,

    pub fn deinit(self: *Plan, allocator: std.mem.Allocator) void {
        for (self.entries) |*entry| entry.deinit(allocator);
        allocator.free(self.entries);
        for (self.gated_ffn_policies) |*policy| policy.deinit(allocator);
        allocator.free(self.gated_ffn_policies);
        self.* = undefined;
    }
};

pub const Calibrator = struct {
    allocator: std.mem.Allocator,
    accumulators: std.StringHashMap(Accumulator),

    pub fn init(allocator: std.mem.Allocator) Calibrator {
        return .{
            .allocator = allocator,
            .accumulators = std.StringHashMap(Accumulator).init(allocator),
        };
    }

    pub fn deinit(self: *Calibrator) void {
        var iterator = self.accumulators.valueIterator();
        while (iterator.next()) |accumulator| self.allocator.free(accumulator.name);
        self.accumulators.deinit();
        self.* = undefined;
    }

    pub fn observe(self: *Calibrator, observation: Observation) !void {
        if (observation.values.len == 0) return;

        const accumulator = try self.ensureAccumulator(observation);
        var sum_sq: f64 = 0;
        var peak_abs = accumulator.peak_abs;
        for (observation.values) |value| {
            peak_abs = @max(peak_abs, @abs(value));
            sum_sq += @as(f64, value) * @as(f64, value);
        }

        accumulator.samples += 1;
        accumulator.total_mean_square += sum_sq / @as(f64, @floatFromInt(observation.values.len));
        accumulator.peak_abs = peak_abs;
        if (observation.role == .ffn_down) {
            observeGatedFfnBlocks(accumulator, observation.values);
        }
    }

    pub fn buildPlan(self: *Calibrator, allocator: std.mem.Allocator) !Plan {
        var entries = try allocator.alloc(Entry, self.accumulators.count());
        errdefer allocator.free(entries);

        var iterator = self.accumulators.valueIterator();
        var index: usize = 0;
        var q4_k_m_count: usize = 0;
        var q5_k_m_count: usize = 0;
        var q6_k_count: usize = 0;
        var q8_0_count: usize = 0;
        var gated_ffn_policy_list = std.ArrayList(GatedFfnPolicy).empty;
        defer gated_ffn_policy_list.deinit(allocator);

        while (iterator.next()) |accumulator| : (index += 1) {
            const importance_score = scoreImportance(accumulator.*);
            const target_format = selectTargetFormat(accumulator.*, importance_score);
            entries[index] = .{
                .name = try allocator.dupe(u8, accumulator.name),
                .role = accumulator.role,
                .rows = accumulator.rows,
                .cols = accumulator.cols,
                .current_format = accumulator.current_format,
                .target_format = target_format,
                .samples = accumulator.samples,
                .mean_square = accumulator.meanSquare(),
                .peak_abs = accumulator.peak_abs,
                .importance_score = importance_score,
            };
            switch (target_format) {
                .q4_k_m => q4_k_m_count += 1,
                .q5_k_m => q5_k_m_count += 1,
                .q6_k => q6_k_count += 1,
                .q8_0 => q8_0_count += 1,
                else => {},
            }
            if (try buildGatedFfnPolicy(allocator, accumulator.*)) |policy| {
                try gated_ffn_policy_list.append(allocator, policy);
            }
        }

        std.mem.sort(Entry, entries, {}, struct {
            fn lessThan(_: void, lhs: Entry, rhs: Entry) bool {
                return std.mem.lessThan(u8, lhs.name, rhs.name);
            }
        }.lessThan);

        return .{
            .entries = entries,
            .gated_ffn_policies = try gated_ffn_policy_list.toOwnedSlice(allocator),
            .q4_k_m_count = q4_k_m_count,
            .q5_k_m_count = q5_k_m_count,
            .q6_k_count = q6_k_count,
            .q8_0_count = q8_0_count,
        };
    }

    fn ensureAccumulator(self: *Calibrator, observation: Observation) !*Accumulator {
        const result = try self.accumulators.getOrPut(observation.name);
        if (!result.found_existing) {
            result.key_ptr.* = try self.allocator.dupe(u8, observation.name);
            result.value_ptr.* = .{
                .name = try self.allocator.dupe(u8, observation.name),
                .role = observation.role,
                .rows = observation.rows,
                .cols = observation.cols,
                .current_format = observation.current_format,
            };
        }
        return result.value_ptr;
    }
};

fn observeGatedFfnBlocks(accumulator: *Accumulator, values: []const f32) void {
    const block_size = moon_quant.q4_k_block_values;
    if (values.len == 0 or values.len % block_size != 0) return;

    const block_count = values.len / block_size;
    accumulator.gate_total_blocks += block_count;
    for (gated_ffn_threshold_candidates, 0..) |threshold, threshold_index| {
        var active_blocks: u64 = 0;
        for (0..block_count) |block_index| {
            const start = block_index * block_size;
            const block = values[start .. start + block_size];
            if (blockHasActivationAboveThreshold(block, threshold)) active_blocks += 1;
        }
        accumulator.gate_active_blocks_by_threshold[threshold_index] += active_blocks;
        const active_ratio = if (block_count == 0)
            1.0
        else
            @as(f32, @floatFromInt(active_blocks)) / @as(f32, @floatFromInt(block_count));
        accumulator.gate_peak_active_ratio_by_threshold[threshold_index] =
            @max(accumulator.gate_peak_active_ratio_by_threshold[threshold_index], active_ratio);
    }
}

fn blockHasActivationAboveThreshold(block: []const f32, threshold: f32) bool {
    for (block) |value| {
        if (@abs(value) > threshold) return true;
    }
    return false;
}

fn buildGatedFfnPolicy(allocator: std.mem.Allocator, accumulator: Accumulator) !?GatedFfnPolicy {
    if (accumulator.role != .ffn_down) return null;
    const layer_index = extractLayerIndex(accumulator.name) orelse return null;
    if (accumulator.gate_total_blocks == 0 or accumulator.samples == 0) return null;

    var best_index: ?usize = null;
    var best_skip_ratio: f32 = 0;
    for (gated_ffn_threshold_candidates, 0..) |threshold, threshold_index| {
        _ = threshold;
        const active_ratio = @as(f32, @floatFromInt(accumulator.gate_active_blocks_by_threshold[threshold_index])) /
            @as(f32, @floatFromInt(accumulator.gate_total_blocks));
        const skip_ratio = 1.0 - active_ratio;
        const peak_ratio = accumulator.gate_peak_active_ratio_by_threshold[threshold_index];
        if (skip_ratio < 0.12) continue;
        if (peak_ratio > 0.94) continue;
        if (best_index == null or skip_ratio > best_skip_ratio) {
            best_index = threshold_index;
            best_skip_ratio = skip_ratio;
        }
    }

    if (best_index == null) return null;
    const chosen_index = best_index.?;
    const avg_total_blocks = @as(f32, @floatFromInt(accumulator.gate_total_blocks)) /
        @as(f32, @floatFromInt(accumulator.samples));
    const avg_active_blocks = @as(f32, @floatFromInt(accumulator.gate_active_blocks_by_threshold[chosen_index])) /
        @as(f32, @floatFromInt(accumulator.samples));
    const active_block_ratio = @as(f32, @floatFromInt(accumulator.gate_active_blocks_by_threshold[chosen_index])) /
        @as(f32, @floatFromInt(accumulator.gate_total_blocks));

    return .{
        .layer_index = layer_index,
        .tensor_name = try allocator.dupe(u8, accumulator.name),
        .threshold = gated_ffn_threshold_candidates[chosen_index],
        .active_block_ratio = active_block_ratio,
        .avg_active_blocks = avg_active_blocks,
        .avg_total_blocks = avg_total_blocks,
        .selected = true,
    };
}

fn extractLayerIndex(name: []const u8) ?u32 {
    const prefix = "blk.";
    if (std.mem.indexOf(u8, name, prefix)) |start| {
        const num_start = start + prefix.len;
        var end = num_start;
        while (end < name.len and std.ascii.isDigit(name[end])) : (end += 1) {}
        if (end > num_start) {
            return std.fmt.parseInt(u32, name[num_start..end], 10) catch null;
        }
    }
    return null;
}

fn scoreImportance(accumulator: Accumulator) f64 {
    const mean_square = accumulator.meanSquare();
    const size_scale = std.math.log2(@as(f64, @floatFromInt(@max(accumulator.cols, 2))));
    return mean_square * roleWeight(accumulator.role) * (1.0 + size_scale * 0.08) +
        @as(f64, accumulator.peak_abs) * 0.05;
}

fn selectTargetFormat(accumulator: Accumulator, importance_score: f64) moon_quant.TargetFormat {
    const minimum_rank: u8 = switch (accumulator.current_format) {
        .q8_0 => 3,
        .q6_k, .legacy_q6_k => 2,
        .q5_k_m => 1,
        else => 0,
    };
    const desired_rank: u8 = if (importance_score >= 0.75)
        3
    else if (importance_score >= 0.30)
        2
    else if (importance_score >= 0.12)
        1
    else
        0;

    return switch (@max(minimum_rank, desired_rank)) {
        0 => .q4_k_m,
        1 => .q5_k_m,
        2 => .q6_k,
        else => .q8_0,
    };
}

fn roleWeight(role: Role) f64 {
    return switch (role) {
        .attn_q, .attn_k, .attn_v => 1.00,
        .attn_output, .ffn_down, .output => 1.35,
        .ffn_gate, .ffn_up => 0.95,
    };
}

test "calibrator produces non-uniform MoonQuant plan" {
    var calibrator = Calibrator.init(std.testing.allocator);
    defer calibrator.deinit();

    const low = [_]f32{ 0.05, -0.03, 0.02, -0.01 };
    const medium = [_]f32{ 0.45, -0.35, 0.25, -0.20 };
    const high = [_]f32{ 1.3, -1.1, 0.9, -0.8 };

    try calibrator.observe(.{
        .name = "blk.0.attn_q.weight",
        .role = .attn_q,
        .rows = 2048,
        .cols = 2048,
        .current_format = .legacy_q4_k,
        .values = &low,
    });
    try calibrator.observe(.{
        .name = "blk.0.ffn_down.weight",
        .role = .ffn_down,
        .rows = 2048,
        .cols = 5632,
        .current_format = .legacy_q4_k,
        .values = &medium,
    });
    try calibrator.observe(.{
        .name = "output.weight",
        .role = .output,
        .rows = 32000,
        .cols = 2048,
        .current_format = .legacy_q6_k,
        .values = &high,
    });

    var plan = try calibrator.buildPlan(std.testing.allocator);
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), plan.q4_k_m_count);
    try std.testing.expectEqual(@as(usize, 1), plan.q6_k_count);
    try std.testing.expectEqual(@as(usize, 1), plan.q8_0_count);
}
