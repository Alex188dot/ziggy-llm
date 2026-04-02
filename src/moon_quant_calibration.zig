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

const Accumulator = struct {
    name: []u8,
    role: Role,
    rows: usize,
    cols: usize,
    current_format: moon_quant.TargetFormat,
    samples: usize = 0,
    total_mean_square: f64 = 0,
    peak_abs: f32 = 0,

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
    q4_k_m_count: usize = 0,
    q5_k_m_count: usize = 0,
    q6_k_count: usize = 0,
    q8_0_count: usize = 0,

    pub fn deinit(self: *Plan, allocator: std.mem.Allocator) void {
        for (self.entries) |*entry| entry.deinit(allocator);
        allocator.free(self.entries);
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
        }

        std.mem.sort(Entry, entries, {}, struct {
            fn lessThan(_: void, lhs: Entry, rhs: Entry) bool {
                return std.mem.lessThan(u8, lhs.name, rhs.name);
            }
        }.lessThan);

        return .{
            .entries = entries,
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
