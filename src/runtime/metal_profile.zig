const std = @import("std");

pub const Category = enum(u8) {
    projections,
    attention,
    kv_writes,
    normalization,
    elementwise_ops,
    readback,
    cpu_sampling,

    pub fn label(self: Category) []const u8 {
        return switch (self) {
            .projections => "projections",
            .attention => "attention",
            .kv_writes => "kv_writes",
            .normalization => "normalization",
            .elementwise_ops => "elementwise_ops",
            .readback => "readback",
            .cpu_sampling => "cpu_sampling",
        };
    }
};

const category_count = @typeInfo(Category).@"enum".fields.len;

pub fn categoryCount() usize {
    return category_count;
}

pub const ShapeDesc = struct {
    rows: usize = 0,
    cols: usize = 0,
    depth: usize = 0,
    extra: usize = 0,
    tensor_type: u32 = 0,
};

pub const TokenBreakdown = struct {
    values: [category_count]u64 = [_]u64{0} ** category_count,

    fn add(self: *TokenBreakdown, category: Category, duration_ns: u64) void {
        self.values[@intFromEnum(category)] += duration_ns;
    }
};

const CategoryStats = struct {
    total_ns: u64 = 0,
    calls: usize = 0,
};

const ShapeKey = struct {
    category: Category,
    rows: usize,
    cols: usize,
    depth: usize,
    extra: usize,
    tensor_type: u32,
};

const ShapeStats = struct {
    total_ns: u64 = 0,
    calls: usize = 0,
};

const TopCategory = struct {
    category: Category,
    total_ns: u64,
};

const ShapeEntry = struct {
    key: ShapeKey,
    stats: ShapeStats,
};

pub const Profiler = struct {
    allocator: std.mem.Allocator,
    enabled: bool = false,
    categories: [category_count]CategoryStats = [_]CategoryStats{.{}} ** category_count,
    token_steps: std.ArrayList(TokenBreakdown),
    shapes: std.AutoHashMap(ShapeKey, ShapeStats),
    current_token: TokenBreakdown = .{},
    moon_quant_total_ns: u64 = 0,
    moon_quant_calls: usize = 0,
    moon_quant_token_steps: std.ArrayList(u64),
    moon_quant_shapes: std.AutoHashMap(ShapeKey, ShapeStats),
    current_moon_quant_token_ns: u64 = 0,
    token_active: bool = false,
    dropped_shape_samples: bool = false,
    dropped_moon_quant_shape_samples: bool = false,

    pub fn init(allocator: std.mem.Allocator, enabled: bool) Profiler {
        return .{
            .allocator = allocator,
            .enabled = enabled,
            .token_steps = .empty,
            .moon_quant_token_steps = .empty,
            .shapes = std.AutoHashMap(ShapeKey, ShapeStats).init(allocator),
            .moon_quant_shapes = std.AutoHashMap(ShapeKey, ShapeStats).init(allocator),
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.token_steps.deinit(self.allocator);
        self.moon_quant_token_steps.deinit(self.allocator);
        self.shapes.deinit();
        self.moon_quant_shapes.deinit();
        self.* = undefined;
    }

    pub fn beginDecodeToken(self: *Profiler) void {
        if (!self.enabled) return;
        self.current_token = .{};
        self.current_moon_quant_token_ns = 0;
        self.token_active = true;
    }

    pub fn endDecodeToken(self: *Profiler) void {
        if (!self.enabled or !self.token_active) return;
        self.token_steps.append(self.allocator, self.current_token) catch {};
        self.moon_quant_token_steps.append(self.allocator, self.current_moon_quant_token_ns) catch {};
        self.current_token = .{};
        self.current_moon_quant_token_ns = 0;
        self.token_active = false;
    }

    pub fn record(self: *Profiler, category: Category, duration_ns: u64) void {
        if (!self.enabled or !self.token_active) return;
        self.categories[@intFromEnum(category)].total_ns += duration_ns;
        self.categories[@intFromEnum(category)].calls += 1;
        self.current_token.add(category, duration_ns);
    }

    pub fn recordWithShape(
        self: *Profiler,
        category: Category,
        duration_ns: u64,
        shape: ShapeDesc,
    ) void {
        self.record(category, duration_ns);
        if (!self.enabled or !self.token_active) return;

        const key = ShapeKey{
            .category = category,
            .rows = shape.rows,
            .cols = shape.cols,
            .depth = shape.depth,
            .extra = shape.extra,
            .tensor_type = shape.tensor_type,
        };
        const entry = self.shapes.getOrPut(key) catch {
            self.dropped_shape_samples = true;
            return;
        };
        if (!entry.found_existing) entry.value_ptr.* = .{};
        entry.value_ptr.total_ns += duration_ns;
        entry.value_ptr.calls += 1;
    }

    pub fn recordMoonQuantProjection(self: *Profiler, duration_ns: u64, shape: ShapeDesc) void {
        if (!self.enabled or !self.token_active) return;
        self.moon_quant_total_ns += duration_ns;
        self.moon_quant_calls += 1;
        self.current_moon_quant_token_ns += duration_ns;

        const key = ShapeKey{
            .category = .projections,
            .rows = shape.rows,
            .cols = shape.cols,
            .depth = shape.depth,
            .extra = shape.extra,
            .tensor_type = shape.tensor_type,
        };
        const entry = self.moon_quant_shapes.getOrPut(key) catch {
            self.dropped_moon_quant_shape_samples = true;
            return;
        };
        if (!entry.found_existing) entry.value_ptr.* = .{};
        entry.value_ptr.total_ns += duration_ns;
        entry.value_ptr.calls += 1;
    }

    pub fn renderSummary(self: *const Profiler, allocator: std.mem.Allocator) ![]u8 {
        if (!self.enabled) return allocator.dupe(u8, "");

        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);

        const total_decode_ns = totalCategoryTime(&self.categories);
        try writer.print("metal_decode_profile_enabled=true\n", .{});
        try writer.print("profiled_decode_tokens={d}\n", .{self.token_steps.items.len});
        try writer.print("profiled_decode_ns={d}\n", .{total_decode_ns});
        try writer.print("moon_quant.profile.total_ns={d}\n", .{self.moon_quant_total_ns});
        try writer.print("moon_quant.profile.calls={d}\n", .{self.moon_quant_calls});
        try writer.print(
            "moon_quant.profile.share_of_decode_pct={d:.3}\n",
            .{percentage(self.moon_quant_total_ns, total_decode_ns)},
        );

        for (std.enums.values(Category)) |category| {
            const stats = self.categories[@intFromEnum(category)];
            const share = percentage(stats.total_ns, total_decode_ns);
            try writer.print(
                "profile.{s}.ns={d}\nprofile.{s}.share_pct={d:.3}\nprofile.{s}.calls={d}\n",
                .{
                    category.label(),
                    stats.total_ns,
                    category.label(),
                    share,
                    category.label(),
                    stats.calls,
                },
            );
        }

        const top_categories = topCategoryBreakdown(self.categories);
        for (top_categories, 0..) |entry, index| {
            const share = percentage(entry.total_ns, total_decode_ns);
            try writer.print(
                "profile.top_bottleneck_{d}={s}:{d}ns:{d:.3}pct\n",
                .{ index + 1, entry.category.label(), entry.total_ns, share },
            );
        }

        var shape_entries = try collectShapes(allocator, self.shapes);
        defer allocator.free(shape_entries);
        std.mem.sort(ShapeEntry, shape_entries, {}, lessShapeEntry);
        const shape_limit = @min(@as(usize, 8), shape_entries.len);
        for (shape_entries[0..shape_limit], 0..) |entry, index| {
            try writer.print(
                "profile.shape_{d}={s}:rows={d}:cols={d}:depth={d}:extra={d}:tensor_type={d}:calls={d}:ns={d}\n",
                .{
                    index + 1,
                    entry.key.category.label(),
                    entry.key.rows,
                    entry.key.cols,
                    entry.key.depth,
                    entry.key.extra,
                    entry.key.tensor_type,
                    entry.stats.calls,
                    entry.stats.total_ns,
                },
            );
        }
        if (self.dropped_shape_samples) {
            try writer.print("profile.shape_samples_dropped=true\n", .{});
        }

        var moon_quant_shape_entries = try collectShapes(allocator, self.moon_quant_shapes);
        defer allocator.free(moon_quant_shape_entries);
        std.mem.sort(ShapeEntry, moon_quant_shape_entries, {}, lessShapeEntry);
        const moon_quant_shape_limit = @min(@as(usize, 4), moon_quant_shape_entries.len);
        for (moon_quant_shape_entries[0..moon_quant_shape_limit], 0..) |entry, index| {
            try writer.print(
                "moon_quant.profile.shape_{d}=rows={d}:cols={d}:depth={d}:extra={d}:tensor_type={d}:calls={d}:ns={d}\n",
                .{
                    index + 1,
                    entry.key.rows,
                    entry.key.cols,
                    entry.key.depth,
                    entry.key.extra,
                    entry.key.tensor_type,
                    entry.stats.calls,
                    entry.stats.total_ns,
                },
            );
        }
        if (self.dropped_moon_quant_shape_samples) {
            try writer.print("moon_quant.profile.shape_samples_dropped=true\n", .{});
        }

        for (self.token_steps.items, 0..) |token, index| {
            try writer.print("profile.token_{d}.total_ns={d}\n", .{ index, totalTokenTime(token) });
            for (std.enums.values(Category)) |category| {
                try writer.print(
                    "profile.token_{d}.{s}.ns={d}\n",
                    .{ index, category.label(), token.values[@intFromEnum(category)] },
                );
            }
            try writer.print(
                "moon_quant.profile.token_{d}.ns={d}\n",
                .{ index, self.moon_quant_token_steps.items[index] },
            );
        }

        return buffer.toOwnedSlice(allocator);
    }
};

fn totalCategoryTime(categories: *const [category_count]CategoryStats) u64 {
    var total: u64 = 0;
    for (categories) |stats| total += stats.total_ns;
    return total;
}

fn totalTokenTime(token: TokenBreakdown) u64 {
    var total: u64 = 0;
    for (token.values) |value| total += value;
    return total;
}

fn percentage(part: u64, whole: u64) f64 {
    if (whole == 0) return 0;
    return (@as(f64, @floatFromInt(part)) * 100.0) / @as(f64, @floatFromInt(whole));
}

fn topCategoryBreakdown(categories: [category_count]CategoryStats) [3]TopCategory {
    var top = [_]TopCategory{
        .{ .category = .projections, .total_ns = 0 },
        .{ .category = .projections, .total_ns = 0 },
        .{ .category = .projections, .total_ns = 0 },
    };

    for (std.enums.values(Category)) |category| {
        const entry = TopCategory{
            .category = category,
            .total_ns = categories[@intFromEnum(category)].total_ns,
        };
        for (0..top.len) |index| {
            if (entry.total_ns > top[index].total_ns) {
                var shift = top.len - 1;
                while (shift > index) : (shift -= 1) {
                    top[shift] = top[shift - 1];
                }
                top[index] = entry;
                break;
            }
        }
    }

    return top;
}

fn collectShapes(
    allocator: std.mem.Allocator,
    shapes: std.AutoHashMap(ShapeKey, ShapeStats),
) ![]ShapeEntry {
    var entries = std.ArrayList(ShapeEntry).empty;
    errdefer entries.deinit(allocator);

    var iterator = shapes.iterator();
    while (iterator.next()) |entry| {
        try entries.append(allocator, .{
            .key = entry.key_ptr.*,
            .stats = entry.value_ptr.*,
        });
    }

    return entries.toOwnedSlice(allocator);
}

fn lessShapeEntry(_: void, lhs: ShapeEntry, rhs: ShapeEntry) bool {
    if (lhs.stats.total_ns == rhs.stats.total_ns) return lhs.stats.calls > rhs.stats.calls;
    return lhs.stats.total_ns > rhs.stats.total_ns;
}

test "profiler summary exposes top bottlenecks and shapes" {
    var profiler = Profiler.init(std.testing.allocator, true);
    defer profiler.deinit();

    profiler.beginDecodeToken();
    profiler.recordWithShape(.projections, 200, .{ .rows = 64, .cols = 128, .tensor_type = 12 });
    profiler.record(.attention, 90);
    profiler.recordWithShape(.kv_writes, 30, .{ .rows = 1, .cols = 128, .depth = 2 });
    profiler.record(.cpu_sampling, 10);
    profiler.recordMoonQuantProjection(150, .{ .rows = 64, .cols = 128, .tensor_type = 12 });
    profiler.endDecodeToken();

    const summary = try profiler.renderSummary(std.testing.allocator);
    defer std.testing.allocator.free(summary);

    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.top_bottleneck_1=projections:200ns") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.shape_1=projections:rows=64:cols=128") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.token_0.cpu_sampling.ns=10") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "moon_quant.profile.total_ns=150") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "moon_quant.profile.token_0.ns=150") != null);
}
