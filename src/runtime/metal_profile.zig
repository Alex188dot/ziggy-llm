const std = @import("std");

pub const Category = enum(u8) {
    projection_dense,
    projection_quantized,
    projection_add_dense,
    projection_add_quantized,
    attention,
    kv_writes,
    normalization,
    rope,
    ffn_activation,
    bias_add,
    output_reduce,
    commit_wait,
    host_readback,
    cpu_sampling,

    pub fn label(self: Category) []const u8 {
        return switch (self) {
            .projection_dense => "projection_dense",
            .projection_quantized => "projection_quantized",
            .projection_add_dense => "projection_add_dense",
            .projection_add_quantized => "projection_add_quantized",
            .attention => "attention",
            .kv_writes => "kv_writes",
            .normalization => "normalization",
            .rope => "rope",
            .ffn_activation => "ffn_activation",
            .bias_add => "bias_add",
            .output_reduce => "output_reduce",
            .commit_wait => "commit_wait",
            .host_readback => "host_readback",
            .cpu_sampling => "cpu_sampling",
        };
    }
};

const category_count = @typeInfo(Category).@"enum".fields.len;

pub fn categoryCount() usize {
    return category_count;
}

pub const SummaryStats = struct {
    enabled: bool = false,
    profiled_decode_tokens: usize = 0,
    profiled_decode_ns: u64 = 0,
    commit_wait_gpu_ns: u64 = 0,
    commit_wait_non_gpu_ns: u64 = 0,
    command_buffer_count: u64 = 0,
    encoder_count: u64 = 0,
    dispatch_count: u64 = 0,
    categories_ns: [category_count]u64 = [_]u64{0} ** category_count,
    categories_calls: [category_count]usize = [_]usize{0} ** category_count,

    pub fn add(self: *SummaryStats, other: SummaryStats) void {
        if (!other.enabled) return;
        self.enabled = true;
        self.profiled_decode_tokens += other.profiled_decode_tokens;
        self.profiled_decode_ns += other.profiled_decode_ns;
        self.commit_wait_gpu_ns += other.commit_wait_gpu_ns;
        self.commit_wait_non_gpu_ns += other.commit_wait_non_gpu_ns;
        self.command_buffer_count += other.command_buffer_count;
        self.encoder_count += other.encoder_count;
        self.dispatch_count += other.dispatch_count;
        for (0..category_count) |index| {
            self.categories_ns[index] += other.categories_ns[index];
            self.categories_calls[index] += other.categories_calls[index];
        }
    }

    pub fn average(self: SummaryStats, runs: usize) SummaryStats {
        if (!self.enabled or runs == 0) return .{};

        var averaged = self;
        averaged.profiled_decode_tokens /= runs;
        averaged.profiled_decode_ns /= runs;
        averaged.commit_wait_gpu_ns /= runs;
        averaged.commit_wait_non_gpu_ns /= runs;
        averaged.command_buffer_count /= runs;
        averaged.encoder_count /= runs;
        averaged.dispatch_count /= runs;
        for (0..category_count) |index| {
            averaged.categories_ns[index] /= runs;
            averaged.categories_calls[index] /= runs;
        }
        return averaged;
    }
};

pub const ShapeDesc = struct {
    rows: usize = 0,
    cols: usize = 0,
    depth: usize = 0,
    extra: usize = 0,
    tensor_type: u32 = 0,
    layout_path: LayoutPath = .none,
};

pub const LayoutPath = enum(u8) {
    none,
    moonq_q4_k,
    packed_q6_k,
    raw_q6_k,

    pub fn label(self: LayoutPath) []const u8 {
        return switch (self) {
            .none => "none",
            .moonq_q4_k => "moonq_q4_k",
            .packed_q6_k => "packed_q6_k",
            .raw_q6_k => "raw_q6_k",
        };
    }
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
    layout_path: LayoutPath,
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
    commit_wait_gpu_total_ns: u64 = 0,
    commit_wait_gpu_token_steps: std.ArrayList(u64),
    current_commit_wait_gpu_ns: u64 = 0,
    commit_wait_non_gpu_total_ns: u64 = 0,
    commit_wait_non_gpu_token_steps: std.ArrayList(u64),
    current_commit_wait_non_gpu_ns: u64 = 0,
    total_command_buffer_count: u64 = 0,
    command_buffer_token_steps: std.ArrayList(u32),
    current_command_buffer_count: u32 = 0,
    total_encoder_count: u64 = 0,
    encoder_token_steps: std.ArrayList(u32),
    current_encoder_count: u32 = 0,
    total_dispatch_count: u64 = 0,
    dispatch_token_steps: std.ArrayList(u32),
    current_dispatch_count: u32 = 0,
    token_active: bool = false,
    dropped_shape_samples: bool = false,
    dropped_moon_quant_shape_samples: bool = false,

    pub fn init(allocator: std.mem.Allocator, enabled: bool) Profiler {
        return .{
            .allocator = allocator,
            .enabled = enabled,
            .token_steps = .empty,
            .moon_quant_token_steps = .empty,
            .commit_wait_gpu_token_steps = .empty,
            .commit_wait_non_gpu_token_steps = .empty,
            .command_buffer_token_steps = .empty,
            .encoder_token_steps = .empty,
            .dispatch_token_steps = .empty,
            .shapes = std.AutoHashMap(ShapeKey, ShapeStats).init(allocator),
            .moon_quant_shapes = std.AutoHashMap(ShapeKey, ShapeStats).init(allocator),
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.token_steps.deinit(self.allocator);
        self.moon_quant_token_steps.deinit(self.allocator);
        self.commit_wait_gpu_token_steps.deinit(self.allocator);
        self.commit_wait_non_gpu_token_steps.deinit(self.allocator);
        self.command_buffer_token_steps.deinit(self.allocator);
        self.encoder_token_steps.deinit(self.allocator);
        self.dispatch_token_steps.deinit(self.allocator);
        self.shapes.deinit();
        self.moon_quant_shapes.deinit();
        self.* = undefined;
    }

    pub fn beginDecodeToken(self: *Profiler) void {
        if (!self.enabled) return;
        self.current_token = .{};
        self.current_moon_quant_token_ns = 0;
        self.current_commit_wait_gpu_ns = 0;
        self.current_commit_wait_non_gpu_ns = 0;
        self.current_command_buffer_count = 0;
        self.current_encoder_count = 0;
        self.current_dispatch_count = 0;
        self.token_active = true;
    }

    pub fn endDecodeToken(self: *Profiler) void {
        if (!self.enabled or !self.token_active) return;
        self.token_steps.append(self.allocator, self.current_token) catch {};
        self.moon_quant_token_steps.append(self.allocator, self.current_moon_quant_token_ns) catch {};
        self.commit_wait_gpu_token_steps.append(self.allocator, self.current_commit_wait_gpu_ns) catch {};
        self.commit_wait_non_gpu_token_steps.append(self.allocator, self.current_commit_wait_non_gpu_ns) catch {};
        self.command_buffer_token_steps.append(self.allocator, self.current_command_buffer_count) catch {};
        self.encoder_token_steps.append(self.allocator, self.current_encoder_count) catch {};
        self.dispatch_token_steps.append(self.allocator, self.current_dispatch_count) catch {};
        self.current_token = .{};
        self.current_moon_quant_token_ns = 0;
        self.current_commit_wait_gpu_ns = 0;
        self.current_commit_wait_non_gpu_ns = 0;
        self.current_command_buffer_count = 0;
        self.current_encoder_count = 0;
        self.current_dispatch_count = 0;
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
            .layout_path = shape.layout_path,
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
            .category = .projection_quantized,
            .rows = shape.rows,
            .cols = shape.cols,
            .depth = shape.depth,
            .extra = shape.extra,
            .tensor_type = shape.tensor_type,
            .layout_path = shape.layout_path,
        };
        const entry = self.moon_quant_shapes.getOrPut(key) catch {
            self.dropped_moon_quant_shape_samples = true;
            return;
        };
        if (!entry.found_existing) entry.value_ptr.* = .{};
        entry.value_ptr.total_ns += duration_ns;
        entry.value_ptr.calls += 1;
    }

    pub fn recordCommitWaitGpu(self: *Profiler, duration_ns: u64) void {
        if (!self.enabled or !self.token_active or duration_ns == 0) return;
        self.commit_wait_gpu_total_ns += duration_ns;
        self.current_commit_wait_gpu_ns += duration_ns;
    }

    pub fn recordCommitWaitNonGpu(self: *Profiler, duration_ns: u64) void {
        if (!self.enabled or !self.token_active or duration_ns == 0) return;
        self.commit_wait_non_gpu_total_ns += duration_ns;
        self.current_commit_wait_non_gpu_ns += duration_ns;
    }

    pub fn recordCommandBuffers(self: *Profiler, count: u32) void {
        if (!self.enabled or !self.token_active or count == 0) return;
        self.total_command_buffer_count += count;
        self.current_command_buffer_count += count;
    }

    pub fn recordEncoders(self: *Profiler, count: u32) void {
        if (!self.enabled or !self.token_active or count == 0) return;
        self.total_encoder_count += count;
        self.current_encoder_count += count;
    }

    pub fn recordDispatches(self: *Profiler, dispatch_count: u32) void {
        if (!self.enabled or !self.token_active or dispatch_count == 0) return;
        self.total_dispatch_count += dispatch_count;
        self.current_dispatch_count += dispatch_count;
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
        try writer.print("profile.command_buffers={d}\n", .{self.total_command_buffer_count});
        try writer.print("profile.encoders={d}\n", .{self.total_encoder_count});
        try writer.print("profile.dispatches={d}\n", .{self.total_dispatch_count});
        try writer.print("moon_quant.profile.total_ns={d}\n", .{self.moon_quant_total_ns});
        try writer.print("moon_quant.profile.calls={d}\n", .{self.moon_quant_calls});
        try writer.print(
            "moon_quant.profile.share_of_decode_pct={d:.3}\n",
            .{percentage(self.moon_quant_total_ns, total_decode_ns)},
        );
        try writer.print(
            "profile.commit_wait.gpu_ns={d}\nprofile.commit_wait.gpu_share_of_decode_pct={d:.3}\nprofile.commit_wait.non_gpu_ns={d}\nprofile.commit_wait.non_gpu_share_of_decode_pct={d:.3}\n",
            .{
                self.commit_wait_gpu_total_ns,
                percentage(self.commit_wait_gpu_total_ns, total_decode_ns),
                self.commit_wait_non_gpu_total_ns,
                percentage(self.commit_wait_non_gpu_total_ns, total_decode_ns),
            },
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
            try writer.print(
                "profile.shape_{d}.layout_path={s}\n",
                .{ index + 1, entry.key.layout_path.label() },
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
            try writer.print(
                "moon_quant.profile.shape_{d}.layout_path={s}\n",
                .{ index + 1, entry.key.layout_path.label() },
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
            try writer.print(
                "profile.token_{d}.commit_wait.gpu_ns={d}\n",
                .{ index, self.commit_wait_gpu_token_steps.items[index] },
            );
            try writer.print(
                "profile.token_{d}.commit_wait.non_gpu_ns={d}\n",
                .{ index, self.commit_wait_non_gpu_token_steps.items[index] },
            );
            try writer.print(
                "profile.token_{d}.command_buffers={d}\n",
                .{ index, self.command_buffer_token_steps.items[index] },
            );
            try writer.print(
                "profile.token_{d}.encoders={d}\n",
                .{ index, self.encoder_token_steps.items[index] },
            );
            try writer.print(
                "profile.token_{d}.dispatches={d}\n",
                .{ index, self.dispatch_token_steps.items[index] },
            );
        }

        return buffer.toOwnedSlice(allocator);
    }
};

pub fn parseSummary(summary: ?[]const u8) SummaryStats {
    const text = summary orelse return .{};
    var stats = SummaryStats{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (std.mem.eql(u8, line, "metal_decode_profile_enabled=true")) {
            stats.enabled = true;
            continue;
        }
        if (parseMetricUsize(line, "profiled_decode_tokens=")) |value| {
            stats.profiled_decode_tokens = value;
            continue;
        }
        if (parseMetricU64(line, "profiled_decode_ns=")) |value| {
            stats.profiled_decode_ns = value;
            continue;
        }
        if (parseMetricU64(line, "profile.commit_wait.gpu_ns=")) |value| {
            stats.commit_wait_gpu_ns = value;
            continue;
        }
        if (parseMetricU64(line, "profile.commit_wait.non_gpu_ns=")) |value| {
            stats.commit_wait_non_gpu_ns = value;
            continue;
        }
        if (parseMetricU64(line, "profile.command_buffers=")) |value| {
            stats.command_buffer_count = value;
            continue;
        }
        if (parseMetricU64(line, "profile.encoders=")) |value| {
            stats.encoder_count = value;
            continue;
        }
        if (parseMetricU64(line, "profile.dispatches=")) |value| {
            stats.dispatch_count = value;
            continue;
        }
        if (parseCategoryMetricU64(line, ".ns=")) |entry| {
            stats.categories_ns[@intFromEnum(entry.category)] = entry.value;
            continue;
        }
        if (parseCategoryMetricUsize(line, ".calls=")) |entry| {
            stats.categories_calls[@intFromEnum(entry.category)] = entry.value;
            continue;
        }
    }
    return stats;
}

pub fn renderBenchStageSummary(
    allocator: std.mem.Allocator,
    prefix: []const u8,
    stats: SummaryStats,
    generated_token_count: usize,
) !?[]u8 {
    if (!stats.enabled) return null;

    var buffer = std.ArrayList(u8).empty;
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    try writer.print("{s}.decode_profile.steps={d}\n", .{ prefix, stats.profiled_decode_tokens });
    try writer.print("{s}.decode_profile.generated_tokens={d}\n", .{ prefix, generated_token_count });
    try writer.print("{s}.decode_profile.total_ms={d:.3}\n", .{ prefix, nsToMs(stats.profiled_decode_ns) });
    try writer.print("{s}.decode_profile.commit_wait_gpu_ms={d:.3}\n", .{ prefix, nsToMs(stats.commit_wait_gpu_ns) });
    try writer.print("{s}.decode_profile.commit_wait_non_gpu_ms={d:.3}\n", .{ prefix, nsToMs(stats.commit_wait_non_gpu_ns) });
    try writer.print("{s}.decode_profile.command_buffers={d}\n", .{ prefix, stats.command_buffer_count });
    try writer.print("{s}.decode_profile.encoders={d}\n", .{ prefix, stats.encoder_count });
    try writer.print(
        "{s}.decode_profile.command_buffers_per_token={d:.3}\n",
        .{ prefix, if (generated_token_count == 0) @as(f64, 0) else @as(f64, @floatFromInt(stats.command_buffer_count)) / @as(f64, @floatFromInt(generated_token_count)) },
    );
    try writer.print(
        "{s}.decode_profile.encoders_per_token={d:.3}\n",
        .{ prefix, if (generated_token_count == 0) @as(f64, 0) else @as(f64, @floatFromInt(stats.encoder_count)) / @as(f64, @floatFromInt(generated_token_count)) },
    );
    try writer.print("{s}.decode_profile.dispatches={d}\n", .{ prefix, stats.dispatch_count });
    try writer.print(
        "{s}.decode_profile.dispatches_per_token={d:.3}\n",
        .{ prefix, if (generated_token_count == 0) @as(f64, 0) else @as(f64, @floatFromInt(stats.dispatch_count)) / @as(f64, @floatFromInt(generated_token_count)) },
    );
    try writer.print(
        "{s}.decode_profile.commit_wait_gpu_share_pct={d:.3}\n",
        .{ prefix, percentage(stats.commit_wait_gpu_ns, stats.profiled_decode_ns) },
    );
    try writer.print(
        "{s}.decode_profile.commit_wait_non_gpu_share_pct={d:.3}\n",
        .{ prefix, percentage(stats.commit_wait_non_gpu_ns, stats.profiled_decode_ns) },
    );

    for (std.enums.values(Category)) |category| {
        const total_ns = stats.categories_ns[@intFromEnum(category)];
        const total_calls = stats.categories_calls[@intFromEnum(category)];
        try writer.print(
            "{s}.decode_profile.{s}_ms={d:.3}\n{s}.decode_profile.{s}_share_pct={d:.3}\n{s}.decode_profile.{s}_calls={d}\n",
            .{
                prefix,
                category.label(),
                nsToMs(total_ns),
                prefix,
                category.label(),
                percentage(total_ns, stats.profiled_decode_ns),
                prefix,
                category.label(),
                total_calls,
            },
        );
    }

    return try buffer.toOwnedSlice(allocator);
}

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

fn nsToMs(value: u64) f64 {
    return @as(f64, @floatFromInt(value)) / std.time.ns_per_ms;
}

fn parseMetricU64(line: []const u8, key: []const u8) ?u64 {
    if (!std.mem.startsWith(u8, line, key)) return null;
    return std.fmt.parseUnsigned(u64, line[key.len..], 10) catch null;
}

fn parseMetricUsize(line: []const u8, key: []const u8) ?usize {
    if (!std.mem.startsWith(u8, line, key)) return null;
    return std.fmt.parseUnsigned(usize, line[key.len..], 10) catch null;
}

fn parseCategoryMetricU64(line: []const u8, suffix: []const u8) ?struct { category: Category, value: u64 } {
    const prefix = "profile.";
    if (!std.mem.startsWith(u8, line, prefix)) return null;
    const rest = line[prefix.len..];
    for (std.enums.values(Category)) |category| {
        const label = category.label();
        if (!std.mem.startsWith(u8, rest, label)) continue;
        const remainder = rest[label.len..];
        if (!std.mem.startsWith(u8, remainder, suffix)) continue;
        return .{
            .category = category,
            .value = std.fmt.parseUnsigned(u64, remainder[suffix.len..], 10) catch return null,
        };
    }
    return null;
}

fn parseCategoryMetricUsize(line: []const u8, suffix: []const u8) ?struct { category: Category, value: usize } {
    const prefix = "profile.";
    if (!std.mem.startsWith(u8, line, prefix)) return null;
    const rest = line[prefix.len..];
    for (std.enums.values(Category)) |category| {
        const label = category.label();
        if (!std.mem.startsWith(u8, rest, label)) continue;
        const remainder = rest[label.len..];
        if (!std.mem.startsWith(u8, remainder, suffix)) continue;
        return .{
            .category = category,
            .value = std.fmt.parseUnsigned(usize, remainder[suffix.len..], 10) catch return null,
        };
    }
    return null;
}

fn topCategoryBreakdown(categories: [category_count]CategoryStats) [3]TopCategory {
    var top = [_]TopCategory{
        .{ .category = .projection_dense, .total_ns = 0 },
        .{ .category = .projection_dense, .total_ns = 0 },
        .{ .category = .projection_dense, .total_ns = 0 },
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
    profiler.recordWithShape(.projection_quantized, 200, .{ .rows = 64, .cols = 128, .tensor_type = 12 });
    profiler.record(.attention, 90);
    profiler.recordWithShape(.kv_writes, 30, .{ .rows = 1, .cols = 128, .depth = 2 });
    profiler.record(.cpu_sampling, 10);
    profiler.record(.commit_wait, 50);
    profiler.recordCommitWaitGpu(25);
    profiler.recordCommitWaitNonGpu(5);
    profiler.recordCommandBuffers(1);
    profiler.recordEncoders(7);
    profiler.recordDispatches(6);
    profiler.recordMoonQuantProjection(150, .{ .rows = 64, .cols = 128, .tensor_type = 12 });
    profiler.endDecodeToken();

    const summary = try profiler.renderSummary(std.testing.allocator);
    defer std.testing.allocator.free(summary);

    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.top_bottleneck_1=projection_quantized:200ns") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.shape_1=projection_quantized:rows=64:cols=128") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.token_0.cpu_sampling.ns=10") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.commit_wait.gpu_ns=25") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.commit_wait.non_gpu_ns=5") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.token_0.commit_wait.gpu_ns=25") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.token_0.commit_wait.non_gpu_ns=5") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.token_0.command_buffers=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.token_0.encoders=7") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "profile.token_0.dispatches=6") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "moon_quant.profile.total_ns=150") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "moon_quant.profile.token_0.ns=150") != null);
}

test "summary parser and bench renderer expose averaged decode stages" {
    const summary =
        \\metal_decode_profile_enabled=true
        \\profiled_decode_tokens=4
        \\profiled_decode_ns=1000
        \\profile.commit_wait.gpu_ns=100
        \\profile.commit_wait.non_gpu_ns=20
        \\profile.command_buffers=4
        \\profile.encoders=60
        \\profile.dispatches=44
        \\profile.projection_quantized.ns=400
        \\profile.projection_quantized.calls=8
        \\profile.attention.ns=300
        \\profile.attention.calls=4
        \\profile.cpu_sampling.ns=50
        \\profile.cpu_sampling.calls=4
        \\
    ;
    const stats = parseSummary(summary);
    try std.testing.expect(stats.enabled);
    try std.testing.expectEqual(@as(usize, 4), stats.profiled_decode_tokens);
    try std.testing.expectEqual(@as(u64, 20), stats.commit_wait_non_gpu_ns);
    try std.testing.expectEqual(@as(u64, 4), stats.command_buffer_count);
    try std.testing.expectEqual(@as(u64, 60), stats.encoder_count);
    try std.testing.expectEqual(@as(u64, 44), stats.dispatch_count);
    try std.testing.expectEqual(@as(u64, 400), stats.categories_ns[@intFromEnum(Category.projection_quantized)]);

    const rendered = try renderBenchStageSummary(std.testing.allocator, "warm", stats, 8);
    defer if (rendered) |text| std.testing.allocator.free(text);

    try std.testing.expect(rendered != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.total_ms=0.001") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.steps=4") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.generated_tokens=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.command_buffers=4") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.command_buffers_per_token=0.500") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.encoders=60") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.encoders_per_token=7.500") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.dispatches=44") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.dispatches_per_token=5.500") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.projection_quantized_ms=0.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.?, "warm.decode_profile.attention_share_pct=30.000") != null);
}
