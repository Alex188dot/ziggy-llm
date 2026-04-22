const std = @import("std");
const backend_api = @import("../../backend.zig");
const metal_backend = @import("../../metal_backend.zig");
const types = @import("../../types.zig");
const llama = @import("../../../model/loader.zig");
const moon_quant = @import("../../../moon_quant.zig");
const llama_fixture = @import("../../llama_fixture.zig");
const offload_policy = @import("offload_policy.zig");

pub const DenseTensorStore = struct {
    allocator: std.mem.Allocator,
    tensors: std.AutoHashMap(u64, []f32),
    raw_tensors: std.AutoHashMap(u64, []const u8),
    moon_quant_tensors: std.AutoHashMap(u64, moon_quant.PackedTensor),
    tensor_meta: std.AutoHashMap(u64, TensorMeta),

    pub fn init(allocator: std.mem.Allocator) DenseTensorStore {
        return .{
            .allocator = allocator,
            .tensors = std.AutoHashMap(u64, []f32).init(allocator),
            .raw_tensors = std.AutoHashMap(u64, []const u8).init(allocator),
            .moon_quant_tensors = std.AutoHashMap(u64, moon_quant.PackedTensor).init(allocator),
            .tensor_meta = std.AutoHashMap(u64, TensorMeta).init(allocator),
        };
    }

    pub fn deinit(self: *DenseTensorStore) void {
        var iterator = self.tensors.valueIterator();
        while (iterator.next()) |values| self.allocator.free(values.*);
        self.tensors.deinit();
        self.raw_tensors.deinit();
        var moon_quant_iterator = self.moon_quant_tensors.valueIterator();
        while (moon_quant_iterator.next()) |tensor| tensor.deinit(self.allocator);
        self.moon_quant_tensors.deinit();
        self.tensor_meta.deinit();
        self.* = undefined;
    }

    pub fn populate(
        self: *DenseTensorStore,
        model: *const llama.Model,
        gpu_layers: types.GpuLayers,
        device_info: ?metal_backend.DeviceInfo,
        moon_quant_mode: types.MoonQuantMode,
        profiler: ?*StartupProfiler,
    ) !void {
        const policy = offload_policy.OffloadPolicy.forModel(model, gpu_layers, device_info);
        try self.addTensor(model, model.output, moon_quant_mode, profiler);
        try self.addTensor(model, model.output_norm, moon_quant_mode, profiler);
        for (model.layers, 0..) |layer, layer_index| {
            if (!policy.offloadsAttention(layer_index)) continue;
            try self.addTensor(model, layer.attn_norm, moon_quant_mode, profiler);
            if (layer.attn_q) |q| try self.addTensor(model, q, moon_quant_mode, profiler);
            if (layer.attn_q_bias) |b| try self.addTensor(model, b, moon_quant_mode, profiler);
            if (layer.attn_q_norm) |n| try self.addTensor(model, n, moon_quant_mode, profiler);
            if (layer.attn_k) |k| try self.addTensor(model, k, moon_quant_mode, profiler);
            if (layer.attn_k_bias) |b| try self.addTensor(model, b, moon_quant_mode, profiler);
            if (layer.attn_k_norm) |n| try self.addTensor(model, n, moon_quant_mode, profiler);
            if (layer.attn_v) |v| try self.addTensor(model, v, moon_quant_mode, profiler);
            if (layer.attn_v_bias) |b| try self.addTensor(model, b, moon_quant_mode, profiler);
            if (layer.attn_output) |o| try self.addTensor(model, o, moon_quant_mode, profiler);
            if (layer.post_attention_norm) |n| try self.addTensor(model, n, moon_quant_mode, profiler);
            if (layer.moe == null or policy.offloadsMoeFfn(layer.moe != null)) {
                try self.addTensor(model, layer.ffn_norm, moon_quant_mode, profiler);
                try self.addTensor(model, layer.ffn_gate, moon_quant_mode, profiler);
                try self.addTensor(model, layer.ffn_down, moon_quant_mode, profiler);
                try self.addTensor(model, layer.ffn_up, moon_quant_mode, profiler);
                if (layer.post_ffw_norm) |n| try self.addTensor(model, n, moon_quant_mode, profiler);
            }
            if (layer.linear_attn) |la| {
                try self.addTensor(model, la.in_proj_qkv, moon_quant_mode, profiler);
                try self.addTensor(model, la.in_proj_z, moon_quant_mode, profiler);
                try self.addTensor(model, la.in_proj_b, moon_quant_mode, profiler);
                try self.addTensor(model, la.in_proj_a, moon_quant_mode, profiler);
                try self.addTensor(model, la.conv1d, moon_quant_mode, profiler);
                try self.addTensor(model, la.dt_bias, moon_quant_mode, profiler);
                try self.addTensor(model, la.A_log, moon_quant_mode, profiler);
                try self.addTensor(model, la.norm_weight, moon_quant_mode, profiler);
                try self.addTensor(model, la.out_proj, moon_quant_mode, profiler);
            }
        }
    }

    pub fn get(self: *const DenseTensorStore, tensor: llama.TensorRef) ?[]const f32 {
        return self.tensors.get(tensor.offset);
    }

    pub fn getByOffset(self: *const DenseTensorStore, offset: u64) ?[]const f32 {
        return self.tensors.get(offset);
    }

    pub fn getRawByOffset(self: *const DenseTensorStore, offset: u64) ?[]const u8 {
        return self.raw_tensors.get(offset);
    }

    pub fn getMoonQuantByOffset(self: *const DenseTensorStore, offset: u64) ?moon_quant.PackedTensor {
        return self.moon_quant_tensors.get(offset);
    }

    pub fn getMoonQuantBytesByOffset(self: *const DenseTensorStore, offset: u64) ?[]const u8 {
        const tensor = self.moon_quant_tensors.get(offset) orelse return null;
        return tensor.bytes;
    }

    pub fn prewarm(self: *const DenseTensorStore, backend: backend_api.MatVecBackend, profiler: ?*StartupProfiler) !void {
        var iterator = self.tensors.valueIterator();
        while (iterator.next()) |matrix| {
            const start = std.time.nanoTimestamp();
            try metal_backend.cacheMatrix(backend, matrix.*);
            if (profiler) |active_profiler| {
                active_profiler.recordMapValue(.metal_prewarm_dense, matrix.*, types.deltaNs(start, std.time.nanoTimestamp()));
            }
        }
        var raw_iterator = self.raw_tensors.iterator();
        while (raw_iterator.next()) |entry| {
            if (self.moon_quant_tensors.contains(entry.key_ptr.*)) {
                if (profiler) |active_profiler| {
                    const meta = self.tensor_meta.get(entry.key_ptr.*);
                    active_profiler.recordSkippedShadowedQ4(entry.value_ptr.*, meta);
                }
                continue;
            }
            const start = std.time.nanoTimestamp();
            try metal_backend.cacheRawMatrix(backend, entry.value_ptr.*);
            if (profiler) |active_profiler| {
                active_profiler.recordOffset(.metal_prewarm_quant_raw, entry.key_ptr.*, entry.value_ptr.*.len, types.deltaNs(start, std.time.nanoTimestamp()), self.tensor_meta.get(entry.key_ptr.*));
            }
        }
        var moon_quant_iterator = self.moon_quant_tensors.iterator();
        while (moon_quant_iterator.next()) |entry| {
            const start = std.time.nanoTimestamp();
            try metal_backend.cacheRawMatrix(backend, entry.value_ptr.bytes);
            if (profiler) |active_profiler| {
                active_profiler.recordOffset(.metal_prewarm_moon_quant, entry.key_ptr.*, entry.value_ptr.bytes.len, types.deltaNs(start, std.time.nanoTimestamp()), self.tensor_meta.get(entry.key_ptr.*));
            }
        }
    }

    pub fn prewarmPlan(self: *const DenseTensorStore) PrewarmPlan {
        var plan = PrewarmPlan{
            .dense_count = self.tensors.count(),
            .moon_quant_count = self.moon_quant_tensors.count(),
        };
        var raw_iterator = self.raw_tensors.iterator();
        while (raw_iterator.next()) |entry| {
            if (self.moon_quant_tensors.contains(entry.key_ptr.*)) {
                plan.skipped_shadowed_raw_count += 1;
                continue;
            }
            plan.raw_count += 1;
        }
        return plan;
    }

    pub fn storesRawQuant(tensor_type: llama.TensorType) bool {
        return switch (tensor_type) {
            .q3_k, .q4_k, .q6_k, .q8_0, .iq3_xxs, .iq4_xs => true,
            else => false,
        };
    }

    fn addTensor(
        self: *DenseTensorStore,
        model: *const llama.Model,
        tensor: llama.TensorRef,
        moon_quant_mode: types.MoonQuantMode,
        profiler: ?*StartupProfiler,
    ) !void {
        if (self.tensors.contains(tensor.offset) or self.raw_tensors.contains(tensor.offset) or self.moon_quant_tensors.contains(tensor.offset)) return;
        const rows = try tensor.rowCount();
        const cols = try tensor.rowLen();
        try self.tensor_meta.put(tensor.offset, .{
            .rows = rows,
            .cols = cols,
            .tensor_type = tensor.tensor_type,
        });

        if (storesRawQuant(tensor.tensor_type)) {
            const prepare_start = std.time.nanoTimestamp();
            const tensor_bytes = try llama.tensorBytes(model, tensor);
            try self.raw_tensors.put(tensor.offset, tensor_bytes);
            if (profiler) |active_profiler| {
                active_profiler.recordTensor(.tensor_prepare_quant_raw, tensor, tensor_bytes.len, types.deltaNs(prepare_start, std.time.nanoTimestamp()));
            }
            if (tensor.tensor_type == .q4_k and moon_quant_mode == .enabled) {
                const pack_start = std.time.nanoTimestamp();
                try self.moon_quant_tensors.put(tensor.offset, try moon_quant.packQ4KTensor(
                    self.allocator,
                    tensor_bytes,
                    try tensor.rowCount(),
                    try tensor.rowLen(),
                ));
                if (profiler) |active_profiler| {
                    const packed_tensor = self.moon_quant_tensors.get(tensor.offset).?;
                    active_profiler.recordTensor(.tensor_prepare_moon_quant_pack, tensor, packed_tensor.bytes.len, types.deltaNs(pack_start, std.time.nanoTimestamp()));
                }
            }
            return;
        }

        const dense_prepare_start = std.time.nanoTimestamp();
        const row_size = try llama.tensorRowByteSize(tensor.tensor_type, cols);
        const bytes = try llama.tensorBytes(model, tensor);
        const dense = try self.allocator.alloc(f32, rows * cols);
        errdefer self.allocator.free(dense);
        @memset(dense, 0);

        for (0..rows) |row_index| {
            const row_bytes = bytes[row_index * row_size ..][0..row_size];
            const row_dense = try self.allocator.alloc(f32, cols);
            defer self.allocator.free(row_dense);
            try llama.dequantizeRow(row_dense, tensor.tensor_type, row_bytes, cols);
            for (0..cols) |col_index| {
                dense[row_index + col_index * rows] = row_dense[col_index];
            }
        }

        try self.tensors.put(tensor.offset, dense);
        if (profiler) |active_profiler| {
            active_profiler.recordOffset(.tensor_prepare_dense, tensor.offset, dense.len * @sizeOf(f32), types.deltaNs(dense_prepare_start, std.time.nanoTimestamp()), .{
                .rows = rows,
                .cols = cols,
                .tensor_type = tensor.tensor_type,
            });
        }
    }
};

const TensorMeta = struct {
    rows: usize,
    cols: usize,
    tensor_type: llama.TensorType,
};

pub const PrewarmPlan = struct {
    dense_count: usize = 0,
    raw_count: usize = 0,
    moon_quant_count: usize = 0,
    skipped_shadowed_raw_count: usize = 0,
};

const StartupCategory = enum(u8) {
    tensor_prepare_dense,
    tensor_prepare_quant_raw,
    tensor_prepare_moon_quant_pack,
    metal_prewarm_dense,
    metal_prewarm_quant_raw,
    metal_prewarm_moon_quant,

    fn label(self: StartupCategory) []const u8 {
        return switch (self) {
            .tensor_prepare_dense => "tensor_prepare_dense",
            .tensor_prepare_quant_raw => "tensor_prepare_quant_raw",
            .tensor_prepare_moon_quant_pack => "tensor_prepare_moon_quant_pack",
            .metal_prewarm_dense => "metal_prewarm_dense",
            .metal_prewarm_quant_raw => "metal_prewarm_quant_raw",
            .metal_prewarm_moon_quant => "metal_prewarm_moon_quant",
        };
    }
};

const startup_category_count = @typeInfo(StartupCategory).@"enum".fields.len;

const StartupCategoryStats = struct {
    total_ns: u64 = 0,
    total_bytes: u64 = 0,
    calls: usize = 0,
};

const StartupEntry = struct {
    category: StartupCategory,
    offset: u64,
    rows: usize,
    cols: usize,
    tensor_type: u32,
    bytes: usize,
    duration_ns: u64,
};

pub const StartupProfiler = struct {
    enabled: bool = false,
    categories: [startup_category_count]StartupCategoryStats = [_]StartupCategoryStats{.{}} ** startup_category_count,
    top_entries: [6]?StartupEntry = [_]?StartupEntry{null} ** 6,
    skipped_shadowed_q4_count: usize = 0,
    skipped_shadowed_q4_bytes: u64 = 0,

    pub fn recordTensor(
        self: *StartupProfiler,
        category: StartupCategory,
        tensor: llama.TensorRef,
        byte_len: usize,
        duration_ns: u64,
    ) void {
        self.recordOffset(category, tensor.offset, byte_len, duration_ns, .{
            .rows = tensor.rowCount() catch 0,
            .cols = tensor.rowLen() catch 0,
            .tensor_type = tensor.tensor_type,
        });
    }

    pub fn recordMapValue(
        self: *StartupProfiler,
        category: StartupCategory,
        matrix: []const f32,
        duration_ns: u64,
    ) void {
        if (!self.enabled) return;
        self.record(category, matrix.len * @sizeOf(f32), duration_ns);
    }

    pub fn recordOffset(
        self: *StartupProfiler,
        category: StartupCategory,
        offset: u64,
        byte_len: usize,
        duration_ns: u64,
        meta: ?TensorMeta,
    ) void {
        if (!self.enabled) return;
        self.record(category, byte_len, duration_ns);
        self.maybeInsertTop(.{
            .category = category,
            .offset = offset,
            .rows = if (meta) |value| value.rows else 0,
            .cols = if (meta) |value| value.cols else 0,
            .tensor_type = if (meta) |value| @intFromEnum(value.tensor_type) else 0,
            .bytes = byte_len,
            .duration_ns = duration_ns,
        });
    }

    pub fn recordSkippedShadowedQ4(self: *StartupProfiler, bytes: []const u8, meta: ?TensorMeta) void {
        if (!self.enabled) return;
        self.skipped_shadowed_q4_count += 1;
        self.skipped_shadowed_q4_bytes += bytes.len;
        _ = meta;
    }

    pub fn renderSummary(self: *const StartupProfiler, allocator: std.mem.Allocator) ![]u8 {
        if (!self.enabled) return allocator.dupe(u8, "");

        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);
        try writer.print("startup_profile_enabled=true\n", .{});

        for (std.enums.values(StartupCategory)) |category| {
            const stats = self.categories[@intFromEnum(category)];
            try writer.print(
                "startup.{s}.ns={d}\nstartup.{s}.bytes={d}\nstartup.{s}.calls={d}\n",
                .{
                    category.label(),
                    stats.total_ns,
                    category.label(),
                    stats.total_bytes,
                    category.label(),
                    stats.calls,
                },
            );
        }
        try writer.print(
            "startup.metal_prewarm_skipped_shadowed_q4.count={d}\nstartup.metal_prewarm_skipped_shadowed_q4.bytes={d}\n",
            .{ self.skipped_shadowed_q4_count, self.skipped_shadowed_q4_bytes },
        );

        var rank: usize = 0;
        for (self.top_entries) |maybe_entry| {
            const entry = maybe_entry orelse continue;
            if (entry.duration_ns == 0) continue;
            rank += 1;
            try writer.print(
                "startup.top_{d}={s}:offset={d}:rows={d}:cols={d}:tensor_type={d}:bytes={d}:ns={d}\n",
                .{
                    rank,
                    entry.category.label(),
                    entry.offset,
                    entry.rows,
                    entry.cols,
                    entry.tensor_type,
                    entry.bytes,
                    entry.duration_ns,
                },
            );
        }

        return buffer.toOwnedSlice(allocator);
    }

    fn record(self: *StartupProfiler, category: StartupCategory, byte_len: usize, duration_ns: u64) void {
        if (!self.enabled) return;
        const stats = &self.categories[@intFromEnum(category)];
        stats.total_ns += duration_ns;
        stats.total_bytes += byte_len;
        stats.calls += 1;
    }

    fn maybeInsertTop(self: *StartupProfiler, entry: StartupEntry) void {
        if (!self.enabled) return;
        var insert_at: ?usize = null;
        for (self.top_entries, 0..) |maybe_existing, index| {
            if (maybe_existing == null) {
                insert_at = index;
                break;
            }
            if (entry.duration_ns > maybe_existing.?.duration_ns) {
                insert_at = index;
                break;
            }
        }
        const index = insert_at orelse return;
        var shift = self.top_entries.len - 1;
        while (shift > index) : (shift -= 1) {
            self.top_entries[shift] = self.top_entries[shift - 1];
        }
        self.top_entries[index] = entry;
    }
};

test "dense tensor store packs q4_k tensors only when MoonQuant is enabled" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaQ4KFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "llama-q4k.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "llama-q4k.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    var packed_store = DenseTensorStore.init(std.testing.allocator);
    defer packed_store.deinit();
    try packed_store.populate(&model, .enabled, null);
    try std.testing.expect(packed_store.getRawByOffset(model.output.offset) != null);
    try std.testing.expect(packed_store.getMoonQuantBytesByOffset(model.output.offset) != null);
    const packed_plan = packed_store.prewarmPlan();
    try std.testing.expectEqual(@as(usize, 0), packed_plan.raw_count);
    try std.testing.expect(packed_plan.skipped_shadowed_raw_count > 0);

    var generic_store = DenseTensorStore.init(std.testing.allocator);
    defer generic_store.deinit();
    try generic_store.populate(&model, .disabled, null);
    try std.testing.expect(generic_store.getRawByOffset(model.output.offset) != null);
    try std.testing.expect(generic_store.getMoonQuantBytesByOffset(model.output.offset) == null);
    const generic_plan = generic_store.prewarmPlan();
    try std.testing.expect(generic_plan.raw_count > 0);
    try std.testing.expectEqual(@as(usize, 0), generic_plan.skipped_shadowed_raw_count);
}

test "dense tensor store keeps initial qwen moe quant targets raw for Metal preparation" {
    const quant_types = [_]llama.TensorType{ .q3_k, .iq3_xxs, .iq4_xs };

    for (quant_types) |tensor_type| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();

        const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, tensor_type);
        defer std.testing.allocator.free(fixture);
        try llama_fixture.writeFixtureFile(tmp.dir, "llama-quant.gguf", fixture);

        const path = try tmp.dir.realpathAlloc(std.testing.allocator, "llama-quant.gguf");
        defer std.testing.allocator.free(path);

        var model = try llama.loadModel(std.testing.allocator, path);
        defer model.deinit(std.testing.allocator);

        var store = DenseTensorStore.init(std.testing.allocator);
        defer store.deinit();
        try store.populate(&model, .disabled, null);

        try std.testing.expectEqual(true, DenseTensorStore.storesRawQuant(tensor_type));
        try std.testing.expect(store.getRawByOffset(model.output.offset) != null);
        try std.testing.expect(store.getByOffset(model.output.offset) == null);

        const plan = store.prewarmPlan();
        try std.testing.expect(plan.raw_count > 0);
    }
}
