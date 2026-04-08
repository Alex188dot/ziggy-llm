const std = @import("std");
const backend_api = @import("backend.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");
const llama = @import("../llama_cpu.zig");
const moon_quant = @import("../moon_quant.zig");
const ziggy_format = @import("../ziggy_format.zig");
const llama_fixture = @import("llama_fixture.zig");

pub const DenseTensorStore = struct {
    allocator: std.mem.Allocator,
    tensors: std.AutoHashMap(u64, []f32),
    raw_tensors: std.AutoHashMap(u64, []const u8),
    moon_quant_tensors: std.AutoHashMap(u64, moon_quant.PackedTensor),
    borrowed_moon_quant_tensors: std.AutoHashMap(u64, []const u8),
    tensor_meta: std.AutoHashMap(u64, TensorMeta),

    pub fn init(allocator: std.mem.Allocator) DenseTensorStore {
        return .{
            .allocator = allocator,
            .tensors = std.AutoHashMap(u64, []f32).init(allocator),
            .raw_tensors = std.AutoHashMap(u64, []const u8).init(allocator),
            .moon_quant_tensors = std.AutoHashMap(u64, moon_quant.PackedTensor).init(allocator),
            .borrowed_moon_quant_tensors = std.AutoHashMap(u64, []const u8).init(allocator),
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
        self.borrowed_moon_quant_tensors.deinit();
        self.tensor_meta.deinit();
        self.* = undefined;
    }

    pub fn populate(
        self: *DenseTensorStore,
        model: *const llama.Model,
        moon_quant_mode: types.MoonQuantMode,
        profiler: ?*StartupProfiler,
    ) !void {
        try self.addTensor(model, model.output, moon_quant_mode, profiler);
        try self.addTensor(model, model.output_norm, moon_quant_mode, profiler);
        for (model.layers) |layer| {
            try self.addTensor(model, layer.attn_norm, moon_quant_mode, profiler);
            try self.addTensor(model, layer.attn_q, moon_quant_mode, profiler);
            if (layer.attn_q_bias) |b| try self.addTensor(model, b, moon_quant_mode, profiler);
            if (layer.attn_q_norm) |n| try self.addTensor(model, n, moon_quant_mode, profiler);
            try self.addTensor(model, layer.attn_k, moon_quant_mode, profiler);
            if (layer.attn_k_bias) |b| try self.addTensor(model, b, moon_quant_mode, profiler);
            if (layer.attn_k_norm) |n| try self.addTensor(model, n, moon_quant_mode, profiler);
            try self.addTensor(model, layer.attn_v, moon_quant_mode, profiler);
            if (layer.attn_v_bias) |b| try self.addTensor(model, b, moon_quant_mode, profiler);
            try self.addTensor(model, layer.attn_output, moon_quant_mode, profiler);
            try self.addTensor(model, layer.ffn_norm, moon_quant_mode, profiler);
            try self.addTensor(model, layer.ffn_gate, moon_quant_mode, profiler);
            try self.addTensor(model, layer.ffn_down, moon_quant_mode, profiler);
            try self.addTensor(model, layer.ffn_up, moon_quant_mode, profiler);
        }
    }

    pub fn populateFromCompiled(
        self: *DenseTensorStore,
        model: *const llama.Model,
        compiled: *const ziggy_format.CompiledModel,
        moon_quant_mode: types.MoonQuantMode,
        profiler: ?*StartupProfiler,
    ) !void {
        try self.addNamedTensorFromCompiled(model, compiled, "output.weight", model.output, moon_quant_mode, profiler);
        try self.addNamedTensorFromCompiled(model, compiled, "output_norm.weight", model.output_norm, moon_quant_mode, profiler);
        for (model.layers, 0..) |layer, index| {
            const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{index});
            defer self.allocator.free(prefix);

            try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_norm.weight", layer.attn_norm, moon_quant_mode, profiler);
            try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_q.weight", layer.attn_q, moon_quant_mode, profiler);
            if (layer.attn_q_bias) |tensor| try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_q.bias", tensor, moon_quant_mode, profiler);
            if (layer.attn_q_norm) |tensor| try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_q_norm.weight", tensor, moon_quant_mode, profiler);
            try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_k.weight", layer.attn_k, moon_quant_mode, profiler);
            if (layer.attn_k_bias) |tensor| try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_k.bias", tensor, moon_quant_mode, profiler);
            if (layer.attn_k_norm) |tensor| try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_k_norm.weight", tensor, moon_quant_mode, profiler);
            try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_v.weight", layer.attn_v, moon_quant_mode, profiler);
            if (layer.attn_v_bias) |tensor| try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_v.bias", tensor, moon_quant_mode, profiler);
            try self.addLayerTensorFromCompiled(model, compiled, prefix, "attn_output.weight", layer.attn_output, moon_quant_mode, profiler);
            try self.addLayerTensorFromCompiled(model, compiled, prefix, "ffn_norm.weight", layer.ffn_norm, moon_quant_mode, profiler);
            try self.addLayerTensorFromCompiled(model, compiled, prefix, "ffn_gate.weight", layer.ffn_gate, moon_quant_mode, profiler);
            try self.addLayerTensorFromCompiled(model, compiled, prefix, "ffn_down.weight", layer.ffn_down, moon_quant_mode, profiler);
            try self.addLayerTensorFromCompiled(model, compiled, prefix, "ffn_up.weight", layer.ffn_up, moon_quant_mode, profiler);
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
        if (self.moon_quant_tensors.get(offset)) |tensor| return tensor.bytes;
        return self.borrowed_moon_quant_tensors.get(offset);
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
            if (self.moon_quant_tensors.contains(entry.key_ptr.*) or self.borrowed_moon_quant_tensors.contains(entry.key_ptr.*)) {
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
        var borrowed_moon_quant_iterator = self.borrowed_moon_quant_tensors.iterator();
        while (borrowed_moon_quant_iterator.next()) |entry| {
            const start = std.time.nanoTimestamp();
            try metal_backend.cacheRawMatrix(backend, entry.value_ptr.*);
            if (profiler) |active_profiler| {
                active_profiler.recordOffset(.metal_prewarm_moon_quant, entry.key_ptr.*, entry.value_ptr.*.len, types.deltaNs(start, std.time.nanoTimestamp()), self.tensor_meta.get(entry.key_ptr.*));
            }
        }
    }

    pub fn prewarmPlan(self: *const DenseTensorStore) PrewarmPlan {
        var plan = PrewarmPlan{
            .dense_count = self.tensors.count(),
            .moon_quant_count = self.moon_quant_tensors.count() + self.borrowed_moon_quant_tensors.count(),
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

    fn addTensor(
        self: *DenseTensorStore,
        model: *const llama.Model,
        tensor: llama.TensorRef,
        moon_quant_mode: types.MoonQuantMode,
        profiler: ?*StartupProfiler,
    ) !void {
        const tensor_bytes = try llama.tensorBytes(model, tensor);
        try self.addTensorBytes(tensor, tensor_bytes, moon_quant_mode, profiler);
    }

    fn addNamedTensorFromCompiled(
        self: *DenseTensorStore,
        model: *const llama.Model,
        compiled: *const ziggy_format.CompiledModel,
        name: []const u8,
        tensor: llama.TensorRef,
        moon_quant_mode: types.MoonQuantMode,
        profiler: ?*StartupProfiler,
    ) !void {
        if (compiled.getTensor(name)) |compiled_tensor| {
            try self.addTensorFromCompiled(model, compiled, tensor, compiled_tensor, moon_quant_mode, profiler);
            return;
        }
        try self.addTensor(model, tensor, moon_quant_mode, profiler);
    }

    fn addLayerTensorFromCompiled(
        self: *DenseTensorStore,
        model: *const llama.Model,
        compiled: *const ziggy_format.CompiledModel,
        prefix: []const u8,
        suffix: []const u8,
        tensor: llama.TensorRef,
        moon_quant_mode: types.MoonQuantMode,
        profiler: ?*StartupProfiler,
    ) !void {
        const name = try std.mem.concat(self.allocator, u8, &.{ prefix, suffix });
        defer self.allocator.free(name);
        try self.addNamedTensorFromCompiled(model, compiled, name, tensor, moon_quant_mode, profiler);
    }

    fn addTensorFromCompiled(
        self: *DenseTensorStore,
        model: *const llama.Model,
        compiled: *const ziggy_format.CompiledModel,
        tensor: llama.TensorRef,
        compiled_tensor: *const ziggy_format.TensorInfo,
        moon_quant_mode: types.MoonQuantMode,
        profiler: ?*StartupProfiler,
    ) !void {
        switch (compiled_tensor.compiled_layout_kind) {
            .q4_k_m_packed => {
                if (moon_quant_mode != .enabled) {
                    try self.addTensor(model, tensor, moon_quant_mode, profiler);
                    return;
                }
                if (self.tensors.contains(tensor.offset) or self.raw_tensors.contains(tensor.offset) or self.moon_quant_tensors.contains(tensor.offset) or self.borrowed_moon_quant_tensors.contains(tensor.offset)) return;
                const rows = try tensor.rowCount();
                const cols = try tensor.rowLen();
                try self.tensor_meta.put(tensor.offset, .{
                    .rows = rows,
                    .cols = cols,
                    .tensor_type = tensor.tensor_type,
                });
                const prepare_start = std.time.nanoTimestamp();
                const compiled_bytes = compiled.tensorBytes(compiled_tensor);
                try self.borrowed_moon_quant_tensors.put(tensor.offset, compiled_bytes);
                if (profiler) |active_profiler| {
                    active_profiler.recordTensor(.tensor_prepare_moon_quant_pack, tensor, compiled_bytes.len, types.deltaNs(prepare_start, std.time.nanoTimestamp()));
                }
                return;
            },
            .f16_raw, .f32_raw, .generic_quant_raw => {
                const compiled_bytes = compiled.tensorBytes(compiled_tensor);
                const expected_row_size = llama.tensorRowByteSize(tensor.tensor_type, try tensor.rowLen()) catch {
                    try self.addTensor(model, tensor, moon_quant_mode, profiler);
                    return;
                };
                const expected_total = (try tensor.rowCount()) * expected_row_size;
                if (compiled_bytes.len == expected_total) {
                    try self.addTensorBytes(tensor, compiled_bytes, moon_quant_mode, profiler);
                    return;
                }
            },
            else => {},
        }

        try self.addTensor(model, tensor, moon_quant_mode, profiler);
    }

    fn addTensorBytes(
        self: *DenseTensorStore,
        tensor: llama.TensorRef,
        tensor_bytes: []const u8,
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

        if (tensor.tensor_type == .q4_k or tensor.tensor_type == .q6_k or tensor.tensor_type == .q8_0) {
            const prepare_start = std.time.nanoTimestamp();
            try self.raw_tensors.put(tensor.offset, tensor_bytes);
            if (profiler) |active_profiler| {
                active_profiler.recordTensor(.tensor_prepare_quant_raw, tensor, tensor_bytes.len, types.deltaNs(prepare_start, std.time.nanoTimestamp()));
            }
            if (tensor.tensor_type == .q4_k and moon_quant_mode == .enabled) {
                const pack_start = std.time.nanoTimestamp();
                try self.moon_quant_tensors.put(tensor.offset, try moon_quant.packQ4KTensor(
                    self.allocator,
                    tensor_bytes,
                    rows,
                    cols,
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
        const dense = try self.allocator.alloc(f32, rows * cols);
        errdefer self.allocator.free(dense);
        @memset(dense, 0);

        for (0..rows) |row_index| {
            const row_bytes = tensor_bytes[row_index * row_size ..][0..row_size];
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

test "dense tensor store can source MoonQuant q4 tensors from compiled ziggy" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaQ4KFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "llama-q4k.gguf", fixture);

    const gguf_path = try tmp.dir.realpathAlloc(std.testing.allocator, "llama-q4k.gguf");
    defer std.testing.allocator.free(gguf_path);
    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const ziggy_path = try std.fmt.allocPrint(std.testing.allocator, "{s}/llama-q4k.ziggy", .{root_path});
    defer std.testing.allocator.free(ziggy_path);

    try ziggy_format.compileFromGGUF(std.testing.allocator, gguf_path, ziggy_path, .{});

    var model = try llama.loadModel(std.testing.allocator, gguf_path);
    defer model.deinit(std.testing.allocator);
    var compiled = try ziggy_format.loadCompiledModel(std.testing.allocator, ziggy_path);
    defer compiled.deinit();

    var store = DenseTensorStore.init(std.testing.allocator);
    defer store.deinit();
    try store.populateFromCompiled(&model, &compiled, .enabled, null);

    try std.testing.expect(store.getMoonQuantBytesByOffset(model.output.offset) != null);
    try std.testing.expect(store.getByOffset(model.output_norm.offset) != null);
    const plan = store.prewarmPlan();
    try std.testing.expectEqual(@as(usize, 0), plan.raw_count);
}
