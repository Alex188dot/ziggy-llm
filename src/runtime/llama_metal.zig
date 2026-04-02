const std = @import("std");
const backend_api = @import("backend.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");
const llama = @import("../llama_cpu.zig");
const moon_quant = @import("../moon_quant.zig");
const llama_fixture = @import("llama_fixture.zig");

pub const DenseTensorStore = struct {
    allocator: std.mem.Allocator,
    tensors: std.AutoHashMap(u64, []f32),
    raw_tensors: std.AutoHashMap(u64, []const u8),
    moon_quant_tensors: std.AutoHashMap(u64, moon_quant.PackedTensor),

    pub fn init(allocator: std.mem.Allocator) DenseTensorStore {
        return .{
            .allocator = allocator,
            .tensors = std.AutoHashMap(u64, []f32).init(allocator),
            .raw_tensors = std.AutoHashMap(u64, []const u8).init(allocator),
            .moon_quant_tensors = std.AutoHashMap(u64, moon_quant.PackedTensor).init(allocator),
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
        self.* = undefined;
    }

    pub fn populate(self: *DenseTensorStore, model: *const llama.Model, moon_quant_mode: types.MoonQuantMode) !void {
        try self.addTensor(model, model.output, moon_quant_mode);
        try self.addTensor(model, model.output_norm, moon_quant_mode);
        for (model.layers) |layer| {
            try self.addTensor(model, layer.attn_norm, moon_quant_mode);
            try self.addTensor(model, layer.attn_q, moon_quant_mode);
            try self.addTensor(model, layer.attn_k, moon_quant_mode);
            try self.addTensor(model, layer.attn_v, moon_quant_mode);
            try self.addTensor(model, layer.attn_output, moon_quant_mode);
            try self.addTensor(model, layer.ffn_norm, moon_quant_mode);
            try self.addTensor(model, layer.ffn_gate, moon_quant_mode);
            try self.addTensor(model, layer.ffn_down, moon_quant_mode);
            try self.addTensor(model, layer.ffn_up, moon_quant_mode);
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

    pub fn prewarm(self: *const DenseTensorStore, backend: backend_api.MatVecBackend) !void {
        var iterator = self.tensors.valueIterator();
        while (iterator.next()) |matrix| {
            try metal_backend.cacheMatrix(backend, matrix.*);
        }
        var raw_iterator = self.raw_tensors.valueIterator();
        while (raw_iterator.next()) |matrix| {
            try metal_backend.cacheRawMatrix(backend, matrix.*);
        }
        var moon_quant_iterator = self.moon_quant_tensors.valueIterator();
        while (moon_quant_iterator.next()) |tensor| {
            try metal_backend.cacheRawMatrix(backend, tensor.bytes);
        }
    }

    fn addTensor(self: *DenseTensorStore, model: *const llama.Model, tensor: llama.TensorRef, moon_quant_mode: types.MoonQuantMode) !void {
        if (self.tensors.contains(tensor.offset) or self.raw_tensors.contains(tensor.offset) or self.moon_quant_tensors.contains(tensor.offset)) return;

        if (tensor.tensor_type == .q4_k or tensor.tensor_type == .q6_k) {
            const tensor_bytes = try llama.tensorBytes(model, tensor);
            try self.raw_tensors.put(tensor.offset, tensor_bytes);
            if (tensor.tensor_type == .q4_k and moon_quant_mode == .enabled) {
                try self.moon_quant_tensors.put(tensor.offset, try moon_quant.packQ4KTensor(
                    self.allocator,
                    tensor_bytes,
                    try tensor.rowCount(),
                    try tensor.rowLen(),
                ));
            }
            return;
        }

        const rows = try tensor.rowCount();
        const cols = try tensor.rowLen();
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
    try packed_store.populate(&model, .enabled);
    try std.testing.expect(packed_store.getRawByOffset(model.output.offset) != null);
    try std.testing.expect(packed_store.getMoonQuantBytesByOffset(model.output.offset) != null);

    var generic_store = DenseTensorStore.init(std.testing.allocator);
    defer generic_store.deinit();
    try generic_store.populate(&model, .disabled);
    try std.testing.expect(generic_store.getRawByOffset(model.output.offset) != null);
    try std.testing.expect(generic_store.getMoonQuantBytesByOffset(model.output.offset) == null);
}
