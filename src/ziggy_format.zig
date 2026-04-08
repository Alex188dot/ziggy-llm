const std = @import("std");
const gguf = @import("gguf.zig");
const moon_quant = @import("moon_quant.zig");
const llama = @import("llama_cpu.zig");
const calibration = @import("moon_quant_calibration.zig");
const llama_fixture = @import("runtime/llama_fixture.zig");

pub const magic = "ZIGY";
pub const current_version: u32 = 3;
pub const current_runtime_version: u32 = 1;
pub const default_alignment: u32 = 32;

pub const CompiledLayoutKind = enum(u32) {
    q4_k_m_packed = 0,
    q4_k_s_packed = 1,
    q5_k_m_packed = 2,
    q5_k_s_packed = 3,
    q6_k_raw = 4,
    q8_0_raw = 5,
    f16_raw = 6,
    f32_raw = 7,
    generic_quant_raw = 8,

    pub fn label(self: CompiledLayoutKind) []const u8 {
        return switch (self) {
            .q4_k_m_packed => "q4_k_m_packed",
            .q4_k_s_packed => "q4_k_s_packed",
            .q5_k_m_packed => "q5_k_m_packed",
            .q5_k_s_packed => "q5_k_s_packed",
            .q6_k_raw => "q6_k_raw",
            .q8_0_raw => "q8_0_raw",
            .f16_raw => "f16_raw",
            .f32_raw => "f32_raw",
            .generic_quant_raw => "generic_quant_raw",
        };
    }
};

pub const KernelFamily = enum(u32) {
    attention = 0,
    ffn = 1,
    output = 2,
    embedding = 3,
    normalization = 4,
    other = 5,

    pub fn label(self: KernelFamily) []const u8 {
        return switch (self) {
            .attention => "attention",
            .ffn => "ffn",
            .output => "output",
            .embedding => "embedding",
            .normalization => "normalization",
            .other => "other",
        };
    }

    pub fn fromTensorName(name: []const u8) KernelFamily {
        if (std.mem.indexOf(u8, name, "attn_q") != null or
            std.mem.indexOf(u8, name, "attn_k") != null or
            std.mem.indexOf(u8, name, "attn_v") != null or
            std.mem.indexOf(u8, name, "attn_output") != null) return .attention;
        if (std.mem.indexOf(u8, name, "ffn_gate") != null or
            std.mem.indexOf(u8, name, "ffn_up") != null or
            std.mem.indexOf(u8, name, "ffn_down") != null) return .ffn;
        if (std.mem.indexOf(u8, name, "output.weight") != null) return .output;
        if (std.mem.indexOf(u8, name, "token_embd") != null) return .embedding;
        if (std.mem.indexOf(u8, name, "norm") != null) return .normalization;
        return .other;
    }
};

pub const DecodeMetadata = extern struct {
    block_size: u32,
    type_size: u32,
    scale_bits: u32,
    zero_point_bits: u32,
    reserved: [16]u8,

    pub fn forLayout(layout: CompiledLayoutKind) DecodeMetadata {
        return switch (layout) {
            .q4_k_m_packed, .q4_k_s_packed => .{
                .block_size = 256,
                .type_size = 144,
                .scale_bits = 6,
                .zero_point_bits = 6,
                .reserved = .{0} ** 16,
            },
            .q5_k_m_packed, .q5_k_s_packed => .{
                .block_size = 256,
                .type_size = 176,
                .scale_bits = 6,
                .zero_point_bits = 6,
                .reserved = .{0} ** 16,
            },
            .q6_k_raw => .{
                .block_size = 256,
                .type_size = 210,
                .scale_bits = 8,
                .zero_point_bits = 0,
                .reserved = .{0} ** 16,
            },
            .q8_0_raw => .{
                .block_size = 32,
                .type_size = 34,
                .scale_bits = 16,
                .zero_point_bits = 0,
                .reserved = .{0} ** 16,
            },
            .f16_raw => .{
                .block_size = 1,
                .type_size = 2,
                .scale_bits = 0,
                .zero_point_bits = 0,
                .reserved = .{0} ** 16,
            },
            .f32_raw => .{
                .block_size = 1,
                .type_size = 4,
                .scale_bits = 0,
                .zero_point_bits = 0,
                .reserved = .{0} ** 16,
            },
            .generic_quant_raw => .{
                .block_size = 32,
                .type_size = 18,
                .scale_bits = 0,
                .zero_point_bits = 0,
                .reserved = .{0} ** 16,
            },
        };
    }
};

pub const ZiggyHeader = extern struct {
    magic: [4]u8,
    version: u32,
    runtime_version: u32,
    architecture_len: u32,
    tensor_count: u64,
    metadata_count: u64,
    metadata_blob_len: u64,
    alignment: u32,
    quantization_version: u32,
    reserved: [8]u8,

    pub fn compiledMetadataBlobLen(self: ZiggyHeader) u64 {
        return std.mem.bytesToValue(u64, &self.reserved);
    }

    pub fn setCompiledMetadataBlobLen(self: *ZiggyHeader, len: u64) void {
        self.reserved = std.mem.toBytes(len);
    }

    pub fn init() ZiggyHeader {
        return .{
            .magic = magic.*,
            .version = current_version,
            .runtime_version = current_runtime_version,
            .architecture_len = 0,
            .tensor_count = 0,
            .metadata_count = 0,
            .metadata_blob_len = 0,
            .alignment = default_alignment,
            .quantization_version = 0,
            .reserved = .{0} ** 8,
        };
    }

    pub fn validate(self: ZiggyHeader) !void {
        if (!std.mem.eql(u8, &self.magic, magic)) return error.InvalidMagic;
        if (self.version != current_version) return error.UnsupportedVersion;
        if (self.runtime_version > current_runtime_version) return error.UnsupportedRuntimeVersion;
        if (self.alignment == 0 or !std.math.isPowerOfTwo(self.alignment)) return error.InvalidAlignment;
    }
};

pub const TensorRecord = extern struct {
    name_len: u32,
    original_gguf_type: u32,
    compiled_layout_kind: u32,
    kernel_family: u32,
    layer_index: u32, // u32 max means no layer
    rows: u64,
    cols: u64,
    byte_offset: u64,
    byte_length: u64,
    row_stride: u64,
    tile_stride: u64,
    decode_metadata: DecodeMetadata,
    reserved: [8]u8,
};

pub const TensorInfo = struct {
    name: []const u8,
    original_gguf_type: gguf.TensorType,
    compiled_layout_kind: CompiledLayoutKind,
    kernel_family: KernelFamily,
    layer_index: ?u32,
    rows: usize,
    cols: usize,
    byte_offset: u64,
    byte_length: usize,
    row_stride: usize,
    tile_stride: usize,
    decode_metadata: DecodeMetadata,

    pub fn fromRecord(allocator: std.mem.Allocator, record: TensorRecord, name_bytes: []const u8) !TensorInfo {
        const name = try allocator.dupe(u8, name_bytes);
        const layer_index: ?u32 = if (record.layer_index == std.math.maxInt(u32)) null else record.layer_index;
        return .{
            .name = name,
            .original_gguf_type = @enumFromInt(record.original_gguf_type),
            .compiled_layout_kind = @enumFromInt(record.compiled_layout_kind),
            .kernel_family = @enumFromInt(record.kernel_family),
            .layer_index = layer_index,
            .rows = record.rows,
            .cols = record.cols,
            .byte_offset = record.byte_offset,
            .byte_length = record.byte_length,
            .row_stride = record.row_stride,
            .tile_stride = record.tile_stride,
            .decode_metadata = record.decode_metadata,
        };
    }

    pub fn deinit(self: *TensorInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.* = undefined;
    }
};

pub const CompiledModel = struct {
    allocator: std.mem.Allocator,
    header: ZiggyHeader,
    architecture: []const u8,
    metadata_blob: []const u8,
    compiled_metadata_blob: []const u8,
    tensors: std.StringHashMap(TensorInfo),
    data_offset: u64,
    bytes: []align(16384) const u8,
    mmap_len: usize,

    pub fn init(allocator: std.mem.Allocator) CompiledModel {
        return .{
            .allocator = allocator,
            .header = undefined,
            .architecture = &.{},
            .metadata_blob = &.{},
            .compiled_metadata_blob = &.{},
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .data_offset = 0,
            .bytes = &.{},
            .mmap_len = 0,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        if (self.mmap_len > 0) {
            std.posix.munmap(@constCast(self.bytes[0..self.mmap_len]));
        }
        var tensor_iter = self.tensors.valueIterator();
        while (tensor_iter.next()) |tensor| {
            var mutable_tensor = tensor.*;
            mutable_tensor.deinit(self.allocator);
        }
        self.tensors.deinit();
        self.allocator.free(self.architecture);
        self.* = undefined;
    }

    pub fn getTensor(self: *const CompiledModel, name: []const u8) ?*const TensorInfo {
        return self.tensors.getPtr(name);
    }

    pub fn tensorBytes(self: *const CompiledModel, tensor: *const TensorInfo) []const u8 {
        const start = self.data_offset + tensor.byte_offset;
        const end = start + tensor.byte_length;
        if (end > self.bytes.len) return &.{};
        return self.bytes[start..end];
    }
};

pub const CompileOptions = struct {
    repack_q4_k: bool = true,
    repack_q6_k: bool = true,
    repack_q8_0: bool = true,
    keep_raw_for_all: bool = false,
    use_calibration_plan: bool = false,
    calibration_plan: ?*const calibration.Plan = null,
    emit_gated_ffn_metadata: bool = false,
};

pub const LayoutPolicy = enum {
    // Always use the original GGUF format (no repacking)
    passthrough,
    // Use packed layouts for all supported formats (Q4_K, Q6_K, Q8_0)
    packed_for_metal,
    // Use calibration plan to determine per-tensor formats
    calibration_based,
};

pub fn deriveCompiledPath(allocator: std.mem.Allocator, model_path: []const u8) ![]u8 {
    if (std.mem.endsWith(u8, model_path, ".ziggy")) return allocator.dupe(u8, model_path);
    if (std.mem.endsWith(u8, model_path, ".gguf")) {
        return std.fmt.allocPrint(allocator, "{s}.ziggy", .{model_path[0 .. model_path.len - 5]});
    }
    return std.fmt.allocPrint(allocator, "{s}.ziggy", .{model_path});
}

pub fn deriveSourceGgufPath(allocator: std.mem.Allocator, model_path: []const u8) ![]u8 {
    if (std.mem.endsWith(u8, model_path, ".gguf")) return allocator.dupe(u8, model_path);
    if (std.mem.endsWith(u8, model_path, ".ziggy")) {
        return std.fmt.allocPrint(allocator, "{s}.gguf", .{model_path[0 .. model_path.len - 6]});
    }
    return allocator.dupe(u8, model_path);
}

pub fn compileFromGGUF(
    allocator: std.mem.Allocator,
    gguf_path: []const u8,
    output_path: []const u8,
    options: CompileOptions,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    const gguf_model = try llama.loadModel(arena_allocator, gguf_path);
    var metadata_blob = try llama.extractMetadataBlob(arena_allocator, gguf_path);
    defer metadata_blob.deinit(arena_allocator);
    var owned_calibration_plan: ?calibration.Plan = null;
    defer if (owned_calibration_plan) |*plan| plan.deinit(allocator);
    const gating_plan = blk: {
        if (!options.emit_gated_ffn_metadata) break :blk null;
        if (options.calibration_plan) |plan| break :blk plan;
        owned_calibration_plan = try llama.buildGatedFfnCalibrationPlan(allocator, &gguf_model);
        break :blk &owned_calibration_plan.?;
    };
    const compiled_metadata_blob = if (gating_plan) |plan|
        try buildCompiledMetadataBlob(arena_allocator, plan)
    else
        &.{};

    var tensor_sources = std.ArrayList(TensorSource).empty;
    defer tensor_sources.deinit(arena_allocator);

    try appendTensorSource(arena_allocator, &tensor_sources, "token_embd.weight", gguf_model.token_embd);
    try appendTensorSource(arena_allocator, &tensor_sources, "output.weight", gguf_model.output);
    try appendTensorSource(arena_allocator, &tensor_sources, "output_norm.weight", gguf_model.output_norm);

    for (gguf_model.layers, 0..) |layer, index| {
        const prefix = try std.fmt.allocPrint(arena_allocator, "blk.{d}.", .{index});
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_norm.weight", layer.attn_norm);
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_q.weight", layer.attn_q);
        if (layer.attn_q_bias) |tensor| try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_q.bias", tensor);
        if (layer.attn_q_norm) |tensor| try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_q_norm.weight", tensor);
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_k.weight", layer.attn_k);
        if (layer.attn_k_bias) |tensor| try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_k.bias", tensor);
        if (layer.attn_k_norm) |tensor| try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_k_norm.weight", tensor);
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_v.weight", layer.attn_v);
        if (layer.attn_v_bias) |tensor| try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_v.bias", tensor);
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "attn_output.weight", layer.attn_output);
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "ffn_norm.weight", layer.ffn_norm);
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "ffn_gate.weight", layer.ffn_gate);
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "ffn_down.weight", layer.ffn_down);
        try appendNamedTensorSource(arena_allocator, &tensor_sources, prefix, "ffn_up.weight", layer.ffn_up);
    }

    var metadata = std.ArrayList(u8).empty;
    defer metadata.deinit(arena_allocator);
    var data = std.ArrayList(u8).empty;
    defer data.deinit(arena_allocator);

    for (tensor_sources.items) |source| {
        const entry = try buildTensorEntry(arena_allocator, source.name, source.tensor, &gguf_model, options);
        const aligned_offset = alignForward(@intCast(data.items.len), default_alignment);
        if (aligned_offset > data.items.len) {
            try data.appendNTimes(arena_allocator, 0, @intCast(aligned_offset - data.items.len));
        }

        var record = entry.record;
        record.byte_offset = aligned_offset;
        record.byte_length = entry.payload.len;

        try metadata.appendSlice(arena_allocator, std.mem.asBytes(&record));
        try metadata.appendSlice(arena_allocator, source.name);
        const aligned_metadata_len = alignForward(@intCast(metadata.items.len), 8);
        if (aligned_metadata_len > metadata.items.len) {
            try metadata.appendNTimes(arena_allocator, 0, @intCast(aligned_metadata_len - metadata.items.len));
        }

        try data.appendSlice(arena_allocator, entry.payload);
    }

    var header = ZiggyHeader.init();
    header.architecture_len = @intCast(gguf_model.architecture.len);
    header.tensor_count = @intCast(tensor_sources.items.len);
    header.metadata_count = metadata_blob.count;
    header.metadata_blob_len = metadata_blob.bytes.len;
    header.setCompiledMetadataBlobLen(compiled_metadata_blob.len);
    header.quantization_version = gguf_model.quantization_version;
    header.alignment = default_alignment;

    var builder = CompiledFileBuilder.init(arena_allocator);
    try builder.writeHeader(header);
    try builder.writeArchitecture(gguf_model.architecture);
    try builder.buffer.appendSlice(arena_allocator, metadata_blob.bytes);
    try builder.buffer.appendSlice(arena_allocator, compiled_metadata_blob);
    try builder.padTo(alignForward(builder.currentPos(), header.alignment));
    try builder.buffer.appendSlice(arena_allocator, metadata.items);
    try builder.padTo(alignForward(builder.currentPos(), header.alignment));
    try builder.buffer.appendSlice(arena_allocator, data.items);
    try builder.writeToFile(output_path);
}

const TensorSource = struct {
    name: []const u8,
    tensor: llama.TensorRef,
};

const TensorEntry = struct {
    record: TensorRecord,
    payload: []const u8,
};

const CompiledFileBuilder = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayList(u8),

    fn init(allocator: std.mem.Allocator) CompiledFileBuilder {
        return .{
            .allocator = allocator,
            .buffer = std.ArrayList(u8).empty,
        };
    }

    fn currentPos(self: *const CompiledFileBuilder) u64 {
        return @intCast(self.buffer.items.len);
    }

    fn writeHeader(self: *CompiledFileBuilder, header: ZiggyHeader) !void {
        try self.buffer.appendSlice(self.allocator, std.mem.asBytes(&header));
    }

    fn writeArchitecture(self: *CompiledFileBuilder, arch: []const u8) !void {
        try self.buffer.appendSlice(self.allocator, arch);
    }

    fn padTo(self: *CompiledFileBuilder, target_pos: u64) !void {
        const current = self.currentPos();
        if (target_pos > current) {
            const padding = target_pos - current;
            try self.buffer.appendNTimes(self.allocator, 0, @intCast(padding));
        }
    }

    fn writeBytesAt(self: *CompiledFileBuilder, offset: u64, bytes: []const u8) !void {
        const end = offset + bytes.len;
        if (end > self.buffer.items.len) {
            try self.buffer.resize(self.allocator, @intCast(end));
        }
        @memcpy(self.buffer.items[@intCast(offset)..@intCast(end)], bytes);
    }

    fn writeToFile(self: *const CompiledFileBuilder, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(self.buffer.items);
    }
};

fn writeNamedTensorRecord(
    builder: *CompiledFileBuilder,
    table_start: u64,
    data_start: u64,
    record_index: *u64,
    current_offset: *u64,
    prefix: []const u8,
    suffix: []const u8,
    tensor: llama.TensorRef,
    model: *const llama.Model,
    options: CompileOptions,
) !void {
    const name = try std.mem.concat(builder.allocator, u8, &.{ prefix, suffix });
    defer builder.allocator.free(name);
    try writeTensorRecord(builder, table_start, data_start, record_index, current_offset, name, tensor, model, options);
}

fn appendTensorSource(
    allocator: std.mem.Allocator,
    tensor_sources: *std.ArrayList(TensorSource),
    name: []const u8,
    tensor: llama.TensorRef,
) !void {
    try tensor_sources.append(allocator, .{
        .name = try allocator.dupe(u8, name),
        .tensor = tensor,
    });
}

fn appendNamedTensorSource(
    allocator: std.mem.Allocator,
    tensor_sources: *std.ArrayList(TensorSource),
    prefix: []const u8,
    suffix: []const u8,
    tensor: llama.TensorRef,
) !void {
    const name = try std.mem.concat(allocator, u8, &.{ prefix, suffix });
    try tensor_sources.append(allocator, .{
        .name = name,
        .tensor = tensor,
    });
}

fn buildTensorEntry(
    allocator: std.mem.Allocator,
    name: []const u8,
    tensor: llama.TensorRef,
    model: *const llama.Model,
    options: CompileOptions,
) !TensorEntry {
    const rows = try tensor.rowCount();
    const cols = try tensor.rowLen();
    const gguf_type = tensor.tensor_type;
    const layer_index = extractLayerIndex(name);
    const kernel_family = KernelFamily.fromTensorName(name);
    const layout_kind = detectLayoutKind(gguf_type, name, options);
    const tensor_bytes = try llama.tensorBytes(model, tensor);

    var record = TensorRecord{
        .name_len = @intCast(name.len),
        .original_gguf_type = @intFromEnum(gguf_type),
        .compiled_layout_kind = @intFromEnum(layout_kind),
        .kernel_family = @intFromEnum(kernel_family),
        .layer_index = layer_index orelse std.math.maxInt(u32),
        .rows = rows,
        .cols = cols,
        .byte_offset = 0,
        .byte_length = 0,
        .row_stride = 0,
        .tile_stride = 0,
        .decode_metadata = DecodeMetadata.forLayout(layout_kind),
        .reserved = .{0} ** 8,
    };

    const payload = if (layout_kind == .q4_k_m_packed and options.repack_q4_k) blk: {
        const packed_result = try moon_quant.packQ4KTensor(allocator, tensor_bytes, rows, cols);
        record.row_stride = moon_quant.q4_k_packed_block_bytes * (cols / moon_quant.q4_k_block_values);
        break :blk packed_result.bytes;
    } else if (layout_kind == .q6_k_raw and options.repack_q6_k) blk: {
        const packed_result = try moon_quant.packQ6KTensor(allocator, tensor_bytes, rows, cols);
        record.row_stride = moon_quant.q6_k_packed_block_bytes * (cols / moon_quant.q6_k_block_values);
        break :blk packed_result.bytes;
    } else if (layout_kind == .q8_0_raw and options.repack_q8_0) blk: {
        const packed_result = try moon_quant.packQ8_0Tensor(allocator, tensor_bytes, rows, cols);
        record.row_stride = moon_quant.q8_0_packed_block_bytes * (cols / moon_quant.q8_0_block_values);
        break :blk packed_result.bytes;
    } else blk: {
        record.row_stride = try llama.tensorRowByteSize(gguf_type, cols);
        break :blk tensor_bytes;
    };
    record.tile_stride = record.row_stride;

    return .{
        .record = record,
        .payload = payload,
    };
}

fn writeTensorRecord(
    builder: *CompiledFileBuilder,
    table_start: u64,
    data_start: u64,
    record_index: *u64,
    current_offset: *u64,
    name: []const u8,
    tensor: llama.TensorRef,
    model: *const llama.Model,
    options: CompileOptions,
) !void {
    const rows = try tensor.rowCount();
    const cols = try tensor.rowLen();
    const gguf_type = tensor.tensor_type;
    const layer_index = extractLayerIndex(name);
    const kernel_family = KernelFamily.fromTensorName(name);
    const layout_kind = detectLayoutKind(gguf_type, name, options);

    var record = TensorRecord{
        .name_len = @intCast(name.len),
        .original_gguf_type = @intFromEnum(gguf_type),
        .compiled_layout_kind = @intFromEnum(layout_kind),
        .kernel_family = @intFromEnum(kernel_family),
        .layer_index = layer_index orelse std.math.maxInt(u32),
        .rows = rows,
        .cols = cols,
        .byte_offset = current_offset.*,
        .byte_length = 0,
        .row_stride = 0,
        .tile_stride = 0,
        .decode_metadata = DecodeMetadata.forLayout(layout_kind),
        .reserved = .{0} ** 8,
    };

    if (layout_kind == .q4_k_m_packed and options.repack_q4_k) {
        record.row_stride = moon_quant.q4_k_packed_block_bytes * (cols / moon_quant.q4_k_block_values);
    } else if (layout_kind == .q6_k_raw and options.repack_q6_k) {
        record.row_stride = moon_quant.q6_k_packed_block_bytes * (cols / moon_quant.q6_k_block_values);
    } else if (layout_kind == .q8_0_raw and options.repack_q8_0) {
        record.row_stride = moon_quant.q8_0_packed_block_bytes * (cols / moon_quant.q8_0_block_values);
    } else {
        record.row_stride = try llama.tensorRowByteSize(gguf_type, cols);
    }
    record.tile_stride = record.row_stride;

    const record_offset = table_start + record_index.* * @sizeOf(TensorRecord);

    const tensor_bytes = try llama.tensorBytes(model, tensor);

    if (layout_kind == .q4_k_m_packed and options.repack_q4_k) {
        const packed_result = try moon_quant.packQ4KTensor(builder.allocator, tensor_bytes, rows, cols);
        record.byte_length = packed_result.bytes.len;

        // Calculate actual data position after alignment
        const data_offset = alignForward(builder.currentPos(), default_alignment);
        try builder.padTo(data_offset);

        // byte_offset is relative to data section start
        record.byte_offset = data_offset - data_start;

        try builder.writeBytesAt(record_offset, std.mem.asBytes(&record));
        try builder.writeBytesAt(record_offset + @sizeOf(TensorRecord), name);

        try builder.buffer.appendSlice(builder.allocator, packed_result.bytes);
        current_offset.* = data_offset - data_start + packed_result.bytes.len;
    } else if (layout_kind == .q6_k_raw and options.repack_q6_k) {
        const packed_result = try moon_quant.packQ6KTensor(builder.allocator, tensor_bytes, rows, cols);
        record.byte_length = packed_result.bytes.len;

        const data_offset = alignForward(builder.currentPos(), default_alignment);
        try builder.padTo(data_offset);

        record.byte_offset = data_offset - data_start;

        try builder.writeBytesAt(record_offset, std.mem.asBytes(&record));
        try builder.writeBytesAt(record_offset + @sizeOf(TensorRecord), name);

        try builder.buffer.appendSlice(builder.allocator, packed_result.bytes);
        current_offset.* = data_offset - data_start + packed_result.bytes.len;
    } else if (layout_kind == .q8_0_raw and options.repack_q8_0) {
        const packed_result = try moon_quant.packQ8_0Tensor(builder.allocator, tensor_bytes, rows, cols);
        record.byte_length = packed_result.bytes.len;

        const data_offset = alignForward(builder.currentPos(), default_alignment);
        try builder.padTo(data_offset);

        record.byte_offset = data_offset - data_start;

        try builder.writeBytesAt(record_offset, std.mem.asBytes(&record));
        try builder.writeBytesAt(record_offset + @sizeOf(TensorRecord), name);

        try builder.buffer.appendSlice(builder.allocator, packed_result.bytes);
        current_offset.* = data_offset - data_start + packed_result.bytes.len;
    } else {
        const row_size = try llama.tensorRowByteSize(gguf_type, cols);
        const total_bytes = rows * row_size;
        record.byte_length = total_bytes;

        const data_offset = alignForward(builder.currentPos(), default_alignment);
        try builder.padTo(data_offset);

        record.byte_offset = data_offset - data_start;

        try builder.writeBytesAt(record_offset, std.mem.asBytes(&record));
        try builder.writeBytesAt(record_offset + @sizeOf(TensorRecord), name);

        try builder.buffer.appendSlice(builder.allocator, tensor_bytes);
        current_offset.* = data_offset - data_start + total_bytes;
    }

    record_index.* += 1;
}

fn extractLayerIndex(name: []const u8) ?u32 {
    // Look for "blk.N." pattern where N is the layer index
    const prefix = "blk.";
    if (std.mem.indexOf(u8, name, prefix)) |start| {
        const num_start = start + prefix.len;
        if (num_start >= name.len) return null;

        var end = num_start;
        while (end < name.len and std.ascii.isDigit(name[end])) {
            end += 1;
        }

        if (end > num_start) {
            return std.fmt.parseInt(u32, name[num_start..end], 10) catch null;
        }
    }
    return null;
}

fn detectLayoutKind(tensor_type: llama.TensorType, name: []const u8, options: CompileOptions) CompiledLayoutKind {
    if (std.mem.eql(u8, name, "token_embd.weight")) {
        return switch (tensor_type) {
            .f16 => .f16_raw,
            .f32 => .f32_raw,
            else => .generic_quant_raw,
        };
    }

    // If calibration-based layout policy is enabled, try to use the calibration plan
    if (options.use_calibration_plan) {
        if (options.calibration_plan) |plan| {
            // Find matching entry in calibration plan
            for (plan.entries) |entry| {
                if (std.mem.eql(u8, entry.name, name)) {
                    // Use the target format from calibration
                    return switch (entry.target_format) {
                        .q4_k_m => .q4_k_m_packed,
                        .q5_k_m => .q5_k_m_packed,
                        .q6_k => .q6_k_raw,
                        .q8_0 => .q8_0_raw,
                        .f16_reference => .f16_raw,
                        .f32_reference => .f32_raw,
                        .legacy_q4_k => if (options.repack_q4_k) .q4_k_m_packed else .generic_quant_raw,
                        .legacy_q6_k => .q6_k_raw,
                        else => .generic_quant_raw,
                    };
                }
            }
        }
    }

    // Default layout detection based on tensor type
    return switch (tensor_type) {
        .q4_k => if (options.repack_q4_k) .q4_k_m_packed else .generic_quant_raw,
        .q6_k => if (options.repack_q6_k) .q6_k_raw else .generic_quant_raw,
        .q8_0 => if (options.repack_q8_0) .q8_0_raw else .generic_quant_raw,
        .f16 => .f16_raw,
        .f32 => .f32_raw,
    };
}

pub fn loadCompiledModel(allocator: std.mem.Allocator, path: []const u8) !CompiledModel {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const size = std.math.cast(usize, stat.size) orelse return error.Overflow;

    const mapped_bytes = try std.posix.mmap(
        null,
        size,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    errdefer std.posix.munmap(mapped_bytes);

    var model = CompiledModel.init(allocator);
    model.mmap_len = size;
    model.bytes = @alignCast(mapped_bytes);

    var pos: usize = 0;
    if (mapped_bytes.len < @sizeOf(ZiggyHeader)) return error.TruncatedFile;
    model.header = std.mem.bytesToValue(ZiggyHeader, mapped_bytes[0..@sizeOf(ZiggyHeader)]);
    try model.header.validate();
    pos = @sizeOf(ZiggyHeader);

    if (pos + model.header.architecture_len > mapped_bytes.len) return error.TruncatedFile;
    model.architecture = try allocator.dupe(u8, mapped_bytes[pos..][0..model.header.architecture_len]);
    pos += model.header.architecture_len;
    if (pos + model.header.metadata_blob_len > mapped_bytes.len) return error.TruncatedFile;
    model.metadata_blob = mapped_bytes[pos .. pos + model.header.metadata_blob_len];
    pos += model.header.metadata_blob_len;
    const compiled_metadata_blob_len: usize = @intCast(model.header.compiledMetadataBlobLen());
    if (pos + compiled_metadata_blob_len > mapped_bytes.len) return error.TruncatedFile;
    model.compiled_metadata_blob = mapped_bytes[pos .. pos + compiled_metadata_blob_len];
    pos += compiled_metadata_blob_len;

    pos = alignForward(pos, model.header.alignment);

    var tensor_index: u64 = 0;
    while (tensor_index < model.header.tensor_count) : (tensor_index += 1) {
        if (pos + @sizeOf(TensorRecord) > mapped_bytes.len) return error.TruncatedFile;
        const record = std.mem.bytesToValue(TensorRecord, mapped_bytes[pos..][0..@sizeOf(TensorRecord)]);
        pos += @sizeOf(TensorRecord);

        if (pos + record.name_len > mapped_bytes.len) return error.TruncatedFile;
        const name_bytes = mapped_bytes[pos..][0..record.name_len];
        pos += record.name_len;

        const tensor_info = try TensorInfo.fromRecord(allocator, record, name_bytes);
        try model.tensors.put(tensor_info.name, tensor_info);

        pos = alignForward(pos, 8);
    }

    model.data_offset = alignForward(pos, model.header.alignment);

    return model;
}

pub fn chatTemplateStyle(allocator: std.mem.Allocator, path: []const u8) !gguf.ChatTemplateStyle {
    var model = try loadCompiledModel(allocator, path);
    defer model.deinit();
    return llama.detectChatTemplateStyleInMetadataBlob(allocator, model.header.metadata_count, model.metadata_blob);
}

pub fn loadExecutionModel(allocator: std.mem.Allocator, path: []const u8) !llama.Model {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const size = std.math.cast(usize, stat.size) orelse return error.Overflow;

    const mapped_bytes = try std.posix.mmap(
        null,
        size,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    errdefer std.posix.munmap(mapped_bytes);

    var pos: usize = 0;
    if (mapped_bytes.len < @sizeOf(ZiggyHeader)) return error.TruncatedFile;
    const header = std.mem.bytesToValue(ZiggyHeader, mapped_bytes[0..@sizeOf(ZiggyHeader)]);
    try header.validate();
    pos = @sizeOf(ZiggyHeader);

    if (pos + header.architecture_len > mapped_bytes.len) return error.TruncatedFile;
    pos += header.architecture_len;

    if (pos + header.metadata_blob_len > mapped_bytes.len) return error.TruncatedFile;
    const metadata_bytes = mapped_bytes[pos .. pos + header.metadata_blob_len];
    pos += header.metadata_blob_len;
    const compiled_metadata_blob_len: usize = @intCast(header.compiledMetadataBlobLen());
    if (pos + compiled_metadata_blob_len > mapped_bytes.len) return error.TruncatedFile;
    pos += compiled_metadata_blob_len;
    pos = @intCast(alignForward(pos, header.alignment));

    var tensor_refs = try allocator.alloc(llama.ExternalTensorRef, @intCast(header.tensor_count));
    errdefer allocator.free(tensor_refs);

    var tensor_index: usize = 0;
    while (tensor_index < tensor_refs.len) : (tensor_index += 1) {
        if (pos + @sizeOf(TensorRecord) > mapped_bytes.len) return error.TruncatedFile;
        const record = std.mem.bytesToValue(TensorRecord, mapped_bytes[pos..][0..@sizeOf(TensorRecord)]);
        pos += @sizeOf(TensorRecord);

        if (pos + record.name_len > mapped_bytes.len) return error.TruncatedFile;
        const name_bytes = mapped_bytes[pos..][0..record.name_len];
        pos += record.name_len;

        var dims = [_]u64{ 1, 1, 1, 1 };
        dims[0] = record.cols;
        dims[1] = record.rows;
        tensor_refs[tensor_index] = .{
            .name = try allocator.dupe(u8, name_bytes),
            .dims = dims,
            .n_dims = 2,
            .tensor_type = @enumFromInt(record.original_gguf_type),
            .offset = record.byte_offset,
        };

        pos = @intCast(alignForward(pos, 8));
    }

    const data_offset: usize = @intCast(alignForward(pos, header.alignment));
    errdefer {
        for (tensor_refs) |tensor_ref| allocator.free(tensor_ref.name);
        allocator.free(tensor_refs);
    }

    const model = try llama.loadModelFromMetadataAndTensors(
        allocator,
        @alignCast(mapped_bytes),
        header.metadata_count,
        metadata_bytes,
        data_offset,
        tensor_refs,
    );

    for (tensor_refs) |tensor_ref| allocator.free(tensor_ref.name);
    allocator.free(tensor_refs);
    return model;
}

fn alignForward(value: u64, alignment: u32) u64 {
    const a = @as(u64, alignment);
    return (value + a - 1) & ~(a - 1);
}

pub const InspectReport = struct {
    path: []const u8,
    version: u32,
    runtime_version: u32,
    architecture: []const u8,
    tensor_count: u64,
    compiled_metadata_bytes: u64,
    quantization_version: u32,
    alignment: u32,
    data_offset: u64,
    total_size: u64,
};

pub fn inspectFile(allocator: std.mem.Allocator, path: []const u8) !InspectReport {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const size = std.math.cast(usize, stat.size) orelse return error.Overflow;

    if (size < @sizeOf(ZiggyHeader)) return error.TruncatedFile;

    var header_buf: [@sizeOf(ZiggyHeader)]u8 = undefined;
    _ = try file.readAll(&header_buf);
    const header = std.mem.bytesToValue(ZiggyHeader, &header_buf);
    try header.validate();

    const arch_buf = try allocator.alloc(u8, header.architecture_len);
    defer allocator.free(arch_buf);
    _ = try file.readAll(arch_buf);

    return .{
        .path = path,
        .version = header.version,
        .runtime_version = header.runtime_version,
        .architecture = try allocator.dupe(u8, arch_buf),
        .tensor_count = header.tensor_count,
        .compiled_metadata_bytes = header.compiledMetadataBlobLen(),
        .quantization_version = header.quantization_version,
        .alignment = header.alignment,
        .data_offset = 0,
        .total_size = size,
    };
}

pub fn printInspectReport(writer: anytype, report: InspectReport) !void {
    try writer.print(
        \\model: {s}
        \\format: ZIGY
        \\version: {d}
        \\runtime_version: {d}
        \\architecture: {s}
        \\tensor_count: {d}
        \\compiled_metadata_bytes: {d}
        \\quantization_version: {d}
        \\alignment: {d}
        \\total_size: {d}
        \\
    , .{
        report.path,
        report.version,
        report.runtime_version,
        report.architecture,
        report.tensor_count,
        report.compiled_metadata_bytes,
        report.quantization_version,
        report.alignment,
        report.total_size,
    });
}

pub const GatedFfnMetadata = struct {
    layer_index: u32,
    threshold: f32,
    active_block_ratio: f32,
    avg_active_blocks: f32,
    avg_total_blocks: f32,
};

pub fn parseGatedFfnMetadata(
    allocator: std.mem.Allocator,
    blob: []const u8,
) ![]GatedFfnMetadata {
    var list = std.ArrayList(GatedFfnMetadata).empty;
    errdefer list.deinit(allocator);

    var lines = std.mem.splitScalar(u8, blob, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "gated_ffn.layer.")) continue;
        var rest = line["gated_ffn.layer.".len..];
        const eq_index = std.mem.indexOfScalar(u8, rest, '=') orelse continue;
        const layer_index = std.fmt.parseUnsigned(u32, rest[0..eq_index], 10) catch continue;
        rest = rest[eq_index + 1 ..];

        var fields = std.mem.splitScalar(u8, rest, ',');
        const threshold = parseFloatField(fields.next() orelse continue, "threshold") orelse continue;
        const active_block_ratio = parseFloatField(fields.next() orelse continue, "active_block_ratio") orelse continue;
        const avg_active_blocks = parseFloatField(fields.next() orelse continue, "avg_active_blocks") orelse continue;
        const avg_total_blocks = parseFloatField(fields.next() orelse continue, "avg_total_blocks") orelse continue;
        try list.append(allocator, .{
            .layer_index = layer_index,
            .threshold = threshold,
            .active_block_ratio = active_block_ratio,
            .avg_active_blocks = avg_active_blocks,
            .avg_total_blocks = avg_total_blocks,
        });
    }
    return try list.toOwnedSlice(allocator);
}

fn buildCompiledMetadataBlob(
    allocator: std.mem.Allocator,
    plan: *const calibration.Plan,
) ![]u8 {
    var blob = std.ArrayList(u8).empty;
    errdefer blob.deinit(allocator);

    for (plan.gated_ffn_policies) |policy| {
        if (!policy.selected) continue;
        try blob.writer(allocator).print(
            "gated_ffn.layer.{d}=threshold={d:.6},active_block_ratio={d:.6},avg_active_blocks={d:.3},avg_total_blocks={d:.3}\n",
            .{
                policy.layer_index,
                policy.threshold,
                policy.active_block_ratio,
                policy.avg_active_blocks,
                policy.avg_total_blocks,
            },
        );
    }
    return try blob.toOwnedSlice(allocator);
}

fn parseFloatField(field: []const u8, key: []const u8) ?f32 {
    if (!std.mem.startsWith(u8, field, key)) return null;
    if (field.len <= key.len + 1 or field[key.len] != '=') return null;
    return std.fmt.parseFloat(f32, field[key.len + 1 ..]) catch null;
}

pub const ZiggyError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedRuntimeVersion,
    InvalidAlignment,
    TruncatedFile,
    Overflow,
};

test "ZIGY header round-trip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const test_path = "test.ziggy";
    const test_architecture = "llama";

    var header = ZiggyHeader.init();
    header.architecture_len = @intCast(test_architecture.len);
    header.tensor_count = 0;
    header.quantization_version = 2;

    const file = try tmp.dir.createFile(test_path, .{});
    defer file.close();
    try file.writeAll(std.mem.asBytes(&header));
    try file.writeAll(test_architecture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, test_path);
    defer std.testing.allocator.free(path);

    const report = try inspectFile(std.testing.allocator, path);
    defer std.testing.allocator.free(report.architecture);

    try std.testing.expectEqual(current_version, report.version);
    try std.testing.expectEqual(current_runtime_version, report.runtime_version);
    try std.testing.expectEqualStrings(test_architecture, report.architecture);
    try std.testing.expectEqual(@as(u32, 2), report.quantization_version);
}

test "ZIGY magic validation rejects invalid files" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("bad.ziggy", .{});
    defer file.close();
    var bytes = [_]u8{0} ** @sizeOf(ZiggyHeader);
    @memcpy(bytes[0..4], "NOPE");
    try file.writeAll(&bytes);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "bad.ziggy");
    defer std.testing.allocator.free(path);

    try std.testing.expectError(error.InvalidMagic, inspectFile(std.testing.allocator, path));
}

test "compileFromGGUF writes a loadable ZIGY container" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "fixture.gguf", fixture);

    const gguf_path = try tmp.dir.realpathAlloc(std.testing.allocator, "fixture.gguf");
    defer std.testing.allocator.free(gguf_path);
    const ziggy_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(ziggy_path);

    const output_path = try std.fmt.allocPrint(std.testing.allocator, "{s}/fixture.ziggy", .{ziggy_path});
    defer std.testing.allocator.free(output_path);

    try compileFromGGUF(std.testing.allocator, gguf_path, output_path, .{});

    var model = try loadCompiledModel(std.testing.allocator, output_path);
    defer model.deinit();

    try std.testing.expectEqualStrings("llama", model.architecture);
    try std.testing.expect(model.header.tensor_count > 0);
    try std.testing.expect(model.getTensor("token_embd.weight") != null);
    try std.testing.expect(model.getTensor("blk.0.attn_q.weight") != null);
}
