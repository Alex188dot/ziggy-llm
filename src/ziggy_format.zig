const std = @import("std");
const gguf = @import("gguf.zig");
const moon_quant = @import("moon_quant.zig");
const llama = @import("llama_cpu.zig");

pub const magic = "ZIGY";
pub const current_version: u32 = 1;
pub const current_runtime_version: u32 = 1;
pub const default_alignment: u32 = 32;

pub const CompiledLayoutKind = enum(u32) {
    q4_k_m_packed = 0,
    q6_k_raw = 1,
    q8_0_raw = 2,
    f16_raw = 3,
    f32_raw = 4,
    generic_quant_raw = 5,

    pub fn label(self: CompiledLayoutKind) []const u8 {
        return switch (self) {
            .q4_k_m_packed => "q4_k_m_packed",
            .q6_k_raw => "q6_k_raw",
            .q8_0_raw => "q8_0_raw",
            .f16_raw => "f16_raw",
            .f32_raw => "f32_raw",
            .generic_quant_raw => "generic_quant_raw",
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
    alignment: u32,
    quantization_version: u32,
    reserved: [16]u8,

    pub fn init() ZiggyHeader {
        return .{
            .magic = magic.*,
            .version = current_version,
            .runtime_version = current_runtime_version,
            .architecture_len = 0,
            .tensor_count = 0,
            .metadata_count = 0,
            .alignment = default_alignment,
            .quantization_version = 0,
            .reserved = .{0} ** 16,
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
    rows: u64,
    cols: u64,
    byte_offset: u64,
    byte_length: u64,
    row_stride: u64,
    tile_stride: u64,
    reserved: [16]u8,
};

pub const TensorInfo = struct {
    name: []const u8,
    original_gguf_type: gguf.TensorType,
    compiled_layout_kind: CompiledLayoutKind,
    rows: usize,
    cols: usize,
    byte_offset: u64,
    byte_length: usize,
    row_stride: usize,
    tile_stride: usize,

    pub fn fromRecord(allocator: std.mem.Allocator, record: TensorRecord, name_bytes: []const u8) !TensorInfo {
        const name = try allocator.dupe(u8, name_bytes);
        return .{
            .name = name,
            .original_gguf_type = @enumFromInt(record.original_gguf_type),
            .compiled_layout_kind = @enumFromInt(record.compiled_layout_kind),
            .rows = record.rows,
            .cols = record.cols,
            .byte_offset = record.byte_offset,
            .byte_length = record.byte_length,
            .row_stride = record.row_stride,
            .tile_stride = record.tile_stride,
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
    tensors: std.StringHashMap(TensorInfo),
    data_offset: u64,
    bytes: []align(16384) const u8,
    mmap_len: usize,

    pub fn init(allocator: std.mem.Allocator) CompiledModel {
        return .{
            .allocator = allocator,
            .header = undefined,
            .architecture = &.{},
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
        if (end > self.bytes.len) return &[]const u8{};
        return self.bytes[start..end];
    }
};

pub const CompileOptions = struct {
    repack_q4_k: bool = true,
    keep_raw_for_all: bool = false,
};

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

    var builder = CompiledFileBuilder.init(arena_allocator);

    var header = ZiggyHeader.init();
    header.architecture_len = @intCast(gguf_model.architecture.len);
    header.tensor_count = @intCast(gguf_model.layers.len * 13 + 3);
    header.metadata_count = 0;
    header.quantization_version = gguf_model.quantization_version;
    header.alignment = default_alignment;

    try builder.writeHeader(header);
    try builder.writeArchitecture(gguf_model.architecture);

    const tensor_table_start = builder.currentPos();
    const tensor_record_size = @sizeOf(TensorRecord);
    const tensor_table_size = tensor_record_size * header.tensor_count;
    try builder.padTo(tensor_table_start + tensor_table_size);

    const data_start = alignForward(builder.currentPos(), default_alignment);
    try builder.padTo(data_start);

    var current_offset: u64 = 0;
    var record_index: u64 = 0;

    try writeTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, "token_embd.weight", gguf_model.token_embd, &gguf_model, options);
    try writeTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, "output.weight", gguf_model.output, &gguf_model, options);
    try writeTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, "output_norm.weight", gguf_model.output_norm, &gguf_model, options);

    for (gguf_model.layers, 0..) |layer, index| {
        const prefix = try std.fmt.allocPrint(arena_allocator, "blk.{d}.", .{index});

        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_norm.weight", layer.attn_norm, &gguf_model, options);
        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_q.weight", layer.attn_q, &gguf_model, options);
        if (layer.attn_q_bias) |t| try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_q.bias", t, &gguf_model, options);
        if (layer.attn_q_norm) |t| try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_q_norm.weight", t, &gguf_model, options);
        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_k.weight", layer.attn_k, &gguf_model, options);
        if (layer.attn_k_bias) |t| try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_k.bias", t, &gguf_model, options);
        if (layer.attn_k_norm) |t| try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_k_norm.weight", t, &gguf_model, options);
        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_v.weight", layer.attn_v, &gguf_model, options);
        if (layer.attn_v_bias) |t| try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_v.bias", t, &gguf_model, options);
        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "attn_output.weight", layer.attn_output, &gguf_model, options);
        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "ffn_norm.weight", layer.ffn_norm, &gguf_model, options);
        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "ffn_gate.weight", layer.ffn_gate, &gguf_model, options);
        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "ffn_down.weight", layer.ffn_down, &gguf_model, options);
        try writeNamedTensorRecord(&builder, tensor_table_start, &record_index, &current_offset, prefix, "ffn_up.weight", layer.ffn_up, &gguf_model, options);
    }

    try builder.writeToFile(output_path);
}

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
    try writeTensorRecord(builder, table_start, record_index, current_offset, name, tensor, model, options);
}

fn writeTensorRecord(
    builder: *CompiledFileBuilder,
    table_start: u64,
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
    const layout_kind = detectLayoutKind(gguf_type, options);

    var record = TensorRecord{
        .name_len = @intCast(name.len),
        .original_gguf_type = @intFromEnum(gguf_type),
        .compiled_layout_kind = @intFromEnum(layout_kind),
        .rows = rows,
        .cols = cols,
        .byte_offset = current_offset.*,
        .byte_length = 0,
        .row_stride = 0,
        .tile_stride = 0,
        .reserved = .{0} ** 16,
    };

    if (layout_kind == .q4_k_m_packed and options.repack_q4_k) {
        record.row_stride = moon_quant.q4_k_packed_block_bytes * (cols / moon_quant.q4_k_block_values);
    } else {
        record.row_stride = try llama.tensorRowByteSize(gguf_type, cols);
    }
    record.tile_stride = record.row_stride;

    const record_offset = table_start + record_index.* * @sizeOf(TensorRecord);

    const tensor_bytes = try llama.tensorBytes(model, tensor);

    if (layout_kind == .q4_k_m_packed and options.repack_q4_k) {
        const packed_result = try moon_quant.packQ4KTensor(builder.allocator, tensor_bytes, rows, cols);
        record.byte_length = packed_result.bytes.len;

        try builder.writeBytesAt(record_offset, std.mem.asBytes(&record));
        try builder.writeBytesAt(record_offset + @sizeOf(TensorRecord), name);

        const data_offset = alignForward(builder.currentPos(), default_alignment);
        try builder.padTo(data_offset);
        try builder.buffer.appendSlice(builder.allocator, packed_result.bytes);
        current_offset.* += packed_result.bytes.len;
    } else {
        const row_size = try llama.tensorRowByteSize(gguf_type, cols);
        const total_bytes = rows * row_size;
        record.byte_length = total_bytes;

        try builder.writeBytesAt(record_offset, std.mem.asBytes(&record));
        try builder.writeBytesAt(record_offset + @sizeOf(TensorRecord), name);

        const data_offset = alignForward(builder.currentPos(), default_alignment);
        try builder.padTo(data_offset);
        try builder.buffer.appendSlice(builder.allocator, tensor_bytes);
        current_offset.* += total_bytes;
    }

    record_index.* += 1;
}

fn detectLayoutKind(tensor_type: llama.TensorType, options: CompileOptions) CompiledLayoutKind {
    return switch (tensor_type) {
        .q4_k => if (options.repack_q4_k) .q4_k_m_packed else .generic_quant_raw,
        .q6_k => .q6_k_raw,
        .q8_0 => .q8_0_raw,
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
        report.quantization_version,
        report.alignment,
        report.total_size,
    });
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

    try std.testing.expectEqual(@as(u32, 1), report.version);
    try std.testing.expectEqual(@as(u32, 1), report.runtime_version);
    try std.testing.expectEqualStrings(test_architecture, report.architecture);
    try std.testing.expectEqual(@as(u32, 2), report.quantization_version);
}

test "ZIGY magic validation rejects invalid files" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("bad.ziggy", .{});
    defer file.close();
    try file.writeAll("NOTZIGY");

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "bad.ziggy");
    defer std.testing.allocator.free(path);

    try std.testing.expectError(error.InvalidMagic, inspectFile(std.testing.allocator, path));
}
