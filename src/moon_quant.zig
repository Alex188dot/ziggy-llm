const std = @import("std");
const gguf = @import("gguf.zig");

pub const supported_quantization =
    "MoonQuant target set: Q4_K_M, Q5_K_M, Q6_K, and Q8_0. Current runnable paths: F32, F16, Q4_K, Q6_K, and Q8_0.";

pub const q4_k_block_values: usize = 256;
pub const q4_k_block_raw_bytes: usize = 144;
pub const q4_k_subblocks: usize = 8;
pub const q4_k_packed_block_bytes: usize = 160;

pub const TargetFormat = enum {
    q4_k_m,
    q5_k_m,
    q6_k,
    q8_0,
    f16_reference,
    f32_reference,
    legacy_q4_k,
    legacy_q6_k,
    other,

    pub fn label(self: TargetFormat) []const u8 {
        return switch (self) {
            .q4_k_m => "Q4_K_M",
            .q5_k_m => "Q5_K_M",
            .q6_k => "Q6_K",
            .q8_0 => "Q8_0",
            .f16_reference => "F16",
            .f32_reference => "F32",
            .legacy_q4_k => "Q4_K (legacy GGUF)",
            .legacy_q6_k => "Q6_K (legacy GGUF)",
            .other => "out-of-scope",
        };
    }
};

pub const Tier = enum {
    optimized_target,
    reference,
    legacy_compatible,
    out_of_scope,

    pub fn label(self: Tier) []const u8 {
        return switch (self) {
            .optimized_target => "optimized-target",
            .reference => "reference",
            .legacy_compatible => "legacy-compatible",
            .out_of_scope => "out-of-scope",
        };
    }
};

pub const RuntimePath = enum {
    available_now,
    planned,
    unsupported,

    pub fn label(self: RuntimePath) []const u8 {
        return switch (self) {
            .available_now => "available-now",
            .planned => "planned",
            .unsupported => "unsupported",
        };
    }
};

pub const Layout = enum {
    apple_decode_packed,
    generic_reference,
    none,

    pub fn label(self: Layout) []const u8 {
        return switch (self) {
            .apple_decode_packed => "apple-decode-packed",
            .generic_reference => "generic-reference",
            .none => "none",
        };
    }
};

pub const Profile = struct {
    format: TargetFormat,
    tier: Tier,
    runtime_path: RuntimePath,
    layout: Layout,
    note: []const u8,
};

pub const PackedTensor = struct {
    format: TargetFormat,
    rows: usize,
    cols: usize,
    row_stride: usize,
    bytes: []u8,

    pub fn deinit(self: *PackedTensor, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        self.* = undefined;
    }
};

pub const Q4KPackedBlock = extern struct {
    d: u16,
    dmin: u16,
    scales: [q4_k_subblocks]u8,
    mins: [q4_k_subblocks]u8,
    quants: [128]u8,
    padding: [12]u8,
};

comptime {
    std.debug.assert(@sizeOf(Q4KPackedBlock) == q4_k_packed_block_bytes);
}

pub fn classify(report: gguf.InspectReport) Profile {
    const format = detectFormat(report);
    return .{
        .format = format,
        .tier = detectTier(format),
        .runtime_path = detectRuntimePath(format),
        .layout = detectLayout(format),
        .note = detectNote(format),
    };
}

pub fn printInspectSummary(writer: anytype, report: gguf.InspectReport) !void {
    const profile = classify(report);
    try writer.print(
        \\moon_quant.format: {s}
        \\moon_quant.tier: {s}
        \\moon_quant.runtime_path: {s}
        \\moon_quant.layout: {s}
        \\moon_quant.note: {s}
        \\
    ,
        .{
            profile.format.label(),
            profile.tier.label(),
            profile.runtime_path.label(),
            profile.layout.label(),
            profile.note,
        },
    );
}

pub fn q4KPackedRowByteLen(row_len: usize) !usize {
    if (row_len == 0 or row_len % q4_k_block_values != 0) return error.InvalidTensorMetadata;
    const block_count = row_len / q4_k_block_values;
    return std.math.mul(usize, block_count, q4_k_packed_block_bytes);
}

pub fn packQ4KTensor(
    allocator: std.mem.Allocator,
    tensor_bytes: []const u8,
    row_count: usize,
    row_len: usize,
) !PackedTensor {
    const raw_row_size = try q4KRawRowByteLen(row_len);
    if (tensor_bytes.len != row_count * raw_row_size) return error.InvalidTensorMetadata;

    const packed_row_size = try q4KPackedRowByteLen(row_len);
    const packed_bytes = try allocator.alloc(u8, row_count * packed_row_size);
    errdefer allocator.free(packed_bytes);

    for (0..row_count) |row_index| {
        const raw_row = tensor_bytes[row_index * raw_row_size ..][0..raw_row_size];
        const packed_row = packed_bytes[row_index * packed_row_size ..][0..packed_row_size];
        try packQ4KRowInto(packed_row, raw_row, row_len);
    }

    return .{
        .format = .q4_k_m,
        .rows = row_count,
        .cols = row_len,
        .row_stride = packed_row_size,
        .bytes = packed_bytes,
    };
}

pub fn packQ4KRowInto(out: []u8, row: []const u8, row_len: usize) !void {
    const raw_row_size = try q4KRawRowByteLen(row_len);
    const packed_row_size = try q4KPackedRowByteLen(row_len);
    if (row.len != raw_row_size or out.len != packed_row_size) return error.InvalidTensorMetadata;

    const block_count = row_len / q4_k_block_values;
    for (0..block_count) |block_index| {
        const src = row[block_index * q4_k_block_raw_bytes ..][0..q4_k_block_raw_bytes];
        const dst = out[block_index * q4_k_packed_block_bytes ..][0..q4_k_packed_block_bytes];
        packQ4KBlockInto(dst, src);
    }
}

fn q4KRawRowByteLen(row_len: usize) !usize {
    if (row_len == 0 or row_len % q4_k_block_values != 0) return error.InvalidTensorMetadata;
    const block_count = row_len / q4_k_block_values;
    return std.math.mul(usize, block_count, q4_k_block_raw_bytes);
}

fn packQ4KBlockInto(out: []u8, block: []const u8) void {
    const block_out = std.mem.bytesAsValue(Q4KPackedBlock, out[0..q4_k_packed_block_bytes]);
    block_out.d = std.mem.readInt(u16, block[0..2], .little);
    block_out.dmin = std.mem.readInt(u16, block[2..4], .little);

    const scales = block[4..16];
    for (0..q4_k_subblocks) |index| {
        const sm = getScaleMinK4(index, scales);
        block_out.scales[index] = sm.scale;
        block_out.mins[index] = sm.min;
    }

    @memcpy(block_out.quants[0..], block[16..144]);
    @memset(block_out.padding[0..], 0);
}

fn detectFormat(report: gguf.InspectReport) TargetFormat {
    if (report.file_type) |file_type| {
        return switch (file_type) {
            15 => .q4_k_m,
            17 => .q5_k_m,
            18 => .q6_k,
            7 => .q8_0,
            1 => .f16_reference,
            0 => .f32_reference,
            14 => .legacy_q4_k,
            else => formatFromDominantTensorType(report.dominant_tensor_type),
        };
    }
    return formatFromDominantTensorType(report.dominant_tensor_type);
}

fn formatFromDominantTensorType(tensor_type: gguf.TensorType) TargetFormat {
    return switch (tensor_type) {
        .f16 => .f16_reference,
        .f32 => .f32_reference,
        .q4_k => .legacy_q4_k,
        .q6_k => .legacy_q6_k,
        .q8_0 => .q8_0,
        else => .other,
    };
}

fn detectTier(format: TargetFormat) Tier {
    return switch (format) {
        .q4_k_m, .q5_k_m, .q6_k, .q8_0 => .optimized_target,
        .f16_reference, .f32_reference => .reference,
        .legacy_q4_k, .legacy_q6_k => .legacy_compatible,
        .other => .out_of_scope,
    };
}

fn detectRuntimePath(format: TargetFormat) RuntimePath {
    return switch (format) {
        .q4_k_m, .q6_k, .q8_0, .f16_reference, .f32_reference, .legacy_q4_k, .legacy_q6_k => .available_now,
        .q5_k_m => .planned,
        .other => .unsupported,
    };
}

fn detectLayout(format: TargetFormat) Layout {
    return switch (format) {
        .q4_k_m, .q5_k_m, .q6_k, .q8_0 => .apple_decode_packed,
        .f16_reference, .f32_reference, .legacy_q4_k, .legacy_q6_k => .generic_reference,
        .other => .none,
    };
}

fn detectNote(format: TargetFormat) []const u8 {
    return switch (format) {
        .q4_k_m => "first MoonQuant target; loader now packs Q4_K blocks into a Metal-oriented fixed-stride layout",
        .q5_k_m => "MoonQuant target format; runtime support still needs a dedicated Q5_K path",
        .q6_k => "MoonQuant target format and current decode path overlap on Q6_K blocks",
        .q8_0 => "MoonQuant target format with a direct raw Metal matvec path and fused decode add support",
        .f16_reference => "reference-validation path kept for correctness and calibration",
        .f32_reference => "full-precision reference path remains useful for correctness checks",
        .legacy_q4_k => "runnable legacy path; loader can repack Q4_K blocks into the MoonQuant row layout",
        .legacy_q6_k => "runnable legacy path that overlaps with a MoonQuant target format",
        .other => "not in the first MoonQuant scope",
    };
}

const ScaleMinK4 = struct {
    scale: u8,
    min: u8,
};

fn getScaleMinK4(index: usize, scale_bytes: []const u8) ScaleMinK4 {
    return if (index < 4)
        .{
            .scale = scale_bytes[index] & 63,
            .min = scale_bytes[index + 4] & 63,
        }
    else
        .{
            .scale = (scale_bytes[index + 4] & 0x0F) | ((scale_bytes[index - 4] >> 6) << 4),
            .min = (scale_bytes[index + 4] >> 4) | ((scale_bytes[index] >> 6) << 4),
        };
}

test "classify exact MoonQuant target formats" {
    const report = gguf.InspectReport{
        .version = 3,
        .metadata_count = 0,
        .tensor_count = 0,
        .alignment = 32,
        .artifact_type = "model",
        .architecture = "llama",
        .quantization_version = 2,
        .file_type = 15,
        .dominant_tensor_type = .q4_k,
        .dominant_tensor_count = 0,
        .data_offset = 0,
        .tokenizer_model = null,
        .tokenizer_pre = null,
        .tokenizer_tokens = null,
        .chat_template = null,
        .bos_token_id = null,
        .eos_token_id = null,
        .unk_token_id = null,
        .pad_token_id = null,
        .add_bos_token = null,
        .add_eos_token = null,
    };

    const profile = classify(report);
    try std.testing.expectEqual(TargetFormat.q4_k_m, profile.format);
    try std.testing.expectEqual(Tier.optimized_target, profile.tier);
    try std.testing.expectEqual(RuntimePath.available_now, profile.runtime_path);
}

test "classify unsupported dominant formats as out of scope" {
    const report = gguf.InspectReport{
        .version = 3,
        .metadata_count = 0,
        .tensor_count = 0,
        .alignment = 32,
        .artifact_type = "model",
        .architecture = "llama",
        .quantization_version = 2,
        .file_type = 32,
        .dominant_tensor_type = .bf16,
        .dominant_tensor_count = 0,
        .data_offset = 0,
        .tokenizer_model = null,
        .tokenizer_pre = null,
        .tokenizer_tokens = null,
        .chat_template = null,
        .bos_token_id = null,
        .eos_token_id = null,
        .unk_token_id = null,
        .pad_token_id = null,
        .add_bos_token = null,
        .add_eos_token = null,
    };

    const profile = classify(report);
    try std.testing.expectEqual(Tier.out_of_scope, profile.tier);
    try std.testing.expectEqual(RuntimePath.unsupported, profile.runtime_path);
}

test "pack q4_k row expands scale and min tables into a fixed-stride MoonQuant block" {
    var raw: [q4_k_block_raw_bytes]u8 = undefined;
    @memset(raw[0..], 0);
    raw[0] = 0x34;
    raw[1] = 0x12;
    raw[2] = 0x78;
    raw[3] = 0x56;

    raw[4] = 1;
    raw[5] = 2;
    raw[6] = 3;
    raw[7] = 4;
    raw[8] = 9;
    raw[9] = 10;
    raw[10] = 11;
    raw[11] = 12;
    raw[12] = 5 | (13 << 4);
    raw[13] = 6 | (14 << 4);
    raw[14] = 7 | (15 << 4);
    raw[15] = 8 | (1 << 4);

    for (16..raw.len, 0..) |index, i| raw[index] = @intCast(i);

    var packed_row: [q4_k_packed_block_bytes]u8 = undefined;
    try packQ4KRowInto(packed_row[0..], raw[0..], q4_k_block_values);

    const block_view = std.mem.bytesAsValue(Q4KPackedBlock, packed_row[0..q4_k_packed_block_bytes]);
    try std.testing.expectEqual(@as(u16, 0x1234), block_view.d);
    try std.testing.expectEqual(@as(u16, 0x5678), block_view.dmin);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, block_view.scales[0..]);
    try std.testing.expectEqualSlices(u8, &.{ 9, 10, 11, 12, 13, 14, 15, 1 }, block_view.mins[0..]);
    try std.testing.expectEqualSlices(u8, raw[16..144], block_view.quants[0..]);
    try std.testing.expectEqual(@as(usize, q4_k_packed_block_bytes), try q4KPackedRowByteLen(q4_k_block_values));
}
