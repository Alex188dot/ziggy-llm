const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");

const rope_metaspace = "\xE2\x96\x81";
const q4_k_block_values = 256;
const q4_k_block_bytes = 144;
const q8_0_block_values = 32;
const q8_0_block_bytes = 34;
const q6_k_block_values = 256;
const q6_k_block_bytes = 210;

const vocab = [_][]const u8{
    "<unk>",               "<s>",                 "</s>",
    rope_metaspace ++ "a", rope_metaspace ++ "b", rope_metaspace ++ "c",
    "!",                   "<0xE2>",              "<0x96>",
    "<0x81>",              "<0x61>",              "<0x62>",
    "<0x63>",              "<0x21>",
};
const token_scores = [_]f32{ 0, 0, 0, 1, 1, 1, 1, -10, -10, -10, -10, -10, -10, -10 };
const token_types = [_]u32{ 2, 3, 3, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6 };

pub const FixtureOptions = struct {
    embedding_length: usize = vocab.len,
    feed_forward_length: usize = 4,
    projection_type: llama_cpu.TensorType = .f16,
    output_type: llama_cpu.TensorType = .f16,
};

pub fn makeLlamaModelFixture(allocator: std.mem.Allocator) ![]u8 {
    return makeLlamaModelFixtureWithOptions(allocator, .{});
}

pub fn makeLlamaQ4KFixture(allocator: std.mem.Allocator) ![]u8 {
    return makeLlamaModelFixtureWithOptions(allocator, .{
        .embedding_length = q4_k_block_values,
        .feed_forward_length = q4_k_block_values,
        .projection_type = .q4_k,
        .output_type = .q4_k,
    });
}

pub fn makeLlamaQ6KFixture(allocator: std.mem.Allocator) ![]u8 {
    return makeLlamaModelFixtureWithOptions(allocator, .{
        .embedding_length = q6_k_block_values,
        .feed_forward_length = q6_k_block_values,
        .projection_type = .q6_k,
        .output_type = .q6_k,
    });
}

pub fn makeLlamaBenchmarkFixture(allocator: std.mem.Allocator, tensor_type: llama_cpu.TensorType) ![]u8 {
    return makeLlamaModelFixtureWithOptions(allocator, .{
        .embedding_length = 2048,
        .feed_forward_length = 5632,
        .projection_type = tensor_type,
        .output_type = tensor_type,
    });
}

pub fn makeLlamaModelFixtureWithOptions(allocator: std.mem.Allocator, options: FixtureOptions) ![]u8 {
    const embed = options.embedding_length;
    const ff = options.feed_forward_length;
    if (embed == 0 or ff == 0) return error.InvalidTensorMetadata;
    if ((options.projection_type == .q4_k or options.output_type == .q4_k) and embed % q4_k_block_values != 0) {
        return error.InvalidTensorMetadata;
    }
    if ((options.projection_type == .q6_k or options.output_type == .q6_k) and embed % q6_k_block_values != 0) {
        return error.InvalidTensorMetadata;
    }
    if (options.projection_type == .q4_k and ff % q4_k_block_values != 0) return error.InvalidTensorMetadata;
    if (options.projection_type == .q6_k and ff % q6_k_block_values != 0) return error.InvalidTensorMetadata;

    const tensor_names = [_][]const u8{
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_up.weight",
    };
    const tensor_dims = [_][2]u64{
        .{ embed, vocab.len },
        .{ embed, vocab.len },
        .{ embed, 1 },
        .{ embed, 1 },
        .{ embed, embed },
        .{ embed, embed },
        .{ embed, embed },
        .{ embed, embed },
        .{ embed, 1 },
        .{ embed, ff },
        .{ ff, embed },
        .{ embed, ff },
    };
    const tensor_types_layout = [_]llama_cpu.TensorType{
        .f16,
        options.output_type,
        .f32,
        .f32,
        options.projection_type,
        options.projection_type,
        options.projection_type,
        options.projection_type,
        .f32,
        options.projection_type,
        options.projection_type,
        options.projection_type,
    };

    var tensor_sizes: [tensor_names.len]u64 = undefined;
    var tensor_offsets: [tensor_names.len]u64 = undefined;
    var running_offset: u64 = 0;
    var tensor_bytes_total: u64 = 0;
    for (tensor_dims, tensor_types_layout, 0..) |dims, tensor_type, index| {
        running_offset = alignForwardU64(running_offset, 32);
        tensor_offsets[index] = running_offset;
        const row_len: usize = @intCast(dims[0]);
        const row_count: usize = @intCast(dims[1]);
        tensor_sizes[index] = try std.math.mul(u64, try llama_cpu.tensorRowByteSize(tensor_type, row_len), row_count);
        running_offset += tensor_sizes[index];
        tensor_bytes_total += tensor_sizes[index];
    }

    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(allocator);
    try list.ensureTotalCapacity(allocator, @intCast(64 * 1024 + tensor_bytes_total));

    writeBytes(&list, "GGUF");
    writeInt(&list, u32, 3);
    writeInt(&list, u64, tensor_names.len);
    writeInt(&list, u64, 21);

    writeStringKv(&list, "general.type", "model");
    writeStringKv(&list, "general.architecture", "llama");
    writeU32Kv(&list, "general.alignment", 32);
    writeStringKv(&list, "tokenizer.ggml.model", "llama");
    writeStringArrayKv(&list, "tokenizer.ggml.tokens", &vocab);
    writeF32ArrayKv(&list, "tokenizer.ggml.scores", &token_scores);
    writeU32ArrayKv(&list, "tokenizer.ggml.token_type", &token_types);
    writeU32Kv(&list, "tokenizer.ggml.bos_token_id", 1);
    writeU32Kv(&list, "tokenizer.ggml.eos_token_id", 2);
    writeU32Kv(&list, "tokenizer.ggml.unknown_token_id", 0);
    writeBoolKv(&list, "tokenizer.ggml.add_bos_token", false);
    writeBoolKv(&list, "tokenizer.ggml.add_eos_token", false);
    writeU32Kv(&list, "llama.context_length", 16);
    writeU32Kv(&list, "llama.embedding_length", @intCast(embed));
    writeU32Kv(&list, "llama.block_count", 1);
    writeU32Kv(&list, "llama.feed_forward_length", @intCast(ff));
    writeU32Kv(&list, "llama.rope.dimension_count", 2);
    writeU32Kv(&list, "llama.attention.head_count", 1);
    writeU32Kv(&list, "llama.attention.head_count_kv", 1);
    writeF32Kv(&list, "llama.attention.layer_norm_rms_epsilon", 0.000001);
    writeF32Kv(&list, "llama.rope.freq_base", 10000);

    for (tensor_names, tensor_dims, tensor_types_layout, tensor_offsets) |name, dims, tensor_type, offset| {
        writeString(&list, name);
        writeInt(&list, u32, 2);
        writeInt(&list, u64, dims[0]);
        writeInt(&list, u64, dims[1]);
        writeInt(&list, u32, @intFromEnum(tensor_type));
        writeInt(&list, u64, offset);
    }

    const aligned_metadata_size = alignForwardU64(list.items.len, 32);
    const padding_len = aligned_metadata_size - list.items.len;
    for (0..padding_len) |_| list.appendAssumeCapacity(0);

    const token_embd = try allocator.alloc(f32, embed * vocab.len);
    defer allocator.free(token_embd);
    @memset(token_embd, 0);
    for (3..vocab.len) |token_id| token_embd[token_id * embed + token_id] = 1;

    const output = try allocator.alloc(f32, vocab.len * embed);
    defer allocator.free(output);
    @memset(output, 0);
    setTransition(output, embed, 3, 4, 1);
    setTransition(output, embed, 4, 5, 1);
    setTransition(output, embed, 5, 6, 1);
    setTransition(output, embed, 6, 3, 1);

    const ones = try allocator.alloc(f32, embed);
    defer allocator.free(ones);
    @memset(ones, 1);

    const zero_square = try allocator.alloc(f32, embed * embed);
    defer allocator.free(zero_square);
    @memset(zero_square, 0);

    const zero_up = try allocator.alloc(f32, ff * embed);
    defer allocator.free(zero_up);
    @memset(zero_up, 0);

    const zero_down = try allocator.alloc(f32, embed * ff);
    defer allocator.free(zero_down);
    @memset(zero_down, 0);

    var data_cursor: u64 = 0;
    for (tensor_offsets, tensor_types_layout, 0..) |offset, tensor_type, index| {
        while (data_cursor < offset) : (data_cursor += 1) list.appendAssumeCapacity(0);
        const values: []const f32 = switch (index) {
            0 => token_embd,
            1 => output,
            2 => ones,
            3 => ones,
            4 => zero_square,
            5 => zero_square,
            6 => zero_square,
            7 => zero_square,
            8 => ones,
            9 => zero_up,
            10 => zero_down,
            11 => zero_up,
            else => unreachable,
        };
        try writeTensorData(&list, tensor_type, values, @intCast(tensor_dims[index][0]));
        data_cursor += tensor_sizes[index];
    }

    return list.toOwnedSlice(allocator);
}

pub fn writeFixtureFile(dir: std.fs.Dir, name: []const u8, contents: []const u8) !void {
    const file = try dir.createFile(name, .{});
    defer file.close();
    try file.writeAll(contents);
}

fn setTransition(output: []f32, cols: usize, source_token: usize, next_token: usize, logit: f32) void {
    output[next_token * cols + source_token] = logit;
}

fn writeTensorData(list: *std.ArrayList(u8), tensor_type: llama_cpu.TensorType, values: []const f32, row_len: usize) !void {
    switch (tensor_type) {
        .f32 => writeTensorDataF32(list, values),
        .f16 => try writeTensorDataF16(list, values),
        .q8_0 => try writeTensorDataQ8_0(list, values, row_len),
        .q4_k => try writeTensorDataQ4K(list, values, row_len),
        .q6_k => try writeTensorDataQ6K(list, values, row_len),
    }
}

fn writeTensorDataQ8_0(list: *std.ArrayList(u8), values: []const f32, row_len: usize) !void {
    if (row_len == 0 or row_len % q8_0_block_values != 0 or values.len % row_len != 0) return error.InvalidTensorMetadata;
    const row_count = values.len / row_len;
    for (0..row_count) |row_index| {
        const row = values[row_index * row_len ..][0..row_len];
        var block_index: usize = 0;
        while (block_index < row_len) : (block_index += q8_0_block_values) {
            appendQ8_0Block(list, row[block_index .. block_index + q8_0_block_values]) catch unreachable;
        }
    }
}

fn writeTensorDataQ4K(list: *std.ArrayList(u8), values: []const f32, row_len: usize) !void {
    if (row_len == 0 or row_len % q4_k_block_values != 0 or values.len % row_len != 0) return error.InvalidTensorMetadata;
    const row_count = values.len / row_len;
    for (0..row_count) |row_index| {
        const row = values[row_index * row_len ..][0..row_len];
        var block_index: usize = 0;
        while (block_index < row_len) : (block_index += q4_k_block_values) {
            appendQ4KBlock(list, row[block_index .. block_index + q4_k_block_values]) catch unreachable;
        }
    }
}

fn writeTensorDataQ6K(list: *std.ArrayList(u8), values: []const f32, row_len: usize) !void {
    if (row_len == 0 or row_len % q6_k_block_values != 0 or values.len % row_len != 0) return error.InvalidTensorMetadata;
    const row_count = values.len / row_len;
    for (0..row_count) |row_index| {
        const row = values[row_index * row_len ..][0..row_len];
        var block_index: usize = 0;
        while (block_index < row_len) : (block_index += q6_k_block_values) {
            appendQ6KBlock(list, row[block_index .. block_index + q6_k_block_values]) catch unreachable;
        }
    }
}

fn appendQ4KBlock(list: *std.ArrayList(u8), values: []const f32) !void {
    var block: [q4_k_block_bytes]u8 = [_]u8{0} ** q4_k_block_bytes;
    writeHalfU16(block[0..2], 1);
    writeHalfU16(block[2..4], 0);
    block[4] = 1;
    block[5] = 1;
    block[6] = 1;
    block[7] = 1;
    block[12] = 1;
    block[13] = 1;
    block[14] = 1;
    block[15] = 1;

    for (values, 0..) |value, index| {
        const nibble = try encodeBinaryValue(value);
        if (index < 32) {
            block[16 + index] |= nibble;
        } else if (index < 64) {
            block[16 + (index - 32)] |= nibble << 4;
        } else if (index < 96) {
            block[48 + (index - 64)] |= nibble;
        } else if (index < 128) {
            block[48 + (index - 96)] |= nibble << 4;
        } else if (index < 160) {
            block[80 + (index - 128)] |= nibble;
        } else if (index < 192) {
            block[80 + (index - 160)] |= nibble << 4;
        } else if (index < 224) {
            block[112 + (index - 192)] |= nibble;
        } else {
            block[112 + (index - 224)] |= nibble << 4;
        }
    }

    list.appendSliceAssumeCapacity(&block);
}

fn appendQ8_0Block(list: *std.ArrayList(u8), values: []const f32) !void {
    var block: [q8_0_block_bytes]u8 = [_]u8{0} ** q8_0_block_bytes;
    writeHalfU16(block[0..2], 1);
    for (values, 0..) |value, index| {
        block[2 + index] = @bitCast(@as(i8, if (value >= 0) 1 else -1));
    }
    list.appendSliceAssumeCapacity(&block);
}

fn appendQ6KBlock(list: *std.ArrayList(u8), values: []const f32) !void {
    var block: [q6_k_block_bytes]u8 = [_]u8{0} ** q6_k_block_bytes;
    @memset(block[128..192], 0xAA);
    @memset(block[192..208], 1);
    writeHalfU16(block[208..210], 1);

    for (values, 0..) |value, index| {
        const low_nibble = try encodeBinaryValue(value);
        setQ6KNibble(&block, index, low_nibble);
    }

    list.appendSliceAssumeCapacity(&block);
}

fn setQ6KNibble(block: *[q6_k_block_bytes]u8, index: usize, nibble: u8) void {
    if (index < 32) {
        block[0 + index] = (block[0 + index] & 0xF0) | nibble;
    } else if (index < 64) {
        block[32 + (index - 32)] = (block[32 + (index - 32)] & 0xF0) | nibble;
    } else if (index < 96) {
        block[0 + (index - 64)] = (block[0 + (index - 64)] & 0x0F) | (nibble << 4);
    } else if (index < 128) {
        block[32 + (index - 96)] = (block[32 + (index - 96)] & 0x0F) | (nibble << 4);
    } else if (index < 160) {
        block[64 + (index - 128)] = (block[64 + (index - 128)] & 0xF0) | nibble;
    } else if (index < 192) {
        block[96 + (index - 160)] = (block[96 + (index - 160)] & 0xF0) | nibble;
    } else if (index < 224) {
        block[64 + (index - 192)] = (block[64 + (index - 192)] & 0x0F) | (nibble << 4);
    } else {
        block[96 + (index - 224)] = (block[96 + (index - 224)] & 0x0F) | (nibble << 4);
    }
}

fn encodeBinaryValue(value: f32) !u8 {
    if (std.math.approxEqAbs(f32, value, 0, 0.0001)) return 0;
    if (std.math.approxEqAbs(f32, value, 1, 0.0001)) return 1;
    return error.UnsupportedFixtureValue;
}

fn writeHalfU16(out: []u8, value: f32) void {
    const half: f16 = @floatCast(value);
    std.mem.writeInt(u16, out[0..2], @as(u16, @bitCast(half)), .little);
}

fn alignForwardU64(value: usize, alignment: u64) u64 {
    const a = alignment;
    return (@as(u64, @intCast(value)) + a - 1) & ~(a - 1);
}

fn writeTensorDataF16(list: *std.ArrayList(u8), values: []const f32) !void {
    for (values) |value| {
        const half: f16 = @floatCast(value);
        writeInt(list, u16, @as(u16, @bitCast(half)));
    }
}

fn writeTensorDataF32(list: *std.ArrayList(u8), values: []const f32) void {
    for (values) |value| writeInt(list, u32, @as(u32, @bitCast(value)));
}

fn writeBytes(list: *std.ArrayList(u8), bytes: []const u8) void {
    list.appendSliceAssumeCapacity(bytes);
}

fn writeInt(list: *std.ArrayList(u8), comptime T: type, value: T) void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    list.appendSliceAssumeCapacity(&bytes);
}

fn writeString(list: *std.ArrayList(u8), value: []const u8) void {
    writeInt(list, u64, value.len);
    list.appendSliceAssumeCapacity(value);
}

fn writeStringKv(list: *std.ArrayList(u8), key: []const u8, value: []const u8) void {
    writeString(list, key);
    writeInt(list, u32, 8);
    writeString(list, value);
}

fn writeU32Kv(list: *std.ArrayList(u8), key: []const u8, value: u32) void {
    writeString(list, key);
    writeInt(list, u32, 4);
    writeInt(list, u32, value);
}

fn writeBoolKv(list: *std.ArrayList(u8), key: []const u8, value: bool) void {
    writeString(list, key);
    writeInt(list, u32, 7);
    writeInt(list, u8, if (value) 1 else 0);
}

fn writeF32Kv(list: *std.ArrayList(u8), key: []const u8, value: f32) void {
    writeString(list, key);
    writeInt(list, u32, 6);
    writeInt(list, u32, @as(u32, @bitCast(value)));
}

fn writeStringArrayKv(list: *std.ArrayList(u8), key: []const u8, values: []const []const u8) void {
    writeString(list, key);
    writeInt(list, u32, 9);
    writeInt(list, u32, 8);
    writeInt(list, u64, values.len);
    for (values) |value| writeString(list, value);
}

fn writeF32ArrayKv(list: *std.ArrayList(u8), key: []const u8, values: []const f32) void {
    writeString(list, key);
    writeInt(list, u32, 9);
    writeInt(list, u32, 6);
    writeInt(list, u64, values.len);
    for (values) |value| writeInt(list, u32, @as(u32, @bitCast(value)));
}

fn writeU32ArrayKv(list: *std.ArrayList(u8), key: []const u8, values: []const u32) void {
    writeString(list, key);
    writeInt(list, u32, 9);
    writeInt(list, u32, 4);
    writeInt(list, u64, values.len);
    for (values) |value| writeInt(list, u32, value);
}
