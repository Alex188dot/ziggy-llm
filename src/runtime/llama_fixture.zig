const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");

const rope_metaspace = "\xE2\x96\x81";

pub fn makeLlamaModelFixture(allocator: std.mem.Allocator) ![]u8 {
    const vocab = [_][]const u8{
        "<unk>", "<s>", "</s>",
        rope_metaspace ++ "a", rope_metaspace ++ "b", rope_metaspace ++ "c", "!",
        "<0xE2>", "<0x96>", "<0x81>", "<0x61>", "<0x62>", "<0x63>", "<0x21>",
    };
    const token_scores = [_]f32{ 0, 0, 0, 1, 1, 1, 1, -10, -10, -10, -10, -10, -10, -10 };
    const token_types = [_]u32{ 2, 3, 3, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6 };
    const embed: usize = vocab.len;
    const ff: usize = 4;
    const context: usize = 16;

    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(allocator);
    try list.ensureTotalCapacity(allocator, 16 * 1024);

    writeBytes(&list, "GGUF");
    writeInt(&list, u32, 3);
    writeInt(&list, u64, 12);
    writeInt(&list, u64, 21);

    writeStringKv(&list, "general.type", "model");
    writeStringKv(&list, "general.architecture", "llama");
    writeU32Kv(&list, "general.alignment", 32);
    writeStringKv(&list, "tokenizer.ggml.model", "llama");
    writeStringArrayKv(&list, "tokenizer.ggml.tokens", &vocab);
    try writeF32ArrayKv(&list, allocator, "tokenizer.ggml.scores", &token_scores);
    try writeU32ArrayKv(&list, allocator, "tokenizer.ggml.token_type", &token_types);
    writeU32Kv(&list, "tokenizer.ggml.bos_token_id", 1);
    writeU32Kv(&list, "tokenizer.ggml.eos_token_id", 2);
    writeU32Kv(&list, "tokenizer.ggml.unknown_token_id", 0);
    writeBoolKv(&list, "tokenizer.ggml.add_bos_token", false);
    writeBoolKv(&list, "tokenizer.ggml.add_eos_token", false);
    writeU32Kv(&list, "llama.context_length", @intCast(context));
    writeU32Kv(&list, "llama.embedding_length", @intCast(embed));
    writeU32Kv(&list, "llama.block_count", 1);
    writeU32Kv(&list, "llama.feed_forward_length", @intCast(ff));
    writeU32Kv(&list, "llama.rope.dimension_count", 2);
    writeU32Kv(&list, "llama.attention.head_count", 1);
    writeU32Kv(&list, "llama.attention.head_count_kv", 1);
    writeF32Kv(&list, "llama.attention.layer_norm_rms_epsilon", 0.000001);
    writeF32Kv(&list, "llama.rope.freq_base", 10000);

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
    const tensor_types = [_]llama_cpu.TensorType{
        .f16,
        .f16,
        .f32,
        .f32,
        .f16,
        .f16,
        .f16,
        .f16,
        .f32,
        .f16,
        .f16,
        .f16,
    };

    var tensor_sizes: [tensor_names.len]u64 = undefined;
    var tensor_offsets: [tensor_names.len]u64 = undefined;
    var running_offset: u64 = 0;
    for (tensor_names, tensor_dims, tensor_types, 0..) |name, dims, tensor_type, index| {
        running_offset = alignForwardU64(running_offset, 32);
        tensor_offsets[index] = running_offset;
        writeString(&list, name);
        writeInt(&list, u32, 2);
        writeInt(&list, u64, dims[0]);
        writeInt(&list, u64, dims[1]);
        writeInt(&list, u32, @intFromEnum(tensor_type));
        writeInt(&list, u64, tensor_offsets[index]);
        const row_len: usize = @intCast(dims[0]);
        const row_count: usize = @intCast(dims[1]);
        tensor_sizes[index] = try std.math.mul(u64, try llama_cpu.tensorRowByteSize(tensor_type, row_len), row_count);
        running_offset += tensor_sizes[index];
    }

    const aligned_metadata_size = alignForwardU64(list.items.len, 32);
    const padding_len = aligned_metadata_size - list.items.len;
    for (0..padding_len) |_| list.appendAssumeCapacity(0);

    const token_embd = try allocator.alloc(f32, embed * vocab.len);
    defer allocator.free(token_embd);
    @memset(token_embd, 0);
    for (3..vocab.len) |token_id| token_embd[token_id + token_id * embed] = 1;

    const output = try allocator.alloc(f32, vocab.len * embed);
    defer allocator.free(output);
    @memset(output, -4);
    setTransition(output, embed, 3, 4, 4);
    setTransition(output, embed, 4, 5, 4);
    setTransition(output, embed, 5, 6, 4);
    setTransition(output, embed, 6, 2, 4);

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
    for (tensor_offsets, tensor_types, 0..) |offset, tensor_type, index| {
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
        switch (tensor_type) {
            .f32 => writeTensorDataF32(&list, values),
            .f16 => try writeTensorDataF16(&list, values),
            else => unreachable,
        }
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

fn writeF32ArrayKv(list: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, values: []const f32) !void {
    _ = allocator;
    writeString(list, key);
    writeInt(list, u32, 9);
    writeInt(list, u32, 6);
    writeInt(list, u64, values.len);
    for (values) |value| writeInt(list, u32, @as(u32, @bitCast(value)));
}

fn writeU32ArrayKv(list: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, values: []const u32) !void {
    _ = allocator;
    writeString(list, key);
    writeInt(list, u32, 9);
    writeInt(list, u32, 4);
    writeInt(list, u64, values.len);
    for (values) |value| writeInt(list, u32, value);
}
