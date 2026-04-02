const std = @import("std");
const tiny_model = @import("tiny_model.zig");
const types = @import("types.zig");

pub const Options = struct {
    file_type: u32 = 1,
    ambiguous_a: bool = false,
};

pub fn makeTinyModelFixture(allocator: std.mem.Allocator, options: Options) ![]u8 {
    const vocab = [_][]const u8{ "<unk>", "<s>", "</s>", "a", "b", "c", "!" };
    const embed: usize = vocab.len;
    const ff: usize = 4;
    const context: usize = 16;

    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(allocator);
    try list.ensureTotalCapacity(allocator, 8192);

    writeBytes(&list, tiny_model.gguf_magic);
    writeInt(&list, u32, 3);
    writeInt(&list, u64, 8);
    writeInt(&list, u64, 16);

    writeStringKv(&list, "general.type", "model");
    writeStringKv(&list, "general.architecture", types.native_architecture);
    writeU32Kv(&list, "general.alignment", tiny_model.default_alignment);
    writeU32Kv(&list, "general.file_type", options.file_type);
    writeStringKv(&list, "tokenizer.ggml.model", "char");
    writeStringKv(&list, "tokenizer.ggml.pre", "identity");
    writeStringArrayKv(&list, "tokenizer.ggml.tokens", &vocab);
    writeU32Kv(&list, "tokenizer.ggml.bos_token_id", 1);
    writeU32Kv(&list, "tokenizer.ggml.eos_token_id", 2);
    writeU32Kv(&list, "tokenizer.ggml.unknown_token_id", 0);
    writeBoolKv(&list, "tokenizer.ggml.add_bos_token", false);
    writeU32Kv(&list, "ziggy.context_length", context);
    writeU32Kv(&list, "ziggy.embedding_length", embed);
    writeU32Kv(&list, "ziggy.feed_forward_length", ff);
    writeU32Kv(&list, "ziggy.block_count", 1);
    writeU32Kv(&list, "ziggy.head_count", 1);

    const tensor_names = [_][]const u8{
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output.weight",
    };
    const tensor_dims = [_][2]u64{
        .{ embed, vocab.len },
        .{ embed, embed },
        .{ embed, embed },
        .{ embed, embed },
        .{ embed, embed },
        .{ ff, embed },
        .{ embed, ff },
        .{ vocab.len, embed },
    };

    var tensor_sizes: [tensor_names.len]u64 = undefined;
    var tensor_offsets: [tensor_names.len]u64 = undefined;
    var running_offset: u64 = 0;
    for (tensor_names, tensor_dims, 0..) |name, dims, index| {
        running_offset = tiny_model.alignForwardU64(running_offset, tiny_model.default_alignment);
        tensor_offsets[index] = running_offset;
        writeString(&list, name);
        writeInt(&list, u32, 2);
        writeInt(&list, u64, dims[0]);
        writeInt(&list, u64, dims[1]);
        writeInt(&list, u32, @intFromEnum(tiny_model.TensorType.f16));
        writeInt(&list, u64, tensor_offsets[index]);
        const count = dims[0] * dims[1];
        const size = count * 2;
        tensor_sizes[index] = size;
        running_offset += size;
    }

    const aligned_metadata_size = tiny_model.alignForwardU64(list.items.len, tiny_model.default_alignment);
    const padding_len = aligned_metadata_size - list.items.len;
    for (0..padding_len) |_| list.appendAssumeCapacity(0);

    var token_embd = try allocator.alloc(f32, embed * vocab.len);
    defer allocator.free(token_embd);
    @memset(token_embd, 0);
    for (3..vocab.len) |token_id| token_embd[token_id + token_id * embed] = 1;

    const zero_square = try allocator.alloc(f32, embed * embed);
    defer allocator.free(zero_square);
    @memset(zero_square, 0);

    const zero_up = try allocator.alloc(f32, ff * embed);
    defer allocator.free(zero_up);
    @memset(zero_up, 0);

    const zero_down = try allocator.alloc(f32, embed * ff);
    defer allocator.free(zero_down);
    @memset(zero_down, 0);

    const output = try allocator.alloc(f32, vocab.len * embed);
    defer allocator.free(output);
    @memset(output, -4);

    setTransition(output, vocab.len, 3, if (options.ambiguous_a) 4 else 4, 4);
    setTransition(output, vocab.len, 4, 5, 4);
    setTransition(output, vocab.len, 5, 6, 4);
    setTransition(output, vocab.len, 6, 2, 4);
    if (options.ambiguous_a) setTransition(output, vocab.len, 3, 5, 4);

    const tensor_data = [_][]const f32{
        token_embd,
        zero_square,
        zero_square,
        zero_square,
        zero_square,
        zero_up,
        zero_down,
        output,
    };
    var data_cursor: u64 = 0;
    for (tensor_data, tensor_offsets) |values, offset| {
        while (data_cursor < offset) : (data_cursor += 1) list.appendAssumeCapacity(0);
        try writeTensorDataF16(&list, values);
        data_cursor += values.len * 2;
    }

    return list.toOwnedSlice(allocator);
}

pub fn writeFixtureFile(dir: std.fs.Dir, name: []const u8, contents: []const u8) !void {
    const file = try dir.createFile(name, .{});
    defer file.close();
    try file.writeAll(contents);
}

fn setTransition(output: []f32, rows: usize, source_token: usize, next_token: usize, logit: f32) void {
    output[next_token + source_token * rows] = logit;
}

fn writeTensorDataF16(list: *std.ArrayList(u8), values: []const f32) !void {
    for (values) |value| {
        const half: f16 = @floatCast(value);
        writeInt(list, u16, @as(u16, @bitCast(half)));
    }
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
    writeInt(list, u32, @intFromEnum(tiny_model.ValueType.string));
    writeString(list, value);
}

fn writeU32Kv(list: *std.ArrayList(u8), key: []const u8, value: u32) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(tiny_model.ValueType.uint32));
    writeInt(list, u32, value);
}

fn writeBoolKv(list: *std.ArrayList(u8), key: []const u8, value: bool) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(tiny_model.ValueType.bool));
    writeInt(list, u8, if (value) 1 else 0);
}

fn writeStringArrayKv(list: *std.ArrayList(u8), key: []const u8, values: []const []const u8) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(tiny_model.ValueType.array));
    writeInt(list, u32, @intFromEnum(tiny_model.ValueType.string));
    writeInt(list, u64, values.len);
    for (values) |value| writeString(list, value);
}
