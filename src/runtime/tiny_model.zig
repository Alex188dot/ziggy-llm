const std = @import("std");
const types = @import("types.zig");

pub const gguf_magic = "GGUF";
pub const default_alignment: u32 = 32;

pub const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
};

const Parser = struct {
    file: std.fs.File,
    pos: u64 = 0,

    fn readBytes(self: *Parser, buffer: []u8) !void {
        const actual = try self.file.preadAll(buffer, self.pos);
        if (actual != buffer.len) return error.TruncatedFile;
        self.pos = try std.math.add(u64, self.pos, buffer.len);
    }

    fn readInt(self: *Parser, comptime T: type) !T {
        var bytes: [@sizeOf(T)]u8 = undefined;
        try self.readBytes(&bytes);
        return std.mem.readInt(T, &bytes, .little);
    }

    fn skipBytes(self: *Parser, bytes: u64) !void {
        self.pos = try std.math.add(u64, self.pos, bytes);
    }
};

const Metadata = struct {
    architecture: ?[]const u8 = null,
    file_type: ?u32 = null,
    alignment: u32 = default_alignment,
    tokenizer_model: ?[]const u8 = null,
    tokenizer_pre: ?[]const u8 = null,
    tokenizer_tokens: std.ArrayList([]const u8),
    bos_token_id: ?u32 = null,
    eos_token_id: ?u32 = null,
    unk_token_id: ?u32 = null,
    add_bos_token: bool = false,
    add_eos_token: bool = false,
    context_length: ?u32 = null,
    embedding_length: ?u32 = null,
    feed_forward_length: ?u32 = null,
    block_count: ?u32 = null,
    head_count: ?u32 = null,

    fn init() Metadata {
        return .{
            .tokenizer_tokens = .empty,
        };
    }

    fn deinit(self: *Metadata, allocator: std.mem.Allocator) void {
        if (self.architecture) |value| allocator.free(value);
        if (self.tokenizer_model) |value| allocator.free(value);
        if (self.tokenizer_pre) |value| allocator.free(value);
        for (self.tokenizer_tokens.items) |token| allocator.free(token);
        self.tokenizer_tokens.deinit(allocator);
        self.* = undefined;
    }
};

const TensorInfo = struct {
    name: []const u8,
    dims: [2]u32,
    tensor_type: TensorType,
    offset: u64,

    fn elementCount(self: TensorInfo) usize {
        return self.dims[0] * self.dims[1];
    }
};

pub const Tokenizer = struct {
    tokens: [][]const u8,
    bos_token_id: ?u32,
    eos_token_id: ?u32,
    unk_token_id: ?u32,
    add_bos_token: bool,
    add_eos_token: bool,

    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        for (self.tokens) |token| allocator.free(token);
        allocator.free(self.tokens);
        self.* = undefined;
    }

    pub fn encodeInto(self: Tokenizer, prompt: []const u8, out: []u32) !usize {
        var count: usize = 0;
        if (self.add_bos_token) {
            const bos = self.bos_token_id orelse return error.MissingRequiredMetadata;
            if (count >= out.len) return error.ContextOverflow;
            out[count] = bos;
            count += 1;
        }

        var remaining = prompt;
        while (remaining.len > 0) {
            var best_id: ?u32 = null;
            var best_len: usize = 0;

            for (self.tokens, 0..) |token, token_id| {
                if (token.len == 0) continue;
                if (token.len < best_len) continue;
                if (!std.mem.startsWith(u8, remaining, token)) continue;
                if (token.len == best_len and best_id != null and token_id >= best_id.?) continue;
                best_len = token.len;
                best_id = @intCast(token_id);
            }

            if (best_id) |token_id| {
                if (count >= out.len) return error.ContextOverflow;
                out[count] = token_id;
                count += 1;
                remaining = remaining[best_len..];
                continue;
            }

            if (self.unk_token_id) |unk| {
                if (count >= out.len) return error.ContextOverflow;
                out[count] = unk;
                count += 1;
                remaining = remaining[1..];
                continue;
            }

            return error.UnknownToken;
        }

        if (self.add_eos_token) {
            const eos = self.eos_token_id orelse return error.MissingRequiredMetadata;
            if (count >= out.len) return error.ContextOverflow;
            out[count] = eos;
            count += 1;
        }

        return count;
    }

    pub fn tokenString(self: Tokenizer, token_id: u32) ![]const u8 {
        if (token_id >= self.tokens.len) return error.InvalidPrompt;
        return self.tokens[token_id];
    }
};

pub const Model = struct {
    tokenizer: Tokenizer,
    context_length: usize,
    embedding_length: usize,
    feed_forward_length: usize,
    token_embd: []f32,
    attn_q: []f32,
    attn_k: []f32,
    attn_v: []f32,
    attn_out: []f32,
    ffn_up: []f32,
    ffn_down: []f32,
    output: []f32,

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        self.tokenizer.deinit(allocator);
        allocator.free(self.token_embd);
        allocator.free(self.attn_q);
        allocator.free(self.attn_k);
        allocator.free(self.attn_v);
        allocator.free(self.attn_out);
        allocator.free(self.ffn_up);
        allocator.free(self.ffn_down);
        allocator.free(self.output);
        self.* = undefined;
    }
};

pub fn loadModel(allocator: std.mem.Allocator, model_path: []const u8) !Model {
    const file = try std.fs.cwd().openFile(model_path, .{});
    defer file.close();

    const stat = try file.stat();
    var parser = Parser{ .file = file };

    const magic = try readExact(&parser, 4);
    if (!std.mem.eql(u8, &magic, gguf_magic)) return error.InvalidMagic;

    const version = try parser.readInt(u32);
    if (version != 2 and version != 3) return error.UnsupportedVersion;

    const tensor_count = try parser.readInt(u64);
    const metadata_count = try parser.readInt(u64);

    var metadata = Metadata.init();
    defer metadata.deinit(allocator);

    var kv_index: u64 = 0;
    while (kv_index < metadata_count) : (kv_index += 1) {
        try parseMetadataEntry(allocator, &parser, &metadata);
    }

    const architecture = metadata.architecture orelse return error.MissingRequiredMetadata;
    if (!std.mem.eql(u8, architecture, types.native_architecture)) return error.UnsupportedArchitecture;

    const file_type = metadata.file_type orelse return error.MissingRequiredMetadata;
    if (file_type != 1) return error.UnsupportedFileType;

    if ((metadata.tokenizer_model == null or !std.mem.eql(u8, metadata.tokenizer_model.?, "wordpiece")) and
        (metadata.tokenizer_model == null or !std.mem.eql(u8, metadata.tokenizer_model.?, "char")))
    {
        return error.UnsupportedTokenizer;
    }

    const context_length = metadata.context_length orelse return error.MissingRequiredMetadata;
    const embedding_length = metadata.embedding_length orelse return error.MissingRequiredMetadata;
    const feed_forward_length = metadata.feed_forward_length orelse return error.MissingRequiredMetadata;
    const block_count = metadata.block_count orelse return error.MissingRequiredMetadata;
    const head_count = metadata.head_count orelse return error.MissingRequiredMetadata;
    if (block_count != 1 or head_count != 1) return error.UnsupportedArchitecture;

    const vocab_size = metadata.tokenizer_tokens.items.len;
    if (vocab_size == 0) return error.MissingRequiredMetadata;

    var tensors = std.StringHashMap(TensorInfo).init(allocator);
    defer {
        var iterator = tensors.iterator();
        while (iterator.next()) |entry| allocator.free(entry.value_ptr.name);
        tensors.deinit();
    }

    var tensor_index: u64 = 0;
    while (tensor_index < tensor_count) : (tensor_index += 1) {
        const tensor = try parseTensorInfo(allocator, &parser);
        try tensors.put(tensor.name, tensor);
    }

    const data_offset = alignForwardU64(parser.pos, metadata.alignment);
    if (data_offset > stat.size) return error.TruncatedFile;

    const tokenizer_tokens = try metadata.tokenizer_tokens.toOwnedSlice(allocator);
    metadata.tokenizer_tokens = .empty;

    return .{
        .tokenizer = .{
            .tokens = tokenizer_tokens,
            .bos_token_id = metadata.bos_token_id,
            .eos_token_id = metadata.eos_token_id,
            .unk_token_id = metadata.unk_token_id,
            .add_bos_token = metadata.add_bos_token,
            .add_eos_token = metadata.add_eos_token,
        },
        .context_length = context_length,
        .embedding_length = embedding_length,
        .feed_forward_length = feed_forward_length,
        .token_embd = try loadTensorF16(allocator, file, data_offset, tensors.get("token_embd.weight") orelse return error.MissingRequiredTensor, embedding_length, vocab_size),
        .attn_q = try loadTensorF16(allocator, file, data_offset, tensors.get("blk.0.attn_q.weight") orelse return error.MissingRequiredTensor, embedding_length, embedding_length),
        .attn_k = try loadTensorF16(allocator, file, data_offset, tensors.get("blk.0.attn_k.weight") orelse return error.MissingRequiredTensor, embedding_length, embedding_length),
        .attn_v = try loadTensorF16(allocator, file, data_offset, tensors.get("blk.0.attn_v.weight") orelse return error.MissingRequiredTensor, embedding_length, embedding_length),
        .attn_out = try loadTensorF16(allocator, file, data_offset, tensors.get("blk.0.attn_output.weight") orelse return error.MissingRequiredTensor, embedding_length, embedding_length),
        .ffn_up = try loadTensorF16(allocator, file, data_offset, tensors.get("blk.0.ffn_up.weight") orelse return error.MissingRequiredTensor, feed_forward_length, embedding_length),
        .ffn_down = try loadTensorF16(allocator, file, data_offset, tensors.get("blk.0.ffn_down.weight") orelse return error.MissingRequiredTensor, embedding_length, feed_forward_length),
        .output = try loadTensorF16(allocator, file, data_offset, tensors.get("output.weight") orelse return error.MissingRequiredTensor, vocab_size, embedding_length),
    };
}

pub fn alignForwardU64(value: u64, alignment: u32) u64 {
    const a = @as(u64, alignment);
    return (value + a - 1) & ~(a - 1);
}

fn parseMetadataEntry(allocator: std.mem.Allocator, parser: *Parser, metadata: *Metadata) !void {
    const key = try readOwnedString(allocator, parser);
    defer allocator.free(key);

    const raw_value_type = try parser.readInt(u32);
    const value_type = std.meta.intToEnum(ValueType, raw_value_type) catch return error.InvalidMetadataType;

    if (std.mem.eql(u8, key, "general.type")) {
        const artifact_type = try readExpectedString(allocator, parser, value_type);
        defer allocator.free(artifact_type);
        if (!std.mem.eql(u8, artifact_type, "model")) return error.UnsupportedArtifactType;
        return;
    }
    if (std.mem.eql(u8, key, "general.architecture")) {
        metadata.architecture = try readExpectedString(allocator, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "general.file_type")) {
        metadata.file_type = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "general.alignment")) {
        metadata.alignment = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.model")) {
        metadata.tokenizer_model = try readExpectedString(allocator, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.pre")) {
        metadata.tokenizer_pre = try readExpectedString(allocator, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
        try readTokenizerTokens(allocator, parser, value_type, &metadata.tokenizer_tokens);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.bos_token_id")) {
        metadata.bos_token_id = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.eos_token_id")) {
        metadata.eos_token_id = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.unknown_token_id")) {
        metadata.unk_token_id = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.add_bos_token")) {
        metadata.add_bos_token = try readExpectedBool(parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.add_eos_token")) {
        metadata.add_eos_token = try readExpectedBool(parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "ziggy.context_length")) {
        metadata.context_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "ziggy.embedding_length")) {
        metadata.embedding_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "ziggy.feed_forward_length")) {
        metadata.feed_forward_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "ziggy.block_count")) {
        metadata.block_count = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "ziggy.head_count")) {
        metadata.head_count = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }

    try skipValue(parser, value_type);
}

fn parseTensorInfo(allocator: std.mem.Allocator, parser: *Parser) !TensorInfo {
    const name = try readOwnedString(allocator, parser);

    const n_dimensions = try parser.readInt(u32);
    if (n_dimensions != 2) return error.InvalidTensorMetadata;

    const dim0 = try parser.readInt(u64);
    const dim1 = try parser.readInt(u64);
    if (dim0 == 0 or dim1 == 0) return error.InvalidTensorMetadata;

    const raw_type = try parser.readInt(u32);
    const tensor_type = std.meta.intToEnum(TensorType, raw_type) catch return error.InvalidTensorMetadata;
    const offset = try parser.readInt(u64);

    return .{
        .name = name,
        .dims = .{
            std.math.cast(u32, dim0) orelse return error.InvalidTensorMetadata,
            std.math.cast(u32, dim1) orelse return error.InvalidTensorMetadata,
        },
        .tensor_type = tensor_type,
        .offset = offset,
    };
}

fn loadTensorF16(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    data_offset: u64,
    tensor: TensorInfo,
    expected_rows: usize,
    expected_cols: usize,
) ![]f32 {
    if (tensor.tensor_type != .f16) return error.UnsupportedTensorType;
    if (tensor.dims[0] != expected_rows or tensor.dims[1] != expected_cols) return error.InvalidTensorMetadata;

    const element_count = tensor.elementCount();
    const byte_count = try std.math.mul(usize, element_count, @sizeOf(u16));

    const bytes = try allocator.alloc(u8, byte_count);
    defer allocator.free(bytes);

    const absolute_offset = try std.math.add(u64, data_offset, tensor.offset);
    const actual = try file.preadAll(bytes, absolute_offset);
    if (actual != byte_count) return error.TruncatedFile;

    const values = try allocator.alloc(f32, element_count);
    for (0..element_count) |index| {
        const raw = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
        const half = @as(f16, @bitCast(raw));
        values[index] = @as(f32, @floatCast(half));
    }
    return values;
}

fn readExact(parser: *Parser, comptime len: usize) ![len]u8 {
    var bytes: [len]u8 = undefined;
    try parser.readBytes(&bytes);
    return bytes;
}

fn readOwnedString(allocator: std.mem.Allocator, parser: *Parser) ![]const u8 {
    const len_u64 = try parser.readInt(u64);
    const len = std.math.cast(usize, len_u64) orelse return error.Overflow;
    const bytes = try allocator.alloc(u8, len);
    try parser.readBytes(bytes);
    return bytes;
}

fn readExpectedString(allocator: std.mem.Allocator, parser: *Parser, value_type: ValueType) ![]const u8 {
    if (value_type != .string) return error.InvalidMetadataType;
    return readOwnedString(allocator, parser);
}

fn readExpectedUnsigned(comptime T: type, parser: *Parser, value_type: ValueType) !T {
    return switch (value_type) {
        .uint8 => std.math.cast(T, try parser.readInt(u8)) orelse error.InvalidMetadataValue,
        .uint16 => std.math.cast(T, try parser.readInt(u16)) orelse error.InvalidMetadataValue,
        .uint32 => std.math.cast(T, try parser.readInt(u32)) orelse error.InvalidMetadataValue,
        .uint64 => std.math.cast(T, try parser.readInt(u64)) orelse error.InvalidMetadataValue,
        else => error.InvalidMetadataType,
    };
}

fn readExpectedBool(parser: *Parser, value_type: ValueType) !bool {
    if (value_type != .bool) return error.InvalidMetadataType;
    return switch (try parser.readInt(u8)) {
        0 => false,
        1 => true,
        else => error.InvalidMetadataValue,
    };
}

fn readTokenizerTokens(
    allocator: std.mem.Allocator,
    parser: *Parser,
    value_type: ValueType,
    out: *std.ArrayList([]const u8),
) !void {
    if (value_type != .array) return error.InvalidMetadataType;
    const raw_element_type = try parser.readInt(u32);
    const element_type = std.meta.intToEnum(ValueType, raw_element_type) catch return error.InvalidMetadataType;
    if (element_type != .string) return error.InvalidMetadataType;

    const count = try parser.readInt(u64);
    try out.ensureTotalCapacity(allocator, std.math.cast(usize, count) orelse return error.Overflow);
    for (0..count) |_| out.appendAssumeCapacity(try readOwnedString(allocator, parser));
}

fn skipValue(parser: *Parser, value_type: ValueType) !void {
    switch (value_type) {
        .uint8, .int8, .bool => try parser.skipBytes(1),
        .uint16, .int16 => try parser.skipBytes(2),
        .uint32, .int32, .float32 => try parser.skipBytes(4),
        .uint64, .int64, .float64 => try parser.skipBytes(8),
        .string => {
            const len = try parser.readInt(u64);
            try parser.skipBytes(len);
        },
        .array => {
            const raw_element_type = try parser.readInt(u32);
            const element_type = std.meta.intToEnum(ValueType, raw_element_type) catch return error.InvalidMetadataType;
            const count = try parser.readInt(u64);
            for (0..count) |_| try skipValue(parser, element_type);
        },
    }
}
