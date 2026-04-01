const std = @import("std");
const gguf = @import("gguf.zig");
const llama_cpu = @import("llama_cpu.zig");
const terminal = @import("terminal.zig");

pub const primary_target = "Apple Silicon + Metal";
pub const fallback_target = "Apple Silicon CPU";
const native_architecture = "ziggy-tiny";
pub const supported_architecture = "ziggy-tiny (native) and llama";
pub const supported_model_family = "ziggy-tiny GGUF reference models and llama-family GGUF models through the native CPU runtime";
pub const supported_quantization = "ziggy-tiny: F16 only; llama native CPU path: F32, F16, Q4_K, and Q6_K";

const gguf_magic = "GGUF";
const default_alignment: u32 = 32;

const ValueType = enum(u32) {
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

const TensorType = enum(u32) {
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

pub const RuntimeError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedArtifactType,
    UnsupportedArchitecture,
    UnsupportedFileType,
    UnsupportedTensorType,
    UnsupportedTokenizer,
    MissingRequiredMetadata,
    MissingRequiredTensor,
    InvalidMetadataType,
    InvalidMetadataValue,
    InvalidTensorMetadata,
    InvalidPrompt,
    TruncatedFile,
    Overflow,
    UnknownToken,
    ContextOverflow,
    EmptyPrompt,
};

pub const GenerationOptions = struct {
    max_tokens: usize = 16,
    seed: u64 = 0,
    temperature: f32 = 0.0,
};

pub const GenerationReport = struct {
    generated_text: []u8,
    prompt_token_count: usize,
    generated_token_count: usize,
    startup_ns: u64,
    prompt_ns: u64,
    decode_ns: u64,
    seed: u64,
    temperature: f32,

    pub fn deinit(self: *GenerationReport, allocator: std.mem.Allocator) void {
        allocator.free(self.generated_text);
        self.* = undefined;
    }

    pub fn decodeTokensPerSecond(self: GenerationReport) f64 {
        if (self.generated_token_count == 0 or self.decode_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.generated_token_count)) / (@as(f64, @floatFromInt(self.decode_ns)) / std.time.ns_per_s);
    }
};

const Model = struct {
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

    fn deinit(self: *Model, allocator: std.mem.Allocator) void {
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

const Tokenizer = struct {
    tokens: [][]const u8,
    bos_token_id: ?u32,
    eos_token_id: ?u32,
    unk_token_id: ?u32,
    add_bos_token: bool,
    add_eos_token: bool,

    fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        for (self.tokens) |token| allocator.free(token);
        allocator.free(self.tokens);
        self.* = undefined;
    }

    fn encodeInto(self: Tokenizer, prompt: []const u8, out: []u32) !usize {
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

        return count;
    }

    fn tokenString(self: Tokenizer, token_id: u32) ![]const u8 {
        if (token_id >= self.tokens.len) return error.InvalidPrompt;
        return self.tokens[token_id];
    }
};

const Session = struct {
    model: *const Model,
    token_buffer: []u32,
    k_cache: []f32,
    v_cache: []f32,
    hidden: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    ff_hidden: []f32,
    logits: []f32,
    scores: []f32,
    position: usize = 0,

    fn init(allocator: std.mem.Allocator, model: *const Model, token_capacity: usize) !Session {
        return .{
            .model = model,
            .token_buffer = try allocator.alloc(u32, token_capacity),
            .k_cache = try allocator.alloc(f32, model.context_length * model.embedding_length),
            .v_cache = try allocator.alloc(f32, model.context_length * model.embedding_length),
            .hidden = try allocator.alloc(f32, model.embedding_length),
            .q = try allocator.alloc(f32, model.embedding_length),
            .k = try allocator.alloc(f32, model.embedding_length),
            .v = try allocator.alloc(f32, model.embedding_length),
            .attn = try allocator.alloc(f32, model.embedding_length),
            .ff_hidden = try allocator.alloc(f32, model.feed_forward_length),
            .logits = try allocator.alloc(f32, model.tokenizer.tokens.len),
            .scores = try allocator.alloc(f32, model.context_length),
        };
    }

    fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        allocator.free(self.token_buffer);
        allocator.free(self.k_cache);
        allocator.free(self.v_cache);
        allocator.free(self.hidden);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.attn);
        allocator.free(self.ff_hidden);
        allocator.free(self.logits);
        allocator.free(self.scores);
        self.* = undefined;
    }

    fn runPrompt(self: *Session, prompt_tokens: []const u32) !void {
        if (prompt_tokens.len == 0) return error.EmptyPrompt;
        for (prompt_tokens) |token_id| _ = try self.step(token_id);
    }

    fn step(self: *Session, token_id: u32) ![]const f32 {
        if (token_id >= self.model.tokenizer.tokens.len) return error.InvalidPrompt;
        if (self.position >= self.model.context_length) return error.ContextOverflow;

        copyEmbedding(self.hidden, self.model.token_embd, self.model.embedding_length, token_id);

        matVec(self.q, self.model.attn_q, self.hidden, self.model.embedding_length, self.model.embedding_length);
        matVec(self.k, self.model.attn_k, self.hidden, self.model.embedding_length, self.model.embedding_length);
        matVec(self.v, self.model.attn_v, self.hidden, self.model.embedding_length, self.model.embedding_length);

        const cache_offset = self.position * self.model.embedding_length;
        @memcpy(self.k_cache[cache_offset .. cache_offset + self.model.embedding_length], self.k);
        @memcpy(self.v_cache[cache_offset .. cache_offset + self.model.embedding_length], self.v);

        const scale = @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(self.model.embedding_length)));
        for (0..self.position + 1) |index| {
            const other_k = self.k_cache[index * self.model.embedding_length ..][0..self.model.embedding_length];
            self.scores[index] = dot(self.q, other_k) * scale;
        }
        softmaxInPlace(self.scores[0 .. self.position + 1]);

        @memset(self.attn, 0);
        for (0..self.position + 1) |index| {
            const weight = self.scores[index];
            const other_v = self.v_cache[index * self.model.embedding_length ..][0..self.model.embedding_length];
            for (self.attn, other_v) |*dst, src| dst.* += weight * src;
        }

        matVec(self.q, self.model.attn_out, self.attn, self.model.embedding_length, self.model.embedding_length);
        for (self.hidden, self.q) |*dst, src| dst.* += src;

        matVec(self.ff_hidden, self.model.ffn_up, self.hidden, self.model.feed_forward_length, self.model.embedding_length);
        for (self.ff_hidden) |*value| value.* = @max(value.*, 0);
        matVec(self.q, self.model.ffn_down, self.ff_hidden, self.model.embedding_length, self.model.feed_forward_length);
        for (self.hidden, self.q) |*dst, src| dst.* += src;

        matVec(self.logits, self.model.output, self.hidden, self.model.tokenizer.tokens.len, self.model.embedding_length);

        self.token_buffer[self.position] = token_id;
        self.position += 1;
        return self.logits;
    }
};

pub fn generate(allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8, options: GenerationOptions) !GenerationReport {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const report = try gguf.inspectFile(arena.allocator(), model_path);
    if (std.mem.eql(u8, report.architecture, "llama")) {
        const llama_report = try llama_cpu.generate(
            allocator,
            model_path,
            prompt,
            options.max_tokens,
            options.seed,
            options.temperature,
        );
        return .{
            .generated_text = llama_report.generated_text,
            .prompt_token_count = llama_report.prompt_token_count,
            .generated_token_count = llama_report.generated_token_count,
            .startup_ns = llama_report.startup_ns,
            .prompt_ns = llama_report.prompt_ns,
            .decode_ns = llama_report.decode_ns,
            .seed = options.seed,
            .temperature = options.temperature,
        };
    }
    if (!std.mem.eql(u8, report.architecture, native_architecture)) return error.UnsupportedArchitecture;

    const startup_begin = std.time.nanoTimestamp();
    var spinner = terminal.Spinner{};
    try spinner.start();
    errdefer spinner.stop();

    var model = try loadModel(allocator, model_path);
    defer model.deinit(allocator);

    const prompt_capacity = prompt.len + options.max_tokens + 4;
    var session = try Session.init(allocator, &model, @min(model.context_length, prompt_capacity));
    defer session.deinit(allocator);

    const startup_end = std.time.nanoTimestamp();
    spinner.stop();

    const prompt_begin = std.time.nanoTimestamp();
    const prompt_token_count = try model.tokenizer.encodeInto(prompt, session.token_buffer);
    try session.runPrompt(session.token_buffer[0..prompt_token_count]);
    const prompt_end = std.time.nanoTimestamp();

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(options.seed);
    const random = rng.random();
    var generated_token_count: usize = 0;

    const decode_begin = std.time.nanoTimestamp();
    while (generated_token_count < options.max_tokens) : (generated_token_count += 1) {
        const next_token = sampleToken(session.logits, options.temperature, random);
        if (model.tokenizer.eos_token_id != null and next_token == model.tokenizer.eos_token_id.?) break;

        const token_text = try model.tokenizer.tokenString(next_token);
        if (!(model.tokenizer.bos_token_id != null and next_token == model.tokenizer.bos_token_id.?)) {
            try output.appendSlice(allocator, token_text);
        }
        _ = try session.step(next_token);
    }
    const decode_end = std.time.nanoTimestamp();

    return .{
        .generated_text = try output.toOwnedSlice(allocator),
        .prompt_token_count = prompt_token_count,
        .generated_token_count = generated_token_count,
        .startup_ns = deltaNs(startup_begin, startup_end),
        .prompt_ns = deltaNs(prompt_begin, prompt_end),
        .decode_ns = deltaNs(decode_begin, decode_end),
        .seed = options.seed,
        .temperature = options.temperature,
    };
}

pub fn runCommand(writer: *std.Io.Writer, allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8, options: GenerationOptions) !void {
    var report = try generate(allocator, model_path, prompt, options);
    defer report.deinit(allocator);

    try writer.print(
        \\generated_text: {s}
        \\prompt_tokens: {d}
        \\generated_tokens: {d}
        \\seed: {d}
        \\temperature: {d:.3}
        \\startup_ms: {d:.3}
        \\prompt_ms: {d:.3}
        \\tps: {d:.3}
        \\decode_tok_s: {d:.3}
        \\
    ,
        .{
            report.generated_text,
            report.prompt_token_count,
            report.generated_token_count,
            report.seed,
            report.temperature,
            nsToMs(report.startup_ns),
            nsToMs(report.prompt_ns),
            report.decodeTokensPerSecond(),
            report.decodeTokensPerSecond(),
        },
    );
}

pub fn benchCommand(writer: *std.Io.Writer, allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8, options: GenerationOptions) !void {
    var report = try generate(allocator, model_path, prompt, options);
    defer report.deinit(allocator);

    try writer.print(
        \\startup_ns={d}
        \\prompt_ns={d}
        \\decode_ns={d}
        \\prompt_tokens={d}
        \\generated_tokens={d}
        \\tps={d:.3}
        \\decode_tok_s={d:.3}
        \\
    ,
        .{
            report.startup_ns,
            report.prompt_ns,
            report.decode_ns,
            report.prompt_token_count,
            report.generated_token_count,
            report.decodeTokensPerSecond(),
            report.decodeTokensPerSecond(),
        },
    );
}

fn loadModel(allocator: std.mem.Allocator, model_path: []const u8) !Model {
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
    if (!std.mem.eql(u8, architecture, native_architecture)) return error.UnsupportedArchitecture;

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

    const model = Model{
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

    return model;
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

fn copyEmbedding(out: []f32, embedding: []const f32, embedding_length: usize, token_id: u32) void {
    const start = token_id * embedding_length;
    @memcpy(out, embedding[start .. start + embedding_length]);
}

fn matVec(out: []f32, matrix: []const f32, input: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(out.len >= rows);
    std.debug.assert(input.len >= cols);
    for (0..rows) |row| {
        var sum: f32 = 0;
        for (0..cols) |col| {
            sum += matrix[row + col * rows] * input[col];
        }
        out[row] = sum;
    }
}

fn dot(a: []const f32, b: []const f32) f32 {
    var total: f32 = 0;
    for (a, b) |lhs, rhs| total += lhs * rhs;
    return total;
}

fn softmaxInPlace(values: []f32) void {
    var max_value = values[0];
    for (values[1..]) |value| max_value = @max(max_value, value);

    var sum: f32 = 0;
    for (values) |*value| {
        value.* = @exp(value.* - max_value);
        sum += value.*;
    }
    for (values) |*value| value.* /= sum;
}

fn sampleToken(logits: []const f32, temperature: f32, random: std.Random) u32 {
    if (temperature <= 0) return argmax(logits);

    var max_logit = logits[0];
    for (logits[1..]) |value| max_logit = @max(max_logit, value);

    var sum: f32 = 0;
    for (logits) |value| sum += @exp((value - max_logit) / temperature);

    var threshold = random.float(f32) * sum;
    for (logits, 0..) |value, index| {
        threshold -= @exp((value - max_logit) / temperature);
        if (threshold <= 0) return @intCast(index);
    }
    return @intCast(logits.len - 1);
}

fn argmax(values: []const f32) u32 {
    var best_index: usize = 0;
    var best_value = values[0];
    for (values[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return @intCast(best_index);
}

fn alignForwardU64(value: u64, alignment: u32) u64 {
    const a = @as(u64, alignment);
    return (value + a - 1) & ~(a - 1);
}

fn deltaNs(start: i128, end: i128) u64 {
    return @intCast(@max(@as(i128, 0), end - start));
}

fn nsToMs(value: u64) f64 {
    return @as(f64, @floatFromInt(value)) / std.time.ns_per_ms;
}

test "ziggy-tiny model runs end to end deterministically" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try makeTinyModelFixture(std.testing.allocator, .{});
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "tiny.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "tiny.gguf");
    defer std.testing.allocator.free(path);

    var report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 123,
        .temperature = 0,
    });
    defer report.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("bc!", report.generated_text);
    try std.testing.expectEqual(@as(usize, 1), report.prompt_token_count);
    try std.testing.expectEqual(@as(usize, 3), report.generated_token_count);
}

test "seeded sampling stays reproducible" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try makeTinyModelFixture(std.testing.allocator, .{
        .ambiguous_a = true,
    });
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "seeded.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "seeded.gguf");
    defer std.testing.allocator.free(path);

    var first = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 2,
        .seed = 77,
        .temperature = 1.0,
    });
    defer first.deinit(std.testing.allocator);

    var second = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 2,
        .seed = 77,
        .temperature = 1.0,
    });
    defer second.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings(first.generated_text, second.generated_text);
}

test "unsupported quantization fails clearly" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try makeTinyModelFixture(std.testing.allocator, .{
        .file_type = 7,
    });
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "bad-file-type.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "bad-file-type.gguf");
    defer std.testing.allocator.free(path);

    try std.testing.expectError(error.UnsupportedFileType, generate(std.testing.allocator, path, "a", .{}));
}

const TinyFixtureOptions = struct {
    file_type: u32 = 1,
    ambiguous_a: bool = false,
};

fn makeTinyModelFixture(allocator: std.mem.Allocator, options: TinyFixtureOptions) ![]u8 {
    const vocab = [_][]const u8{ "<unk>", "<s>", "</s>", "a", "b", "c", "!" };
    const embed: usize = vocab.len;
    const ff: usize = 4;
    const context: usize = 16;

    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(allocator);
    try list.ensureTotalCapacity(allocator, 8192);

    writeBytes(&list, gguf_magic);
    writeInt(&list, u32, 3);
    writeInt(&list, u64, 8);
    writeInt(&list, u64, 16);

    writeStringKv(&list, "general.type", "model");
    writeStringKv(&list, "general.architecture", native_architecture);
    writeU32Kv(&list, "general.alignment", default_alignment);
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
        running_offset = alignForwardU64(running_offset, default_alignment);
        tensor_offsets[index] = running_offset;
        writeString(&list, name);
        writeInt(&list, u32, 2);
        writeInt(&list, u64, dims[0]);
        writeInt(&list, u64, dims[1]);
        writeInt(&list, u32, @intFromEnum(TensorType.f16));
        writeInt(&list, u64, tensor_offsets[index]);
        const count = dims[0] * dims[1];
        const size = count * 2;
        tensor_sizes[index] = size;
        running_offset += size;
    }

    const aligned_metadata_size = alignForwardU64(list.items.len, default_alignment);
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

    setTransition(output, vocab.len, embed, 3, if (options.ambiguous_a) 4 else 4, 4);
    setTransition(output, vocab.len, embed, 4, 5, 4);
    setTransition(output, vocab.len, embed, 5, 6, 4);
    setTransition(output, vocab.len, embed, 6, 2, 4);
    if (options.ambiguous_a) setTransition(output, vocab.len, embed, 3, 5, 4);

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

fn setTransition(output: []f32, rows: usize, cols: usize, source_token: usize, next_token: usize, logit: f32) void {
    _ = cols;
    output[next_token + source_token * rows] = logit;
}

fn writeTensorDataF16(list: *std.ArrayList(u8), values: []const f32) !void {
    for (values) |value| {
        const half: f16 = @floatCast(value);
        writeInt(list, u16, @as(u16, @bitCast(half)));
    }
}

fn writeFixtureFile(dir: std.fs.Dir, name: []const u8, contents: []const u8) !void {
    const file = try dir.createFile(name, .{});
    defer file.close();
    try file.writeAll(contents);
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
    writeInt(list, u32, @intFromEnum(ValueType.string));
    writeString(list, value);
}

fn writeU32Kv(list: *std.ArrayList(u8), key: []const u8, value: u32) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(ValueType.uint32));
    writeInt(list, u32, value);
}

fn writeBoolKv(list: *std.ArrayList(u8), key: []const u8, value: bool) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(ValueType.bool));
    writeInt(list, u8, if (value) 1 else 0);
}

fn writeStringArrayKv(list: *std.ArrayList(u8), key: []const u8, values: []const []const u8) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(ValueType.array));
    writeInt(list, u32, @intFromEnum(ValueType.string));
    writeInt(list, u64, values.len);
    for (values) |value| writeString(list, value);
}
