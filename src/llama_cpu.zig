const std = @import("std");
const terminal = @import("terminal.zig");
const backend_api = @import("runtime/backend.zig");
const llama_gpu = @import("runtime/llama_gpu.zig");
const metal_profile = @import("runtime/metal_profile.zig");
const runtime_types = @import("runtime/types.zig");

pub const GenerateError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedArtifactType,
    UnsupportedArchitecture,
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

pub const GenerateReport = struct {
    generated_text: []u8,
    prompt_token_count: usize,
    reused_prompt_token_count: usize = 0,
    generated_token_count: usize,
    startup_ns: u64,
    prompt_ns: u64,
    ttft_ns: u64,
    decode_ns: u64,
    backend: runtime_types.BackendUsed,
    startup_breakdown: runtime_types.StartupBreakdown = .{},
    metal_profile_summary: ?[]u8 = null,

    pub fn deinit(self: *GenerateReport, allocator: std.mem.Allocator) void {
        allocator.free(self.generated_text);
        if (self.metal_profile_summary) |summary| allocator.free(summary);
        self.* = undefined;
    }
};

pub const StreamCallback = *const fn (?*anyopaque, []const u8) anyerror!void;

pub const DenseTensorLookup = struct {
    ctx: ?*const anyopaque,
    get_fn: *const fn (?*const anyopaque, TensorRef) ?[]const f32,
    get_by_offset_fn: *const fn (?*const anyopaque, u64) ?[]const f32,
    get_raw_by_offset_fn: *const fn (?*const anyopaque, u64) ?[]const u8,

    pub fn get(self: DenseTensorLookup, tensor: TensorRef) ?[]const f32 {
        return self.get_fn(self.ctx, tensor);
    }

    pub fn getByOffset(self: DenseTensorLookup, offset: u64) ?[]const f32 {
        return self.get_by_offset_fn(self.ctx, offset);
    }

    pub fn getRawByOffset(self: DenseTensorLookup, offset: u64) ?[]const u8 {
        return self.get_raw_by_offset_fn(self.ctx, offset);
    }
};

const gguf_magic = "GGUF";
const default_alignment: u32 = 32;
const max_tensor_dims: usize = 4;
const rope_metaspace = "\xE2\x96\x81";
const negative_infinity = -std.math.inf(f32);
const simd_lane_count: usize = 8;
const parallel_matvec_min_rows: usize = 2048;
const parallel_matvec_min_work: usize = 4_000_000;
const max_matvec_helper_threads: usize = 3;

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

pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_k = 12,
    q6_k = 14,
};

const TokenType = enum(u32) {
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,

    fn isEncodable(value: u32) bool {
        return value != @intFromEnum(TokenType.control) and
            value != @intFromEnum(TokenType.unused) and
            value != @intFromEnum(TokenType.unknown);
    }
};

const Parser = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn readBytes(self: *Parser, len: usize) ![]const u8 {
        const end = std.math.add(usize, self.pos, len) catch return error.Overflow;
        if (end > self.bytes.len) return error.TruncatedFile;
        const slice = self.bytes[self.pos..end];
        self.pos = end;
        return slice;
    }

    fn readInt(self: *Parser, comptime T: type) !T {
        const bytes = try self.readBytes(@sizeOf(T));
        return std.mem.readInt(T, bytes[0..@sizeOf(T)], .little);
    }

    fn skipBytes(self: *Parser, len: usize) !void {
        _ = try self.readBytes(len);
    }
};

pub const TensorRef = struct {
    name: []const u8,
    dims: [max_tensor_dims]u64,
    n_dims: usize,
    tensor_type: TensorType,
    offset: u64,

    pub fn rowLen(self: TensorRef) !usize {
        return std.math.cast(usize, self.dims[0]) orelse error.InvalidTensorMetadata;
    }

    pub fn rowCount(self: TensorRef) !usize {
        var total: u64 = 1;
        var index: usize = 1;
        while (index < self.n_dims) : (index += 1) {
            total = try std.math.mul(u64, total, self.dims[index]);
        }
        return std.math.cast(usize, total) orelse error.Overflow;
    }
};

const ScaleMinK4 = struct {
    scale: u8,
    min: u8,
};

pub const LayerRefs = struct {
    attn_norm: TensorRef,
    attn_q: TensorRef,
    attn_k: TensorRef,
    attn_v: TensorRef,
    attn_output: TensorRef,
    ffn_norm: TensorRef,
    ffn_gate: TensorRef,
    ffn_down: TensorRef,
    ffn_up: TensorRef,
};

const Metadata = struct {
    architecture: ?[]u8 = null,
    alignment: u32 = default_alignment,
    file_type: ?u32 = null,
    context_length: ?u32 = null,
    embedding_length: ?u32 = null,
    block_count: ?u32 = null,
    feed_forward_length: ?u32 = null,
    rope_dimension_count: ?u32 = null,
    head_count: ?u32 = null,
    head_count_kv: ?u32 = null,
    rms_norm_eps: ?f32 = null,
    rope_freq_base: ?f32 = null,
    tokenizer_model: ?[]u8 = null,
    tokenizer_tokens: std.ArrayList([]u8) = .empty,
    tokenizer_scores: std.ArrayList(f32) = .empty,
    tokenizer_types: std.ArrayList(u32) = .empty,
    bos_token_id: ?u32 = null,
    eos_token_id: ?u32 = null,
    unk_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,
    add_bos_token: ?bool = null,
    add_eos_token: ?bool = null,

    fn deinit(self: *Metadata, allocator: std.mem.Allocator) void {
        if (self.architecture) |value| allocator.free(value);
        if (self.tokenizer_model) |value| allocator.free(value);
        for (self.tokenizer_tokens.items) |token| allocator.free(token);
        self.tokenizer_tokens.deinit(allocator);
        self.tokenizer_scores.deinit(allocator);
        self.tokenizer_types.deinit(allocator);
        self.* = undefined;
    }
};

const Tokenizer = struct {
    tokens: [][]u8,
    scores: []f32,
    token_types: []u32,
    byte_fallback: [256]?u32,
    bos_token_id: ?u32,
    eos_token_id: ?u32,
    unk_token_id: ?u32,
    pad_token_id: ?u32,
    add_bos_token: bool,
    add_eos_token: bool,

    fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        for (self.tokens) |token| allocator.free(token);
        allocator.free(self.tokens);
        allocator.free(self.scores);
        allocator.free(self.token_types);
        self.* = undefined;
    }

    fn isByteFallbackToken(self: Tokenizer, token_id: u32) ?u8 {
        if (token_id >= self.tokens.len) return null;
        return parseByteFallback(self.tokens[token_id]);
    }

    fn canEncodeToken(self: Tokenizer, token_id: u32) bool {
        if (token_id >= self.tokens.len) return false;
        if (self.bos_token_id != null and token_id == self.bos_token_id.?) return false;
        if (self.eos_token_id != null and token_id == self.eos_token_id.?) return false;
        if (self.unk_token_id != null and token_id == self.unk_token_id.?) return false;
        if (self.pad_token_id != null and token_id == self.pad_token_id.?) return false;
        if (self.token_types.len == self.tokens.len) {
            if (!TokenType.isEncodable(self.token_types[token_id])) return false;
            if (self.token_types[token_id] == @intFromEnum(TokenType.byte)) return false;
        }
        return self.tokens[token_id].len > 0;
    }

    fn encodeInto(self: Tokenizer, allocator: std.mem.Allocator, prompt: []const u8, out: []u32) !usize {
        if (prompt.len == 0) return error.EmptyPrompt;

        var normalized = std.ArrayList(u8).empty;
        defer normalized.deinit(allocator);
        try normalized.appendSlice(allocator, rope_metaspace);
        for (prompt) |byte| {
            if (byte == ' ') {
                try normalized.appendSlice(allocator, rope_metaspace);
            } else {
                try normalized.append(allocator, byte);
            }
        }

        const text = normalized.items;
        const n = text.len;
        var best_scores = try allocator.alloc(f32, n + 1);
        defer allocator.free(best_scores);
        var best_tokens = try allocator.alloc(?u32, n);
        defer allocator.free(best_tokens);
        var best_next = try allocator.alloc(usize, n);
        defer allocator.free(best_next);

        @memset(best_scores, negative_infinity);
        best_scores[n] = 0;

        var pos: usize = n;
        while (pos > 0) {
            pos -= 1;
            var best_score = negative_infinity;
            var best_token: ?u32 = null;
            var best_end = pos;

            for (self.tokens, 0..) |token, token_id_usize| {
                const token_id: u32 = @intCast(token_id_usize);
                if (!self.canEncodeToken(token_id)) continue;
                if (token.len == 0 or token[0] != text[pos]) continue;
                if (token.len > text.len - pos) continue;
                if (!std.mem.startsWith(u8, text[pos..], token)) continue;

                const end = pos + token.len;
                if (!std.math.isFinite(best_scores[end])) continue;
                const token_score = if (token_id < self.scores.len) self.scores[token_id] else 0;
                const candidate = token_score + best_scores[end];
                if (candidate > best_score) {
                    best_score = candidate;
                    best_token = token_id;
                    best_end = end;
                }
            }

            if (best_token == null) {
                const fallback = self.byte_fallback[text[pos]] orelse return error.UnknownToken;
                const end = pos + 1;
                if (!std.math.isFinite(best_scores[end])) return error.UnknownToken;
                best_score = (if (fallback < self.scores.len) self.scores[fallback] else -10) + best_scores[end];
                best_token = fallback;
                best_end = end;
            }

            best_scores[pos] = best_score;
            best_tokens[pos] = best_token;
            best_next[pos] = best_end;
        }

        var count: usize = 0;
        if (self.add_bos_token) {
            const bos = self.bos_token_id orelse return error.MissingRequiredMetadata;
            if (count >= out.len) return error.ContextOverflow;
            out[count] = bos;
            count += 1;
        }

        pos = 0;
        while (pos < n) {
            const token_id = best_tokens[pos] orelse return error.UnknownToken;
            if (count >= out.len) return error.ContextOverflow;
            out[count] = token_id;
            count += 1;
            pos = best_next[pos];
        }

        if (self.add_eos_token and self.eos_token_id != null) {
            if (count >= out.len) return error.ContextOverflow;
            out[count] = self.eos_token_id.?;
            count += 1;
        }

        return count;
    }

    fn appendDecodedToken(self: Tokenizer, output: *std.ArrayList(u8), allocator: std.mem.Allocator, token_id: u32) !void {
        if (self.bos_token_id != null and token_id == self.bos_token_id.?) return;
        if (self.eos_token_id != null and token_id == self.eos_token_id.?) return;
        if (self.pad_token_id != null and token_id == self.pad_token_id.?) return;
        if (self.unk_token_id != null and token_id == self.unk_token_id.?) return;
        if (token_id >= self.tokens.len) return error.InvalidPrompt;

        if (self.isByteFallbackToken(token_id)) |byte| {
            try output.append(allocator, byte);
            return;
        }

        const token = self.tokens[token_id];
        var index: usize = 0;
        while (index < token.len) {
            if (index + rope_metaspace.len <= token.len and std.mem.eql(u8, token[index .. index + rope_metaspace.len], rope_metaspace)) {
                try output.append(allocator, ' ');
                index += rope_metaspace.len;
            } else {
                try output.append(allocator, token[index]);
                index += 1;
            }
        }
    }
};

pub const Model = struct {
    bytes: []const u8,
    mapped_bytes: ?[]align(std.heap.page_size_min) const u8,
    tokenizer: Tokenizer,
    context_length: usize,
    embedding_length: usize,
    block_count: usize,
    feed_forward_length: usize,
    rope_dimension_count: usize,
    head_count: usize,
    head_count_kv: usize,
    head_dimension: usize,
    kv_dimension: usize,
    rms_norm_eps: f32,
    rope_freq_base: f32,
    data_offset: usize,
    token_embd: TensorRef,
    output: TensorRef,
    output_norm: TensorRef,
    layers: []LayerRefs,

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        self.tokenizer.deinit(allocator);
        for (self.layers) |layer| {
            allocator.free(layer.attn_norm.name);
            allocator.free(layer.attn_q.name);
            allocator.free(layer.attn_k.name);
            allocator.free(layer.attn_v.name);
            allocator.free(layer.attn_output.name);
            allocator.free(layer.ffn_norm.name);
            allocator.free(layer.ffn_gate.name);
            allocator.free(layer.ffn_down.name);
            allocator.free(layer.ffn_up.name);
        }
        allocator.free(self.layers);
        allocator.free(self.token_embd.name);
        allocator.free(self.output.name);
        allocator.free(self.output_norm.name);
        if (self.mapped_bytes) |mapped_bytes| {
            std.posix.munmap(mapped_bytes);
        } else {
            allocator.free(self.bytes);
        }
        self.* = undefined;
    }
};

pub const ReusableSession = struct {
    session: Session,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *const Model,
        backend: ?backend_api.MatVecBackend,
        dense_tensors: ?DenseTensorLookup,
    ) !ReusableSession {
        var session = try Session.init(
                allocator,
                model,
                backend,
                dense_tensors,
                model.context_length,
                null,
            );
        errdefer session.deinit(allocator);
        return .{ .session = session };
    }

    pub fn deinit(self: *ReusableSession, allocator: std.mem.Allocator) void {
        self.session.deinit(allocator);
        self.* = undefined;
    }

    pub fn reset(self: *ReusableSession) void {
        self.session.position = 0;
    }
};

const Session = struct {
    model: *const Model,
    backend: ?backend_api.MatVecBackend,
    dense_tensors: ?DenseTensorLookup,
    gpu_session: ?llama_gpu.Session = null,
    token_buffer: []u32,
    hidden: []f32,
    normed: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    attn_tmp: []f32,
    gate: []f32,
    up: []f32,
    logits: []f32,
    scores: []f32,
    k_cache: []f32,
    v_cache: []f32,
    matvec_workers: ?*MatVecWorkers = null,
    position: usize = 0,

    fn init(
        allocator: std.mem.Allocator,
        model: *const Model,
        backend: ?backend_api.MatVecBackend,
        dense_tensors: ?DenseTensorLookup,
        token_capacity: usize,
        profiler: ?*metal_profile.Profiler,
    ) !Session {
        var session = Session{
            .model = model,
            .backend = backend,
            .dense_tensors = dense_tensors,
            .token_buffer = try allocator.alloc(u32, token_capacity),
            .hidden = try allocator.alloc(f32, model.embedding_length),
            .normed = try allocator.alloc(f32, model.embedding_length),
            .q = try allocator.alloc(f32, model.embedding_length),
            .k = try allocator.alloc(f32, model.kv_dimension),
            .v = try allocator.alloc(f32, model.kv_dimension),
            .attn_out = try allocator.alloc(f32, model.embedding_length),
            .attn_tmp = try allocator.alloc(f32, model.embedding_length),
            .gate = try allocator.alloc(f32, model.feed_forward_length),
            .up = try allocator.alloc(f32, model.feed_forward_length),
            .logits = try allocator.alloc(f32, model.tokenizer.tokens.len),
            .scores = try allocator.alloc(f32, model.context_length),
            .k_cache = try allocator.alloc(f32, model.block_count * model.context_length * model.kv_dimension),
            .v_cache = try allocator.alloc(f32, model.block_count * model.context_length * model.kv_dimension),
        };
        errdefer session.deinit(allocator);

        if (backend) |selected_backend| {
            if (selected_backend.label == .metal) {
                const lookup = dense_tensors orelse return error.InvalidTensorMetadata;
                session.gpu_session = try llama_gpu.Session.init(selected_backend, adaptDenseLookup(lookup), adaptModelDesc(model), profiler);
            }
        }

        if (backend == null) {
            const cpu_count = std.Thread.getCpuCount() catch 1;
            const helper_count = @min(max_matvec_helper_threads, cpu_count -| 1);
            if (helper_count > 0) {
                session.matvec_workers = try MatVecWorkers.init(allocator, helper_count);
            }
        }

        return session;
    }

    fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        if (self.gpu_session) |*gpu_session| gpu_session.deinit();
        if (self.matvec_workers) |workers| {
            workers.deinit(allocator);
            allocator.destroy(workers);
        }
        allocator.free(self.token_buffer);
        allocator.free(self.hidden);
        allocator.free(self.normed);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.attn_out);
        allocator.free(self.attn_tmp);
        allocator.free(self.gate);
        allocator.free(self.up);
        allocator.free(self.logits);
        allocator.free(self.scores);
        allocator.free(self.k_cache);
        allocator.free(self.v_cache);
        self.* = undefined;
    }

    fn runPrompt(self: *Session, prompt_tokens: []const u32) !void {
        if (prompt_tokens.len == 0) return error.EmptyPrompt;
        for (prompt_tokens) |token_id| _ = try self.step(token_id);
    }

    fn step(self: *Session, token_id: u32) ![]const f32 {
        if (self.position >= self.model.context_length) return error.ContextOverflow;
        try embeddingLookup(self.hidden, self.model, self.model.token_embd, token_id);

        for (self.model.layers, 0..) |layer, layer_index| {
            if (self.gpu_session) |*gpu_session| {
                if (layer_index == 0) try gpu_session.beginToken(self.hidden);
                try gpu_session.runAttentionBlock(adaptLayerDesc(layer), layer_index, self.position);
            } else {
                try rmsNorm(self.normed, self.hidden, self.model, layer.attn_norm);
                try self.matVec(self.q, layer.attn_q, self.normed);
                try self.matVec(self.k, layer.attn_k, self.normed);
                try self.matVec(self.v, layer.attn_v, self.normed);
                applyRoPE(self.q, self.model.head_count, self.model.head_dimension, self.model.rope_dimension_count, self.position, self.model.rope_freq_base);
                applyRoPE(self.k, self.model.head_count_kv, self.model.head_dimension, self.model.rope_dimension_count, self.position, self.model.rope_freq_base);
                self.storeKv(layer_index);
                self.computeAttention(layer_index);
                try self.matVec(self.attn_tmp, layer.attn_output, self.attn_out);
                addInPlace(self.hidden, self.attn_tmp);
            }

            if (self.gpu_session) |*gpu_session| {
                try gpu_session.runFfnBlock(adaptLayerDesc(layer));
            } else {
                try rmsNorm(self.normed, self.hidden, self.model, layer.ffn_norm);
                try self.matVec(self.gate, layer.ffn_gate, self.normed);
                try self.matVec(self.up, layer.ffn_up, self.normed);
                for (self.gate, self.up) |*gate, up| {
                    gate.* = silu(gate.*) * up;
                }
                try self.matVec(self.attn_tmp, layer.ffn_down, self.gate);
            }
            addInPlace(self.hidden, self.attn_tmp);
        }

        if (self.gpu_session) |*gpu_session| {
            try gpu_session.runOutput(adaptTensorDesc(self.model.output_norm), adaptTensorDesc(self.model.output), self.logits);
        } else {
            try rmsNorm(self.normed, self.hidden, self.model, self.model.output_norm);
            try self.matVec(self.logits, self.model.output, self.normed);
        }

        self.token_buffer[self.position] = token_id;
        self.position += 1;
        return self.logits;
    }

    fn storeKv(self: *Session, layer_index: usize) void {
        const stride = self.model.context_length * self.model.kv_dimension;
        const base = layer_index * stride + self.position * self.model.kv_dimension;
        @memcpy(self.k_cache[base .. base + self.model.kv_dimension], self.k);
        @memcpy(self.v_cache[base .. base + self.model.kv_dimension], self.v);
    }

    fn computeAttention(self: *Session, layer_index: usize) void {
        @memset(self.attn_out, 0);
        const head_dim = self.model.head_dimension;
        const kv_group_size = self.model.head_count / self.model.head_count_kv;
        const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
        const scale = @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..self.model.head_count) |head_index| {
            const q_head = self.q[head_index * head_dim ..][0..head_dim];
            const kv_head = head_index / kv_group_size;
            const kv_offset = kv_head * head_dim;

            for (0..self.position + 1) |token_index| {
                const k_base = layer_base + token_index * self.model.kv_dimension + kv_offset;
                const k_head = self.k_cache[k_base..][0..head_dim];
                self.scores[token_index] = dot(q_head, k_head) * scale;
            }
            softmaxInPlace(self.scores[0 .. self.position + 1]);

            const out_head = self.attn_out[head_index * head_dim ..][0..head_dim];
            @memset(out_head, 0);
            for (0..self.position + 1) |token_index| {
                const weight = self.scores[token_index];
                const v_base = layer_base + token_index * self.model.kv_dimension + kv_offset;
                const v_head = self.v_cache[v_base..][0..head_dim];
                for (out_head, v_head) |*dst, src| {
                    dst.* += weight * src;
                }
            }
        }
    }

    fn matVec(self: *Session, out: []f32, tensor: TensorRef, input: []const f32) !void {
        const row_len = try tensor.rowLen();
        const row_count = try tensor.rowCount();
        if (row_len != input.len or row_count != out.len) return error.InvalidTensorMetadata;

        if (self.backend) |backend| {
            const dense_tensors = self.dense_tensors orelse return error.InvalidTensorMetadata;
            const matrix = dense_tensors.get(tensor) orelse return error.InvalidTensorMetadata;
            try backend.matVec(out, matrix, input, row_count, row_len);
            return;
        }

        const bytes = try tensorBytes(self.model, tensor);
        const row_size = try tensorRowByteSize(tensor.tensor_type, row_len);
        const work = row_count * row_len;

        if (self.matvec_workers) |workers| {
            if (row_count >= parallel_matvec_min_rows and work >= parallel_matvec_min_work) {
                workers.run(.{
                    .out = out,
                    .bytes = bytes,
                    .row_size = row_size,
                    .row_len = row_len,
                    .tensor_type = tensor.tensor_type,
                    .input = input,
                    .row_count = row_count,
                });
                return;
            }
        }

        const range = workerRowRange(0, 1, row_count);
        for (range.start..range.end) |row_index| {
            const row = bytes[row_index * row_size ..][0..row_size];
            out[row_index] = dotRowAssumeValid(tensor.tensor_type, row, row_len, input);
        }
    }
};

fn adaptDenseLookup(lookup: DenseTensorLookup) llama_gpu.DenseLookup {
    return .{
        .ctx = lookup.ctx,
        .get_dense_fn = lookup.get_by_offset_fn,
        .get_raw_fn = lookup.get_raw_by_offset_fn,
    };
}

fn adaptTensorDesc(tensor: TensorRef) llama_gpu.TensorDesc {
    return .{
        .offset = tensor.offset,
        .rows = tensor.rowCount() catch unreachable,
        .cols = tensor.rowLen() catch unreachable,
        .tensor_type = @intFromEnum(tensor.tensor_type),
    };
}

fn adaptLayerDesc(layer: LayerRefs) llama_gpu.LayerDesc {
    return .{
        .attn_norm = adaptTensorDesc(layer.attn_norm),
        .attn_q = adaptTensorDesc(layer.attn_q),
        .attn_k = adaptTensorDesc(layer.attn_k),
        .attn_v = adaptTensorDesc(layer.attn_v),
        .attn_output = adaptTensorDesc(layer.attn_output),
        .ffn_norm = adaptTensorDesc(layer.ffn_norm),
        .ffn_gate = adaptTensorDesc(layer.ffn_gate),
        .ffn_down = adaptTensorDesc(layer.ffn_down),
        .ffn_up = adaptTensorDesc(layer.ffn_up),
    };
}

fn adaptModelDesc(model: *const Model) llama_gpu.ModelDesc {
    return .{
        .embedding_length = model.embedding_length,
        .block_count = model.block_count,
        .context_length = model.context_length,
        .feed_forward_length = model.feed_forward_length,
        .rope_dimension_count = model.rope_dimension_count,
        .head_count = model.head_count,
        .head_count_kv = model.head_count_kv,
        .head_dimension = model.head_dimension,
        .kv_dimension = model.kv_dimension,
        .rope_freq_base = model.rope_freq_base,
        .vocab_size = model.tokenizer.tokens.len,
        .rms_norm_eps = model.rms_norm_eps,
    };
}

const MatVecTask = struct {
    out: []f32,
    bytes: []const u8,
    row_size: usize,
    row_len: usize,
    tensor_type: TensorType,
    input: []const f32,
    row_count: usize,
};

const RowRange = struct {
    start: usize,
    end: usize,
};

const MatVecWorkers = struct {
    mutex: std.Thread.Mutex = .{},
    work_ready: std.Thread.Condition = .{},
    work_done: std.Thread.Condition = .{},
    threads: []std.Thread,
    task: ?MatVecTask = null,
    generation: u64 = 0,
    active_workers: usize = 0,
    shutting_down: bool = false,

    fn init(allocator: std.mem.Allocator, helper_count: usize) !*MatVecWorkers {
        const workers = try allocator.create(MatVecWorkers);
        errdefer allocator.destroy(workers);

        workers.* = .{
            .threads = try allocator.alloc(std.Thread, helper_count),
        };
        errdefer allocator.free(workers.threads);

        var spawned: usize = 0;
        errdefer {
            workers.mutex.lock();
            workers.shutting_down = true;
            workers.work_ready.broadcast();
            workers.mutex.unlock();
            for (workers.threads[0..spawned]) |thread| thread.join();
        }

        for (workers.threads, 0..) |*thread, index| {
            thread.* = try std.Thread.spawn(.{}, matVecWorkerMain, .{ workers, index });
            spawned += 1;
        }

        return workers;
    }

    fn deinit(self: *MatVecWorkers, allocator: std.mem.Allocator) void {
        self.mutex.lock();
        self.shutting_down = true;
        self.work_ready.broadcast();
        self.mutex.unlock();

        for (self.threads) |thread| thread.join();
        allocator.free(self.threads);
        self.* = undefined;
    }

    fn run(self: *MatVecWorkers, task: MatVecTask) void {
        const participant_count = self.threads.len + 1;

        self.mutex.lock();
        self.task = task;
        self.active_workers = self.threads.len;
        self.generation += 1;
        self.work_ready.broadcast();
        self.mutex.unlock();

        runMatVecTask(task, self.threads.len, participant_count);

        self.mutex.lock();
        while (self.active_workers != 0) {
            self.work_done.wait(&self.mutex);
        }
        self.task = null;
        self.mutex.unlock();
    }
};

fn workerRowRange(worker_index: usize, participant_count: usize, row_count: usize) RowRange {
    const rows_per_participant = std.math.divCeil(usize, row_count, participant_count) catch unreachable;
    const start = @min(row_count, worker_index * rows_per_participant);
    const end = @min(row_count, start + rows_per_participant);
    return .{ .start = start, .end = end };
}

fn runMatVecTask(task: MatVecTask, worker_index: usize, participant_count: usize) void {
    const range = workerRowRange(worker_index, participant_count, task.row_count);
    for (range.start..range.end) |row_index| {
        const row = task.bytes[row_index * task.row_size ..][0..task.row_size];
        task.out[row_index] = dotRowAssumeValid(task.tensor_type, row, task.row_len, task.input);
    }
}

fn matVecWorkerMain(workers: *MatVecWorkers, worker_index: usize) void {
    var seen_generation: u64 = 0;

    while (true) {
        workers.mutex.lock();
        while (!workers.shutting_down and seen_generation == workers.generation) {
            workers.work_ready.wait(&workers.mutex);
        }
        if (workers.shutting_down) {
            workers.mutex.unlock();
            return;
        }

        seen_generation = workers.generation;
        const task = workers.task.?;
        const participant_count = workers.threads.len + 1;
        workers.mutex.unlock();

        runMatVecTask(task, worker_index, participant_count);

        workers.mutex.lock();
        workers.active_workers -= 1;
        if (workers.active_workers == 0) workers.work_done.signal();
        workers.mutex.unlock();
    }
}

pub fn generateLoaded(
    allocator: std.mem.Allocator,
    model: *const Model,
    prompt: []const u8,
    options: runtime_types.GenerationOptions,
    backend: ?backend_api.MatVecBackend,
    dense_tensors: ?DenseTensorLookup,
) !GenerateReport {
    return generateLoadedStreaming(allocator, model, prompt, options, backend, dense_tensors, null, null);
}

pub fn generateLoadedStreaming(
    allocator: std.mem.Allocator,
    model: *const Model,
    prompt: []const u8,
    options: runtime_types.GenerationOptions,
    backend: ?backend_api.MatVecBackend,
    dense_tensors: ?DenseTensorLookup,
    stream_ctx: ?*anyopaque,
    stream_callback: ?StreamCallback,
) !GenerateReport {
    const startup_begin = std.time.nanoTimestamp();
    var spinner = terminal.Spinner{};
    try spinner.start();
    errdefer spinner.stop();
    var profiler = metal_profile.Profiler.init(allocator, backend != null and backend.?.label == .metal and options.metal_profile);
    defer profiler.deinit();
    var prng = std.Random.DefaultPrng.init(options.seed);
    const random = prng.random();

    const prompt_capacity = prompt.len * 4 + options.max_tokens + 8;
    const session_init_begin = std.time.nanoTimestamp();
    var session = try Session.init(
        allocator,
        model,
        backend,
        dense_tensors,
        @min(model.context_length, prompt_capacity),
        if (profiler.enabled) &profiler else null,
    );
    const session_init_ns = deltaNs(session_init_begin, std.time.nanoTimestamp());
    defer session.deinit(allocator);
    const startup_end = std.time.nanoTimestamp();
    spinner.stop();

    const prompt_begin = std.time.nanoTimestamp();
    const prompt_token_count = try model.tokenizer.encodeInto(allocator, prompt, session.token_buffer);
    try session.runPrompt(session.token_buffer[0..prompt_token_count]);
    const prompt_end = std.time.nanoTimestamp();

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);
    const sample_candidates = try allocator.alloc(SampleCandidate, session.logits.len);
    defer allocator.free(sample_candidates);
    var generated_token_count: usize = 0;
    var ttft_ns = deltaNs(startup_begin, prompt_end);
    var first_decode_step_ns: u64 = 0;

    const decode_begin = std.time.nanoTimestamp();
    while (generated_token_count < options.max_tokens) : (generated_token_count += 1) {
        profiler.beginDecodeToken();
        const sample_begin = std.time.nanoTimestamp();
        const next_token = sampleToken(session.logits, session.token_buffer[0..session.position], options, random, sample_candidates);
        profiler.record(.cpu_sampling, runtime_types.deltaNs(sample_begin, std.time.nanoTimestamp()));
        if (model.tokenizer.eos_token_id != null and next_token == model.tokenizer.eos_token_id.?) {
            profiler.endDecodeToken();
            break;
        }
        const chunk_start = output.items.len;
        try model.tokenizer.appendDecodedToken(&output, allocator, next_token);
        if (stream_callback) |callback| {
            const chunk = output.items[chunk_start..];
            if (chunk.len > 0) try callback(stream_ctx, chunk);
        }
        if (generated_token_count == 0) {
            ttft_ns = deltaNs(startup_begin, std.time.nanoTimestamp());
        }
        const step_begin = std.time.nanoTimestamp();
        _ = try session.step(next_token);
        if (generated_token_count == 0) {
            first_decode_step_ns = deltaNs(step_begin, std.time.nanoTimestamp());
        }
        profiler.endDecodeToken();
    }
    const decode_end = std.time.nanoTimestamp();
    const profile_summary = if (profiler.enabled) try profiler.renderSummary(allocator) else null;

    return .{
        .generated_text = try output.toOwnedSlice(allocator),
        .prompt_token_count = prompt_token_count,
        .reused_prompt_token_count = 0,
        .generated_token_count = generated_token_count,
        .startup_ns = deltaNs(startup_begin, startup_end),
        .prompt_ns = deltaNs(prompt_begin, prompt_end),
        .ttft_ns = ttft_ns,
        .decode_ns = deltaNs(decode_begin, decode_end),
        .backend = if (backend == null) .cpu else .metal,
        .startup_breakdown = .{
            .session_init_ns = session_init_ns,
            .first_decode_step_ns = first_decode_step_ns,
        },
        .metal_profile_summary = profile_summary,
    };
}

pub fn generateLoadedStreamingCached(
    allocator: std.mem.Allocator,
    model: *const Model,
    prompt: []const u8,
    options: runtime_types.GenerationOptions,
    backend: ?backend_api.MatVecBackend,
    _: ?DenseTensorLookup,
    reusable_session: *ReusableSession,
    stream_ctx: ?*anyopaque,
    stream_callback: ?StreamCallback,
) !GenerateReport {
    const startup_begin = std.time.nanoTimestamp();
    var spinner = terminal.Spinner{};
    try spinner.start();
    errdefer spinner.stop();
    var profiler = metal_profile.Profiler.init(allocator, backend != null and backend.?.label == .metal and options.metal_profile);
    defer profiler.deinit();
    var prng = std.Random.DefaultPrng.init(options.seed);
    const random = prng.random();
    const startup_end = std.time.nanoTimestamp();
    spinner.stop();

    const prompt_tokens = try allocator.alloc(u32, model.context_length);
    defer allocator.free(prompt_tokens);

    const prompt_begin = std.time.nanoTimestamp();
    const prompt_token_count = try model.tokenizer.encodeInto(allocator, prompt, prompt_tokens);
    if (prompt_token_count == 0) return error.EmptyPrompt;

    var session = &reusable_session.session;
    if (session.model != model) return error.InvalidPrompt;

    const cached_token_count = session.position;
    const reused_prompt_token_count = commonPrefixLen(
        session.token_buffer[0..cached_token_count],
        prompt_tokens[0..prompt_token_count],
    );
    if (reused_prompt_token_count != cached_token_count) {
        reusable_session.reset();
        session = &reusable_session.session;
    }

    if (session.position < prompt_token_count) {
        try session.runPrompt(prompt_tokens[session.position..prompt_token_count]);
    }
    const prompt_end = std.time.nanoTimestamp();

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);
    const sample_candidates = try allocator.alloc(SampleCandidate, session.logits.len);
    defer allocator.free(sample_candidates);
    var generated_token_count: usize = 0;
    var ttft_ns = deltaNs(startup_begin, prompt_end);
    var first_decode_step_ns: u64 = 0;

    const decode_begin = std.time.nanoTimestamp();
    while (generated_token_count < options.max_tokens) : (generated_token_count += 1) {
        profiler.beginDecodeToken();
        const sample_begin = std.time.nanoTimestamp();
        const next_token = sampleToken(session.logits, session.token_buffer[0..session.position], options, random, sample_candidates);
        profiler.record(.cpu_sampling, runtime_types.deltaNs(sample_begin, std.time.nanoTimestamp()));
        if (model.tokenizer.eos_token_id != null and next_token == model.tokenizer.eos_token_id.?) {
            profiler.endDecodeToken();
            break;
        }
        const chunk_start = output.items.len;
        try model.tokenizer.appendDecodedToken(&output, allocator, next_token);
        if (stream_callback) |callback| {
            const chunk = output.items[chunk_start..];
            if (chunk.len > 0) try callback(stream_ctx, chunk);
        }
        if (generated_token_count == 0) {
            ttft_ns = deltaNs(startup_begin, std.time.nanoTimestamp());
        }
        const step_begin = std.time.nanoTimestamp();
        _ = try session.step(next_token);
        if (generated_token_count == 0) {
            first_decode_step_ns = deltaNs(step_begin, std.time.nanoTimestamp());
        }
        profiler.endDecodeToken();
    }
    const decode_end = std.time.nanoTimestamp();
    const profile_summary = if (profiler.enabled) try profiler.renderSummary(allocator) else null;

    return .{
        .generated_text = try output.toOwnedSlice(allocator),
        .prompt_token_count = prompt_token_count,
        .reused_prompt_token_count = reused_prompt_token_count,
        .generated_token_count = generated_token_count,
        .startup_ns = deltaNs(startup_begin, startup_end),
        .prompt_ns = deltaNs(prompt_begin, prompt_end),
        .ttft_ns = ttft_ns,
        .decode_ns = deltaNs(decode_begin, decode_end),
        .backend = if (backend == null) .cpu else .metal,
        .startup_breakdown = .{
            .session_init_ns = 0,
            .first_decode_step_ns = first_decode_step_ns,
        },
        .metal_profile_summary = profile_summary,
    };
}

fn commonPrefixLen(lhs: []const u32, rhs: []const u32) usize {
    const max_len = @min(lhs.len, rhs.len);
    var index: usize = 0;
    while (index < max_len and lhs[index] == rhs[index]) : (index += 1) {}
    return index;
}

pub fn loadModel(allocator: std.mem.Allocator, model_path: []const u8) !Model {
    const file = try std.fs.cwd().openFile(model_path, .{});
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

    var parser = Parser{ .bytes = mapped_bytes };
    const magic = try parser.readBytes(4);
    if (!std.mem.eql(u8, magic, gguf_magic)) return error.InvalidMagic;

    const version = try parser.readInt(u32);
    if (version != 2 and version != 3) return error.UnsupportedVersion;

    const tensor_count = try parser.readInt(u64);
    const metadata_count = try parser.readInt(u64);

    var metadata = Metadata{};
    defer metadata.deinit(allocator);

    var kv_index: u64 = 0;
    while (kv_index < metadata_count) : (kv_index += 1) {
        try parseMetadataEntry(allocator, &parser, &metadata);
    }

    const architecture = metadata.architecture orelse return error.MissingRequiredMetadata;
    if (!std.mem.eql(u8, architecture, "llama")) return error.UnsupportedArchitecture;
    const tokenizer_model = metadata.tokenizer_model orelse return error.MissingRequiredMetadata;
    if (!std.mem.eql(u8, tokenizer_model, "llama")) return error.UnsupportedTokenizer;

    var tensors = std.StringHashMap(TensorRef).init(allocator);
    defer tensors.deinit();

    var tensor_index: u64 = 0;
    while (tensor_index < tensor_count) : (tensor_index += 1) {
        const tensor = try parseTensorInfo(allocator, &parser, metadata.alignment);
        try tensors.put(tensor.name, tensor);
    }

    const data_offset = alignForward(parser.pos, metadata.alignment);
    if (data_offset > mapped_bytes.len) return error.TruncatedFile;

    const block_count = metadata.block_count orelse return error.MissingRequiredMetadata;
    const embedding_length = metadata.embedding_length orelse return error.MissingRequiredMetadata;
    const context_length = metadata.context_length orelse return error.MissingRequiredMetadata;
    const feed_forward_length = metadata.feed_forward_length orelse return error.MissingRequiredMetadata;
    const head_count = metadata.head_count orelse return error.MissingRequiredMetadata;
    const head_count_kv = metadata.head_count_kv orelse return error.MissingRequiredMetadata;
    const rope_dimension_count = metadata.rope_dimension_count orelse return error.MissingRequiredMetadata;
    const rms_norm_eps = metadata.rms_norm_eps orelse return error.MissingRequiredMetadata;
    const rope_freq_base = metadata.rope_freq_base orelse 10000;
    if (head_count == 0 or head_count_kv == 0 or embedding_length % head_count != 0) return error.InvalidMetadataValue;

    const token_embd = try takeTensor(allocator, &tensors, "token_embd.weight");
    const output = try takeTensor(allocator, &tensors, "output.weight");
    const output_norm = try takeTensor(allocator, &tensors, "output_norm.weight");

    const layers = try allocator.alloc(LayerRefs, block_count);
    errdefer allocator.free(layers);
    for (0..block_count) |index| {
        layers[index] = .{
            .attn_norm = try takeLayerTensor(allocator, &tensors, index, "attn_norm.weight"),
            .attn_q = try takeLayerTensor(allocator, &tensors, index, "attn_q.weight"),
            .attn_k = try takeLayerTensor(allocator, &tensors, index, "attn_k.weight"),
            .attn_v = try takeLayerTensor(allocator, &tensors, index, "attn_v.weight"),
            .attn_output = try takeLayerTensor(allocator, &tensors, index, "attn_output.weight"),
            .ffn_norm = try takeLayerTensor(allocator, &tensors, index, "ffn_norm.weight"),
            .ffn_gate = try takeLayerTensor(allocator, &tensors, index, "ffn_gate.weight"),
            .ffn_down = try takeLayerTensor(allocator, &tensors, index, "ffn_down.weight"),
            .ffn_up = try takeLayerTensor(allocator, &tensors, index, "ffn_up.weight"),
        };
    }

    const tokenizer = try buildTokenizer(allocator, &metadata);
    const head_dimension = embedding_length / head_count;
    const kv_dimension = head_dimension * head_count_kv;

    return .{
        .bytes = mapped_bytes,
        .mapped_bytes = mapped_bytes,
        .tokenizer = tokenizer,
        .context_length = context_length,
        .embedding_length = embedding_length,
        .block_count = block_count,
        .feed_forward_length = feed_forward_length,
        .rope_dimension_count = rope_dimension_count,
        .head_count = head_count,
        .head_count_kv = head_count_kv,
        .head_dimension = head_dimension,
        .kv_dimension = kv_dimension,
        .rms_norm_eps = rms_norm_eps,
        .rope_freq_base = rope_freq_base,
        .data_offset = data_offset,
        .token_embd = token_embd,
        .output = output,
        .output_norm = output_norm,
        .layers = layers,
    };
}

pub fn countPromptTokens(allocator: std.mem.Allocator, model: *const Model, prompt: []const u8) !usize {
    const token_buf = try allocator.alloc(u32, model.context_length);
    defer allocator.free(token_buf);
    return model.tokenizer.encodeInto(allocator, prompt, token_buf);
}

fn buildTokenizer(allocator: std.mem.Allocator, metadata: *Metadata) !Tokenizer {
    const token_count = metadata.tokenizer_tokens.items.len;
    if (token_count == 0) return error.MissingRequiredMetadata;
    if (metadata.tokenizer_scores.items.len != token_count) return error.MissingRequiredMetadata;

    const tokens = try metadata.tokenizer_tokens.toOwnedSlice(allocator);
    metadata.tokenizer_tokens = .empty;

    const scores = try metadata.tokenizer_scores.toOwnedSlice(allocator);
    metadata.tokenizer_scores = .empty;

    const token_types = if (metadata.tokenizer_types.items.len == token_count)
        try metadata.tokenizer_types.toOwnedSlice(allocator)
    else blk: {
        const fallback = try allocator.alloc(u32, token_count);
        @memset(fallback, 1);
        break :blk fallback;
    };
    metadata.tokenizer_types = .empty;

    var byte_fallback = [_]?u32{null} ** 256;
    for (tokens, 0..) |token, token_id| {
        if (parseByteFallback(token)) |byte| byte_fallback[byte] = @intCast(token_id);
    }

    return .{
        .tokens = tokens,
        .scores = scores,
        .token_types = token_types,
        .byte_fallback = byte_fallback,
        .bos_token_id = metadata.bos_token_id,
        .eos_token_id = metadata.eos_token_id,
        .unk_token_id = metadata.unk_token_id,
        .pad_token_id = metadata.pad_token_id,
        .add_bos_token = metadata.add_bos_token orelse true,
        .add_eos_token = metadata.add_eos_token orelse false,
    };
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
    if (std.mem.eql(u8, key, "general.alignment")) {
        metadata.alignment = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "general.file_type")) {
        metadata.file_type = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.context_length")) {
        metadata.context_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.embedding_length")) {
        metadata.embedding_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.block_count")) {
        metadata.block_count = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.feed_forward_length")) {
        metadata.feed_forward_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.rope.dimension_count")) {
        metadata.rope_dimension_count = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.attention.head_count")) {
        metadata.head_count = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.attention.head_count_kv")) {
        metadata.head_count_kv = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.attention.layer_norm_rms_epsilon")) {
        metadata.rms_norm_eps = try readExpectedFloat(parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "llama.rope.freq_base")) {
        metadata.rope_freq_base = try readExpectedFloat(parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.model")) {
        metadata.tokenizer_model = try readExpectedString(allocator, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
        try readStringArray(allocator, parser, value_type, &metadata.tokenizer_tokens);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.scores")) {
        try readFloatArray(allocator, parser, value_type, &metadata.tokenizer_scores);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.token_type")) {
        try readU32Array(allocator, parser, value_type, &metadata.tokenizer_types);
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
    if (std.mem.eql(u8, key, "tokenizer.ggml.padding_token_id")) {
        metadata.pad_token_id = try readExpectedUnsigned(u32, parser, value_type);
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

    try skipValue(parser, value_type);
}

fn parseTensorInfo(allocator: std.mem.Allocator, parser: *Parser, alignment: u32) !TensorRef {
    const name = try readOwnedString(allocator, parser);
    const raw_dims = try parser.readInt(u32);
    if (raw_dims == 0 or raw_dims > max_tensor_dims) return error.InvalidTensorMetadata;
    const n_dims: usize = @intCast(raw_dims);

    var dims = [_]u64{1} ** max_tensor_dims;
    for (0..n_dims) |index| {
        dims[index] = try parser.readInt(u64);
        if (dims[index] == 0) return error.InvalidTensorMetadata;
    }

    const raw_type = try parser.readInt(u32);
    const tensor_type = std.meta.intToEnum(TensorType, raw_type) catch return error.UnsupportedTensorType;
    const offset = try parser.readInt(u64);
    if (offset % alignment != 0) return error.InvalidTensorMetadata;

    return .{
        .name = name,
        .dims = dims,
        .n_dims = n_dims,
        .tensor_type = tensor_type,
        .offset = offset,
    };
}

fn takeTensor(allocator: std.mem.Allocator, tensors: *std.StringHashMap(TensorRef), name: []const u8) !TensorRef {
    const tensor = tensors.get(name) orelse return error.MissingRequiredTensor;
    _ = allocator;
    _ = tensors.remove(name);
    return tensor;
}

fn takeLayerTensor(allocator: std.mem.Allocator, tensors: *std.StringHashMap(TensorRef), layer_index: usize, suffix: []const u8) !TensorRef {
    var buffer: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&buffer, "blk.{d}.{s}", .{ layer_index, suffix });
    return takeTensor(allocator, tensors, name);
}

pub fn tensorBytes(model: *const Model, tensor: TensorRef) ![]const u8 {
    const start = std.math.add(usize, model.data_offset, std.math.cast(usize, tensor.offset) orelse return error.Overflow) catch return error.Overflow;
    const row_len = try tensor.rowLen();
    const row_count = try tensor.rowCount();
    const row_size = try tensorRowByteSize(tensor.tensor_type, row_len);
    const total = try std.math.mul(usize, row_size, row_count);
    const end = std.math.add(usize, start, total) catch return error.Overflow;
    if (end > model.bytes.len) return error.TruncatedFile;
    return model.bytes[start..end];
}

pub fn tensorRowByteSize(tensor_type: TensorType, row_len: usize) !usize {
    return switch (tensor_type) {
        .f32 => try std.math.mul(usize, row_len, 4),
        .f16 => try std.math.mul(usize, row_len, 2),
        .q4_k => blk: {
            if (row_len % 256 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 256, 144);
        },
        .q6_k => blk: {
            if (row_len % 256 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 256, 210);
        },
    };
}

fn embeddingLookup(out: []f32, model: *const Model, tensor: TensorRef, token_id: u32) !void {
    const row_count = try tensor.rowCount();
    if (token_id >= row_count) return error.InvalidPrompt;
    const row_len = try tensor.rowLen();
    if (out.len != row_len) return error.InvalidTensorMetadata;

    const bytes = try tensorBytes(model, tensor);
    const row_size = try tensorRowByteSize(tensor.tensor_type, row_len);
    const start = @as(usize, token_id) * row_size;
    const row = bytes[start .. start + row_size];
    try dequantizeRow(out, tensor.tensor_type, row, row_len);
}

fn rmsNorm(out: []f32, input: []const f32, model: *const Model, tensor: TensorRef) !void {
    const weights = try tensorBytes(model, tensor);
    if (tensor.tensor_type != .f32) return error.UnsupportedTensorType;
    if (try tensor.rowLen() != input.len) return error.InvalidTensorMetadata;

    var mean_square = dot(input, input);
    mean_square /= @as(f32, @floatFromInt(input.len));
    const scale = @as(f32, 1.0) / @sqrt(mean_square + model.rms_norm_eps);

    for (input, 0..) |value, index| {
        const weight = readF32(weights[index * 4 ..][0..4]);
        out[index] = value * scale * weight;
    }
}

pub fn dequantizeRow(out: []f32, tensor_type: TensorType, row: []const u8, row_len: usize) !void {
    switch (tensor_type) {
        .f32 => {
            for (0..row_len) |index| out[index] = readF32(row[index * 4 ..][0..4]);
        },
        .f16 => {
            for (0..row_len) |index| out[index] = readF16AsF32(row[index * 2 ..][0..2]);
        },
        .q4_k => try dequantizeRowQ4K(out, row, row_len),
        .q6_k => try dequantizeRowQ6K(out, row, row_len),
    }
}

fn dotRow(tensor_type: TensorType, row: []const u8, row_len: usize, input: []const f32) !f32 {
    return switch (tensor_type) {
        .f32 => dotF32Row(row, input),
        .f16 => dotF16Row(row, input),
        .q4_k => try dotQ4KRow(row, row_len, input),
        .q6_k => try dotQ6KRow(row, row_len, input),
    };
}

fn dotRowAssumeValid(tensor_type: TensorType, row: []const u8, row_len: usize, input: []const f32) f32 {
    return switch (tensor_type) {
        .f32 => dotF32Row(row, input),
        .f16 => dotF16Row(row, input),
        .q4_k => dotQ4KRow(row, row_len, input) catch unreachable,
        .q6_k => dotQ6KRow(row, row_len, input) catch unreachable,
    };
}

fn dotF32Row(row: []const u8, input: []const f32) f32 {
    var acc = zeroSimd();
    var index: usize = 0;
    while (index + simd_lane_count <= input.len) : (index += simd_lane_count) {
        acc += loadF32Vec(row[index * 4 ..][0 .. simd_lane_count * 4]) * loadInputVec(input, index);
    }

    var sum = reduceVec(acc);
    while (index < input.len) : (index += 1) {
        sum += readF32(row[index * 4 ..][0..4]) * input[index];
    }
    return sum;
}

fn dotF16Row(row: []const u8, input: []const f32) f32 {
    var acc = zeroSimd();
    var index: usize = 0;
    while (index + simd_lane_count <= input.len) : (index += simd_lane_count) {
        acc += loadF16Vec(row[index * 2 ..][0 .. simd_lane_count * 2]) * loadInputVec(input, index);
    }

    var sum = reduceVec(acc);
    while (index < input.len) : (index += 1) {
        sum += readF16AsF32(row[index * 2 ..][0..2]) * input[index];
    }
    return sum;
}

fn dequantizeRowQ4K(out: []f32, row: []const u8, row_len: usize) !void {
    if (row_len % 256 != 0) return error.InvalidTensorMetadata;
    var out_index: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += 144) {
        const block = row[block_index .. block_index + 144];
        const d = readF16AsF32(block[0..2]);
        const dmin = readF16AsF32(block[2..4]);
        const scales: []const u8 = block[4..16];
        var q: []const u8 = block[16..144];
        var scale_index: usize = 0;

        var group: usize = 0;
        while (group < 4) : (group += 1) {
            const sm0 = getScaleMinK4(scale_index + 0, scales);
            const sm1 = getScaleMinK4(scale_index + 1, scales);
            const d1 = d * @as(f32, @floatFromInt(sm0.scale));
            const m1 = dmin * @as(f32, @floatFromInt(sm0.min));
            const d2 = d * @as(f32, @floatFromInt(sm1.scale));
            const m2 = dmin * @as(f32, @floatFromInt(sm1.min));

            for (0..32) |l| out[out_index + l] = d1 * @as(f32, @floatFromInt(q[l] & 0x0F)) - m1;
            for (0..32) |l| out[out_index + 32 + l] = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
            out_index += 64;
            q = q[32..];
            scale_index += 2;
        }
    }
}

fn dotQ4KRow(row: []const u8, row_len: usize, input: []const f32) !f32 {
    if (row_len % 256 != 0) return error.InvalidTensorMetadata;
    var sum: f32 = 0;
    var input_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += 144) {
        const block = row[block_index .. block_index + 144];
        const d = readF16AsF32(block[0..2]);
        const dmin = readF16AsF32(block[2..4]);
        const scales: []const u8 = block[4..16];
        var q: []const u8 = block[16..144];
        var scale_index: usize = 0;

        var group: usize = 0;
        while (group < 4) : (group += 1) {
            const sm0 = getScaleMinK4(scale_index + 0, scales);
            const sm1 = getScaleMinK4(scale_index + 1, scales);
            const d1 = d * @as(f32, @floatFromInt(sm0.scale));
            const m1 = dmin * @as(f32, @floatFromInt(sm0.min));
            const d2 = d * @as(f32, @floatFromInt(sm1.scale));
            const m2 = dmin * @as(f32, @floatFromInt(sm1.min));

            var low_q_dot: f32 = 0;
            var low_input_sum: f32 = 0;
            var high_q_dot: f32 = 0;
            var high_input_sum: f32 = 0;

            var l: usize = 0;
            while (l < 32) : (l += simd_lane_count) {
                const q_chunk = q[l..][0..simd_lane_count];
                const input_low = loadInputVec(input, input_offset + l);
                const input_high = loadInputVec(input, input_offset + 32 + l);

                low_q_dot += reduceVec(loadQ4LowVec(q_chunk) * input_low);
                low_input_sum += reduceVec(input_low);
                high_q_dot += reduceVec(loadQ4HighVec(q_chunk) * input_high);
                high_input_sum += reduceVec(input_high);
            }

            sum += d1 * low_q_dot - m1 * low_input_sum;
            sum += d2 * high_q_dot - m2 * high_input_sum;
            input_offset += 64;
            q = q[32..];
            scale_index += 2;
        }
    }
    return sum;
}

fn dequantizeRowQ6K(out: []f32, row: []const u8, row_len: usize) !void {
    if (row_len % 256 != 0) return error.InvalidTensorMetadata;
    var out_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += 210) {
        const block = row[block_index .. block_index + 210];
        const ql: []const u8 = block[0..128];
        const qh: []const u8 = block[128..192];
        const scales: []const u8 = block[192..208];
        const d = readF16AsF32(block[208..210]);

        var ql_half = ql;
        var qh_half = qh;
        var sc_half = scales;
        var half: usize = 0;
        while (half < 2) : (half += 1) {
            for (0..32) |l| {
                const is = l / 16;
                const q1 = (@as(i32, ql_half[l + 0] & 0x0F) | (@as(i32, (qh_half[l] >> 0) & 0x03) << 4)) - 32;
                const q2 = (@as(i32, ql_half[l + 32] & 0x0F) | (@as(i32, (qh_half[l] >> 2) & 0x03) << 4)) - 32;
                const q3 = (@as(i32, ql_half[l + 0] >> 4) | (@as(i32, (qh_half[l] >> 4) & 0x03) << 4)) - 32;
                const q4 = (@as(i32, ql_half[l + 32] >> 4) | (@as(i32, (qh_half[l] >> 6) & 0x03) << 4)) - 32;
                out[out_offset + l + 0] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_half[is + 0])))) * @as(f32, @floatFromInt(q1));
                out[out_offset + l + 32] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_half[is + 2])))) * @as(f32, @floatFromInt(q2));
                out[out_offset + l + 64] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_half[is + 4])))) * @as(f32, @floatFromInt(q3));
                out[out_offset + l + 96] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_half[is + 6])))) * @as(f32, @floatFromInt(q4));
            }
            out_offset += 128;
            ql_half = ql_half[64..];
            qh_half = qh_half[32..];
            sc_half = sc_half[8..];
        }
    }
}

fn dotQ6KRow(row: []const u8, row_len: usize, input: []const f32) !f32 {
    if (row_len % 256 != 0) return error.InvalidTensorMetadata;
    var sum: f32 = 0;
    var input_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += 210) {
        const block = row[block_index .. block_index + 210];
        const ql: []const u8 = block[0..128];
        const qh: []const u8 = block[128..192];
        const scales: []const u8 = block[192..208];
        const d = readF16AsF32(block[208..210]);

        var ql_half = ql;
        var qh_half = qh;
        var sc_half = scales;
        var half: usize = 0;
        while (half < 2) : (half += 1) {
            for (0..2) |group16| {
                const l0 = group16 * 16;
                const s0 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_half[group16 + 0]))));
                const s2 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_half[group16 + 2]))));
                const s4 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_half[group16 + 4]))));
                const s6 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_half[group16 + 6]))));

                var q1_dot: f32 = 0;
                var q2_dot: f32 = 0;
                var q3_dot: f32 = 0;
                var q4_dot: f32 = 0;

                for (0..2) |sub| {
                    const base = l0 + sub * simd_lane_count;
                    q1_dot += reduceVec(loadQ6Q1Vec(ql_half, qh_half, base) * loadInputVec(input, input_offset + base + 0));
                    q2_dot += reduceVec(loadQ6Q2Vec(ql_half, qh_half, base) * loadInputVec(input, input_offset + base + 32));
                    q3_dot += reduceVec(loadQ6Q3Vec(ql_half, qh_half, base) * loadInputVec(input, input_offset + base + 64));
                    q4_dot += reduceVec(loadQ6Q4Vec(ql_half, qh_half, base) * loadInputVec(input, input_offset + base + 96));
                }

                sum += s0 * q1_dot;
                sum += s2 * q2_dot;
                sum += s4 * q3_dot;
                sum += s6 * q4_dot;
            }
            input_offset += 128;
            ql_half = ql_half[64..];
            qh_half = qh_half[32..];
            sc_half = sc_half[8..];
        }
    }
    return sum;
}

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

fn applyRoPE(values: []f32, head_count: usize, head_dim: usize, rope_dim: usize, position: usize, freq_base: f32) void {
    const n_rot = @min(rope_dim, head_dim);
    for (0..head_count) |head_index| {
        const head = values[head_index * head_dim ..][0..head_dim];
        var pair: usize = 0;
        while (pair + 1 < n_rot) : (pair += 2) {
            const exponent = @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(n_rot));
            const theta = @as(f32, @floatFromInt(position)) / std.math.pow(f32, freq_base, exponent);
            const cos_theta = @cos(theta);
            const sin_theta = @sin(theta);
            const x0 = head[pair];
            const x1 = head[pair + 1];
            head[pair] = x0 * cos_theta - x1 * sin_theta;
            head[pair + 1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

const SampleCandidate = struct {
    token_id: u32 = 0,
    logit: f32 = 0,
    weight: f64 = 0,
};

fn sampleToken(
    logits: []const f32,
    recent_tokens: []const u32,
    options: runtime_types.GenerationOptions,
    random: std.Random,
    candidates: []SampleCandidate,
) u32 {
    if (options.temperature <= 0) return argmax(logits);

    const top_k = if (options.top_k == 0) logits.len else @min(options.top_k, logits.len);
    const candidate_count = if (top_k < logits.len)
        collectTopKCandidates(logits, top_k, candidates[0..top_k])
    else
        collectAllCandidates(logits, candidates[0..logits.len]);

    var active = candidates[0..candidate_count];
    if (active.len == 0) return argmax(logits);

    var max_logit = -std.math.inf(f32);
    var total_weight: f64 = 0;
    for (active) |*candidate| {
        const adjusted_logit = applyRepeatPenalty(candidate.logit, candidate.token_id, recent_tokens, options.repeat_penalty);
        candidate.logit = adjusted_logit;
        max_logit = @max(max_logit, adjusted_logit);
    }
    for (active) |*candidate| {
        const shifted = (@as(f64, candidate.logit) - @as(f64, max_logit)) / @as(f64, options.temperature);
        candidate.weight = @exp(shifted);
        total_weight += candidate.weight;
    }
    if (total_weight <= 0 or !std.math.isFinite(total_weight)) return argmax(logits);

    if (options.min_p > 0) {
        const kept = truncateByMinP(active, options.min_p);
        active = active[0..kept];
        total_weight = 0;
        for (active) |candidate| total_weight += candidate.weight;
        if (total_weight <= 0 or !std.math.isFinite(total_weight)) return argmax(logits);
    }

    if (options.top_p < 1.0) {
        std.mem.sort(SampleCandidate, active, {}, lessSampleCandidate);
        const kept = truncateByTopP(active, total_weight, options.top_p);
        active = active[0..kept];
        total_weight = 0;
        for (active) |candidate| total_weight += candidate.weight;
        if (total_weight <= 0 or !std.math.isFinite(total_weight)) return argmax(logits);
    }

    const target = random.float(f64) * total_weight;
    var cumulative: f64 = 0;
    for (active) |candidate| {
        cumulative += candidate.weight;
        if (target <= cumulative) return candidate.token_id;
    }

    return active[active.len - 1].token_id;
}

fn applyRepeatPenalty(logit: f32, token_id: u32, recent_tokens: []const u32, repeat_penalty: f32) f32 {
    if (repeat_penalty <= 1.0) return logit;
    for (recent_tokens) |recent| {
        if (recent != token_id) continue;
        return if (logit < 0) logit * repeat_penalty else logit / repeat_penalty;
    }
    return logit;
}

fn collectAllCandidates(logits: []const f32, out: []SampleCandidate) usize {
    for (logits, 0..) |logit, index| {
        out[index] = .{
            .token_id = @intCast(index),
            .logit = logit,
        };
    }
    return logits.len;
}

fn collectTopKCandidates(logits: []const f32, top_k: usize, out: []SampleCandidate) usize {
    var count: usize = 0;
    for (logits, 0..) |logit, index| {
        const candidate = SampleCandidate{
            .token_id = @intCast(index),
            .logit = logit,
        };

        if (count < top_k) {
            out[count] = candidate;
            count += 1;
            var pos = count - 1;
            while (pos > 0 and out[pos].logit > out[pos - 1].logit) : (pos -= 1) {
                std.mem.swap(SampleCandidate, &out[pos], &out[pos - 1]);
            }
            continue;
        }

        if (candidate.logit <= out[top_k - 1].logit) continue;
        out[top_k - 1] = candidate;
        var pos = top_k - 1;
        while (pos > 0 and out[pos].logit > out[pos - 1].logit) : (pos -= 1) {
            std.mem.swap(SampleCandidate, &out[pos], &out[pos - 1]);
        }
    }
    return count;
}

fn truncateByTopP(candidates: []SampleCandidate, total_weight: f64, top_p: f32) usize {
    var cumulative: f64 = 0;
    for (candidates, 0..) |candidate, index| {
        cumulative += candidate.weight;
        if (cumulative / total_weight >= @as(f64, top_p)) return index + 1;
    }
    return candidates.len;
}

fn truncateByMinP(candidates: []SampleCandidate, min_p: f32) usize {
    var max_weight: f64 = 0;
    for (candidates) |candidate| {
        max_weight = @max(max_weight, candidate.weight);
    }
    const threshold = max_weight * @as(f64, min_p);
    var kept: usize = 0;
    for (candidates) |candidate| {
        if (candidate.weight >= threshold) {
            candidates[kept] = candidate;
            kept += 1;
        }
    }
    return @max(@as(usize, 1), kept);
}

fn lessSampleCandidate(_: void, lhs: SampleCandidate, rhs: SampleCandidate) bool {
    return lhs.weight > rhs.weight;
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

test "sampleToken uses seed-driven randomness when temperature is enabled" {
    const logits = [_]f32{ 2.0, 1.8, 1.7, 1.6 };
    var candidates_a: [logits.len]SampleCandidate = undefined;
    var candidates_b: [logits.len]SampleCandidate = undefined;
    var candidates_c: [logits.len]SampleCandidate = undefined;

    var prng_a = std.Random.DefaultPrng.init(42);
    var prng_b = std.Random.DefaultPrng.init(42);
    var prng_c = std.Random.DefaultPrng.init(99);

    const sample_a = sampleToken(&logits, &.{}, .{ .temperature = 0.8 }, prng_a.random(), &candidates_a);
    const sample_b = sampleToken(&logits, &.{}, .{ .temperature = 0.8 }, prng_b.random(), &candidates_b);
    const sample_c = sampleToken(&logits, &.{}, .{ .temperature = 0.8 }, prng_c.random(), &candidates_c);

    try std.testing.expectEqual(sample_a, sample_b);
    try std.testing.expect(sample_a < logits.len);
    try std.testing.expect(sample_c < logits.len);
}

test "sampleToken falls back to argmax for non-positive temperature" {
    const logits = [_]f32{ -1.0, 3.0, 0.5 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    try std.testing.expectEqual(@as(u32, 1), sampleToken(&logits, &.{}, .{ .temperature = 0 }, prng.random(), &candidates));
}

test "sampleToken top-k restricts sampling set" {
    const logits = [_]f32{ 10.0, 9.0, 1.0, 0.5 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(123);

    for (0..32) |_| {
        const sample = sampleToken(&logits, &.{}, .{ .temperature = 1.0, .top_k = 2 }, prng.random(), &candidates);
        try std.testing.expect(sample == 0 or sample == 1);
    }
}

test "sampleToken top-p trims low-mass tail" {
    const logits = [_]f32{ 8.0, 7.5, 1.0, 0.2 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(321);

    for (0..32) |_| {
        const sample = sampleToken(&logits, &.{}, .{ .temperature = 1.0, .top_p = 0.9 }, prng.random(), &candidates);
        try std.testing.expect(sample == 0 or sample == 1);
    }
}

test "sampleToken repeat penalty demotes repeated tokens" {
    const logits = [_]f32{ 4.0, 3.9, 1.0 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(5);

    for (0..32) |_| {
        const sample = sampleToken(&logits, &.{0}, .{ .temperature = 0.7, .repeat_penalty = 1.5 }, prng.random(), &candidates);
        try std.testing.expect(sample != 2);
    }
}

test "sampleToken min-p drops low relative probability tail" {
    const logits = [_]f32{ 6.0, 5.5, 0.5, 0.2 };
    var candidates: [logits.len]SampleCandidate = undefined;
    var prng = std.Random.DefaultPrng.init(777);

    for (0..32) |_| {
        const sample = sampleToken(&logits, &.{}, .{ .temperature = 1.0, .min_p = 0.5 }, prng.random(), &candidates);
        try std.testing.expect(sample == 0 or sample == 1);
    }
}

fn addInPlace(dst: []f32, src: []const f32) void {
    for (dst, src) |*lhs, rhs| lhs.* += rhs;
}

fn silu(value: f32) f32 {
    return value / (1 + @exp(-value));
}

const F32x = @Vector(simd_lane_count, f32);

fn zeroSimd() F32x {
    return @splat(0);
}

fn reduceVec(value: F32x) f32 {
    return @reduce(.Add, value);
}

fn loadInputVec(values: []const f32, index: usize) F32x {
    return @as(F32x, @bitCast(values[index..][0..simd_lane_count].*));
}

fn loadF32Vec(bytes: []const u8) F32x {
    var values: [simd_lane_count]f32 = undefined;
    for (0..simd_lane_count) |lane| {
        values[lane] = readF32(bytes[lane * 4 ..][0..4]);
    }
    return @as(F32x, @bitCast(values));
}

fn loadF16Vec(bytes: []const u8) F32x {
    var values: [simd_lane_count]f32 = undefined;
    for (0..simd_lane_count) |lane| {
        values[lane] = readF16AsF32(bytes[lane * 2 ..][0..2]);
    }
    return @as(F32x, @bitCast(values));
}

fn loadQ4LowVec(bytes: []const u8) F32x {
    var values: [simd_lane_count]f32 = undefined;
    for (0..simd_lane_count) |lane| {
        values[lane] = @floatFromInt(bytes[lane] & 0x0F);
    }
    return @as(F32x, @bitCast(values));
}

fn loadQ4HighVec(bytes: []const u8) F32x {
    var values: [simd_lane_count]f32 = undefined;
    for (0..simd_lane_count) |lane| {
        values[lane] = @floatFromInt(bytes[lane] >> 4);
    }
    return @as(F32x, @bitCast(values));
}

fn loadQ6Q1Vec(ql: []const u8, qh: []const u8, base: usize) F32x {
    var values: [simd_lane_count]f32 = undefined;
    for (0..simd_lane_count) |lane| {
        const idx = base + lane;
        const q = (@as(i32, ql[idx] & 0x0F) | (@as(i32, (qh[idx] >> 0) & 0x03) << 4)) - 32;
        values[lane] = @floatFromInt(q);
    }
    return @as(F32x, @bitCast(values));
}

fn loadQ6Q2Vec(ql: []const u8, qh: []const u8, base: usize) F32x {
    var values: [simd_lane_count]f32 = undefined;
    for (0..simd_lane_count) |lane| {
        const idx = base + lane;
        const q = (@as(i32, ql[idx + 32] & 0x0F) | (@as(i32, (qh[idx] >> 2) & 0x03) << 4)) - 32;
        values[lane] = @floatFromInt(q);
    }
    return @as(F32x, @bitCast(values));
}

fn loadQ6Q3Vec(ql: []const u8, qh: []const u8, base: usize) F32x {
    var values: [simd_lane_count]f32 = undefined;
    for (0..simd_lane_count) |lane| {
        const idx = base + lane;
        const q = (@as(i32, ql[idx] >> 4) | (@as(i32, (qh[idx] >> 4) & 0x03) << 4)) - 32;
        values[lane] = @floatFromInt(q);
    }
    return @as(F32x, @bitCast(values));
}

fn loadQ6Q4Vec(ql: []const u8, qh: []const u8, base: usize) F32x {
    var values: [simd_lane_count]f32 = undefined;
    for (0..simd_lane_count) |lane| {
        const idx = base + lane;
        const q = (@as(i32, ql[idx + 32] >> 4) | (@as(i32, (qh[idx] >> 6) & 0x03) << 4)) - 32;
        values[lane] = @floatFromInt(q);
    }
    return @as(F32x, @bitCast(values));
}

fn dot(a: []const f32, b: []const f32) f32 {
    var acc = zeroSimd();
    var index: usize = 0;
    while (index + simd_lane_count <= a.len) : (index += simd_lane_count) {
        acc += loadInputVec(a, index) * loadInputVec(b, index);
    }

    var sum = reduceVec(acc);
    while (index < a.len) : (index += 1) {
        sum += a[index] * b[index];
    }
    return sum;
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

fn parseByteFallback(token: []const u8) ?u8 {
    if (token.len != 6) return null;
    if (!std.mem.startsWith(u8, token, "<0x")) return null;
    if (token[5] != '>') return null;
    const hi = std.fmt.charToDigit(token[3], 16) catch return null;
    const lo = std.fmt.charToDigit(token[4], 16) catch return null;
    return @intCast((hi << 4) | lo);
}

fn readOwnedString(allocator: std.mem.Allocator, parser: *Parser) ![]u8 {
    const len_u64 = try parser.readInt(u64);
    const len = std.math.cast(usize, len_u64) orelse return error.Overflow;
    const bytes = try allocator.alloc(u8, len);
    const src = try parser.readBytes(len);
    @memcpy(bytes, src);
    return bytes;
}

fn readExpectedString(allocator: std.mem.Allocator, parser: *Parser, value_type: ValueType) ![]u8 {
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

fn readExpectedFloat(parser: *Parser, value_type: ValueType) !f32 {
    return switch (value_type) {
        .float32 => @bitCast(try parser.readInt(u32)),
        .float64 => @floatCast(@as(f64, @bitCast(try parser.readInt(u64)))),
        else => error.InvalidMetadataType,
    };
}

fn readExpectedBool(parser: *Parser, value_type: ValueType) !bool {
    if (value_type != .bool) return error.InvalidMetadataType;
    const raw = try parser.readInt(u8);
    return switch (raw) {
        0 => false,
        1 => true,
        else => error.InvalidMetadataValue,
    };
}

fn readStringArray(allocator: std.mem.Allocator, parser: *Parser, value_type: ValueType, out: *std.ArrayList([]u8)) !void {
    if (value_type != .array) return error.InvalidMetadataType;
    const element_type = std.meta.intToEnum(ValueType, try parser.readInt(u32)) catch return error.InvalidMetadataType;
    if (element_type != .string) return error.InvalidMetadataType;
    const count = try parser.readInt(u64);
    try out.ensureTotalCapacity(allocator, std.math.cast(usize, count) orelse return error.Overflow);
    for (0..count) |_| out.appendAssumeCapacity(try readOwnedString(allocator, parser));
}

fn readFloatArray(allocator: std.mem.Allocator, parser: *Parser, value_type: ValueType, out: *std.ArrayList(f32)) !void {
    if (value_type != .array) return error.InvalidMetadataType;
    const element_type = std.meta.intToEnum(ValueType, try parser.readInt(u32)) catch return error.InvalidMetadataType;
    if (element_type != .float32 and element_type != .float64) return error.InvalidMetadataType;
    const count = try parser.readInt(u64);
    try out.ensureTotalCapacity(allocator, std.math.cast(usize, count) orelse return error.Overflow);
    for (0..count) |_| {
        const value = if (element_type == .float32)
            @as(f32, @bitCast(try parser.readInt(u32)))
        else
            @as(f32, @floatCast(@as(f64, @bitCast(try parser.readInt(u64)))));
        out.appendAssumeCapacity(value);
    }
}

fn readU32Array(allocator: std.mem.Allocator, parser: *Parser, value_type: ValueType, out: *std.ArrayList(u32)) !void {
    if (value_type != .array) return error.InvalidMetadataType;
    const element_type = std.meta.intToEnum(ValueType, try parser.readInt(u32)) catch return error.InvalidMetadataType;
    if (element_type != .uint8 and element_type != .uint16 and element_type != .uint32 and element_type != .uint64 and element_type != .int8 and element_type != .int16 and element_type != .int32 and element_type != .int64) return error.InvalidMetadataType;
    const count = try parser.readInt(u64);
    try out.ensureTotalCapacity(allocator, std.math.cast(usize, count) orelse return error.Overflow);
    for (0..count) |_| {
        const value = switch (element_type) {
            .uint8 => @as(u32, try parser.readInt(u8)),
            .uint16 => @as(u32, try parser.readInt(u16)),
            .uint32 => try parser.readInt(u32),
            .uint64 => std.math.cast(u32, try parser.readInt(u64)) orelse return error.InvalidMetadataValue,
            .int8 => std.math.cast(u32, try parser.readInt(i8)) orelse return error.InvalidMetadataValue,
            .int16 => std.math.cast(u32, try parser.readInt(i16)) orelse return error.InvalidMetadataValue,
            .int32 => std.math.cast(u32, try parser.readInt(i32)) orelse return error.InvalidMetadataValue,
            .int64 => std.math.cast(u32, try parser.readInt(i64)) orelse return error.InvalidMetadataValue,
            else => unreachable,
        };
        out.appendAssumeCapacity(value);
    }
}

fn skipValue(parser: *Parser, value_type: ValueType) !void {
    switch (value_type) {
        .uint8, .int8, .bool => try parser.skipBytes(1),
        .uint16, .int16 => try parser.skipBytes(2),
        .uint32, .int32, .float32 => try parser.skipBytes(4),
        .uint64, .int64, .float64 => try parser.skipBytes(8),
        .string => {
            const len = try parser.readInt(u64);
            try parser.skipBytes(std.math.cast(usize, len) orelse return error.Overflow);
        },
        .array => {
            const element_type = std.meta.intToEnum(ValueType, try parser.readInt(u32)) catch return error.InvalidMetadataType;
            if (element_type == .array) return error.InvalidMetadataType;
            const count = try parser.readInt(u64);
            for (0..count) |_| try skipValue(parser, element_type);
        },
    }
}

fn readF32(bytes: []const u8) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[0..4], .little));
}

fn readF16AsF32(bytes: []const u8) f32 {
    const raw = std.mem.readInt(u16, bytes[0..2], .little);
    const half = @as(f16, @bitCast(raw));
    return @as(f32, @floatCast(half));
}

fn alignForward(value: usize, alignment: u32) usize {
    const a = @as(usize, alignment);
    return (value + a - 1) & ~(a - 1);
}

fn deltaNs(start: i128, end: i128) u64 {
    return @intCast(@max(@as(i128, 0), end - start));
}

test "readU32Array accepts int32 token type arrays" {
    var bytes = std.ArrayList(u8).empty;
    defer bytes.deinit(std.testing.allocator);

    try bytes.writer(std.testing.allocator).writeInt(u32, @intFromEnum(ValueType.int32), .little);
    try bytes.writer(std.testing.allocator).writeInt(u64, 4, .little);
    try bytes.writer(std.testing.allocator).writeInt(i32, 1, .little);
    try bytes.writer(std.testing.allocator).writeInt(i32, 2, .little);
    try bytes.writer(std.testing.allocator).writeInt(i32, 3, .little);
    try bytes.writer(std.testing.allocator).writeInt(i32, 6, .little);

    var parser = Parser{ .bytes = bytes.items };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);

    try readU32Array(std.testing.allocator, &parser, .array, &out);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 6 }, out.items);
}

test "readU32Array rejects negative signed values" {
    var bytes = std.ArrayList(u8).empty;
    defer bytes.deinit(std.testing.allocator);

    try bytes.writer(std.testing.allocator).writeInt(u32, @intFromEnum(ValueType.int32), .little);
    try bytes.writer(std.testing.allocator).writeInt(u64, 1, .little);
    try bytes.writer(std.testing.allocator).writeInt(i32, -1, .little);

    var parser = Parser{ .bytes = bytes.items };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);

    try std.testing.expectError(error.InvalidMetadataValue, readU32Array(std.testing.allocator, &parser, .array, &out));
}
