const std = @import("std");
const terminal = @import("../terminal.zig");
const backend_api = @import("../runtime/backend.zig");
const gpu = @import("../runtime/gpu/session.zig");
const metal_profile = @import("../runtime/metal_profile.zig");
const moon_quant_calibration = @import("../moon_quant_calibration.zig");
const runtime_types = @import("../runtime/types.zig");
const sampling = @import("../sampling.zig");

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
    sampling_strategy: runtime_types.SamplingStrategy = .auto,
    sampling_path: runtime_types.EffectiveSamplingPath = .cpu_logits,
    readback_mode: runtime_types.ReadbackMode = .none,
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
    get_moon_quant_by_offset_fn: *const fn (?*const anyopaque, u64) ?[]const u8,

    pub fn get(self: DenseTensorLookup, tensor: TensorRef) ?[]const f32 {
        return self.get_fn(self.ctx, tensor);
    }

    pub fn getByOffset(self: DenseTensorLookup, offset: u64) ?[]const f32 {
        return self.get_by_offset_fn(self.ctx, offset);
    }

    pub fn getRawByOffset(self: DenseTensorLookup, offset: u64) ?[]const u8 {
        return self.get_raw_by_offset_fn(self.ctx, offset);
    }

    pub fn getMoonQuantByOffset(self: DenseTensorLookup, offset: u64) ?[]const u8 {
        return self.get_moon_quant_by_offset_fn(self.ctx, offset);
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

pub const LayerType = enum {
    full_attention,
    linear_attention,
};

fn chooseSamplingPath(has_gpu_session: bool, options: runtime_types.GenerationOptions) runtime_types.EffectiveSamplingPath {
    if (!has_gpu_session) return .cpu_logits;
    const path = runtime_types.resolveSamplingPath(has_gpu_session, options.temperature, options.sampling_strategy);
    if (path == .gpu_topk_sampler and !runtime_types.canUseGpuTopKSampling(options)) return .cpu_logits;
    return path;
}

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
    float16 = 13,
};

pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
};

pub const RopeStyle = enum(u32) {
    interleaved = 0,
    neox = 1,
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
    attn_q: ?TensorRef = null,
    attn_q_bias: ?TensorRef = null,
    attn_q_norm: ?TensorRef = null,
    attn_k: ?TensorRef = null,
    attn_k_bias: ?TensorRef = null,
    attn_k_norm: ?TensorRef = null,
    attn_v: ?TensorRef = null,
    attn_v_bias: ?TensorRef = null,
    attn_output: ?TensorRef = null,
    ffn_norm: TensorRef,
    ffn_gate: TensorRef,
    ffn_down: TensorRef,
    ffn_up: TensorRef,
    post_attention_norm: ?TensorRef = null,
    post_ffw_norm: ?TensorRef = null,
    linear_attn: ?LinearAttnTensors = null,
};

pub const LinearAttnTensors = struct {
    in_proj_qkv: TensorRef,
    in_proj_z: TensorRef,
    in_proj_b: TensorRef,
    in_proj_a: TensorRef,
    conv1d: TensorRef,
    dt_bias: TensorRef,
    A_log: TensorRef,
    norm_weight: TensorRef,
    out_proj: TensorRef,
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
    attention_key_length: ?u32 = null,
    attention_value_length: ?u32 = null,
    rms_norm_eps: ?f32 = null,
    rope_freq_base: ?f32 = null,
    rope_scaling_type: ?[]u8 = null,
    rope_scaling_factor: ?f32 = null,
    sliding_window: ?u32 = null,
    attn_logit_softcapping: ?f32 = null,
    final_logit_softcapping: ?f32 = null,
    partial_rotary_factor: ?f32 = null,
    linear_num_key_heads: ?u32 = null,
    linear_num_value_heads: ?u32 = null,
    linear_key_head_dim: ?u32 = null,
    linear_value_head_dim: ?u32 = null,
    linear_conv_kernel_dim: ?u32 = null,
    layer_types: std.ArrayList([]u8) = .empty,
    tokenizer_model: ?[]u8 = null,
    tokenizer_pre: ?[]u8 = null,
    tokenizer_tokens: std.ArrayList([]u8) = .empty,
    tokenizer_merges: std.ArrayList([]u8) = .empty,
    tokenizer_scores: std.ArrayList(f32) = .empty,
    tokenizer_types: std.ArrayList(u32) = .empty,
    bos_token_id: ?u32 = null,
    eos_token_id: ?u32 = null,
    unk_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,
    add_bos_token: ?bool = null,
    add_eos_token: ?bool = null,
    add_space_prefix: ?bool = null,

    fn deinit(self: *Metadata, allocator: std.mem.Allocator) void {
        if (self.architecture) |value| allocator.free(value);
        if (self.tokenizer_model) |value| allocator.free(value);
        if (self.tokenizer_pre) |value| allocator.free(value);
        for (self.tokenizer_tokens.items) |token| allocator.free(token);
        for (self.tokenizer_merges.items) |merge| allocator.free(merge);
        for (self.layer_types.items) |lt| allocator.free(lt);
        self.tokenizer_tokens.deinit(allocator);
        self.tokenizer_merges.deinit(allocator);
        self.tokenizer_scores.deinit(allocator);
        self.tokenizer_types.deinit(allocator);
        self.layer_types.deinit(allocator);
        self.* = undefined;
    }
};

const Tokenizer = struct {
    mode: Mode,
    tokens: [][]u8,
    scores: []f32,
    token_types: []u32,
    special_tokens: []u32,
    byte_fallback: [256]?u32,
    merge_table: std.AutoHashMap(MergeKey, MergeValue),
    bos_token_id: ?u32,
    eos_token_id: ?u32,
    unk_token_id: ?u32,
    pad_token_id: ?u32,
    add_bos_token: bool,
    add_eos_token: bool,
    add_space_prefix: bool,
    prefer_longest_match: bool = false,

    const Mode = enum {
        score_dp,
        gpt2_bpe,
    };

    const MergeKey = struct {
        left: u32,
        right: u32,
    };

    const MergeValue = struct {
        merged: u32,
        rank: u32,
    };

    fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        for (self.tokens) |token| allocator.free(token);
        allocator.free(self.tokens);
        allocator.free(self.scores);
        allocator.free(self.token_types);
        allocator.free(self.special_tokens);
        self.merge_table.deinit();
        self.* = undefined;
    }

    fn isByteFallbackToken(self: Tokenizer, token_id: u32) ?u8 {
        if (self.mode == .gpt2_bpe or token_id >= self.tokens.len) return null;
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
            if (self.mode == .score_dp and self.token_types[token_id] == @intFromEnum(TokenType.byte)) return false;
        }
        return self.tokens[token_id].len > 0;
    }

    fn encodeInto(self: Tokenizer, allocator: std.mem.Allocator, prompt: []const u8, out: []u32) !usize {
        if (prompt.len == 0) return error.EmptyPrompt;

        return switch (self.mode) {
            .score_dp => self.encodeScoreDp(allocator, prompt, out),
            .gpt2_bpe => self.encodeGpt2Bpe(allocator, prompt, out),
        };
    }

    fn encodeScoreDp(self: Tokenizer, allocator: std.mem.Allocator, prompt: []const u8, out: []u32) !usize {
        var pieces = std.ArrayList(u32).empty;
        defer pieces.deinit(allocator);
        try pieces.ensureTotalCapacity(allocator, prompt.len);

        var pos: usize = 0;
        var is_first_chunk = true;
        while (pos < prompt.len) {
            var best_match: ?u32 = null;
            var best_match_pos: usize = prompt.len;
            var best_match_len: usize = 0;

            for (self.special_tokens) |special_id| {
                const special_str = self.tokens[special_id];
                if (special_str.len == 0) continue;
                if (std.mem.indexOf(u8, prompt[pos..], special_str)) |offset| {
                    const abs_pos = pos + offset;
                    if (abs_pos < best_match_pos or (abs_pos == best_match_pos and special_str.len > best_match_len)) {
                        best_match_pos = abs_pos;
                        best_match = special_id;
                        best_match_len = special_str.len;
                    }
                }
            }

            if (best_match_pos > pos) {
                try self.encodeScoreDpChunk(allocator, prompt[pos..best_match_pos], &pieces, is_first_chunk);
                is_first_chunk = false;
            }
            if (best_match) |token_id| {
                pieces.appendAssumeCapacity(token_id);
                pos = best_match_pos + best_match_len;
                is_first_chunk = false;
            } else {
                pos = prompt.len;
            }
        }

        var count: usize = 0;
        if (self.add_bos_token) {
            const bos = self.bos_token_id orelse return error.MissingRequiredMetadata;
            if (count >= out.len) return error.ContextOverflow;
            out[count] = bos;
            count += 1;
        }

        for (pieces.items) |token_id| {
            if (count >= out.len) return error.ContextOverflow;
            out[count] = token_id;
            count += 1;
        }

        if (self.add_eos_token and self.eos_token_id != null) {
            if (count >= out.len) return error.ContextOverflow;
            out[count] = self.eos_token_id.?;
            count += 1;
        }

        return count;
    }

    fn encodeScoreDpChunk(self: Tokenizer, allocator: std.mem.Allocator, chunk: []const u8, pieces: *std.ArrayList(u32), add_initial_space_prefix: bool) !void {
        if (chunk.len == 0) return;

        var normalized = std.ArrayList(u8).empty;
        defer normalized.deinit(allocator);
        if (self.add_space_prefix and add_initial_space_prefix) {
            try normalized.appendSlice(allocator, rope_metaspace);
        }
        for (chunk) |byte| {
            if (byte == ' ') {
                try normalized.appendSlice(allocator, rope_metaspace);
            } else {
                try normalized.append(allocator, byte);
            }
        }

        const text = normalized.items;
        if (self.prefer_longest_match) {
            try self.encodeScoreLongestChunk(text, pieces);
            return;
        }

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

        pos = 0;
        while (pos < n) {
            const token_id = best_tokens[pos] orelse return error.UnknownToken;
            try pieces.append(allocator, token_id);
            pos = best_next[pos];
        }
    }

    fn encodeScoreLongestChunk(self: Tokenizer, text: []const u8, pieces: *std.ArrayList(u32)) !void {
        var pos: usize = 0;
        while (pos < text.len) {
            var best_token: ?u32 = null;
            var best_len: usize = 0;

            for (self.tokens, 0..) |token, token_id_usize| {
                const token_id: u32 = @intCast(token_id_usize);
                if (!self.canEncodeToken(token_id)) continue;
                if (token.len == 0 or token.len > text.len - pos) continue;
                if (!std.mem.startsWith(u8, text[pos..], token)) continue;
                if (token.len > best_len) {
                    best_len = token.len;
                    best_token = token_id;
                }
            }

            if (best_token) |token_id| {
                pieces.appendAssumeCapacity(token_id);
                pos += best_len;
                continue;
            }

            const fallback = self.byte_fallback[text[pos]] orelse return error.UnknownToken;
            pieces.appendAssumeCapacity(fallback);
            pos += 1;
        }
    }

    fn encodeGpt2Bpe(self: Tokenizer, allocator: std.mem.Allocator, prompt: []const u8, out: []u32) !usize {
        var pieces = std.ArrayList(u32).empty;
        defer pieces.deinit(allocator);
        try pieces.ensureTotalCapacity(allocator, prompt.len);

        var pos: usize = 0;
        while (pos < prompt.len) {
            var best_match: ?u32 = null;
            var best_match_pos: usize = prompt.len;
            var best_match_len: usize = 0;

            for (self.special_tokens) |special_id| {
                const special_str = self.tokens[special_id];
                if (special_str.len == 0) continue;
                if (std.mem.indexOf(u8, prompt[pos..], special_str)) |offset| {
                    const abs_pos = pos + offset;
                    if (abs_pos < best_match_pos or (abs_pos == best_match_pos and special_str.len > best_match_len)) {
                        best_match_pos = abs_pos;
                        best_match = special_id;
                        best_match_len = special_str.len;
                    }
                }
            }

            if (best_match_pos > pos) {
                try self.encodeGpt2BpeChunk(allocator, prompt[pos..best_match_pos], &pieces);
            }
            if (best_match != null) {
                pieces.appendAssumeCapacity(best_match.?);
                pos = best_match_pos + best_match_len;
            } else {
                pos = prompt.len;
            }
        }

        var count: usize = 0;
        if (self.add_bos_token) {
            const bos = self.bos_token_id orelse return error.MissingRequiredMetadata;
            if (count >= out.len) return error.ContextOverflow;
            out[count] = bos;
            count += 1;
        }

        for (pieces.items) |token_id| {
            if (count >= out.len) return error.ContextOverflow;
            out[count] = token_id;
            count += 1;
        }

        if (self.add_eos_token and self.eos_token_id != null) {
            if (count >= out.len) return error.ContextOverflow;
            out[count] = self.eos_token_id.?;
            count += 1;
        }

        return count;
    }

    fn encodeGpt2BpeChunk(self: Tokenizer, allocator: std.mem.Allocator, chunk: []const u8, pieces: *std.ArrayList(u32)) !void {
        if (chunk.len == 0) return;
        var chunk_pieces = std.ArrayList(u32).empty;
        defer chunk_pieces.deinit(allocator);
        try chunk_pieces.ensureTotalCapacity(allocator, chunk.len);

        for (chunk) |byte| {
            const token_id = self.byte_fallback[byte] orelse return error.UnknownToken;
            chunk_pieces.appendAssumeCapacity(token_id);
        }

        while (chunk_pieces.items.len >= 2) {
            var best_index: ?usize = null;
            var best_rank: u32 = std.math.maxInt(u32);
            var best_merged: u32 = undefined;

            for (0..chunk_pieces.items.len - 1) |index| {
                const key = MergeKey{
                    .left = chunk_pieces.items[index],
                    .right = chunk_pieces.items[index + 1],
                };
                if (self.merge_table.get(key)) |merge| {
                    if (best_index == null or merge.rank < best_rank) {
                        best_index = index;
                        best_rank = merge.rank;
                        best_merged = merge.merged;
                    }
                }
            }

            const index = best_index orelse break;
            chunk_pieces.items[index] = best_merged;
            _ = chunk_pieces.orderedRemove(index + 1);
        }

        try pieces.appendSlice(allocator, chunk_pieces.items);
    }

    fn appendDecodedToken(self: Tokenizer, output: *std.ArrayList(u8), allocator: std.mem.Allocator, token_id: u32) !void {
        if (self.bos_token_id != null and token_id == self.bos_token_id.?) return;
        if (self.eos_token_id != null and token_id == self.eos_token_id.?) return;
        if (self.pad_token_id != null and token_id == self.pad_token_id.?) return;
        if (self.unk_token_id != null and token_id == self.unk_token_id.?) return;
        if (token_id >= self.tokens.len) return error.InvalidPrompt;

        return switch (self.mode) {
            .score_dp => self.appendDecodedScoreToken(output, allocator, token_id),
            .gpt2_bpe => self.appendDecodedGpt2Token(output, allocator, token_id),
        };
    }

    fn appendDecodedScoreToken(self: Tokenizer, output: *std.ArrayList(u8), allocator: std.mem.Allocator, token_id: u32) !void {
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

    fn appendDecodedGpt2Token(self: Tokenizer, output: *std.ArrayList(u8), allocator: std.mem.Allocator, token_id: u32) !void {
        const token = self.tokens[token_id];
        var view = try std.unicode.Utf8View.init(token);
        var it = view.iterator();
        while (it.nextCodepoint()) |codepoint| {
            const byte = gpt2DecodeByte(codepoint) orelse return error.InvalidPrompt;
            try output.append(allocator, byte);
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
    key_head_dimension: usize,
    value_head_dimension: usize,
    q_projection_size: usize,
    kv_projection_size: usize,
    kv_dimension: usize,
    rms_norm_eps: f32,
    rope_freq_base: f32,
    rope_scaling_factor: f32 = 0.0,
    sliding_window: usize = 0,
    attn_logit_softcapping: ?f32 = null,
    final_logit_softcapping: ?f32 = null,
    global_attention_interval: usize = 0,
    use_gelu_ffn: bool = false,
    embedding_scale: f32 = 1.0,
    rms_norm_weight_offset: f32 = 0.0,
    rope_style: RopeStyle,
    data_offset: usize,
    token_embd: TensorRef,
    output: TensorRef,
    output_norm: TensorRef,
    layers: []LayerRefs,
    is_qwen35_text: bool = false,
    partial_rotary_factor: f32 = 1.0,
    linear_num_key_heads: u32 = 16,
    linear_num_value_heads: u32 = 16,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    layer_types: []LayerType = &.{},

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        self.tokenizer.deinit(allocator);
        for (self.layers) |layer| {
            allocator.free(layer.attn_norm.name);
            if (layer.attn_q) |q| allocator.free(q.name);
            if (layer.attn_q_bias) |b| allocator.free(b.name);
            if (layer.attn_q_norm) |n| allocator.free(n.name);
            if (layer.attn_k) |k| allocator.free(k.name);
            if (layer.attn_k_bias) |b| allocator.free(b.name);
            if (layer.attn_k_norm) |n| allocator.free(n.name);
            if (layer.attn_v) |v| allocator.free(v.name);
            if (layer.attn_v_bias) |b| allocator.free(b.name);
            if (layer.attn_output) |o| allocator.free(o.name);
            allocator.free(layer.ffn_norm.name);
            allocator.free(layer.ffn_gate.name);
            allocator.free(layer.ffn_down.name);
            allocator.free(layer.ffn_up.name);
            if (layer.post_attention_norm) |n| allocator.free(n.name);
            if (layer.post_ffw_norm) |n| allocator.free(n.name);
            if (layer.linear_attn) |la| {
                allocator.free(la.in_proj_qkv.name);
                allocator.free(la.in_proj_z.name);
                allocator.free(la.in_proj_b.name);
                allocator.free(la.in_proj_a.name);
                allocator.free(la.conv1d.name);
                allocator.free(la.dt_bias.name);
                allocator.free(la.A_log.name);
                allocator.free(la.norm_weight.name);
                allocator.free(la.out_proj.name);
            }
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
        context_length: usize,
        dense_tensors: ?DenseTensorLookup,
    ) !ReusableSession {
        var session = try Session.init(
            allocator,
            model,
            backend,
            dense_tensors,
            context_length,
            context_length,
            null,
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
        self.session.pending_greedy_token = null;
    }
};

const Session = struct {
    model: *const Model,
    context_length: usize,
    backend: ?backend_api.MatVecBackend,
    dense_tensors: ?DenseTensorLookup,
    gpu_session: ?gpu.Session = null,
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
    pending_greedy_token: ?u32 = null,
    pending_shortlist: [gpu.max_shortlist_len]gpu.ShortlistEntry = undefined,
    pending_shortlist_len: usize = 0,
    scores: []f32,
    k_cache: []f32,
    v_cache: []f32,
    linear_conv_state: []f32,
    linear_recurrent_state: []f32,
    linear_qkv: []f32,
    linear_z: []f32,
    linear_a: []f32,
    linear_b: []f32,
    linear_g: []f32,
    linear_conv_tmp: []f32,
    moon_quant_calibrator: ?*moon_quant_calibration.Calibrator = null,
    matvec_workers: ?*MatVecWorkers = null,
    position: usize = 0,

    fn init(
        allocator: std.mem.Allocator,
        model: *const Model,
        backend: ?backend_api.MatVecBackend,
        dense_tensors: ?DenseTensorLookup,
        context_length: usize,
        token_capacity: usize,
        profiler: ?*metal_profile.Profiler,
        moon_quant_calibrator: ?*moon_quant_calibration.Calibrator,
    ) !Session {
        const linear_num_v_heads = model.linear_num_value_heads;
        const linear_value_dim_per_head = model.linear_value_head_dim;
        const linear_key_dim_per_head = model.linear_key_head_dim;
        const conv_state_per_layer = linear_num_v_heads * linear_value_dim_per_head * model.linear_conv_kernel_dim;
        const recurrent_state_per_layer = linear_num_v_heads * model.linear_key_head_dim * model.linear_value_head_dim;
        const linear_qkv_dim = 2 * model.linear_num_key_heads * linear_key_dim_per_head + linear_num_v_heads * linear_value_dim_per_head;
        const linear_z_dim = linear_num_v_heads * linear_value_dim_per_head;

        var session = Session{
            .model = model,
            .context_length = context_length,
            .backend = backend,
            .dense_tensors = dense_tensors,
            .token_buffer = try allocator.alloc(u32, token_capacity),
            .hidden = try allocator.alloc(f32, model.embedding_length),
            .normed = try allocator.alloc(f32, model.embedding_length),
            .q = try allocator.alloc(f32, model.q_projection_size),
            .k = try allocator.alloc(f32, model.kv_projection_size),
            .v = try allocator.alloc(f32, model.kv_projection_size),
            .attn_out = try allocator.alloc(f32, model.q_projection_size),
            .attn_tmp = try allocator.alloc(f32, model.embedding_length),
            .gate = try allocator.alloc(f32, model.feed_forward_length),
            .up = try allocator.alloc(f32, model.feed_forward_length),
            .logits = try allocator.alloc(f32, model.tokenizer.tokens.len),
            .scores = try allocator.alloc(f32, context_length),
            .k_cache = try allocator.alloc(f32, model.block_count * context_length * model.kv_dimension),
            .v_cache = try allocator.alloc(f32, model.block_count * context_length * model.kv_dimension),
            .linear_conv_state = try allocator.alloc(f32, model.block_count * conv_state_per_layer),
            .linear_recurrent_state = try allocator.alloc(f32, model.block_count * recurrent_state_per_layer),
            .linear_qkv = try allocator.alloc(f32, linear_qkv_dim),
            .linear_z = try allocator.alloc(f32, linear_z_dim),
            .linear_a = try allocator.alloc(f32, model.linear_num_key_heads),
            .linear_b = try allocator.alloc(f32, model.linear_num_key_heads),
            .linear_g = try allocator.alloc(f32, model.linear_num_key_heads),
            .linear_conv_tmp = try allocator.alloc(f32, linear_z_dim),
            .moon_quant_calibrator = moon_quant_calibrator,
        };
        errdefer session.deinit(allocator);

        if (backend) |selected_backend| {
            if (selected_backend.label == .metal) {
                const lookup = dense_tensors orelse return error.InvalidTensorMetadata;
                session.gpu_session = try gpu.Session.init(selected_backend, adaptDenseLookup(lookup), adaptModelDesc(model, context_length), profiler);
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
        allocator.free(self.linear_conv_state);
        allocator.free(self.linear_recurrent_state);
        allocator.free(self.linear_qkv);
        allocator.free(self.linear_z);
        allocator.free(self.linear_a);
        allocator.free(self.linear_b);
        allocator.free(self.linear_g);
        allocator.free(self.linear_conv_tmp);
        self.* = undefined;
    }

    fn runPrompt(
        self: *Session,
        prompt_tokens: []const u32,
        sampling_path: runtime_types.EffectiveSamplingPath,
        shortlist_len: usize,
    ) !void {
        if (prompt_tokens.len == 0) return error.EmptyPrompt;
        if (prompt_tokens.len > 1) {
            for (prompt_tokens[0 .. prompt_tokens.len - 1]) |token_id| {
                try self.stepNoOutput(token_id);
            }
        }

        const last_token = prompt_tokens[prompt_tokens.len - 1];
        switch (sampling_path) {
            .gpu_greedy_argmax => _ = try self.stepGreedy(last_token),
            .gpu_topk_sampler => unreachable,
            .gpu_shortlist_cpu_sampler => _ = try self.stepShortlist(last_token, shortlist_len),
            .cpu_logits => _ = try self.step(last_token),
        }
    }

    fn stepShortlist(self: *Session, token_id: u32, shortlist_len: usize) ![]const gpu.ShortlistEntry {
        if (self.gpu_session == null) {
            _ = try self.step(token_id);
            self.pending_shortlist_len = 0;
            return self.pending_shortlist[0..0];
        }
        try self.runTokenCore(token_id);
        const gpu_session = &self.gpu_session.?;
        const shortlist = try gpu_session.runOutputShortlist(
            adaptTensorDesc(self.model.output_norm),
            adaptTensorDesc(self.model.output),
            shortlist_len,
            self.pending_shortlist[0..shortlist_len],
        );
        self.pending_shortlist_len = shortlist.len;
        self.pending_greedy_token = null;
        self.finishToken(token_id);
        return shortlist;
    }

    fn stepGpuTopK(self: *Session, token_id: u32, top_k: usize, temperature: f32, random_uniform: f32) !u32 {
        if (self.gpu_session == null) {
            _ = try self.step(token_id);
            return argmax(self.logits);
        }
        try self.runTokenCore(token_id);
        const gpu_session = &self.gpu_session.?;
        const next_token = try gpu_session.runOutputSampleTopK(
            adaptTensorDesc(self.model.output_norm),
            adaptTensorDesc(self.model.output),
            top_k,
            temperature,
            random_uniform,
        );
        self.pending_greedy_token = next_token;
        self.pending_shortlist_len = 0;
        self.finishToken(token_id);
        return next_token;
    }

    fn stepGreedy(self: *Session, token_id: u32) !u32 {
        if (self.gpu_session == null) {
            _ = try self.step(token_id);
            return argmax(self.logits);
        }
        try self.runTokenCore(token_id);
        const gpu_session = &self.gpu_session.?;
        const next_token = try gpu_session.runOutputArgmax(adaptTensorDesc(self.model.output_norm), adaptTensorDesc(self.model.output));
        self.pending_greedy_token = next_token;
        self.finishToken(token_id);
        return next_token;
    }

    fn stepNoOutput(self: *Session, token_id: u32) !void {
        try self.runTokenCore(token_id);
        if (self.gpu_session) |*gpu_session| {
            try gpu_session.commitToken();
        }
        self.pending_greedy_token = null;
        self.pending_shortlist_len = 0;
        self.finishToken(token_id);
    }

    fn findDraftTokens(self: *Session, next_token: u32, max_draft: usize) []const u32 {
        if (self.position < 3) return &.{};

        const prev_token = self.token_buffer[self.position - 1];
        var i: usize = self.position - 3;
        while (true) {
            if (self.token_buffer[i] == prev_token and self.token_buffer[i + 1] == next_token) {
                const match_end = i + 2;
                const available = self.position - match_end;
                const draft_len = @min(max_draft, available);
                if (draft_len > 0) {
                    return self.token_buffer[match_end .. match_end + draft_len];
                }
            }
            if (i == 0) break;
            i -= 1;
        }
        return &.{};
    }

    fn verifyDraftTokensSequential(self: *Session, current_token: u32, draft_tokens: []const u32, out_accepted: []u32) !usize {
        var accepted_count: usize = 0;
        var next_input = current_token;

        for (draft_tokens) |draft| {
            const predicted = try self.stepGreedy(next_input);
            out_accepted[accepted_count] = predicted;
            accepted_count += 1;

            if (predicted != draft) {
                return accepted_count;
            }
            next_input = draft;
        }

        const bonus_predicted = try self.stepGreedy(next_input);
        out_accepted[accepted_count] = bonus_predicted;
        accepted_count += 1;

        return accepted_count;
    }

    fn verifyDraftTokensBatchGpu(self: *Session, current_token: u32, draft_tokens: []const u32, out_accepted: []u32) !usize {
        if (self.gpu_session == null) return 0;
        if (draft_tokens.len == 0) return 0;

        const gpu_session = &self.gpu_session.?;
        var layers: [64]gpu.LayerDesc = undefined;
        for (self.model.layers, 0..) |layer, i| {
            layers[i] = adaptLayerDesc(layer);
        }

        var all_drafts: [gpu.max_draft_len + 1]u32 = undefined;
        all_drafts[0] = current_token;
        for (draft_tokens, 0..) |dt, i| {
            all_drafts[i + 1] = dt;
        }
        const batch_count = draft_tokens.len + 1;

        var predicted: [gpu.max_draft_len + 1]u32 = undefined;
        const accepted_count = try gpu_session.runBatchSpeculativeDecode(
            layers[0..self.model.layers.len],
            all_drafts[0..batch_count],
            self.position,
            &predicted,
        );

        for (0..accepted_count) |i| {
            out_accepted[i] = predicted[i];
        }

        if (accepted_count > 0) {
            for (0..accepted_count) |i| {
                self.token_buffer[self.position + i] = predicted[i];
            }
            self.position += accepted_count;
        }

        return accepted_count;
    }

    fn step(self: *Session, token_id: u32) ![]const f32 {
        try self.runTokenCore(token_id);
        self.pending_greedy_token = null;
        self.pending_shortlist_len = 0;
        if (self.gpu_session) |*gpu_session| {
            try gpu_session.runOutput(adaptTensorDesc(self.model.output_norm), adaptTensorDesc(self.model.output), self.logits);
        } else {
            try rmsNorm(self.normed, self.hidden, self.model, self.model.output_norm);
            try self.recordCalibration(self.model.output, .output, self.normed);
            try self.matVec(self.logits, self.model.output, self.normed);
        }
        if (self.model.final_logit_softcapping) |cap| applySoftcapInPlace(self.logits, cap);

        self.finishToken(token_id);
        return self.logits;
    }

    fn runTokenCore(self: *Session, token_id: u32) !void {
        if (self.position >= self.context_length) return error.ContextOverflow;
        try embeddingLookup(self.hidden, self.model, self.model.token_embd, token_id);
        scaleInPlace(self.hidden, self.model.embedding_scale);

        for (self.model.layers, 0..) |layer, layer_index| {
            const is_linear_attn = layer.linear_attn != null;

            if (self.gpu_session) |*gpu_session| {
                if (layer_index == 0) try gpu_session.beginToken(self.hidden);
                try gpu_session.runAttentionBlock(adaptLayerDesc(layer), layer_index, self.position);
            } else if (is_linear_attn) {
                try rmsNorm(self.normed, self.hidden, self.model, layer.attn_norm);
                try self.computeLinearAttention(layer_index, layer.linear_attn.?);
                addInPlace(self.hidden, self.attn_tmp);
            } else {
                try rmsNorm(self.normed, self.hidden, self.model, layer.attn_norm);
                if (layer.attn_q) |q| {
                    try self.recordCalibration(q, .attn_q, self.normed);
                    try self.matVec(self.q, q, self.normed);
                }
                if (layer.attn_q_bias) |b| try self.addBiasCpu(self.q, b);
                if (layer.attn_q_norm) |n| try self.rmsNormPerHead(self.q, self.q, n, self.model.head_count, self.model.key_head_dimension);

                if (layer.attn_k) |k| {
                    try self.recordCalibration(k, .attn_k, self.normed);
                    try self.matVec(self.k, k, self.normed);
                }
                if (layer.attn_k_bias) |b| try self.addBiasCpu(self.k, b);
                if (layer.attn_k_norm) |n| try self.rmsNormPerHead(self.k, self.k, n, self.model.head_count_kv, self.model.key_head_dimension);

                if (layer.attn_v) |v| {
                    try self.recordCalibration(v, .attn_v, self.normed);
                    try self.matVec(self.v, v, self.normed);
                }
                if (layer.attn_v_bias) |b| try self.addBiasCpu(self.v, b);

                const rotary_dim = @as(usize, @intFromFloat(@as(f32, @floatFromInt(self.model.rope_dimension_count)) * self.model.partial_rotary_factor));
                applyRoPE(self.q, self.model.head_count, rotary_dim, self.model.key_head_dimension, self.position, self.model.rope_freq_base, self.model.rope_style);
                applyRoPE(self.k, self.model.head_count_kv, rotary_dim, self.model.key_head_dimension, self.position, self.model.rope_freq_base, self.model.rope_style);
                self.storeKv(layer_index);
                self.computeAttention(layer_index);
                if (layer.attn_output) |o| {
                    try self.recordCalibration(o, .attn_output, self.attn_out);
                    try self.matVec(self.attn_tmp, o, self.attn_out);
                }
                if (layer.post_attention_norm) |n| try rmsNorm(self.attn_tmp, self.attn_tmp, self.model, n);
            }

            if (self.gpu_session) |*gpu_session| {
                try gpu_session.runFfnBlock(adaptLayerDesc(layer));
            } else {
                try rmsNorm(self.normed, self.hidden, self.model, layer.ffn_norm);
                try self.recordCalibration(layer.ffn_gate, .ffn_gate, self.normed);
                try self.recordCalibration(layer.ffn_up, .ffn_up, self.normed);
                try self.matVec(self.gate, layer.ffn_gate, self.normed);
                try self.matVec(self.up, layer.ffn_up, self.normed);
                if (self.model.use_gelu_ffn) {
                    geluInPlace(self.gate);
                } else {
                    siluInPlace(self.gate);
                }
                for (self.gate, self.up) |*gate, up| {
                    gate.* *= up;
                }
                try self.recordCalibration(layer.ffn_down, .ffn_down, self.gate);
                try self.matVec(self.attn_tmp, layer.ffn_down, self.gate);
                if (layer.post_ffw_norm) |n| try rmsNorm(self.attn_tmp, self.attn_tmp, self.model, n);
            }
            addInPlace(self.hidden, self.attn_tmp);
        }
    }

    fn finishToken(self: *Session, token_id: u32) void {
        self.token_buffer[self.position] = token_id;
        self.position += 1;
    }

    fn recordCalibration(self: *Session, tensor: TensorRef, role: moon_quant_calibration.Role, input: []const f32) !void {
        const calibrator = self.moon_quant_calibrator orelse return;
        try calibrator.observe(.{
            .name = tensor.name,
            .role = role,
            .rows = tensor.rowCount() catch return error.InvalidTensorMetadata,
            .cols = tensor.rowLen() catch return error.InvalidTensorMetadata,
            .current_format = switch (tensor.tensor_type) {
                .f32 => .f32_reference,
                .f16 => .f16_reference,
                .q8_0 => .q8_0,
                .q4_k => .legacy_q4_k,
                .q6_k => .legacy_q6_k,
                .q5_k => .q5_k_m,
                .q4_0, .q4_1, .q5_0, .q5_1, .q8_1, .q2_k, .q3_k, .q8_k, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => return error.UnsupportedTensorType,
                .i8, .i16, .i32, .i64, .f64, .bf16 => return error.UnsupportedTensorType,
                .tq1_0, .tq2_0, .mxfp4, .nvfp4 => return error.UnsupportedTensorType,
            },
            .values = input,
        });
    }

    fn storeKv(self: *Session, layer_index: usize) void {
        const stride = self.context_length * self.model.kv_dimension;
        const base = layer_index * stride + self.position * self.model.kv_dimension;
        @memcpy(self.k_cache[base .. base + self.model.kv_dimension], self.k);
        @memcpy(self.v_cache[base .. base + self.model.kv_dimension], self.v);
    }

    fn computeAttention(self: *Session, layer_index: usize) void {
        @memset(self.attn_out, 0);
        const head_dim = self.model.key_head_dimension;
        const kv_group_size = self.model.head_count / self.model.head_count_kv;
        const layer_base = layer_index * self.context_length * self.model.kv_dimension;
        const scale = @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const window_start = self.attentionWindowStart(layer_index);
        const token_count = self.position - window_start + 1;

        for (0..self.model.head_count) |head_index| {
            const q_head = self.q[head_index * head_dim ..][0..head_dim];
            const kv_head = head_index / kv_group_size;
            const kv_offset = kv_head * head_dim;

            for (window_start..self.position + 1) |token_index| {
                const k_base = layer_base + token_index * self.model.kv_dimension + kv_offset;
                const k_head = self.k_cache[k_base..][0..head_dim];
                self.scores[token_index - window_start] = dotSimd(q_head, k_head) * scale;
            }
            if (self.model.attn_logit_softcapping) |cap| applySoftcapInPlace(self.scores[0..token_count], cap);
            softmaxInPlace(self.scores[0..token_count]);

            const out_head = self.attn_out[head_index * head_dim ..][0..head_dim];
            for (window_start..self.position + 1) |token_index| {
                const weight = self.scores[token_index - window_start];
                const v_base = layer_base + token_index * self.model.kv_dimension + kv_offset;
                const v_head = self.v_cache[v_base..][0..head_dim];

                var i: usize = 0;
                while (i + simd_lane_count <= head_dim) : (i += simd_lane_count) {
                    const w_vec: F32x = @splat(weight);
                    const v_vec = loadInputVec(v_head, i);
                    const out_vec = loadInputVec(out_head, i);
                    const result = @mulAdd(F32x, w_vec, v_vec, out_vec);
                    @as(*[simd_lane_count]f32, @ptrCast(out_head[i..])).* = result;
                }
                while (i < head_dim) : (i += 1) {
                    out_head[i] = @mulAdd(f32, weight, v_head[i], out_head[i]);
                }
            }
        }
    }

    fn attentionWindowStart(self: *const Session, layer_index: usize) usize {
        if (!self.layerUsesSlidingWindow(layer_index)) return 0;
        const window = self.model.sliding_window;
        return (self.position + 1) - @min(self.position + 1, window);
    }

    fn layerUsesSlidingWindow(self: *const Session, layer_index: usize) bool {
        if (self.model.sliding_window == 0) return false;
        if (self.model.global_attention_interval == 0) return true;
        return ((layer_index + 1) % self.model.global_attention_interval) != 0;
    }

    fn computeLinearAttention(self: *Session, layer_index: usize, linear_attn: LinearAttnTensors) !void {
        const model = self.model;
        const num_key_heads = model.linear_num_key_heads;
        const num_value_heads = model.linear_num_value_heads;
        const key_head_dim = model.linear_key_head_dim;
        const value_head_dim = model.linear_value_head_dim;
        const kernel_dim = model.linear_conv_kernel_dim;
        const q_dim = num_key_heads * key_head_dim;
        const v_dim = num_value_heads * value_head_dim;

        const qkv_dim = q_dim + q_dim + v_dim;
        const num_heads = num_key_heads;

        try self.matVec(self.linear_qkv[0..qkv_dim], linear_attn.in_proj_qkv, self.normed);

        const q_out = self.linear_qkv[0..q_dim];
        const v_out = self.linear_qkv[q_dim + q_dim ..][0..v_dim];

        try self.matVec(self.linear_z[0..v_dim], linear_attn.in_proj_z, self.normed);

        try self.matVec(self.linear_a[0..num_heads], linear_attn.in_proj_a, q_out);

        const dt_bias_bytes = try tensorBytes(model, linear_attn.dt_bias);
        const a_log_bytes = try tensorBytes(model, linear_attn.A_log);

        for (0..num_heads) |h| {
            const a_val = self.linear_a[h];
            const dt_bias_val = switch (linear_attn.dt_bias.tensor_type) {
                .f32 => readF32(dt_bias_bytes[h * 4 ..][0..4]),
                .f16 => readF16AsF32(dt_bias_bytes[h * 2 ..][0..2]),
                else => return error.UnsupportedTensorType,
            };
            const a_log_val = switch (linear_attn.A_log.tensor_type) {
                .f32 => readF32(a_log_bytes[h * 4 ..][0..4]),
                .f16 => readF16AsF32(a_log_bytes[h * 2 ..][0..2]),
                else => return error.UnsupportedTensorType,
            };

            const softplus_val = if (a_val + dt_bias_val > 20.0) a_val + dt_bias_val else std.math.log1p(@exp(a_val + dt_bias_val));
            const g_val = -@exp(a_log_val) * softplus_val;
            self.linear_g[h] = g_val;
        }

        const conv_state_per_layer = num_value_heads * value_head_dim * kernel_dim;
        const recurrent_state_per_layer = num_value_heads * key_head_dim * value_head_dim;
        const conv_state_base = layer_index * conv_state_per_layer;
        const recurrent_state_base = layer_index * recurrent_state_per_layer;

        const conv1d_bytes = try tensorBytes(model, linear_attn.conv1d);
        const conv_out = self.linear_conv_tmp[0..v_dim];

        @memset(conv_out, 0);

        var head_idx: usize = 0;
        while (head_idx < num_value_heads) : (head_idx += 1) {
            const v_head_base = head_idx * value_head_dim;
            const v_head = v_out[v_head_base..][0..value_head_dim];
            const conv_head_base = head_idx * value_head_dim;
            const conv_head_out = conv_out[conv_head_base..][0..value_head_dim];

            var offset: usize = 0;
            while (offset < value_head_dim) : (offset += 1) {
                var kernel_idx: usize = 0;
                while (kernel_idx < kernel_dim - 1) : (kernel_idx += 1) {
                    const state_idx = kernel_idx * value_head_dim + offset;
                    const state_offset = conv_state_base + head_idx * value_head_dim * kernel_dim + state_idx;
                    const w_idx = kernel_idx * value_head_dim * value_head_dim + offset * value_head_dim + offset;
                    const w_val = switch (linear_attn.conv1d.tensor_type) {
                        .f32 => readF32(conv1d_bytes[w_idx * 4 ..][0..4]),
                        .f16 => readF16AsF32(conv1d_bytes[w_idx * 2 ..][0..2]),
                        else => return error.UnsupportedTensorType,
                    };
                    conv_head_out[offset] += self.linear_conv_state[state_offset] * w_val;
                }

                const w_idx = (kernel_dim - 1) * value_head_dim * value_head_dim + offset * value_head_dim + offset;
                const w_val = switch (linear_attn.conv1d.tensor_type) {
                    .f32 => readF32(conv1d_bytes[w_idx * 4 ..][0..4]),
                    .f16 => readF16AsF32(conv1d_bytes[w_idx * 2 ..][0..2]),
                    else => return error.UnsupportedTensorType,
                };
                conv_head_out[offset] += v_head[offset] * w_val;
            }
        }

        var shift_offset: usize = 0;
        while (shift_offset < value_head_dim * (kernel_dim - 1)) : (shift_offset += value_head_dim) {
            const src_offset = conv_state_base + shift_offset;
            const dst_offset = conv_state_base + value_head_dim + shift_offset;
            @memcpy(self.linear_conv_state[dst_offset..][0..value_head_dim], self.linear_conv_state[src_offset..][0..value_head_dim]);
        }
        @memcpy(self.linear_conv_state[conv_state_base..][0..value_head_dim], v_out[0..v_dim]);

        var g_idx: usize = 0;
        while (g_idx < num_heads) : (g_idx += 1) {
            const g_val = self.linear_g[g_idx];
            const recurrent_offset = recurrent_state_base + g_idx * key_head_dim * value_head_dim;
            const conv_head_offset = g_idx * value_head_dim;

            var j: usize = 0;
            while (j < key_head_dim * value_head_dim) : (j += 1) {
                const conv_idx = conv_head_offset + j;
                self.linear_recurrent_state[recurrent_offset + j] = g_val * self.linear_recurrent_state[recurrent_offset + j] + (1 - g_val) * conv_out[conv_idx];
            }
        }

        const norm_bytes = try tensorBytes(model, linear_attn.norm_weight);
        var norm_idx: usize = 0;
        while (norm_idx < v_dim) : (norm_idx += 1) {
            const norm_w = switch (linear_attn.norm_weight.tensor_type) {
                .f32 => readF32(norm_bytes[norm_idx * 4 ..][0..4]),
                .f16 => readF16AsF32(norm_bytes[norm_idx * 2 ..][0..2]),
                else => return error.UnsupportedTensorType,
            };
            self.linear_z[norm_idx] *= norm_w;
        }

        try self.matVec(self.attn_tmp, linear_attn.out_proj, self.linear_z[0..v_dim]);
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

    fn addBiasCpu(self: *Session, out: []f32, bias_tensor: TensorRef) !void {
        const bytes = try tensorBytes(self.model, bias_tensor);
        const cols = try bias_tensor.rowLen();
        if (out.len != cols) return error.InvalidTensorMetadata;

        if (bias_tensor.tensor_type == .f32) {
            for (0..cols) |i| out[i] += readF32(bytes[i * 4 ..][0..4]);
        } else if (bias_tensor.tensor_type == .f16) {
            for (0..cols) |i| out[i] += readF16AsF32(bytes[i * 2 ..][0..2]);
        } else {
            return error.UnsupportedTensorType;
        }
    }

    fn rmsNormPerHead(self: *Session, out: []f32, input: []const f32, tensor: TensorRef, head_count: usize, head_dim: usize) !void {
        const weights = try tensorBytes(self.model, tensor);
        if (tensor.tensor_type != .f32) return error.UnsupportedTensorType;
        if (try tensor.rowLen() != head_dim) return error.InvalidTensorMetadata;

        for (0..head_count) |head| {
            const h_in = input[head * head_dim ..][0..head_dim];
            const h_out = out[head * head_dim ..][0..head_dim];

            var mean_square: f32 = 0;
            for (h_in) |val| mean_square += val * val;
            mean_square /= @as(f32, @floatFromInt(head_dim));
            const scale = @as(f32, 1.0) / @sqrt(mean_square + self.model.rms_norm_eps);

            var i: usize = 0;
            while (i + simd_lane_count <= head_dim) : (i += simd_lane_count) {
                const input_vec = loadInputVec(h_in, i);
                const weight_vec = loadF32Vec(weights[i * 4 ..][0 .. simd_lane_count * 4]);
                const scale_vec: F32x = @splat(scale);
                const weight_offset_vec: F32x = @splat(self.model.rms_norm_weight_offset);
                const result = scale_vec * input_vec * (weight_vec + weight_offset_vec);
                @as(*[simd_lane_count]f32, @ptrCast(h_out[i..])).* = result;
            }
            while (i < head_dim) : (i += 1) {
                const weight = readF32(weights[i * 4 ..][0..4]) + self.model.rms_norm_weight_offset;
                h_out[i] = h_in[i] * scale * weight;
            }
        }
    }
};

fn adaptDenseLookup(lookup: DenseTensorLookup) gpu.DenseLookup {
    return .{
        .ctx = lookup.ctx,
        .get_dense_fn = lookup.get_by_offset_fn,
        .get_raw_fn = lookup.get_raw_by_offset_fn,
        .get_moon_quant_fn = lookup.get_moon_quant_by_offset_fn,
    };
}

fn adaptTensorDesc(tensor: TensorRef) gpu.TensorDesc {
    return .{
        .offset = tensor.offset,
        .rows = tensor.rowCount() catch unreachable,
        .cols = tensor.rowLen() catch unreachable,
        .tensor_type = @intFromEnum(tensor.tensor_type),
    };
}

fn adaptLinearAttnDesc(linear_attn: LinearAttnTensors) gpu.LinearAttnDesc {
    return .{
        .in_proj_qkv = adaptTensorDesc(linear_attn.in_proj_qkv),
        .in_proj_z = adaptTensorDesc(linear_attn.in_proj_z),
        .in_proj_b = adaptTensorDesc(linear_attn.in_proj_b),
        .in_proj_a = adaptTensorDesc(linear_attn.in_proj_a),
        .conv1d = adaptTensorDesc(linear_attn.conv1d),
        .dt_bias = adaptTensorDesc(linear_attn.dt_bias),
        .A_log = adaptTensorDesc(linear_attn.A_log),
        .norm_weight = adaptTensorDesc(linear_attn.norm_weight),
        .out_proj = adaptTensorDesc(linear_attn.out_proj),
    };
}

fn adaptLayerDesc(layer: LayerRefs) gpu.LayerDesc {
    return .{
        .attn_norm = adaptTensorDesc(layer.attn_norm),
        .attn_q = if (layer.attn_q) |q| adaptTensorDesc(q) else null,
        .attn_q_bias = if (layer.attn_q_bias) |b| adaptTensorDesc(b) else null,
        .attn_q_norm = if (layer.attn_q_norm) |n| adaptTensorDesc(n) else null,
        .attn_k = if (layer.attn_k) |k| adaptTensorDesc(k) else null,
        .attn_k_bias = if (layer.attn_k_bias) |b| adaptTensorDesc(b) else null,
        .attn_k_norm = if (layer.attn_k_norm) |n| adaptTensorDesc(n) else null,
        .attn_v = if (layer.attn_v) |v| adaptTensorDesc(v) else null,
        .attn_v_bias = if (layer.attn_v_bias) |b| adaptTensorDesc(b) else null,
        .attn_output = if (layer.attn_output) |o| adaptTensorDesc(o) else null,
        .ffn_norm = adaptTensorDesc(layer.ffn_norm),
        .ffn_gate = adaptTensorDesc(layer.ffn_gate),
        .ffn_down = adaptTensorDesc(layer.ffn_down),
        .ffn_up = adaptTensorDesc(layer.ffn_up),
        .post_attention_norm = if (layer.post_attention_norm) |n| adaptTensorDesc(n) else null,
        .post_ffw_norm = if (layer.post_ffw_norm) |n| adaptTensorDesc(n) else null,
        .linear_attn = if (layer.linear_attn) |la| adaptLinearAttnDesc(la) else null,
    };
}

fn adaptModelDesc(model: *const Model, context_length: usize) gpu.ModelDesc {
    return .{
        .embedding_length = model.embedding_length,
        .block_count = model.block_count,
        .context_length = context_length,
        .feed_forward_length = model.feed_forward_length,
        .rope_dimension_count = model.rope_dimension_count,
        .head_count = model.head_count,
        .head_count_kv = model.head_count_kv,
        .head_dimension = model.head_dimension,
        .key_head_dimension = model.key_head_dimension,
        .value_head_dimension = model.value_head_dimension,
        .q_projection_size = model.q_projection_size,
        .kv_projection_size = model.kv_projection_size,
        .kv_dimension = model.kv_dimension,
        .rope_freq_base = model.rope_freq_base,
        .rope_scaling_factor = model.rope_scaling_factor,
        .sliding_window = model.sliding_window,
        .attn_logit_softcapping = model.attn_logit_softcapping,
        .final_logit_softcapping = model.final_logit_softcapping,
        .global_attention_interval = model.global_attention_interval,
        .use_gelu_ffn = model.use_gelu_ffn,
        .embedding_scale = model.embedding_scale,
        .rms_norm_weight_offset = model.rms_norm_weight_offset,
        .vocab_size = model.tokenizer.tokens.len,
        .rms_norm_eps = model.rms_norm_eps,
        .token_embd_offset = model.token_embd.offset,
        .rope_style = @intFromEnum(model.rope_style),
        .linear_num_key_heads = model.linear_num_key_heads,
        .linear_num_value_heads = model.linear_num_value_heads,
        .linear_key_head_dim = model.linear_key_head_dim,
        .linear_value_head_dim = model.linear_value_head_dim,
        .linear_conv_kernel_dim = model.linear_conv_kernel_dim,
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
    const context_length = effectiveContextLength(model, options);

    const prompt_capacity = prompt.len * 4 + options.max_tokens + 8;
    const session_init_begin = std.time.nanoTimestamp();
    var session = try Session.init(
        allocator,
        model,
        backend,
        dense_tensors,
        context_length,
        @min(context_length, prompt_capacity),
        if (profiler.enabled) &profiler else null,
        null,
    );
    const session_init_ns = deltaNs(session_init_begin, std.time.nanoTimestamp());
    defer session.deinit(allocator);
    const startup_end = std.time.nanoTimestamp();
    spinner.stop();
    const backend_used: runtime_types.BackendUsed = if (backend == null) .cpu else .metal;
    const sampling_path = chooseSamplingPath(session.gpu_session != null, options);
    const shortlist_len = sampling.shortlistLenFor(options, session.logits.len);
    const gpu_top_k = if (options.top_k == 0) @min(gpu.max_shortlist_len, session.logits.len) else @min(options.top_k, session.logits.len);

    const prompt_begin = std.time.nanoTimestamp();
    const prompt_token_count = try model.tokenizer.encodeInto(allocator, prompt, session.token_buffer);
    switch (sampling_path) {
        .gpu_topk_sampler => {
            if (prompt_token_count == 0) return error.EmptyPrompt;
            if (prompt_token_count > 1) {
                for (session.token_buffer[0 .. prompt_token_count - 1]) |token_id| {
                    try session.stepNoOutput(token_id);
                }
            }
            _ = try session.stepGpuTopK(session.token_buffer[prompt_token_count - 1], gpu_top_k, options.temperature, random.float(f32));
        },
        else => try session.runPrompt(session.token_buffer[0..prompt_token_count], sampling_path, shortlist_len),
    }
    const prompt_end = std.time.nanoTimestamp();

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);
    const shortlist_sampling = sampling_path == .gpu_shortlist_cpu_sampler;
    const candidate_capacity = if (shortlist_sampling) gpu.max_shortlist_len else if (sampling_path == .cpu_logits) session.logits.len else 0;
    const sample_candidates = try allocator.alloc(sampling.SampleCandidate, candidate_capacity);
    defer allocator.free(sample_candidates);
    var generated_token_count: usize = 0;
    var ttft_ns = deltaNs(startup_begin, prompt_end);
    var first_decode_step_ns: u64 = 0;
    const gpu_greedy = sampling_path == .gpu_greedy_argmax;
    const gpu_topk = sampling_path == .gpu_topk_sampler;
    const gpu_shortlist = sampling_path == .gpu_shortlist_cpu_sampler;
    const needs_sample_timing = profiler.enabled;
    var greedy_next_token: ?u32 = if (gpu_greedy) (session.pending_greedy_token orelse argmax(session.logits)) else null;
    const max_draft_len = 3;
    var accepted_tokens: [max_draft_len + 1]u32 = undefined;

    const decode_begin = std.time.nanoTimestamp();
    while (generated_token_count < options.max_tokens) : (generated_token_count += 1) {
        profiler.beginDecodeToken();
        const next_token = if (gpu_greedy or gpu_topk) blk: {
            break :blk greedy_next_token.?;
        } else if (gpu_shortlist) blk: {
            const sample_begin = if (needs_sample_timing) std.time.nanoTimestamp() else 0;
            const sampled = sampling.sampleShortlist(
                session.pending_shortlist[0..session.pending_shortlist_len],
                session.token_buffer[0..session.position],
                options,
                random,
                sample_candidates,
            );
            if (needs_sample_timing) profiler.record(.cpu_sampling, runtime_types.deltaNs(sample_begin, std.time.nanoTimestamp()));
            break :blk sampled;
        } else blk: {
            const sample_begin = if (needs_sample_timing) std.time.nanoTimestamp() else 0;
            const sampled = sampling.sampleToken(session.logits, session.token_buffer[0..session.position], options, random, sample_candidates);
            if (needs_sample_timing) profiler.record(.cpu_sampling, runtime_types.deltaNs(sample_begin, std.time.nanoTimestamp()));
            break :blk sampled;
        };

        if (model.tokenizer.eos_token_id != null and next_token == model.tokenizer.eos_token_id.?) {
            profiler.endDecodeToken();
            break;
        }

        const chunk_start = output.items.len;
        try model.tokenizer.appendDecodedToken(&output, allocator, next_token);
        if (stream_callback) |callback| {
            const chunk = output.items[chunk_start..];
            if (chunk.len > 0) callback(stream_ctx, chunk) catch |err| switch (err) {
                error.StopStreaming => {
                    profiler.endDecodeToken();
                    generated_token_count += 1;
                    break;
                },
                else => return err,
            };
        }

        if (generated_token_count == 0) {
            ttft_ns = deltaNs(startup_begin, std.time.nanoTimestamp());
        }

        const needs_step_timing = generated_token_count == 0;
        const step_begin = if (needs_step_timing) std.time.nanoTimestamp() else 0;
        if (gpu_greedy) {
            const draft_tokens = session.findDraftTokens(next_token, max_draft_len);
            if (draft_tokens.len > 0) {
                const accepted_count = try session.verifyDraftTokensSequential(next_token, draft_tokens, &accepted_tokens);
                var i: usize = 0;
                while (i < accepted_count - 1) : (i += 1) {
                    const t = accepted_tokens[i];
                    if (model.tokenizer.eos_token_id != null and t == model.tokenizer.eos_token_id.?) {
                        greedy_next_token = t;
                        break;
                    }
                    if (generated_token_count + 1 >= options.max_tokens) {
                        greedy_next_token = t;
                        break;
                    }
                    generated_token_count += 1;
                    const inner_chunk_start = output.items.len;
                    try model.tokenizer.appendDecodedToken(&output, allocator, t);
                    if (stream_callback) |inner_cb| {
                        const chunk = output.items[inner_chunk_start..];
                        if (chunk.len > 0) inner_cb(stream_ctx, chunk) catch |err| switch (err) {
                            error.StopStreaming => {
                                greedy_next_token = model.tokenizer.eos_token_id orelse 0;
                                break;
                            },
                            else => return err,
                        };
                    }
                }
                greedy_next_token = accepted_tokens[i];
            } else if (draft_tokens.len > 0) {
                const accepted_count = try session.verifyDraftTokensSequential(next_token, draft_tokens, &accepted_tokens);
                var i: usize = 0;
                while (i < accepted_count - 1) : (i += 1) {
                    const t = accepted_tokens[i];
                    if (model.tokenizer.eos_token_id != null and t == model.tokenizer.eos_token_id.?) {
                        greedy_next_token = t;
                        break;
                    }
                    if (generated_token_count + 1 >= options.max_tokens) {
                        greedy_next_token = t;
                        break;
                    }
                    generated_token_count += 1;
                    const inner_chunk_start = output.items.len;
                    try model.tokenizer.appendDecodedToken(&output, allocator, t);
                    if (stream_callback) |inner_cb| {
                        const chunk = output.items[inner_chunk_start..];
                        if (chunk.len > 0) inner_cb(stream_ctx, chunk) catch |err| switch (err) {
                            error.StopStreaming => {
                                greedy_next_token = model.tokenizer.eos_token_id orelse 0;
                                break;
                            },
                            else => return err,
                        };
                    }
                }
                greedy_next_token = accepted_tokens[i];
            } else {
                greedy_next_token = try session.stepGreedy(next_token);
            }
        } else if (gpu_topk) {
            greedy_next_token = try session.stepGpuTopK(next_token, gpu_top_k, options.temperature, random.float(f32));
        } else if (gpu_shortlist) {
            _ = try session.stepShortlist(next_token, shortlist_len);
        } else {
            _ = try session.step(next_token);
        }

        if (needs_step_timing) {
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
        .backend = backend_used,
        .sampling_strategy = options.sampling_strategy,
        .sampling_path = sampling_path,
        .readback_mode = runtime_types.readbackModeFor(backend_used, sampling_path),
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
    const context_length = effectiveContextLength(model, options);

    const prompt_tokens = try allocator.alloc(u32, context_length);
    defer allocator.free(prompt_tokens);

    const prompt_begin = std.time.nanoTimestamp();
    const prompt_token_count = try model.tokenizer.encodeInto(allocator, prompt, prompt_tokens);
    if (prompt_token_count == 0) return error.EmptyPrompt;

    var session = &reusable_session.session;
    if (session.model != model) return error.InvalidPrompt;
    if (session.context_length != context_length) return error.InvalidPrompt;

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
        const sampling_path = chooseSamplingPath(session.gpu_session != null, options);
        const shortlist_len = sampling.shortlistLenFor(options, session.logits.len);
        const gpu_top_k = if (options.top_k == 0) @min(gpu.max_shortlist_len, session.logits.len) else @min(options.top_k, session.logits.len);
        if (sampling_path == .gpu_topk_sampler) {
            const remaining = prompt_tokens[session.position..prompt_token_count];
            if (remaining.len > 1) {
                for (remaining[0 .. remaining.len - 1]) |token_id| {
                    try session.stepNoOutput(token_id);
                }
            }
            _ = try session.stepGpuTopK(remaining[remaining.len - 1], gpu_top_k, options.temperature, random.float(f32));
        } else {
            try session.runPrompt(prompt_tokens[session.position..prompt_token_count], sampling_path, shortlist_len);
        }
    }
    const prompt_end = std.time.nanoTimestamp();

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);
    const backend_used: runtime_types.BackendUsed = if (backend == null) .cpu else .metal;
    const sampling_path = chooseSamplingPath(session.gpu_session != null, options);
    const shortlist_sampling = sampling_path == .gpu_shortlist_cpu_sampler;
    const candidate_capacity = if (shortlist_sampling) gpu.max_shortlist_len else if (sampling_path == .cpu_logits) session.logits.len else 0;
    const sample_candidates = try allocator.alloc(sampling.SampleCandidate, candidate_capacity);
    defer allocator.free(sample_candidates);
    var generated_token_count: usize = 0;
    var ttft_ns = deltaNs(startup_begin, prompt_end);
    var first_decode_step_ns: u64 = 0;
    const gpu_greedy = sampling_path == .gpu_greedy_argmax;
    const gpu_topk = sampling_path == .gpu_topk_sampler;
    const gpu_shortlist = sampling_path == .gpu_shortlist_cpu_sampler;
    var greedy_next_token: ?u32 = if (gpu_greedy) (session.pending_greedy_token orelse argmax(session.logits)) else null;
    const shortlist_len = sampling.shortlistLenFor(options, session.logits.len);
    const gpu_top_k = if (options.top_k == 0) @min(gpu.max_shortlist_len, session.logits.len) else @min(options.top_k, session.logits.len);

    const max_draft_len = 3;
    var accepted_tokens: [max_draft_len + 1]u32 = undefined;

    const decode_begin = std.time.nanoTimestamp();
    while (generated_token_count < options.max_tokens) : (generated_token_count += 1) {
        profiler.beginDecodeToken();
        const next_token = if (gpu_greedy or gpu_topk) blk: {
            break :blk greedy_next_token.?;
        } else if (gpu_shortlist) blk: {
            const sample_begin = std.time.nanoTimestamp();
            const sampled = sampling.sampleShortlist(
                session.pending_shortlist[0..session.pending_shortlist_len],
                session.token_buffer[0..session.position],
                options,
                random,
                sample_candidates,
            );
            profiler.record(.cpu_sampling, runtime_types.deltaNs(sample_begin, std.time.nanoTimestamp()));
            break :blk sampled;
        } else blk: {
            const sample_begin = std.time.nanoTimestamp();
            const sampled = sampling.sampleToken(session.logits, session.token_buffer[0..session.position], options, random, sample_candidates);
            profiler.record(.cpu_sampling, runtime_types.deltaNs(sample_begin, std.time.nanoTimestamp()));
            break :blk sampled;
        };

        if (model.tokenizer.eos_token_id != null and next_token == model.tokenizer.eos_token_id.?) {
            profiler.endDecodeToken();
            break;
        }

        const chunk_start = output.items.len;
        try model.tokenizer.appendDecodedToken(&output, allocator, next_token);
        if (stream_callback) |callback| {
            const chunk = output.items[chunk_start..];
            if (chunk.len > 0) callback(stream_ctx, chunk) catch |err| switch (err) {
                error.StopStreaming => {
                    profiler.endDecodeToken();
                    generated_token_count += 1;
                    break;
                },
                else => return err,
            };
        }

        if (generated_token_count == 0) {
            ttft_ns = deltaNs(startup_begin, std.time.nanoTimestamp());
        }

        const step_begin = std.time.nanoTimestamp();
        if (gpu_greedy) {
            const draft_tokens = session.findDraftTokens(next_token, max_draft_len);
            if (draft_tokens.len > 0) {
                const accepted_count = try session.verifyDraftTokensSequential(next_token, draft_tokens, &accepted_tokens);
                var i: usize = 0;
                while (i < accepted_count - 1) : (i += 1) {
                    const t = accepted_tokens[i];
                    if (model.tokenizer.eos_token_id != null and t == model.tokenizer.eos_token_id.?) {
                        greedy_next_token = t;
                        break;
                    }
                    if (generated_token_count + 1 >= options.max_tokens) {
                        greedy_next_token = t;
                        break;
                    }
                    generated_token_count += 1;
                    const inner_chunk_start = output.items.len;
                    try model.tokenizer.appendDecodedToken(&output, allocator, t);
                    if (stream_callback) |inner_cb| {
                        const chunk = output.items[inner_chunk_start..];
                        if (chunk.len > 0) inner_cb(stream_ctx, chunk) catch |err| switch (err) {
                            error.StopStreaming => {
                                greedy_next_token = model.tokenizer.eos_token_id orelse 0;
                                break;
                            },
                            else => return err,
                        };
                    }
                }
                greedy_next_token = accepted_tokens[i];
            } else {
                greedy_next_token = try session.stepGreedy(next_token);
            }
        } else if (gpu_topk) {
            greedy_next_token = try session.stepGpuTopK(next_token, gpu_top_k, options.temperature, random.float(f32));
        } else if (gpu_shortlist) {
            _ = try session.stepShortlist(next_token, shortlist_len);
        } else {
            _ = try session.step(next_token);
        }

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
        .backend = backend_used,
        .sampling_strategy = options.sampling_strategy,
        .sampling_path = sampling_path,
        .readback_mode = runtime_types.readbackModeFor(backend_used, sampling_path),
        .startup_breakdown = .{
            .session_init_ns = 0,
            .first_decode_step_ns = first_decode_step_ns,
        },
        .metal_profile_summary = profile_summary,
    };
}

fn effectiveContextLength(model: *const Model, options: runtime_types.GenerationOptions) usize {
    return @min(model.context_length, options.context_length);
}

pub const CalibrationSession = struct {
    session: Session,

    pub fn deinit(self: *CalibrationSession, allocator: std.mem.Allocator) void {
        self.session.deinit(allocator);
        self.* = undefined;
    }

    pub fn tokenBuffer(self: *CalibrationSession) []u32 {
        return self.session.token_buffer;
    }

    pub fn runPrompt(self: *CalibrationSession, prompt_tokens: []const u32) !void {
        try self.session.runPrompt(prompt_tokens, .cpu_logits, 0);
    }
};

pub fn initCalibrationSession(
    allocator: std.mem.Allocator,
    model: *const Model,
    token_capacity: usize,
    calibrator: *moon_quant_calibration.Calibrator,
) !CalibrationSession {
    var session = try Session.init(
        allocator,
        model,
        null,
        null,
        @min(model.context_length, token_capacity),
        token_capacity,
        null,
        calibrator,
    );
    errdefer session.deinit(allocator);
    return .{ .session = session };
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
    const is_mistral = std.mem.startsWith(u8, architecture, "mistral");
    const is_qwen = std.mem.startsWith(u8, architecture, "qwen");
    const is_llama = std.mem.eql(u8, architecture, "llama") or std.mem.startsWith(u8, architecture, "llama");
    const is_qwen35_text = std.mem.eql(u8, architecture, "qwen3_5_text") or std.mem.eql(u8, architecture, "qwen35");
    const is_gemma = std.mem.eql(u8, architecture, "gemma") or std.mem.eql(u8, architecture, "gemma2") or std.mem.eql(u8, architecture, "gemma3");
    if (!is_llama and !is_qwen and !is_mistral and !is_qwen35_text and !is_gemma) return error.UnsupportedArchitecture;
    const rope_style: RopeStyle = if (is_qwen or is_mistral or is_qwen35_text or is_gemma) .neox else .interleaved;

    if (metadata.tokenizer_model) |tm| {
        if (!isSupportedTokenizerModel(tm)) return error.UnsupportedTokenizer;
    } else if (!std.mem.startsWith(u8, architecture, "qwen")) {
        return error.MissingRequiredMetadata;
    }

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

    const head_dimension = embedding_length / head_count;
    const key_head_dimension = metadata.attention_key_length orelse head_dimension;
    const value_head_dimension = metadata.attention_value_length orelse key_head_dimension;
    const rope_dimension_count = metadata.rope_dimension_count orelse key_head_dimension;
    const rms_norm_eps = metadata.rms_norm_eps orelse 1e-6;
    const rope_freq_base = metadata.rope_freq_base orelse 10000;
    const rope_scaling_factor = metadata.rope_scaling_factor orelse 0.0;
    const effective_freq_base = if (rope_scaling_factor > 0 and rope_freq_base > 0)
        rope_freq_base / rope_scaling_factor
    else
        rope_freq_base;
    if (head_count == 0 or head_count_kv == 0 or embedding_length % head_count != 0) return error.InvalidMetadataValue;
    if (key_head_dimension == 0 or value_head_dimension == 0 or rope_dimension_count > key_head_dimension) return error.InvalidMetadataValue;

    const token_embd = try takeTensor(allocator, &tensors, "token_embd.weight");
    const output = takeTensor(allocator, &tensors, "output.weight") catch |err| switch (err) {
        // Newer llama-family GGUFs can omit the output projection when embeddings are tied.
        error.MissingRequiredTensor => try cloneTensorRef(allocator, token_embd),
        else => return err,
    };
    const output_norm = try takeTensor(allocator, &tensors, "output_norm.weight");

    const partial_rotary_factor = metadata.partial_rotary_factor orelse 1.0;
    const linear_num_key_heads = metadata.linear_num_key_heads orelse 16;
    const linear_num_value_heads = metadata.linear_num_value_heads orelse 16;
    const linear_key_head_dim = metadata.linear_key_head_dim orelse 128;
    const linear_value_head_dim = metadata.linear_value_head_dim orelse 128;
    const linear_conv_kernel_dim = metadata.linear_conv_kernel_dim orelse 4;
    const sliding_window = metadata.sliding_window orelse 0;
    const global_attention_interval: usize = if (std.mem.eql(u8, architecture, "gemma2"))
        2
    else if (std.mem.eql(u8, architecture, "gemma3"))
        6
    else
        0;
    const use_gelu_ffn = std.mem.eql(u8, architecture, "gemma") or
        std.mem.eql(u8, architecture, "gemma2") or
        std.mem.eql(u8, architecture, "gemma3");
    const embedding_scale: f32 = if (std.mem.eql(u8, architecture, "gemma") or std.mem.eql(u8, architecture, "gemma2") or std.mem.eql(u8, architecture, "gemma3"))
        @sqrt(@as(f32, @floatFromInt(embedding_length)))
    else
        1.0;
    const rms_norm_weight_offset: f32 = 0.0;

    var layer_types: []LayerType = &.{};
    defer if (layer_types.len > 0) allocator.free(layer_types);

    if (is_qwen35_text and metadata.layer_types.items.len > 0) {
        layer_types = try allocator.alloc(LayerType, metadata.layer_types.items.len);
        for (metadata.layer_types.items, 0..) |lt, i| {
            layer_types[i] = if (std.mem.eql(u8, lt, "linear_attention")) .linear_attention else .full_attention;
        }
    }

    const layers = try allocator.alloc(LayerRefs, block_count);
    errdefer allocator.free(layers);
    for (0..block_count) |index| {
        layers[index] = .{
            .attn_norm = try takeLayerTensor(allocator, &tensors, index, "attn_norm.weight"),
            .attn_q = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_q.weight"),
            .attn_q_bias = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_q.bias"),
            .attn_q_norm = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_q_norm.weight"),
            .attn_k = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_k.weight"),
            .attn_k_bias = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_k.bias"),
            .attn_k_norm = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_k_norm.weight"),
            .attn_v = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_v.weight"),
            .attn_v_bias = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_v.bias"),
            .attn_output = try takeOptionalLayerTensor(allocator, &tensors, index, "attn_output.weight"),
            .ffn_norm = try takeLayerTensor(allocator, &tensors, index, "ffn_norm.weight"),
            .ffn_gate = try takeLayerTensor(allocator, &tensors, index, "ffn_gate.weight"),
            .ffn_down = try takeLayerTensor(allocator, &tensors, index, "ffn_down.weight"),
            .ffn_up = try takeLayerTensor(allocator, &tensors, index, "ffn_up.weight"),
            .post_attention_norm = try takeOptionalLayerTensor(allocator, &tensors, index, "post_attention_norm.weight"),
            .post_ffw_norm = try takeOptionalLayerTensor(allocator, &tensors, index, "post_ffw_norm.weight"),
            .linear_attn = null,
        };
    }

    const tokenizer = try buildTokenizer(allocator, &metadata);
    const q_projection_size = head_count * key_head_dimension;
    const kv_projection_size = head_count_kv * value_head_dimension;
    const kv_dimension = kv_projection_size;

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
        .key_head_dimension = key_head_dimension,
        .value_head_dimension = value_head_dimension,
        .q_projection_size = q_projection_size,
        .kv_projection_size = kv_projection_size,
        .kv_dimension = kv_dimension,
        .rms_norm_eps = rms_norm_eps,
        .rope_freq_base = effective_freq_base,
        .rope_scaling_factor = rope_scaling_factor,
        .sliding_window = sliding_window,
        .attn_logit_softcapping = metadata.attn_logit_softcapping,
        .final_logit_softcapping = metadata.final_logit_softcapping,
        .global_attention_interval = global_attention_interval,
        .use_gelu_ffn = use_gelu_ffn,
        .embedding_scale = embedding_scale,
        .rms_norm_weight_offset = rms_norm_weight_offset,
        .rope_style = rope_style,
        .data_offset = data_offset,
        .token_embd = token_embd,
        .output = output,
        .output_norm = output_norm,
        .layers = layers,
        .is_qwen35_text = is_qwen35_text,
        .partial_rotary_factor = partial_rotary_factor,
        .linear_num_key_heads = linear_num_key_heads,
        .linear_num_value_heads = linear_num_value_heads,
        .linear_key_head_dim = linear_key_head_dim,
        .linear_value_head_dim = linear_value_head_dim,
        .linear_conv_kernel_dim = linear_conv_kernel_dim,
        .layer_types = layer_types,
    };
}

pub fn countPromptTokens(allocator: std.mem.Allocator, model: *const Model, prompt: []const u8) !usize {
    const token_buf = try allocator.alloc(u32, model.context_length);
    defer allocator.free(token_buf);
    return model.tokenizer.encodeInto(allocator, prompt, token_buf);
}

fn isSupportedTokenizerModel(tokenizer_model: []const u8) bool {
    return std.mem.eql(u8, tokenizer_model, "llama") or
        std.mem.eql(u8, tokenizer_model, "gpt2") or
        std.mem.startsWith(u8, tokenizer_model, "qwen");
}

pub fn encodePromptInto(allocator: std.mem.Allocator, model: *const Model, prompt: []const u8, out: []u32) !usize {
    return model.tokenizer.encodeInto(allocator, prompt, out);
}

fn buildTokenizer(allocator: std.mem.Allocator, metadata: *Metadata) !Tokenizer {
    const token_count = metadata.tokenizer_tokens.items.len;
    if (token_count == 0) return error.MissingRequiredMetadata;

    const is_gemma_tokenizer = if (metadata.architecture) |arch|
        std.mem.eql(u8, arch, "gemma") or std.mem.eql(u8, arch, "gemma2") or std.mem.eql(u8, arch, "gemma3")
    else
        false;

    if (is_gemma_tokenizer) {
        if (token_count == 0) {
            return error.MissingRequiredMetadata;
        }
        const tokens = try metadata.tokenizer_tokens.toOwnedSlice(allocator);
        metadata.tokenizer_tokens = .empty;
        const scores = if (metadata.tokenizer_scores.items.len > 0)
            try metadata.tokenizer_scores.toOwnedSlice(allocator)
        else
            try allocator.alloc(f32, 0);
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
        var special_tokens = std.ArrayList(u32).empty;
        for (token_types, 0..) |t, id| {
            if (t == @intFromEnum(TokenType.control) or t == @intFromEnum(TokenType.user_defined)) {
                try special_tokens.append(allocator, @intCast(id));
            }
        }
        return .{
            .mode = .score_dp,
            .tokens = tokens,
            .scores = scores,
            .token_types = token_types,
            .special_tokens = try special_tokens.toOwnedSlice(allocator),
            .byte_fallback = byte_fallback,
            .merge_table = std.AutoHashMap(Tokenizer.MergeKey, Tokenizer.MergeValue).init(allocator),
            .bos_token_id = metadata.bos_token_id,
            .eos_token_id = metadata.eos_token_id,
            .unk_token_id = metadata.unk_token_id,
            .pad_token_id = metadata.pad_token_id,
            .add_bos_token = metadata.add_bos_token orelse true,
            .add_eos_token = metadata.add_eos_token orelse false,
            .add_space_prefix = metadata.add_space_prefix orelse true,
            .prefer_longest_match = true,
        };
    }

    const is_bpe = if (metadata.tokenizer_model) |m|
        std.mem.eql(u8, m, "gpt2") or std.mem.startsWith(u8, m, "qwen")
    else if (metadata.architecture) |arch|
        std.mem.startsWith(u8, arch, "qwen")
    else
        false;

    if (is_bpe) {
        return buildGpt2Tokenizer(allocator, metadata, token_count);
    }

    if (!is_gemma_tokenizer and metadata.tokenizer_scores.items.len != token_count) return error.MissingRequiredMetadata;

    const tokens = if (is_gemma_tokenizer and metadata.tokenizer_tokens.items.len == 0)
        try allocator.alloc([]u8, 1)
    else
        try metadata.tokenizer_tokens.toOwnedSlice(allocator);
    metadata.tokenizer_tokens = .empty;

    const scores = if (is_gemma_tokenizer)
        try allocator.alloc(f32, 0)
    else
        try metadata.tokenizer_scores.toOwnedSlice(allocator);
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

    var special_tokens = std.ArrayList(u32).empty;
    for (token_types, 0..) |t, id| {
        if (t == @intFromEnum(TokenType.control) or t == @intFromEnum(TokenType.user_defined)) {
            try special_tokens.append(allocator, @intCast(id));
        }
    }

    return .{
        .mode = .score_dp,
        .tokens = tokens,
        .scores = scores,
        .token_types = token_types,
        .special_tokens = try special_tokens.toOwnedSlice(allocator),
        .byte_fallback = byte_fallback,
        .merge_table = std.AutoHashMap(Tokenizer.MergeKey, Tokenizer.MergeValue).init(allocator),
        .bos_token_id = metadata.bos_token_id,
        .eos_token_id = metadata.eos_token_id,
        .unk_token_id = metadata.unk_token_id,
        .pad_token_id = metadata.pad_token_id,
        .add_bos_token = metadata.add_bos_token orelse true,
        .add_eos_token = metadata.add_eos_token orelse false,
        .add_space_prefix = metadata.add_space_prefix orelse true,
        .prefer_longest_match = false,
    };
}

fn buildGpt2Tokenizer(allocator: std.mem.Allocator, metadata: *Metadata, token_count: usize) !Tokenizer {
    if (metadata.tokenizer_merges.items.len == 0) return error.MissingRequiredMetadata;

    const tokens = try metadata.tokenizer_tokens.toOwnedSlice(allocator);
    metadata.tokenizer_tokens = .empty;

    const scores = try allocator.alloc(f32, 0);

    const token_types = if (metadata.tokenizer_types.items.len == token_count)
        try metadata.tokenizer_types.toOwnedSlice(allocator)
    else blk: {
        const fallback = try allocator.alloc(u32, token_count);
        @memset(fallback, 1);
        break :blk fallback;
    };
    metadata.tokenizer_types = .empty;

    var token_lookup = std.StringHashMap(u32).init(allocator);
    defer token_lookup.deinit();
    try token_lookup.ensureTotalCapacity(std.math.cast(u32, token_count) orelse return error.Overflow);
    for (tokens, 0..) |token, token_id| {
        try token_lookup.put(token, @intCast(token_id));
    }

    var byte_fallback = [_]?u32{null} ** 256;
    for (0..256) |byte_usize| {
        const byte: u8 = @intCast(byte_usize);
        var encoded: [4]u8 = undefined;
        const encoded_slice = gpt2EncodeByte(&encoded, byte);
        byte_fallback[byte] = token_lookup.get(encoded_slice);
    }

    var merge_table = std.AutoHashMap(Tokenizer.MergeKey, Tokenizer.MergeValue).init(allocator);
    errdefer merge_table.deinit();

    var merged_buffer = std.ArrayList(u8).empty;
    defer merged_buffer.deinit(allocator);

    for (metadata.tokenizer_merges.items, 0..) |merge, rank| {
        const split_index = std.mem.indexOfScalar(u8, merge, ' ') orelse continue;
        if (split_index == 0 or split_index + 1 >= merge.len) continue;

        const left = merge[0..split_index];
        const right = merge[split_index + 1 ..];
        const left_id = token_lookup.get(left) orelse continue;
        const right_id = token_lookup.get(right) orelse continue;

        merged_buffer.clearRetainingCapacity();
        try merged_buffer.ensureTotalCapacity(allocator, left.len + right.len);
        merged_buffer.appendSliceAssumeCapacity(left);
        merged_buffer.appendSliceAssumeCapacity(right);
        const merged_id = token_lookup.get(merged_buffer.items) orelse continue;

        try merge_table.put(.{
            .left = left_id,
            .right = right_id,
        }, .{
            .merged = merged_id,
            .rank = @intCast(rank),
        });
    }

    for (metadata.tokenizer_merges.items) |merge| allocator.free(merge);
    metadata.tokenizer_merges.deinit(allocator);
    metadata.tokenizer_merges = .empty;

    var special_tokens = std.ArrayList(u32).empty;
    for (token_types, 0..) |t, id| {
        if (t == @intFromEnum(TokenType.control) or t == @intFromEnum(TokenType.user_defined)) {
            try special_tokens.append(allocator, @intCast(id));
        }
    }

    return .{
        .mode = .gpt2_bpe,
        .tokens = tokens,
        .scores = scores,
        .token_types = token_types,
        .special_tokens = try special_tokens.toOwnedSlice(allocator),
        .byte_fallback = byte_fallback,
        .merge_table = merge_table,
        .bos_token_id = metadata.bos_token_id,
        .eos_token_id = metadata.eos_token_id,
        .unk_token_id = metadata.unk_token_id,
        .pad_token_id = metadata.pad_token_id,
        .add_bos_token = metadata.add_bos_token orelse false,
        .add_eos_token = metadata.add_eos_token orelse false,
        .add_space_prefix = metadata.add_space_prefix orelse true,
        .prefer_longest_match = false,
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
    if (std.mem.endsWith(u8, key, ".context_length")) {
        metadata.context_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".embedding_length")) {
        metadata.embedding_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".block_count")) {
        metadata.block_count = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".feed_forward_length")) {
        metadata.feed_forward_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".rope.dimension_count")) {
        metadata.rope_dimension_count = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".attention.head_count")) {
        metadata.head_count = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".attention.head_count_kv")) {
        metadata.head_count_kv = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".attention.key_length")) {
        metadata.attention_key_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".attention.value_length")) {
        metadata.attention_value_length = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".attention.layer_norm_rms_epsilon") or std.mem.endsWith(u8, key, ".attention.layer_norm_epsilon")) {
        metadata.rms_norm_eps = try readExpectedFloat(parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".rope.freq_base")) {
        metadata.rope_freq_base = try readExpectedFloat(parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".rope.scaling.type")) {
        metadata.rope_scaling_type = try readExpectedString(allocator, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".rope.scaling.factor")) {
        metadata.rope_scaling_factor = try readExpectedFloat(parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".attention.sliding_window")) {
        metadata.sliding_window = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".attn_logit_softcapping")) {
        metadata.attn_logit_softcapping = try readExpectedFloat(parser, value_type);
        return;
    }
    if (std.mem.endsWith(u8, key, ".final_logit_softcapping")) {
        metadata.final_logit_softcapping = try readExpectedFloat(parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "partial_rotary_factor")) {
        metadata.partial_rotary_factor = try readExpectedFloat(parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "linear_num_key_heads")) {
        metadata.linear_num_key_heads = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "linear_num_value_heads")) {
        metadata.linear_num_value_heads = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "linear_key_head_dim")) {
        metadata.linear_key_head_dim = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "linear_value_head_dim")) {
        metadata.linear_value_head_dim = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "linear_conv_kernel_dim")) {
        metadata.linear_conv_kernel_dim = try readExpectedUnsigned(u32, parser, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "layer_types")) {
        try readStringArray(allocator, parser, value_type, &metadata.layer_types);
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
        try readStringArray(allocator, parser, value_type, &metadata.tokenizer_tokens);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.merges")) {
        try readStringArray(allocator, parser, value_type, &metadata.tokenizer_merges);
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
    if (std.mem.eql(u8, key, "tokenizer.ggml.add_space_prefix")) {
        metadata.add_space_prefix = try readExpectedBool(parser, value_type);
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

fn cloneTensorRef(allocator: std.mem.Allocator, tensor: TensorRef) !TensorRef {
    return .{
        .name = try allocator.dupe(u8, tensor.name),
        .dims = tensor.dims,
        .n_dims = tensor.n_dims,
        .tensor_type = tensor.tensor_type,
        .offset = tensor.offset,
    };
}

fn takeLayerTensor(allocator: std.mem.Allocator, tensors: *std.StringHashMap(TensorRef), layer_index: usize, suffix: []const u8) !TensorRef {
    var buffer: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&buffer, "blk.{d}.{s}", .{ layer_index, suffix });
    return takeTensor(allocator, tensors, name);
}

fn takeOptionalLayerTensor(allocator: std.mem.Allocator, tensors: *std.StringHashMap(TensorRef), layer_index: usize, suffix: []const u8) !?TensorRef {
    var buffer: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&buffer, "blk.{d}.{s}", .{ layer_index, suffix });
    if (tensors.get(name)) |tensor| {
        _ = allocator;
        _ = tensors.remove(name);
        return tensor;
    }
    return null;
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
        .q4_0 => blk: {
            if (row_len % 32 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 32, 36);
        },
        .q4_1 => blk: {
            if (row_len % 32 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 32, 50);
        },
        .q5_0 => blk: {
            if (row_len % 32 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 32, 44);
        },
        .q5_1 => blk: {
            if (row_len % 32 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 32, 58);
        },
        .q8_0 => blk: {
            if (row_len % 32 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 32, 34);
        },
        .q8_1 => blk: {
            if (row_len % 32 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 32, 66);
        },
        .q2_k => blk: {
            if (row_len % 256 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 256, 80);
        },
        .q3_k => blk: {
            if (row_len % 256 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 256, 110);
        },
        .q4_k => blk: {
            if (row_len % 256 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 256, 144);
        },
        .q5_k => blk: {
            if (row_len % 256 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 256, 176);
        },
        .q6_k => blk: {
            if (row_len % 256 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 256, 210);
        },
        .q8_k => blk: {
            if (row_len % 256 != 0) return error.InvalidTensorMetadata;
            break :blk try std.math.mul(usize, row_len / 256, 256);
        },
        .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => return error.UnsupportedTensorType,
        .i8, .i16, .i32, .i64, .f64, .bf16 => return error.UnsupportedTensorType,
        .tq1_0, .tq2_0, .mxfp4, .nvfp4 => return error.UnsupportedTensorType,
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

    var i: usize = 0;
    while (i + simd_lane_count <= input.len) : (i += simd_lane_count) {
        const input_vec = loadInputVec(input, i);
        const weight_vec = loadF32Vec(weights[i * 4 ..][0 .. simd_lane_count * 4]);
        const scale_vec: F32x = @splat(scale);
        const weight_offset_vec: F32x = @splat(model.rms_norm_weight_offset);
        const result = scale_vec * input_vec * (weight_vec + weight_offset_vec);
        @as(*[simd_lane_count]f32, @ptrCast(out[i..])).* = result;
    }
    while (i < input.len) : (i += 1) {
        const weight = readF32(weights[i * 4 ..][0..4]) + model.rms_norm_weight_offset;
        out[i] = input[i] * scale * weight;
    }
}

pub fn dequantizeRow(out: []f32, tensor_type: TensorType, row: []const u8, row_len: usize) !void {
    switch (tensor_type) {
        .f32 => {
            var index: usize = 0;
            while (index + simd_lane_count <= row_len) : (index += simd_lane_count) {
                @as(*[simd_lane_count]f32, @ptrCast(out[index..])).* = loadF32Vec(row[index * 4 ..][0 .. simd_lane_count * 4]);
            }
            while (index < row_len) : (index += 1) {
                out[index] = readF32(row[index * 4 ..][0..4]);
            }
        },
        .f16 => {
            var index: usize = 0;
            while (index + simd_lane_count <= row_len) : (index += simd_lane_count) {
                @as(*[simd_lane_count]f32, @ptrCast(out[index..])).* = loadF16Vec(row[index * 2 ..][0 .. simd_lane_count * 2]);
            }
            while (index < row_len) : (index += 1) {
                out[index] = readF16AsF32(row[index * 2 ..][0..2]);
            }
        },
        .q8_0 => try dequantizeRowQ8_0(out, row, row_len),
        .q4_k => try dequantizeRowQ4K(out, row, row_len),
        .q6_k => try dequantizeRowQ6K(out, row, row_len),
        .q5_k => try dequantizeRowQ5K(out, row, row_len),
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_1, .q2_k, .q3_k, .q8_k, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => return error.UnsupportedTensorType,
        .i8, .i16, .i32, .i64, .f64, .bf16 => return error.UnsupportedTensorType,
        .tq1_0, .tq2_0, .mxfp4, .nvfp4 => return error.UnsupportedTensorType,
    }
}

fn dequantizeRowQ5K(out: []f32, row: []const u8, row_len: usize) !void {
    if (row_len % 256 != 0) return error.InvalidTensorMetadata;
    var out_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += 176) {
        const block = row[block_index .. block_index + 176];
        const d = readF16AsF32(block[0..2]);
        for (0..32) |index| {
            out[out_offset + index] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + index]))));
        }
        out_offset += 32;
    }
}

fn dotRow(tensor_type: TensorType, row: []const u8, row_len: usize, input: []const f32) !f32 {
    return switch (tensor_type) {
        .f32 => dotF32Row(row, input),
        .f16 => dotF16Row(row, input),
        .q8_0 => try dotQ8_0Row(row, row_len, input),
        .q4_k => try dotQ4KRow(row, row_len, input),
        .q6_k => try dotQ6KRow(row, row_len, input),
        .q5_k => try dotQ5KRow(row, row_len, input),
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_1, .q2_k, .q3_k, .q8_k, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => return error.UnsupportedTensorType,
        .i8, .i16, .i32, .i64, .f64, .bf16 => return error.UnsupportedTensorType,
        .tq1_0, .tq2_0, .mxfp4, .nvfp4 => return error.UnsupportedTensorType,
    };
}

fn dotQ5KRow(row: []const u8, row_len: usize, input: []const f32) !f32 {
    if (row_len % 256 != 0) return error.InvalidTensorMetadata;
    var sum: f32 = 0;
    var input_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += 176) {
        const block = row[block_index .. block_index + 176];
        const d = readF16AsF32(block[0..2]);
        for (0..32) |index| {
            sum = @mulAdd(f32, d * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + index])))), input[input_offset + index], sum);
        }
        input_offset += 32;
    }
    return sum;
}

fn dotRowAssumeValid(tensor_type: TensorType, row: []const u8, row_len: usize, input: []const f32) f32 {
    return switch (tensor_type) {
        .f32 => dotF32Row(row, input),
        .f16 => dotF16Row(row, input),
        .q8_0 => dotQ8_0Row(row, row_len, input) catch unreachable,
        .q4_k => dotQ4KRow(row, row_len, input) catch unreachable,
        .q6_k => dotQ6KRow(row, row_len, input) catch unreachable,
        .q5_k => dotQ5KRow(row, row_len, input) catch unreachable,
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_1, .q2_k, .q3_k, .q8_k, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => unreachable,
        .i8, .i16, .i32, .i64, .f64, .bf16 => unreachable,
        .tq1_0, .tq2_0, .mxfp4, .nvfp4 => unreachable,
    };
}

fn dequantizeRowQ8_0(out: []f32, row: []const u8, row_len: usize) !void {
    if (row_len % 32 != 0) return error.InvalidTensorMetadata;
    var out_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += 34) {
        const block = row[block_index .. block_index + 34];
        const d = readF16AsF32(block[0..2]);
        for (0..32) |index| {
            out[out_offset + index] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + index]))));
        }
        out_offset += 32;
    }
}

fn dotQ8_0Row(row: []const u8, row_len: usize, input: []const f32) !f32 {
    if (row_len % 32 != 0) return error.InvalidTensorMetadata;
    var sum: f32 = 0;
    var input_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += 34) {
        const block = row[block_index .. block_index + 34];
        const d = readF16AsF32(block[0..2]);
        for (0..32) |index| {
            sum = @mulAdd(f32, d * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + index])))), input[input_offset + index], sum);
        }
        input_offset += 32;
    }
    return sum;
}

fn dotF32Row(row: []const u8, input: []const f32) f32 {
    var acc = zeroSimd();
    var index: usize = 0;
    while (index + simd_lane_count <= input.len) : (index += simd_lane_count) {
        acc += loadF32Vec(row[index * 4 ..][0 .. simd_lane_count * 4]) * loadInputVec(input, index);
    }

    var sum = reduceVec(acc);
    while (index < input.len) : (index += 1) {
        sum = @mulAdd(f32, readF32(row[index * 4 ..][0..4]), input[index], sum);
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
        sum = @mulAdd(f32, readF16AsF32(row[index * 2 ..][0..2]), input[index], sum);
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

            sum = @mulAdd(f32, d1, low_q_dot, sum);
            sum = @mulAdd(f32, -m1, low_input_sum, sum);
            sum = @mulAdd(f32, d2, high_q_dot, sum);
            sum = @mulAdd(f32, -m2, high_input_sum, sum);
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

                sum = @mulAdd(f32, s0, q1_dot, sum);
                sum = @mulAdd(f32, s2, q2_dot, sum);
                sum = @mulAdd(f32, s4, q3_dot, sum);
                sum = @mulAdd(f32, s6, q4_dot, sum);
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

fn applyRoPE(values: []f32, head_count: usize, head_dim: usize, rope_dim: usize, position: usize, freq_base: f32, rope_style: RopeStyle) void {
    const n_rot = @min(rope_dim, head_dim);
    const pos_f32 = @as(f32, @floatFromInt(position));
    for (0..head_count) |head_index| {
        const head = values[head_index * head_dim ..][0..head_dim];
        if (rope_style == .interleaved) {
            var pair: usize = 0;
            while (pair + 1 < n_rot) : (pair += 2) {
                const exponent = @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(n_rot));
                const theta = pos_f32 / std.math.pow(f32, freq_base, exponent);
                const cos_theta = @cos(theta);
                const sin_theta = @sin(theta);
                const x0 = head[pair];
                const x1 = head[pair + 1];
                head[pair] = @mulAdd(f32, x0, cos_theta, -x1 * sin_theta);
                head[pair + 1] = @mulAdd(f32, x0, sin_theta, x1 * cos_theta);
            }
        } else {
            const half_rot = n_rot / 2;
            for (0..half_rot) |i| {
                const exponent = @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(n_rot));
                const theta = pos_f32 / std.math.pow(f32, freq_base, exponent);
                const cos_theta = @cos(theta);
                const sin_theta = @sin(theta);
                const x0 = head[i];
                const x1 = head[i + half_rot];
                head[i] = @mulAdd(f32, x0, cos_theta, -x1 * sin_theta);
                head[i + half_rot] = @mulAdd(f32, x0, sin_theta, x1 * cos_theta);
            }
        }
    }
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

fn addInPlace(dst: []f32, src: []const f32) void {
    var i: usize = 0;
    while (i + simd_lane_count <= dst.len) : (i += simd_lane_count) {
        const dst_vec = loadInputVec(dst, i);
        const src_vec = loadInputVec(src, i);
        const result = dst_vec + src_vec;
        @as(*[simd_lane_count]f32, @ptrCast(dst[i..])).* = result;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] += src[i];
    }
}

fn scaleInPlace(values: []f32, scale: f32) void {
    if (scale == 1.0) return;

    var i: usize = 0;
    const scale_vec: F32x = @splat(scale);
    while (i + simd_lane_count <= values.len) : (i += simd_lane_count) {
        const value_vec = loadInputVec(values, i);
        @as(*[simd_lane_count]f32, @ptrCast(values[i..])).* = value_vec * scale_vec;
    }
    while (i < values.len) : (i += 1) values[i] *= scale;
}

fn siluInPlace(values: []f32) void {
    var i: usize = 0;
    while (i + simd_lane_count <= values.len) : (i += simd_lane_count) {
        const x = loadInputVec(values, i);
        const neg_x: F32x = -x;
        const one: F32x = @splat(1.0);
        const result = x / (one + @exp(neg_x));
        @as(*[simd_lane_count]f32, @ptrCast(values[i..])).* = result;
    }
    while (i < values.len) : (i += 1) {
        values[i] = values[i] / (1 + @exp(-values[i]));
    }
}

fn geluInPlace(values: []f32) void {
    for (values) |*value| value.* = geluTanh(value.*);
}

fn geluTanh(value: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    return 0.5 * value * (1.0 + std.math.tanh(sqrt_2_over_pi * (value + 0.044715 * value * value * value)));
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
    return dotSimd(a, b);
}

fn dotSimd(a: []const f32, b: []const f32) f32 {
    var acc = zeroSimd();
    var index: usize = 0;
    while (index + simd_lane_count <= a.len) : (index += simd_lane_count) {
        acc += loadInputVec(a, index) * loadInputVec(b, index);
    }

    var sum = reduceVec(acc);
    while (index < a.len) : (index += 1) {
        sum = @mulAdd(f32, a[index], b[index], sum);
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

fn applySoftcapInPlace(values: []f32, cap: f32) void {
    if (!(cap > 0)) return;
    for (values) |*value| {
        value.* = std.math.tanh(value.* / cap) * cap;
    }
}

fn gpt2ByteToCodepoint(byte: u8) u21 {
    if ((byte >= '!' and byte <= '~') or
        (byte >= 0xA1 and byte <= 0xAC) or
        (byte >= 0xAE and byte <= 0xFF))
    {
        return byte;
    }

    var offset: u21 = 0;
    var candidate: u16 = 0;
    while (candidate < byte) : (candidate += 1) {
        const b: u8 = @intCast(candidate);
        if ((b >= '!' and b <= '~') or
            (b >= 0xA1 and b <= 0xAC) or
            (b >= 0xAE and b <= 0xFF))
        {
            continue;
        }
        offset += 1;
    }
    return 256 + offset;
}

fn gpt2DecodeByte(codepoint: u21) ?u8 {
    if ((codepoint >= '!' and codepoint <= '~') or
        (codepoint >= 0xA1 and codepoint <= 0xAC) or
        (codepoint >= 0xAE and codepoint <= 0xFF))
    {
        return @intCast(codepoint);
    }
    if (codepoint < 256) return null;

    const target_offset = codepoint - 256;
    var offset: u21 = 0;
    var candidate: u16 = 0;
    while (candidate < 256) : (candidate += 1) {
        const byte: u8 = @intCast(candidate);
        if ((byte >= '!' and byte <= '~') or
            (byte >= 0xA1 and byte <= 0xAC) or
            (byte >= 0xAE and byte <= 0xFF))
        {
            continue;
        }
        if (offset == target_offset) return byte;
        offset += 1;
    }
    return null;
}

fn gpt2EncodeByte(buffer: *[4]u8, byte: u8) []const u8 {
    const codepoint = gpt2ByteToCodepoint(byte);
    const len = std.unicode.utf8Encode(codepoint, buffer) catch unreachable;
    return buffer[0..len];
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
    const count = try parser.readInt(u64);
    try out.ensureTotalCapacity(allocator, std.math.cast(usize, count) orelse return error.Overflow);
    for (0..count) |_| {
        const value: f32 = switch (element_type) {
            .float32 => @as(f32, @bitCast(try parser.readInt(u32))),
            .float64 => @as(f32, @floatCast(@as(f64, @bitCast(try parser.readInt(u64))))),
            .float16 => @as(f32, @floatCast(@as(f16, @bitCast(try parser.readInt(u16))))),
            .int32 => @floatFromInt(try parser.readInt(i32)),
            .int16 => @floatFromInt(try parser.readInt(i16)),
            .int8 => @floatFromInt(try parser.readInt(i8)),
            .uint32 => @floatFromInt(try parser.readInt(u32)),
            .uint16 => @floatFromInt(try parser.readInt(u16)),
            .uint8 => @floatFromInt(try parser.readInt(u8)),
            else => return error.InvalidMetadataType,
        };
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
        .uint16, .int16, .float16 => try parser.skipBytes(2),
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

test "supported tokenizer model whitelist accepts llama 3 style gguf metadata" {
    try std.testing.expect(isSupportedTokenizerModel("llama"));
    try std.testing.expect(isSupportedTokenizerModel("gpt2"));
    try std.testing.expect(!isSupportedTokenizerModel("sentencepiece"));
}

test "gpt2 byte mapping round trips common bytes" {
    try std.testing.expectEqual(@as(?u8, 'A'), gpt2DecodeByte(gpt2ByteToCodepoint('A')));
    try std.testing.expectEqual(@as(?u8, ' '), gpt2DecodeByte(gpt2ByteToCodepoint(' ')));
    try std.testing.expectEqual(@as(?u8, '\n'), gpt2DecodeByte(gpt2ByteToCodepoint('\n')));
    try std.testing.expectEqual(@as(?u8, 0), gpt2DecodeByte(gpt2ByteToCodepoint(0)));
}

test "gpt2 tokenizer merges byte pieces with rank order" {
    const allocator = std.testing.allocator;

    const tokens = try allocator.alloc([]u8, 5);
    errdefer allocator.free(tokens);
    tokens[0] = try allocator.dupe(u8, "a");
    tokens[1] = try allocator.dupe(u8, "b");
    tokens[2] = try allocator.dupe(u8, "c");
    tokens[3] = try allocator.dupe(u8, "ab");
    tokens[4] = try allocator.dupe(u8, "abc");

    const scores = try allocator.alloc(f32, 0);
    const token_types = try allocator.alloc(u32, 5);
    @memset(token_types, 1);

    var byte_fallback = [_]?u32{null} ** 256;
    byte_fallback['a'] = 0;
    byte_fallback['b'] = 1;
    byte_fallback['c'] = 2;

    var merge_table = std.AutoHashMap(Tokenizer.MergeKey, Tokenizer.MergeValue).init(allocator);
    try merge_table.put(.{ .left = 0, .right = 1 }, .{ .merged = 3, .rank = 0 });
    try merge_table.put(.{ .left = 3, .right = 2 }, .{ .merged = 4, .rank = 1 });

    var tokenizer = Tokenizer{
        .mode = .gpt2_bpe,
        .tokens = tokens,
        .scores = scores,
        .token_types = token_types,
        .special_tokens = &.{},
        .byte_fallback = byte_fallback,
        .merge_table = merge_table,
        .bos_token_id = null,
        .eos_token_id = null,
        .unk_token_id = null,
        .pad_token_id = null,
        .add_bos_token = false,
        .add_eos_token = false,
    };
    defer tokenizer.deinit(allocator);

    var encoded: [3]u32 = undefined;
    const count = try tokenizer.encodeInto(allocator, "abc", &encoded);
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expectEqual(@as(u32, 4), encoded[0]);

    var decoded = std.ArrayList(u8).empty;
    defer decoded.deinit(allocator);
    try tokenizer.appendDecodedToken(&decoded, allocator, 4);
    try std.testing.expectEqualStrings("abc", decoded.items);
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
