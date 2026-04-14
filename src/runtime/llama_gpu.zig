const std = @import("std");
const backend_api = @import("backend.zig");
const gguf = @import("../gguf.zig");
const metal_backend = @import("metal_backend.zig");
const metal_profile = @import("metal_profile.zig");

pub const DenseLookup = struct {
    ctx: ?*const anyopaque,
    get_dense_fn: *const fn (?*const anyopaque, u64) ?[]const f32,
    get_raw_fn: *const fn (?*const anyopaque, u64) ?[]const u8,
    get_moon_quant_fn: *const fn (?*const anyopaque, u64) ?[]const u8,

    pub fn getDense(self: DenseLookup, offset: u64) ?[]const f32 {
        return self.get_dense_fn(self.ctx, offset);
    }

    pub fn getRaw(self: DenseLookup, offset: u64) ?[]const u8 {
        return self.get_raw_fn(self.ctx, offset);
    }

    pub fn getMoonQuant(self: DenseLookup, offset: u64) ?[]const u8 {
        return self.get_moon_quant_fn(self.ctx, offset);
    }
};

pub const TensorDesc = struct {
    offset: u64,
    rows: usize,
    cols: usize,
    tensor_type: u32,
};

pub const LayerDesc = struct {
    attn_norm: TensorDesc,
    attn_q: TensorDesc,
    attn_q_bias: ?TensorDesc = null,
    attn_q_norm: ?TensorDesc = null,
    attn_k: TensorDesc,
    attn_k_bias: ?TensorDesc = null,
    attn_k_norm: ?TensorDesc = null,
    attn_v: TensorDesc,
    attn_v_bias: ?TensorDesc = null,
    attn_output: TensorDesc,
    ffn_norm: TensorDesc,
    ffn_gate: TensorDesc,
    ffn_down: TensorDesc,
    ffn_up: TensorDesc,
};

pub const ModelDesc = struct {
    embedding_length: usize,
    block_count: usize,
    context_length: usize,
    feed_forward_length: usize,
    rope_dimension_count: usize,
    head_count: usize,
    head_count_kv: usize,
    head_dimension: usize,
    kv_dimension: usize,
    rope_freq_base: f32,
    vocab_size: usize,
    rms_norm_eps: f32,
    token_embd_offset: u64,
    rope_style: u32,
};

pub const GatedFfnLayerPolicy = struct {
    threshold: f32 = 0,
    active_block_ratio: f32 = 1.0,
    avg_active_blocks: f32 = 0,
    avg_total_blocks: f32 = 0,

    pub fn enabled(self: GatedFfnLayerPolicy) bool {
        return self.threshold > 0;
    }
};

pub const ShortlistEntry = struct {
    token_id: u32,
    logit: f32,
};

const FusedKFallbackReason = enum(u8) {
    success,
    tensor_type,
    k_bias,
    k_norm,
    rope_style,
    rope_dim,
    raw_missing,
};

const FusedQFallbackReason = enum(u8) {
    success,
    tensor_type,
    q_bias,
    q_norm,
    rope_style,
    rope_dim,
    matrix_missing,
};

const FusedQkFallbackReason = enum(u8) {
    success,
    tensor_type,
    shape,
    q_bias,
    q_norm,
    k_bias,
    k_norm,
    rope_style,
    rope_dim,
    matrix_missing,
};

const FusedKvFallbackReason = enum(u8) {
    success,
    tensor_type,
    shape,
    k_bias,
    k_norm,
    v_bias,
    rope_style,
    rope_dim,
    matrix_missing,
};

pub const max_shortlist_len: usize = 64;
pub const max_draft_len: usize = 4;
pub const max_prompt_prefill_batch_len: usize = 32;
const ggml_type_count: usize = 41;
const q4_k_tensor_type: u32 = @intFromEnum(gguf.TensorType.q4_k);
const q6_k_tensor_type: u32 = @intFromEnum(gguf.TensorType.q6_k);

const FusedKvKernel = enum {
    q4k_q4k,
    q4k_q6k,
};

pub const Session = struct {
    backend: backend_api.MatVecBackend,
    dense_lookup: DenseLookup,
    model: ModelDesc,
    hidden: metal_backend.BufferHandle,
    normed: metal_backend.BufferHandle,
    norm_scale: metal_backend.BufferHandle,
    q: metal_backend.BufferHandle,
    k: metal_backend.BufferHandle,
    v: metal_backend.BufferHandle,
    attn: metal_backend.BufferHandle,
    gate: metal_backend.BufferHandle,
    up: metal_backend.BufferHandle,
    tmp: metal_backend.BufferHandle,
    logits_readback: metal_backend.BufferHandle,
    sampled_token: metal_backend.BufferHandle,
    sampled_token_packed: metal_backend.BufferHandle,
    shortlist_entries: metal_backend.BufferHandle,
    k_cache: metal_backend.BufferHandle,
    v_cache: metal_backend.BufferHandle,
    batch_logits: metal_backend.BufferHandle,
    batch_tokens: metal_backend.BufferHandle,
    prompt_prefill_inputs: metal_backend.BufferHandle,
    prompt_prefill_normed: metal_backend.BufferHandle,
    prompt_prefill_q: metal_backend.BufferHandle,
    prompt_prefill_k: metal_backend.BufferHandle,
    prompt_prefill_v: metal_backend.BufferHandle,
    prompt_prefill_gate: metal_backend.BufferHandle,
    prompt_prefill_up: metal_backend.BufferHandle,
    ffn_block_mask: metal_backend.BufferHandle,
    ffn_gate_stats: metal_backend.BufferHandle,
    gated_ffn_policies: []const GatedFfnLayerPolicy,
    gated_ffn_enabled: bool,
    gated_ffn_total_blocks: u64 = 0,
    gated_ffn_active_blocks: u64 = 0,
    token_norm_scale_ready: bool = false,
    fused_q_attempts: u64 = 0,
    fused_q_successes: u64 = 0,
    fused_q_fallback_counts: [@typeInfo(FusedQFallbackReason).@"enum".fields.len]u64 =
        [_]u64{0} ** @typeInfo(FusedQFallbackReason).@"enum".fields.len,
    fused_qk_attempts: u64 = 0,
    fused_qk_successes: u64 = 0,
    fused_qk_fallback_counts: [@typeInfo(FusedQkFallbackReason).@"enum".fields.len]u64 =
        [_]u64{0} ** @typeInfo(FusedQkFallbackReason).@"enum".fields.len,
    fused_kv_attempts: u64 = 0,
    fused_kv_successes: u64 = 0,
    fused_kv_fallback_counts: [@typeInfo(FusedKvFallbackReason).@"enum".fields.len]u64 =
        [_]u64{0} ** @typeInfo(FusedKvFallbackReason).@"enum".fields.len,
    fused_kv_tensor_pair_counts: [ggml_type_count * ggml_type_count]u64 =
        [_]u64{0} ** (ggml_type_count * ggml_type_count),
    fused_k_attempts: u64 = 0,
    fused_k_successes: u64 = 0,
    fused_k_fallback_counts: [@typeInfo(FusedKFallbackReason).@"enum".fields.len]u64 =
        [_]u64{0} ** @typeInfo(FusedKFallbackReason).@"enum".fields.len,
    profiler: ?*metal_profile.Profiler = null,

    pub fn init(
        backend: backend_api.MatVecBackend,
        dense_lookup: DenseLookup,
        model: ModelDesc,
        gated_ffn_policies: []const GatedFfnLayerPolicy,
        gated_ffn_enabled: bool,
        profiler: ?*metal_profile.Profiler,
    ) !Session {
        const max_input = @max(model.embedding_length, model.feed_forward_length);
        const max_vec = @max(model.embedding_length, @max(model.feed_forward_length, model.vocab_size));
        const cache_len = model.block_count * model.context_length * model.kv_dimension;
        const hidden = try metal_backend.createScratchBuffer(backend, max_input);
        errdefer metal_backend.destroyBuffer(hidden);
        const normed = try metal_backend.createGpuScratchBuffer(backend, model.embedding_length);
        errdefer metal_backend.destroyBuffer(normed);
        const norm_scale = try metal_backend.createScratchBuffer(backend, 1);
        errdefer metal_backend.destroyBuffer(norm_scale);
        const q = try metal_backend.createGpuScratchBuffer(backend, model.embedding_length);
        errdefer metal_backend.destroyBuffer(q);
        const k = try metal_backend.createGpuScratchBuffer(backend, model.kv_dimension);
        errdefer metal_backend.destroyBuffer(k);
        const v = try metal_backend.createGpuScratchBuffer(backend, model.kv_dimension);
        errdefer metal_backend.destroyBuffer(v);
        const attn = try metal_backend.createGpuScratchBuffer(backend, model.embedding_length);
        errdefer metal_backend.destroyBuffer(attn);
        const gate = try metal_backend.createGpuScratchBuffer(backend, model.feed_forward_length);
        errdefer metal_backend.destroyBuffer(gate);
        const up = try metal_backend.createGpuScratchBuffer(backend, model.feed_forward_length);
        errdefer metal_backend.destroyBuffer(up);
        const tmp = try metal_backend.createGpuScratchBuffer(backend, max_vec);
        errdefer metal_backend.destroyBuffer(tmp);
        const logits_readback = try metal_backend.createScratchBuffer(backend, model.vocab_size);
        errdefer metal_backend.destroyBuffer(logits_readback);
        const sampled_token = try metal_backend.createScratchBuffer(backend, 1);
        errdefer metal_backend.destroyBuffer(sampled_token);
        const sampled_token_packed = try metal_backend.createByteScratchBuffer(backend, 3 * @sizeOf(u32));
        errdefer metal_backend.destroyBuffer(sampled_token_packed);
        const shortlist_entries = try metal_backend.createByteScratchBuffer(backend, max_shortlist_len * @sizeOf(metal_backend.ShortlistEntry));
        errdefer metal_backend.destroyBuffer(shortlist_entries);
        const k_cache = try metal_backend.createGpuByteScratchBuffer(backend, cache_len * @sizeOf(f16));
        errdefer metal_backend.destroyBuffer(k_cache);
        const v_cache = try metal_backend.createGpuByteScratchBuffer(backend, cache_len * @sizeOf(f16));
        errdefer metal_backend.destroyBuffer(v_cache);
        const batch_logits = try metal_backend.createGpuScratchBuffer(backend, max_draft_len * model.vocab_size);
        errdefer metal_backend.destroyBuffer(batch_logits);
        const batch_tokens = try metal_backend.createByteScratchBuffer(backend, max_draft_len * @sizeOf(u32));
        errdefer metal_backend.destroyBuffer(batch_tokens);
        const prompt_prefill_inputs = try metal_backend.createScratchBuffer(backend, max_prompt_prefill_batch_len * model.embedding_length);
        errdefer metal_backend.destroyBuffer(prompt_prefill_inputs);
        const prompt_prefill_normed = try metal_backend.createGpuScratchBuffer(backend, max_prompt_prefill_batch_len * model.embedding_length);
        errdefer metal_backend.destroyBuffer(prompt_prefill_normed);
        const prompt_prefill_q = try metal_backend.createGpuScratchBuffer(backend, max_prompt_prefill_batch_len * model.embedding_length);
        errdefer metal_backend.destroyBuffer(prompt_prefill_q);
        const prompt_prefill_k = try metal_backend.createGpuScratchBuffer(backend, max_prompt_prefill_batch_len * model.kv_dimension);
        errdefer metal_backend.destroyBuffer(prompt_prefill_k);
        const prompt_prefill_v = try metal_backend.createGpuScratchBuffer(backend, max_prompt_prefill_batch_len * model.kv_dimension);
        errdefer metal_backend.destroyBuffer(prompt_prefill_v);
        const prompt_prefill_gate = try metal_backend.createGpuScratchBuffer(backend, max_prompt_prefill_batch_len * model.feed_forward_length);
        errdefer metal_backend.destroyBuffer(prompt_prefill_gate);
        const prompt_prefill_up = try metal_backend.createGpuScratchBuffer(backend, max_prompt_prefill_batch_len * model.feed_forward_length);
        errdefer metal_backend.destroyBuffer(prompt_prefill_up);
        const ffn_block_count = std.math.divCeil(usize, model.feed_forward_length, 256) catch unreachable;
        const ffn_block_mask = try metal_backend.createGpuByteScratchBuffer(backend, ffn_block_count * @sizeOf(u32));
        errdefer metal_backend.destroyBuffer(ffn_block_mask);
        const ffn_gate_stats = try metal_backend.createByteScratchBuffer(backend, 2 * @sizeOf(u32));
        errdefer metal_backend.destroyBuffer(ffn_gate_stats);

        return .{
            .backend = backend,
            .dense_lookup = dense_lookup,
            .model = model,
            .hidden = hidden,
            .normed = normed,
            .norm_scale = norm_scale,
            .q = q,
            .k = k,
            .v = v,
            .attn = attn,
            .gate = gate,
            .up = up,
            .tmp = tmp,
            .logits_readback = logits_readback,
            .sampled_token = sampled_token,
            .sampled_token_packed = sampled_token_packed,
            .shortlist_entries = shortlist_entries,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .batch_logits = batch_logits,
            .batch_tokens = batch_tokens,
            .prompt_prefill_inputs = prompt_prefill_inputs,
            .prompt_prefill_normed = prompt_prefill_normed,
            .prompt_prefill_q = prompt_prefill_q,
            .prompt_prefill_k = prompt_prefill_k,
            .prompt_prefill_v = prompt_prefill_v,
            .prompt_prefill_gate = prompt_prefill_gate,
            .prompt_prefill_up = prompt_prefill_up,
            .ffn_block_mask = ffn_block_mask,
            .ffn_gate_stats = ffn_gate_stats,
            .gated_ffn_policies = gated_ffn_policies,
            .gated_ffn_enabled = gated_ffn_enabled,
            .profiler = profiler,
        };
    }

    pub fn deinit(self: *Session) void {
        metal_backend.destroyBuffer(self.hidden);
        metal_backend.destroyBuffer(self.normed);
        metal_backend.destroyBuffer(self.norm_scale);
        metal_backend.destroyBuffer(self.q);
        metal_backend.destroyBuffer(self.k);
        metal_backend.destroyBuffer(self.v);
        metal_backend.destroyBuffer(self.attn);
        metal_backend.destroyBuffer(self.gate);
        metal_backend.destroyBuffer(self.up);
        metal_backend.destroyBuffer(self.tmp);
        metal_backend.destroyBuffer(self.logits_readback);
        metal_backend.destroyBuffer(self.sampled_token);
        metal_backend.destroyBuffer(self.sampled_token_packed);
        metal_backend.destroyBuffer(self.shortlist_entries);
        metal_backend.destroyBuffer(self.k_cache);
        metal_backend.destroyBuffer(self.v_cache);
        metal_backend.destroyBuffer(self.batch_logits);
        metal_backend.destroyBuffer(self.batch_tokens);
        metal_backend.destroyBuffer(self.prompt_prefill_inputs);
        metal_backend.destroyBuffer(self.prompt_prefill_normed);
        metal_backend.destroyBuffer(self.prompt_prefill_q);
        metal_backend.destroyBuffer(self.prompt_prefill_k);
        metal_backend.destroyBuffer(self.prompt_prefill_v);
        metal_backend.destroyBuffer(self.prompt_prefill_gate);
        metal_backend.destroyBuffer(self.prompt_prefill_up);
        metal_backend.destroyBuffer(self.ffn_block_mask);
        metal_backend.destroyBuffer(self.ffn_gate_stats);
        self.* = undefined;
    }

    pub fn beginToken(self: *Session, input: []const f32) !void {
        try metal_backend.writeBufferF32(self.hidden, input);
        self.token_norm_scale_ready = false;
        try metal_backend.beginSequence(self.backend);
    }

    pub fn beginTokenWithNormScale(self: *Session, input: []const f32, norm_scale_value: f32) !void {
        var scale = [_]f32{norm_scale_value};
        try metal_backend.writeBufferF32(self.hidden, input);
        try metal_backend.writeBufferF32(self.norm_scale, &scale);
        try metal_backend.beginSequence(self.backend);
        self.token_norm_scale_ready = true;
    }

    pub fn stagePromptPrefillEmbedding(self: *Session, batch_index: usize, input: []const f32) !void {
        if (batch_index >= max_prompt_prefill_batch_len) return error.InvalidTensorMetadata;
        if (input.len != self.model.embedding_length) return error.InvalidTensorMetadata;
        try metal_backend.writeBufferF32At(
            self.prompt_prefill_inputs,
            batch_index * self.model.embedding_length,
            input,
        );
    }

    pub fn runPromptPrefillChunk(
        self: *Session,
        layers: []const LayerDesc,
        batch_count: usize,
        base_position: usize,
    ) !void {
        if (batch_count == 0 or batch_count > max_prompt_prefill_batch_len) return error.InvalidTensorMetadata;
        if (self.canRunTrueBatchPromptPrefill(layers)) {
            return self.runPromptPrefillChunkBatched(layers, batch_count, base_position);
        }
        return self.runPromptPrefillChunkSequential(layers, batch_count, base_position);
    }

    fn runPromptPrefillChunkSequential(
        self: *Session,
        layers: []const LayerDesc,
        batch_count: usize,
        base_position: usize,
    ) !void {
        try metal_backend.beginSequence(self.backend);

        const embedding_bytes = self.model.embedding_length * @sizeOf(f32);
        for (0..batch_count) |i| {
            const position = base_position + i;
            try metal_backend.copyBufferRegion(
                self.backend,
                self.prompt_prefill_inputs,
                i * embedding_bytes,
                self.hidden,
                0,
                embedding_bytes,
            );

            for (layers, 0..) |layer, layer_index| {
                try self.runAttentionBlock(layer, layer_index, position);
            }

            for (layers) |layer| {
                if (!try self.runFusedFfnFanout(
                    layer.ffn_gate,
                    layer.ffn_up,
                    self.hidden,
                    layer.ffn_norm,
                    self.norm_scale,
                    self.gate,
                    self.up,
                )) {
                    try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);
                    try self.runProjection(layer.ffn_gate, self.normed, self.gate);
                    try self.runProjection(layer.ffn_up, self.normed, self.up);
                }
                try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
                try self.runProjectionAdd(layer.ffn_down, self.gate, self.hidden);
            }
        }

        try metal_backend.commitSequence(self.backend);
    }

    fn canRunTrueBatchPromptPrefill(self: *Session, layers: []const LayerDesc) bool {
        if (self.model.rope_style != 0) return false;
        if (self.model.rope_dimension_count != self.model.head_dimension) return false;
        for (layers) |layer| {
            if (layer.attn_q_bias != null or layer.attn_q_norm != null) return false;
            if (layer.attn_k_bias != null or layer.attn_k_norm != null) return false;
            if (layer.attn_v_bias != null) return false;
            if (self.dense_lookup.getMoonQuant(layer.attn_q.offset) != null) return false;
            if (self.dense_lookup.getMoonQuant(layer.attn_k.offset) != null) return false;
            if (self.dense_lookup.getMoonQuant(layer.attn_v.offset) != null) return false;
            if (self.dense_lookup.getMoonQuant(layer.attn_output.offset) != null) return false;
            if (self.dense_lookup.getMoonQuant(layer.ffn_gate.offset) != null) return false;
            if (self.dense_lookup.getMoonQuant(layer.ffn_up.offset) != null) return false;
            if (self.dense_lookup.getMoonQuant(layer.ffn_down.offset) != null) return false;
            if (!supportsBatchProjectionTensor(layer.attn_q.tensor_type)) return false;
            if (!supportsBatchProjectionTensor(layer.attn_k.tensor_type)) return false;
            if (!supportsBatchProjectionTensor(layer.attn_v.tensor_type)) return false;
            if (!supportsBatchProjectionTensor(layer.attn_output.tensor_type)) return false;
            if (!supportsBatchProjectionTensor(layer.ffn_gate.tensor_type)) return false;
            if (!supportsBatchProjectionTensor(layer.ffn_up.tensor_type)) return false;
            if (!supportsBatchProjectionTensor(layer.ffn_down.tensor_type)) return false;
        }
        return true;
    }

    fn runPromptPrefillChunkBatched(
        self: *Session,
        layers: []const LayerDesc,
        batch_count: usize,
        base_position: usize,
    ) !void {
        try metal_backend.beginSequence(self.backend);

        for (layers, 0..) |layer, layer_index| {
            try self.runPromptAttentionLayerBatched(layer, layer_index, batch_count, base_position);
            try self.runPromptFfnLayerBatched(layer, batch_count);
        }

        try metal_backend.commitSequence(self.backend);
    }

    pub fn runAttentionBlock(
        self: *Session,
        layer: LayerDesc,
        layer_index: usize,
        position: usize,
    ) !void {
        var normed_ready = false;
        var norm_scale_ready = layer_index == 0 and self.token_norm_scale_ready;
        _ = &norm_scale_ready;
        if (layer_index == 0 and norm_scale_ready) self.token_norm_scale_ready = false;

        if (true) {
            if (true) {
                try self.runRmsNorm(layer.attn_norm, self.hidden, self.normed);
                normed_ready = true;
                try self.runProjection(layer.attn_q, self.normed, self.q);
                if (layer.attn_q_bias) |b| try self.runBiasAdd(b, self.q);
                if (layer.attn_q_norm) |n| {
                    if (self.model.rope_style != 0 and self.model.head_count != self.model.head_count_kv) {
                        try self.runRmsNormPerHead(n, self.q, self.attn, self.model.head_count, self.model.head_dimension);
                        try metal_backend.copyBufferRegion(self.backend, self.attn, 0, self.q, 0, self.model.embedding_length * @sizeOf(f32));
                    } else {
                        try self.runRmsNormPerHead(n, self.q, self.q, self.model.head_count, self.model.head_dimension);
                    }
                }

                const q_rope_start = std.time.nanoTimestamp();
                try metal_backend.applyRoPE(
                    self.backend,
                    self.q,
                    self.model.head_count,
                    self.model.head_dimension,
                    self.model.rope_dimension_count,
                    position,
                    self.model.rope_freq_base,
                    self.model.rope_style,
                );
                self.recordCategoryWithShape(.rope, q_rope_start, .{
                    .rows = self.model.head_count,
                    .cols = self.model.head_dimension,
                    .depth = self.model.rope_dimension_count,
                    .extra = position + 1,
                });
            }

            const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
            const kv_offset_elements = layer_base + position * self.model.kv_dimension;

            if (true) {
                if (!normed_ready) {
                    try self.runRmsNorm(layer.attn_norm, self.hidden, self.normed);
                    normed_ready = true;
                }

                const kv_k_start = std.time.nanoTimestamp();
                const fused_k_written = false;
                if (fused_k_written) {
                    try self.runProjection(layer.attn_v, self.normed, self.v);
                } else if (!try self.runFusedKvFanout(layer.attn_k, layer.attn_v, self.normed, self.k, self.v)) {
                    try self.runProjection(layer.attn_k, self.normed, self.k);
                    try self.runProjection(layer.attn_v, self.normed, self.v);
                }

                if (!fused_k_written) {
                    if (layer.attn_k_bias) |b| try self.runBiasAdd(b, self.k);
                    if (layer.attn_k_norm) |n| {
                        if (self.model.rope_style != 0 and self.model.head_count != self.model.head_count_kv) {
                            try self.runRmsNormPerHead(n, self.k, self.gate, self.model.head_count_kv, self.model.head_dimension);
                            try metal_backend.copyBufferRegion(self.backend, self.gate, 0, self.k, 0, self.model.kv_dimension * @sizeOf(f32));
                        } else {
                            try self.runRmsNormPerHead(n, self.k, self.k, self.model.head_count_kv, self.model.head_dimension);
                        }
                    }
                }
                if (layer.attn_v_bias) |b| try self.runBiasAdd(b, self.v);

                if (!fused_k_written) {
                    try metal_backend.packKvHalf(
                        self.backend,
                        self.k,
                        self.v,
                        self.k_cache,
                        self.v_cache,
                        kv_offset_elements,
                        self.model.head_count_kv,
                        self.model.head_dimension,
                        self.model.rope_dimension_count,
                        position,
                        self.model.rope_freq_base,
                        self.model.rope_style,
                    );
                    self.recordCategoryWithShape(.rope, kv_k_start, .{
                        .rows = self.model.head_count_kv,
                        .cols = self.model.head_dimension,
                        .depth = self.model.rope_dimension_count,
                        .extra = position + 1,
                    });
                } else {
                    try metal_backend.storeKvHalf(
                        self.backend,
                        self.v,
                        self.v_cache,
                        kv_offset_elements,
                        self.model.kv_dimension,
                    );
                }
                self.recordCategoryWithShape(.kv_writes, kv_k_start, .{
                    .rows = 1,
                    .cols = self.model.kv_dimension,
                    .depth = layer_index,
                    .extra = position + 1,
                });
            }
        }

        const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
        const attention_start = std.time.nanoTimestamp();
        try metal_backend.attentionFused(
            self.backend,
            self.q,
            self.k_cache,
            self.v_cache,
            self.attn,
            self.model.head_count,
            self.model.head_count_kv,
            self.model.head_dimension,
            self.model.kv_dimension,
            self.model.context_length,
            position,
            layer_base,
            @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(self.model.head_dimension))),
        );
        self.recordCategoryWithShape(.attention, attention_start, .{
            .rows = self.model.head_count,
            .cols = self.model.head_dimension,
            .depth = position + 1,
            .extra = self.model.head_count_kv,
        });
        try self.runProjectionAdd(layer.attn_output, self.attn, self.hidden);
    }

    pub fn runFfnBlock(self: *Session, layer: LayerDesc, layer_index: usize) !void {
        const fused_ffn_fanout = try self.runFusedFfnFanout(
            layer.ffn_gate,
            layer.ffn_up,
            self.hidden,
            layer.ffn_norm,
            self.norm_scale,
            self.gate,
            self.up,
        );
        if (!fused_ffn_fanout) {
            try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);
            try self.runProjection(layer.ffn_gate, self.normed, self.gate);
            try self.runProjection(layer.ffn_up, self.normed, self.up);
        }

        const tensor = layer.ffn_down;
        var handled_fused = false;
        const gated_policy = if (self.gated_ffn_enabled and layer_index < self.gated_ffn_policies.len)
            self.gated_ffn_policies[layer_index]
        else
            GatedFfnLayerPolicy{};

        if (tensor.tensor_type == 12) {
            if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                const start = std.time.nanoTimestamp();
                if (gated_policy.enabled()) {
                    const stats = try metal_backend.buildFfnGateBlockMask(self.backend, self.gate, self.up, self.ffn_block_mask, self.ffn_gate_stats, tensor.cols, gated_policy.threshold);
                    self.gated_ffn_total_blocks += stats.total_blocks;
                    self.gated_ffn_active_blocks += stats.active_blocks;
                    try metal_backend.runMatVecMoonQuantQ4KGatedSiluDownAddToBuffer(self.backend, matrix, self.gate, self.up, self.ffn_block_mask, self.hidden, tensor.rows, tensor.cols);
                } else {
                    try metal_backend.runMatVecMoonQuantQ4KSiluDownAddToBuffer(self.backend, matrix, self.gate, self.up, self.hidden, tensor.rows, tensor.cols);
                }
                const shape = metal_profile.ShapeDesc{
                    .rows = tensor.rows,
                    .cols = tensor.cols,
                    .tensor_type = tensor.tensor_type,
                    .extra = 2,
                    .layout_path = .moonq_q4_k,
                };
                self.recordCategoryWithShape(.projection_add_quantized, start, shape);
                self.recordQuantizedLayoutProjection(start, shape);
                handled_fused = true;
            } else {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                const start = std.time.nanoTimestamp();
                if (gated_policy.enabled()) {
                    const stats = try metal_backend.buildFfnGateBlockMask(self.backend, self.gate, self.up, self.ffn_block_mask, self.ffn_gate_stats, tensor.cols, gated_policy.threshold);
                    self.gated_ffn_total_blocks += stats.total_blocks;
                    self.gated_ffn_active_blocks += stats.active_blocks;
                    try metal_backend.runMatVecQ4KGatedSiluDownAddToBuffer(self.backend, matrix, self.gate, self.up, self.ffn_block_mask, self.hidden, tensor.rows, tensor.cols);
                } else {
                    try metal_backend.runMatVecQ4KSiluDownAddToBuffer(self.backend, matrix, self.gate, self.up, self.hidden, tensor.rows, tensor.cols);
                }
                const shape = metal_profile.ShapeDesc{ .rows = tensor.rows, .cols = tensor.cols, .tensor_type = tensor.tensor_type, .extra = 2 };
                self.recordCategoryWithShape(.projection_add_quantized, start, shape);
                handled_fused = true;
            }
        }

        if (!handled_fused) {
            const silu_start = std.time.nanoTimestamp();
            try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
            self.recordCategoryWithShape(.ffn_activation, silu_start, .{
                .rows = 1,
                .cols = self.model.feed_forward_length,
                .depth = 2,
            });
            try self.runProjectionAdd(layer.ffn_down, self.gate, self.hidden);
        }
    }

    pub fn runOutput(self: *Session, norm: TensorDesc, tensor: TensorDesc, out: []f32) !void {
        try self.runRmsNorm(norm, self.hidden, self.normed);
        try self.runProjection(tensor, self.normed, self.tmp);
        try metal_backend.copyBufferRegion(
            self.backend,
            self.tmp,
            0,
            self.logits_readback,
            0,
            out.len * @sizeOf(f32),
        );
        const shape = metal_profile.ShapeDesc{
            .rows = 1,
            .cols = out.len,
        };
        try self.commitOutputSequence(shape);
        const host_readback_start = std.time.nanoTimestamp();
        try metal_backend.readBufferF32(self.logits_readback, out);
        self.recordCategoryWithShape(.host_readback, host_readback_start, shape);
    }

    pub fn commitToken(self: *Session) !void {
        try metal_backend.commitSequence(self.backend);
    }

    pub fn renderGatedFfnSummary(self: *const Session, allocator: std.mem.Allocator) !?[]u8 {
        if (!self.gated_ffn_enabled or self.gated_ffn_total_blocks == 0) return null;
        const active_ratio = @as(f64, @floatFromInt(self.gated_ffn_active_blocks)) /
            @as(f64, @floatFromInt(self.gated_ffn_total_blocks));
        return try std.fmt.allocPrint(
            allocator,
            "gated_ffn.profile.active_blocks={d}\ngated_ffn.profile.total_blocks={d}\ngated_ffn.profile.active_block_ratio={d:.6}\ngated_ffn.profile.estimated_weight_skip_pct={d:.3}\n",
            .{
                self.gated_ffn_active_blocks,
                self.gated_ffn_total_blocks,
                active_ratio,
                (1.0 - active_ratio) * 100.0,
            },
        );
    }

    pub fn renderFusedKSummary(self: *const Session, allocator: std.mem.Allocator) !?[]u8 {
        if (self.fused_k_attempts == 0) return null;

        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);
        try writer.print(
            "fused_k.profile.attempts={d}\nfused_k.profile.successes={d}\nfused_k.profile.success_pct={d:.3}\n",
            .{
                self.fused_k_attempts,
                self.fused_k_successes,
                percent(self.fused_k_successes, self.fused_k_attempts),
            },
        );
        inline for ([_]FusedKFallbackReason{
            .success,
            .tensor_type,
            .k_bias,
            .k_norm,
            .rope_style,
            .rope_dim,
            .raw_missing,
        }) |reason| {
            try writer.print(
                "fused_k.profile.reason.{s}={d}\n",
                .{ @tagName(reason), self.fused_k_fallback_counts[@intFromEnum(reason)] },
            );
        }
        return try buffer.toOwnedSlice(allocator);
    }

    pub fn renderFusedKvSummary(self: *const Session, allocator: std.mem.Allocator) !?[]u8 {
        if (self.fused_kv_attempts == 0) return null;

        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);
        try writer.print(
            "fused_kv_half.profile.attempts={d}\nfused_kv_half.profile.successes={d}\nfused_kv_half.profile.success_pct={d:.3}\n",
            .{
                self.fused_kv_attempts,
                self.fused_kv_successes,
                percent(self.fused_kv_successes, self.fused_kv_attempts),
            },
        );
        inline for ([_]FusedKvFallbackReason{
            .success,
            .tensor_type,
            .shape,
            .k_bias,
            .k_norm,
            .v_bias,
            .rope_style,
            .rope_dim,
            .matrix_missing,
        }) |reason| {
            try writer.print(
                "fused_kv_half.profile.reason.{s}={d}\n",
                .{ @tagName(reason), self.fused_kv_fallback_counts[@intFromEnum(reason)] },
            );
        }
        for (self.fused_kv_tensor_pair_counts, 0..) |count, index| {
            if (count == 0) continue;
            const k_type: u32 = @intCast(index / ggml_type_count);
            const v_type: u32 = @intCast(index % ggml_type_count);
            try writer.print(
                "fused_kv_half.profile.tensor_pair.{s}.{s}={d}\n",
                .{ tensorTypeName(k_type), tensorTypeName(v_type), count },
            );
        }
        return try buffer.toOwnedSlice(allocator);
    }

    pub fn renderFusedQSummary(self: *const Session, allocator: std.mem.Allocator) !?[]u8 {
        if (self.fused_q_attempts == 0) return null;

        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);
        try writer.print(
            "fused_q_rope.profile.attempts={d}\nfused_q_rope.profile.successes={d}\nfused_q_rope.profile.success_pct={d:.3}\n",
            .{
                self.fused_q_attempts,
                self.fused_q_successes,
                percent(self.fused_q_successes, self.fused_q_attempts),
            },
        );
        inline for ([_]FusedQFallbackReason{
            .success,
            .tensor_type,
            .q_bias,
            .q_norm,
            .rope_style,
            .rope_dim,
            .matrix_missing,
        }) |reason| {
            try writer.print(
                "fused_q_rope.profile.reason.{s}={d}\n",
                .{ @tagName(reason), self.fused_q_fallback_counts[@intFromEnum(reason)] },
            );
        }
        return try buffer.toOwnedSlice(allocator);
    }

    pub fn renderFusedQkSummary(self: *const Session, allocator: std.mem.Allocator) !?[]u8 {
        if (self.fused_qk_attempts == 0) return null;

        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);
        try writer.print(
            "fused_qk_half.profile.attempts={d}\nfused_qk_half.profile.successes={d}\nfused_qk_half.profile.success_pct={d:.3}\n",
            .{
                self.fused_qk_attempts,
                self.fused_qk_successes,
                percent(self.fused_qk_successes, self.fused_qk_attempts),
            },
        );
        inline for ([_]FusedQkFallbackReason{
            .success,
            .tensor_type,
            .shape,
            .q_bias,
            .q_norm,
            .k_bias,
            .k_norm,
            .rope_style,
            .rope_dim,
            .matrix_missing,
        }) |reason| {
            try writer.print(
                "fused_qk_half.profile.reason.{s}={d}\n",
                .{ @tagName(reason), self.fused_qk_fallback_counts[@intFromEnum(reason)] },
            );
        }
        return try buffer.toOwnedSlice(allocator);
    }

    pub fn runOutputArgmax(self: *Session, norm: TensorDesc, tensor: TensorDesc) !u32 {
        const shape = metal_profile.ShapeDesc{
            .rows = 1,
            .cols = 1,
            .depth = self.model.vocab_size,
            .tensor_type = tensor.tensor_type,
        };
        const output_reduce_start = std.time.nanoTimestamp();
        const use_fused_output_argmax = tensor.tensor_type == 12 or tensor.tensor_type == 14;
        if (use_fused_output_argmax) {
            const norm_weights = self.dense_lookup.getDense(norm.offset) orelse return error.InvalidTensorMetadata;
            try self.runRmsNormScale(self.hidden, self.norm_scale);
            const initial_state = [_]u32{ 0, 0, std.math.maxInt(u32) };
            try metal_backend.writeBufferU32(self.sampled_token_packed, &initial_state);
            const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
            if (tensor.tensor_type == 12) {
                try metal_backend.runMatVecQ4KArgmaxRmsToBuffer(
                    self.backend,
                    matrix,
                    self.hidden,
                    norm_weights,
                    self.norm_scale,
                    self.sampled_token_packed,
                    tensor.rows,
                    tensor.cols,
                );
            } else {
                try metal_backend.runMatVecQ6KArgmaxRmsToBuffer(
                    self.backend,
                    matrix,
                    self.hidden,
                    norm_weights,
                    self.norm_scale,
                    self.sampled_token_packed,
                    tensor.rows,
                    tensor.cols,
                );
            }
        } else {
            try self.runRmsNorm(norm, self.hidden, self.normed);
            try self.runProjection(tensor, self.normed, self.tmp);
            try metal_backend.argmax(self.backend, self.tmp, self.sampled_token, self.model.vocab_size);
        }
        self.recordCategoryWithShape(.output_reduce, output_reduce_start, shape);
        try self.commitOutputSequence(shape);
        const host_readback_start = std.time.nanoTimestamp();
        var token: [1]u32 = .{0};
        if (use_fused_output_argmax) {
            var argmax_state: [3]u32 = .{ 0, 0, 0 };
            try metal_backend.readBufferU32(self.sampled_token_packed, &argmax_state);
            token[0] = argmax_state[2];
        } else {
            try metal_backend.readBufferU32(self.sampled_token, &token);
        }
        self.recordCategoryWithShape(.host_readback, host_readback_start, shape);
        return token[0];
    }

    pub fn runOutputSampleTopK(
        self: *Session,
        norm: TensorDesc,
        tensor: TensorDesc,
        top_k: usize,
        temperature: f32,
        random_uniform: f32,
    ) !u32 {
        try self.runRmsNorm(norm, self.hidden, self.normed);
        try self.runProjection(tensor, self.normed, self.tmp);
        const shape = metal_profile.ShapeDesc{
            .rows = 1,
            .cols = 1,
            .depth = self.model.vocab_size,
            .extra = top_k,
        };
        const output_reduce_start = std.time.nanoTimestamp();
        try metal_backend.sampleTopK(
            self.backend,
            self.tmp,
            self.sampled_token,
            self.model.vocab_size,
            top_k,
            temperature,
            random_uniform,
        );
        self.recordCategoryWithShape(.output_reduce, output_reduce_start, shape);
        try self.commitOutputSequence(shape);
        var token: [1]u32 = .{0};
        const host_readback_start = std.time.nanoTimestamp();
        try metal_backend.readBufferU32(self.sampled_token, &token);
        self.recordCategoryWithShape(.host_readback, host_readback_start, shape);
        return token[0];
    }

    pub fn runOutputShortlist(
        self: *Session,
        norm: TensorDesc,
        tensor: TensorDesc,
        shortlist_len: usize,
        out: []ShortlistEntry,
    ) ![]const ShortlistEntry {
        if (shortlist_len == 0 or shortlist_len > max_shortlist_len) return error.InvalidTensorMetadata;
        if (out.len < shortlist_len) return error.InvalidTensorMetadata;

        try self.runRmsNorm(norm, self.hidden, self.normed);
        try self.runProjection(tensor, self.normed, self.tmp);
        const shape = metal_profile.ShapeDesc{
            .rows = 2,
            .cols = shortlist_len,
            .depth = self.model.vocab_size,
        };
        const output_reduce_start = std.time.nanoTimestamp();
        try metal_backend.topKShortlist(
            self.backend,
            self.tmp,
            self.shortlist_entries,
            self.model.vocab_size,
            shortlist_len,
        );
        self.recordCategoryWithShape(.output_reduce, output_reduce_start, shape);
        try self.commitOutputSequence(shape);

        var entries: [max_shortlist_len]metal_backend.ShortlistEntry = undefined;
        const host_readback_start = std.time.nanoTimestamp();
        try metal_backend.readShortlistEntries(self.shortlist_entries, entries[0..shortlist_len]);
        for (0..shortlist_len) |index| {
            out[index] = .{
                .token_id = entries[index].token_id,
                .logit = entries[index].score,
            };
        }
        self.recordCategoryWithShape(.host_readback, host_readback_start, shape);
        return out[0..shortlist_len];
    }

    fn runRmsNorm(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
    ) !void {
        const weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
        const start = std.time.nanoTimestamp();
        try metal_backend.rmsNorm(
            self.backend,
            input,
            weights,
            output,
            self.model.embedding_length,
            self.model.rms_norm_eps,
        );
        self.recordCategoryWithShape(.normalization, start, .{
            .rows = 1,
            .cols = self.model.embedding_length,
        });
    }

    fn runRmsNormScale(
        self: *Session,
        input: metal_backend.BufferHandle,
        output_scale: metal_backend.BufferHandle,
    ) !void {
        const start = std.time.nanoTimestamp();
        try metal_backend.rmsNormScale(
            self.backend,
            input,
            output_scale,
            self.model.embedding_length,
            self.model.rms_norm_eps,
        );
        self.recordCategoryWithShape(.normalization, start, .{
            .rows = 1,
            .cols = self.model.embedding_length,
            .depth = 1,
        });
    }

    fn runRmsNormPerHead(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
        head_count: usize,
        head_dim: usize,
    ) !void {
        const weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
        if (tensor.cols != head_dim) return error.InvalidTensorMetadata;
        const start = std.time.nanoTimestamp();
        try metal_backend.rmsNormPerHeadF32(
            self.backend,
            input,
            weights,
            output,
            head_count,
            head_dim,
            self.model.rms_norm_eps,
        );
        self.recordCategoryWithShape(.normalization, start, .{
            .rows = head_count,
            .cols = head_dim,
        });
    }

    fn runProjection(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
    ) !void {
        const start = std.time.nanoTimestamp();
        var quant_layout_path: metal_profile.LayoutPath = .none;
        switch (tensor.tensor_type) {
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    quant_layout_path = .moonq_q4_k;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                }
            },
            14 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ6KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    quant_layout_path = .packed_q6_k;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ6KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    quant_layout_path = .raw_q6_k;
                }
            },
            8 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ8_0ToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
            },
            else => {
                const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecToBuffer(
                    self.backend,
                    matrix,
                    input,
                    output,
                    tensor.rows,
                    tensor.cols,
                );
            },
        }
        const shape = metal_profile.ShapeDesc{
            .rows = tensor.rows,
            .cols = tensor.cols,
            .tensor_type = tensor.tensor_type,
            .layout_path = quant_layout_path,
        };
        self.recordCategoryWithShape(projectionCategoryFor(tensor.tensor_type), start, shape);
        if (quant_layout_path != .none) {
            self.recordQuantizedLayoutProjection(start, shape);
        }
    }

    fn runFusedQProjectionRoPE(
        self: *Session,
        layer: LayerDesc,
        input: metal_backend.BufferHandle,
        norm: TensorDesc,
        norm_scale: metal_backend.BufferHandle,
        norm_scale_ready: *bool,
        position: usize,
    ) !bool {
        self.fused_q_attempts += 1;
        if (layer.attn_q.tensor_type != 12) {
            self.recordFusedQFallback(.tensor_type);
            return false;
        }
        if (layer.attn_q_bias != null) {
            self.recordFusedQFallback(.q_bias);
            return false;
        }
        if (layer.attn_q_norm != null) {
            self.recordFusedQFallback(.q_norm);
            return false;
        }
        if (self.model.rope_style != 0) {
            self.recordFusedQFallback(.rope_style);
            return false;
        }
        if (self.model.rope_dimension_count != self.model.head_dimension) {
            self.recordFusedQFallback(.rope_dim);
            return false;
        }

        const start = std.time.nanoTimestamp();
        const norm_weights = self.dense_lookup.getDense(norm.offset) orelse return error.InvalidTensorMetadata;
        var used_moon_quant = false;
        if (self.dense_lookup.getMoonQuant(layer.attn_q.offset)) |matrix| {
            try self.ensureNormScale(input, norm_scale, norm_scale_ready);
            try metal_backend.runMatVecMoonQuantQ4KQRopeRms(
                self.backend,
                matrix,
                input,
                norm_weights,
                norm_scale,
                self.q,
                self.model.head_count,
                self.model.head_dimension,
                self.model.rope_dimension_count,
                layer.attn_q.cols,
                position,
                self.model.rope_freq_base,
                self.model.rope_style,
            );
            used_moon_quant = true;
        } else if (self.dense_lookup.getRaw(layer.attn_q.offset)) |matrix| {
            try self.ensureNormScale(input, norm_scale, norm_scale_ready);
            try metal_backend.runMatVecQ4KQRopeRms(
                self.backend,
                matrix,
                input,
                norm_weights,
                norm_scale,
                self.q,
                self.model.head_count,
                self.model.head_dimension,
                self.model.rope_dimension_count,
                layer.attn_q.cols,
                position,
                self.model.rope_freq_base,
                self.model.rope_style,
            );
        } else {
            self.recordFusedQFallback(.matrix_missing);
            return false;
        }

        const projection_shape = metal_profile.ShapeDesc{
            .rows = layer.attn_q.rows,
            .cols = layer.attn_q.cols,
            .tensor_type = layer.attn_q.tensor_type,
            .layout_path = if (used_moon_quant) .moonq_q4_k else .none,
        };
        self.recordCategoryWithShape(.projection_quantized, start, projection_shape);
        self.recordCategoryWithShape(.rope, start, .{
            .rows = self.model.head_count,
            .cols = self.model.head_dimension,
            .depth = self.model.rope_dimension_count,
            .extra = position + 1,
        });
        if (used_moon_quant) self.recordQuantizedLayoutProjection(start, projection_shape);
        self.fused_q_successes += 1;
        self.recordFusedQFallback(.success);
        return true;
    }

    fn runFusedFfnFanout(
        self: *Session,
        gate_tensor: TensorDesc,
        up_tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        norm: TensorDesc,
        norm_scale: metal_backend.BufferHandle,
        gate_output: metal_backend.BufferHandle,
        up_output: metal_backend.BufferHandle,
    ) !bool {
        if (gate_tensor.tensor_type != 12 or up_tensor.tensor_type != 12) return false;
        if (gate_tensor.rows != up_tensor.rows or gate_tensor.cols != up_tensor.cols) return false;
        const norm_weights = self.dense_lookup.getDense(norm.offset) orelse return error.InvalidTensorMetadata;

        if (self.dense_lookup.getMoonQuant(gate_tensor.offset)) |gate_matrix| {
            const up_matrix = self.dense_lookup.getMoonQuant(up_tensor.offset) orelse return false;
            const start = std.time.nanoTimestamp();
            try self.runRmsNormScale(input, norm_scale);
            try metal_backend.runMatVecMoonQuantQ4KDualRmsToBuffers(
                self.backend,
                gate_matrix,
                up_matrix,
                input,
                norm_weights,
                norm_scale,
                gate_output,
                up_output,
                gate_tensor.rows,
                gate_tensor.cols,
            );
            const shape = metal_profile.ShapeDesc{
                .rows = gate_tensor.rows,
                .cols = gate_tensor.cols,
                .tensor_type = gate_tensor.tensor_type,
                .extra = 2,
                .layout_path = .moonq_q4_k,
            };
            self.recordCategoryWithShape(.projection_quantized, start, shape);
            self.recordQuantizedLayoutProjection(start, shape);
            return true;
        }

        const gate_matrix = self.dense_lookup.getRaw(gate_tensor.offset) orelse return false;
        const up_matrix = self.dense_lookup.getRaw(up_tensor.offset) orelse return false;

        const start = std.time.nanoTimestamp();
        try self.runRmsNormScale(input, norm_scale);
        try metal_backend.runMatVecQ4KDualRmsToBuffers(
            self.backend,
            gate_matrix,
            up_matrix,
            input,
            norm_weights,
            norm_scale,
            gate_output,
            up_output,
            gate_tensor.rows,
            gate_tensor.cols,
        );
        self.recordCategoryWithShape(.projection_quantized, start, .{
            .rows = gate_tensor.rows,
            .cols = gate_tensor.cols,
            .tensor_type = gate_tensor.tensor_type,
            .extra = 2,
        });
        return true;
    }

    fn runFusedKvFanout(
        self: *Session,
        k_tensor: TensorDesc,
        v_tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        k_output: metal_backend.BufferHandle,
        v_output: metal_backend.BufferHandle,
    ) !bool {
        if (k_tensor.tensor_type != 12 or v_tensor.tensor_type != 12) return false;
        if (k_tensor.rows != v_tensor.rows or k_tensor.cols != v_tensor.cols) return false;

        if (self.dense_lookup.getMoonQuant(k_tensor.offset)) |k_matrix| {
            const v_matrix = self.dense_lookup.getMoonQuant(v_tensor.offset) orelse return false;
            const start = std.time.nanoTimestamp();
            try metal_backend.runMatVecMoonQuantQ4KDualToBuffers(
                self.backend,
                k_matrix,
                v_matrix,
                input,
                k_output,
                v_output,
                k_tensor.rows,
                k_tensor.cols,
            );
            const shape = metal_profile.ShapeDesc{
                .rows = k_tensor.rows,
                .cols = k_tensor.cols,
                .tensor_type = k_tensor.tensor_type,
                .extra = 2,
                .layout_path = .moonq_q4_k,
            };
            self.recordCategoryWithShape(.projection_quantized, start, shape);
            self.recordQuantizedLayoutProjection(start, shape);
            return true;
        }

        const k_matrix = self.dense_lookup.getRaw(k_tensor.offset) orelse return false;
        const v_matrix = self.dense_lookup.getRaw(v_tensor.offset) orelse return false;
        const start = std.time.nanoTimestamp();
        try metal_backend.runMatVecQ4KDualToBuffers(
            self.backend,
            k_matrix,
            v_matrix,
            input,
            k_output,
            v_output,
            k_tensor.rows,
            k_tensor.cols,
        );
        self.recordCategoryWithShape(.projection_quantized, start, .{
            .rows = k_tensor.rows,
            .cols = k_tensor.cols,
            .tensor_type = k_tensor.tensor_type,
            .extra = 2,
        });
        return true;
    }

    fn runFusedQKCacheWrite(
        self: *Session,
        layer: LayerDesc,
        layer_index: usize,
        position: usize,
        input: metal_backend.BufferHandle,
        norm: TensorDesc,
        norm_scale: metal_backend.BufferHandle,
        norm_scale_ready: *bool,
    ) !bool {
        self.fused_qk_attempts += 1;
        if (layer.attn_q.tensor_type != 12 or layer.attn_k.tensor_type != 12) {
            self.recordFusedQkFallback(.tensor_type);
            return false;
        }
        if (layer.attn_q.rows != layer.attn_k.rows or layer.attn_q.cols != layer.attn_k.cols or self.model.head_count != self.model.head_count_kv) {
            self.recordFusedQkFallback(.shape);
            return false;
        }
        if (layer.attn_q_bias != null) {
            self.recordFusedQkFallback(.q_bias);
            return false;
        }
        if (layer.attn_q_norm != null) {
            self.recordFusedQkFallback(.q_norm);
            return false;
        }
        if (layer.attn_k_bias != null) {
            self.recordFusedQkFallback(.k_bias);
            return false;
        }
        if (layer.attn_k_norm != null) {
            self.recordFusedQkFallback(.k_norm);
            return false;
        }
        if (self.model.rope_style != 0) {
            self.recordFusedQkFallback(.rope_style);
            return false;
        }
        if (self.model.rope_dimension_count != self.model.head_dimension) {
            self.recordFusedQkFallback(.rope_dim);
            return false;
        }

        const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
        const kv_offset_elements = layer_base + position * self.model.kv_dimension;
        const q_matrix = self.dense_lookup.getRaw(layer.attn_q.offset) orelse {
            self.recordFusedQkFallback(.matrix_missing);
            return false;
        };
        const k_matrix = self.dense_lookup.getRaw(layer.attn_k.offset) orelse {
            self.recordFusedQkFallback(.matrix_missing);
            return false;
        };
        const norm_weights = self.dense_lookup.getDense(norm.offset) orelse return error.InvalidTensorMetadata;
        try self.ensureNormScale(input, norm_scale, norm_scale_ready);

        const start = std.time.nanoTimestamp();
        try metal_backend.runMatVecQ4KQKKHalfRms(
            self.backend,
            q_matrix,
            k_matrix,
            input,
            norm_weights,
            norm_scale,
            self.q,
            self.k_cache,
            kv_offset_elements,
            self.model.head_count,
            self.model.head_dimension,
            self.model.rope_dimension_count,
            layer.attn_q.cols,
            position,
            self.model.rope_freq_base,
            self.model.rope_style,
        );
        self.recordCategoryWithShape(.projection_quantized, start, .{
            .rows = layer.attn_q.rows,
            .cols = layer.attn_q.cols,
            .tensor_type = layer.attn_q.tensor_type,
            .extra = 2,
        });
        self.recordCategoryWithShape(.rope, start, .{
            .rows = self.model.head_count,
            .cols = self.model.head_dimension,
            .depth = self.model.rope_dimension_count,
            .extra = position + 1,
        });
        self.recordCategoryWithShape(.kv_writes, start, .{
            .rows = 1,
            .cols = self.model.kv_dimension,
            .depth = layer_index,
            .extra = position + 1,
        });
        self.fused_qk_successes += 1;
        self.recordFusedQkFallback(.success);
        return true;
    }

    fn runFusedKvCacheWrite(
        self: *Session,
        layer: LayerDesc,
        layer_index: usize,
        position: usize,
        input: metal_backend.BufferHandle,
        norm: TensorDesc,
        norm_scale: metal_backend.BufferHandle,
        norm_scale_ready: *bool,
    ) !bool {
        self.fused_kv_attempts += 1;
        self.recordFusedKvTensorPair(layer.attn_k.tensor_type, layer.attn_v.tensor_type);
        const kernel = fusedKvKernelFor(layer.attn_k.tensor_type, layer.attn_v.tensor_type) orelse {
            self.recordFusedKvFallback(.tensor_type);
            return false;
        };
        if (layer.attn_k.rows != layer.attn_v.rows or layer.attn_k.cols != layer.attn_v.cols) {
            self.recordFusedKvFallback(.shape);
            return false;
        }
        if (layer.attn_k_bias != null) {
            self.recordFusedKvFallback(.k_bias);
            return false;
        }
        if (layer.attn_k_norm != null) {
            self.recordFusedKvFallback(.k_norm);
            return false;
        }
        if (layer.attn_v_bias != null) {
            self.recordFusedKvFallback(.v_bias);
            return false;
        }
        if (self.model.rope_style != 0) {
            self.recordFusedKvFallback(.rope_style);
            return false;
        }
        if (self.model.rope_dimension_count != self.model.head_dimension) {
            self.recordFusedKvFallback(.rope_dim);
            return false;
        }

        const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
        const kv_offset_elements = layer_base + position * self.model.kv_dimension;
        const start = std.time.nanoTimestamp();
        const norm_weights = self.dense_lookup.getDense(norm.offset) orelse return error.InvalidTensorMetadata;
        var used_moon_quant = false;
        if (self.dense_lookup.getMoonQuant(layer.attn_k.offset)) |k_matrix| {
            try self.ensureNormScale(input, norm_scale, norm_scale_ready);
            switch (kernel) {
                .q4k_q4k => {
                    const v_matrix = self.dense_lookup.getMoonQuant(layer.attn_v.offset) orelse {
                        self.recordFusedKvFallback(.matrix_missing);
                        return false;
                    };
                    try metal_backend.runMatVecMoonQuantQ4KDualKvHalfRms(
                        self.backend,
                        k_matrix,
                        v_matrix,
                        input,
                        norm_weights,
                        norm_scale,
                        self.k_cache,
                        self.v_cache,
                        kv_offset_elements,
                        self.model.head_count_kv,
                        self.model.head_dimension,
                        self.model.rope_dimension_count,
                        layer.attn_k.cols,
                        position,
                        self.model.rope_freq_base,
                        self.model.rope_style,
                    );
                },
                .q4k_q6k => {
                    const v_matrix = self.dense_lookup.getRaw(layer.attn_v.offset) orelse {
                        self.recordFusedKvFallback(.matrix_missing);
                        return false;
                    };
                    try metal_backend.runMatVecMoonQuantQ4KQ6KDualKvHalfRms(
                        self.backend,
                        k_matrix,
                        v_matrix,
                        input,
                        norm_weights,
                        norm_scale,
                        self.k_cache,
                        self.v_cache,
                        kv_offset_elements,
                        self.model.head_count_kv,
                        self.model.head_dimension,
                        self.model.rope_dimension_count,
                        layer.attn_k.cols,
                        position,
                        self.model.rope_freq_base,
                        self.model.rope_style,
                    );
                },
            }
            used_moon_quant = true;
        } else {
            const k_matrix = self.dense_lookup.getRaw(layer.attn_k.offset) orelse {
                self.recordFusedKvFallback(.matrix_missing);
                return false;
            };
            const v_matrix = self.dense_lookup.getRaw(layer.attn_v.offset) orelse {
                self.recordFusedKvFallback(.matrix_missing);
                return false;
            };
            try self.ensureNormScale(input, norm_scale, norm_scale_ready);
            switch (kernel) {
                .q4k_q4k => try metal_backend.runMatVecQ4KDualKvHalfRms(
                    self.backend,
                    k_matrix,
                    v_matrix,
                    input,
                    norm_weights,
                    norm_scale,
                    self.k_cache,
                    self.v_cache,
                    kv_offset_elements,
                    self.model.head_count_kv,
                    self.model.head_dimension,
                    self.model.rope_dimension_count,
                    layer.attn_k.cols,
                    position,
                    self.model.rope_freq_base,
                    self.model.rope_style,
                ),
                .q4k_q6k => try metal_backend.runMatVecQ4KQ6KDualKvHalfRms(
                    self.backend,
                    k_matrix,
                    v_matrix,
                    input,
                    norm_weights,
                    norm_scale,
                    self.k_cache,
                    self.v_cache,
                    kv_offset_elements,
                    self.model.head_count_kv,
                    self.model.head_dimension,
                    self.model.rope_dimension_count,
                    layer.attn_k.cols,
                    position,
                    self.model.rope_freq_base,
                    self.model.rope_style,
                ),
            }
        }

        const shape = metal_profile.ShapeDesc{
            .rows = layer.attn_k.rows,
            .cols = layer.attn_k.cols,
            .tensor_type = layer.attn_k.tensor_type,
            .extra = 2,
            .layout_path = if (used_moon_quant) .moonq_q4_k else .none,
        };
        self.recordCategoryWithShape(.projection_quantized, start, shape);
        self.recordCategoryWithShape(.kv_writes, start, .{
            .rows = 1,
            .cols = self.model.kv_dimension,
            .depth = layer_index,
            .extra = position + 1,
        });
        self.recordCategoryWithShape(.rope, start, .{
            .rows = self.model.head_count_kv,
            .cols = self.model.head_dimension,
            .depth = self.model.rope_dimension_count,
            .extra = position + 1,
        });
        if (used_moon_quant) self.recordQuantizedLayoutProjection(start, shape);
        self.fused_kv_successes += 1;
        self.recordFusedKvFallback(.success);
        return true;
    }

    fn runFusedKCacheWrite(
        self: *Session,
        layer: LayerDesc,
        layer_index: usize,
        position: usize,
        input: metal_backend.BufferHandle,
        norm: TensorDesc,
        norm_scale: metal_backend.BufferHandle,
        norm_scale_ready: *bool,
    ) !bool {
        self.fused_k_attempts += 1;
        if (layer.attn_k.tensor_type != 12) {
            self.recordFusedKFallback(.tensor_type);
            return false;
        }
        if (layer.attn_k_bias != null) {
            self.recordFusedKFallback(.k_bias);
            return false;
        }
        if (layer.attn_k_norm != null) {
            self.recordFusedKFallback(.k_norm);
            return false;
        }
        if (self.model.rope_style != 0) {
            self.recordFusedKFallback(.rope_style);
            return false;
        }
        if (self.model.rope_dimension_count != self.model.head_dimension) {
            self.recordFusedKFallback(.rope_dim);
            return false;
        }

        const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
        const kv_offset_elements = layer_base + position * self.model.kv_dimension;
        const start = std.time.nanoTimestamp();
        const norm_weights = self.dense_lookup.getDense(norm.offset) orelse return error.InvalidTensorMetadata;
        var used_moon_quant = false;
        if (self.dense_lookup.getMoonQuant(layer.attn_k.offset)) |k_matrix| {
            try self.ensureNormScale(input, norm_scale, norm_scale_ready);
            try metal_backend.runMatVecMoonQuantQ4KKHalfRms(
                self.backend,
                k_matrix,
                input,
                norm_weights,
                norm_scale,
                self.k_cache,
                kv_offset_elements,
                self.model.head_count_kv,
                self.model.head_dimension,
                self.model.rope_dimension_count,
                layer.attn_k.cols,
                position,
                self.model.rope_freq_base,
                self.model.rope_style,
            );
            used_moon_quant = true;
        } else if (self.dense_lookup.getRaw(layer.attn_k.offset)) |k_matrix| {
            try self.ensureNormScale(input, norm_scale, norm_scale_ready);
            try metal_backend.runMatVecQ4KKHalfRms(
                self.backend,
                k_matrix,
                input,
                norm_weights,
                norm_scale,
                self.k_cache,
                kv_offset_elements,
                self.model.head_count_kv,
                self.model.head_dimension,
                self.model.rope_dimension_count,
                layer.attn_k.cols,
                position,
                self.model.rope_freq_base,
                self.model.rope_style,
            );
        } else {
            self.recordFusedKFallback(.raw_missing);
            return false;
        }

        const shape = metal_profile.ShapeDesc{
            .rows = layer.attn_k.rows,
            .cols = layer.attn_k.cols,
            .tensor_type = layer.attn_k.tensor_type,
            .extra = 1,
            .layout_path = if (used_moon_quant) .moonq_q4_k else .none,
        };
        self.recordCategoryWithShape(.projection_quantized, start, shape);
        if (used_moon_quant) self.recordQuantizedLayoutProjection(start, shape);
        self.fused_k_successes += 1;
        self.recordFusedKFallback(.success);
        return true;
    }

    fn runFusedQKvCacheWrite(
        self: *Session,
        layer: LayerDesc,
        layer_index: usize,
        position: usize,
        input: metal_backend.BufferHandle,
        norm: TensorDesc,
        norm_scale: metal_backend.BufferHandle,
        norm_scale_ready: *bool,
    ) !bool {
        _ = self;
        _ = layer;
        _ = layer_index;
        _ = position;
        _ = input;
        _ = norm;
        _ = norm_scale;
        _ = norm_scale_ready;
        return false;
    }

    fn ensureNormScale(
        self: *Session,
        input: metal_backend.BufferHandle,
        norm_scale: metal_backend.BufferHandle,
        ready: *bool,
    ) !void {
        if (ready.*) return;
        try self.runRmsNormScale(input, norm_scale);
        ready.* = true;
    }

    fn runProjectionToDst(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
        output_offset_bytes: usize,
    ) !void {
        const start = std.time.nanoTimestamp();
        var quant_layout_path: metal_profile.LayoutPath = .none;
        switch (tensor.tensor_type) {
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                    quant_layout_path = .moonq_q4_k;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                }
            },
            14 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ6KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                    quant_layout_path = .packed_q6_k;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ6KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                    quant_layout_path = .raw_q6_k;
                }
            },
            8 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ8_0ToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
            },
            else => {
                const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
            },
        }
        const shape = metal_profile.ShapeDesc{
            .rows = tensor.rows,
            .cols = tensor.cols,
            .tensor_type = tensor.tensor_type,
            .layout_path = quant_layout_path,
        };
        self.recordCategoryWithShape(projectionCategoryFor(tensor.tensor_type), start, shape);
        if (quant_layout_path != .none) {
            self.recordQuantizedLayoutProjection(start, shape);
        }
    }

    fn runProjectionAdd(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
    ) !void {
        const start = std.time.nanoTimestamp();
        var quant_layout_path: metal_profile.LayoutPath = .none;
        switch (tensor.tensor_type) {
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    quant_layout_path = .moonq_q4_k;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                }
            },
            14 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ6KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    quant_layout_path = .packed_q6_k;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ6KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    quant_layout_path = .raw_q6_k;
                }
            },
            8 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ8_0AddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
            },
            else => {
                const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
            },
        }
        const shape = metal_profile.ShapeDesc{
            .rows = tensor.rows,
            .cols = tensor.cols,
            .tensor_type = tensor.tensor_type,
            .extra = 1,
            .layout_path = quant_layout_path,
        };
        self.recordCategoryWithShape(projectionAddCategoryFor(tensor.tensor_type), start, shape);
        if (quant_layout_path != .none) {
            self.recordQuantizedLayoutProjection(start, shape);
        }
    }

    fn runBatchProjection(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
        batch_count: usize,
    ) !void {
        const start = std.time.nanoTimestamp();
        var quant_layout_path: metal_profile.LayoutPath = .none;
        switch (tensor.tensor_type) {
            12 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.batchMatvecQ4KAll(self.backend, matrix, input, output, tensor.rows, tensor.cols, batch_count);
            },
            14 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.batchMatvecQ6KAll(self.backend, matrix, input, output, tensor.rows, tensor.cols, batch_count);
                quant_layout_path = .raw_q6_k;
            },
            else => {
                const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.batchMatvecAll(self.backend, matrix, input, output, tensor.rows, tensor.cols, batch_count);
            },
        }
        const shape = metal_profile.ShapeDesc{
            .rows = tensor.rows,
            .cols = tensor.cols,
            .depth = batch_count,
            .tensor_type = tensor.tensor_type,
            .layout_path = quant_layout_path,
        };
        self.recordCategoryWithShape(projectionCategoryFor(tensor.tensor_type), start, shape);
        if (quant_layout_path != .none) self.recordQuantizedLayoutProjection(start, shape);
    }

    fn runBatchRmsNorm(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
        batch_count: usize,
    ) !void {
        const weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
        const start = std.time.nanoTimestamp();
        try metal_backend.batchRmsNormAll(
            self.backend,
            input,
            weights,
            output,
            self.model.embedding_length,
            self.model.rms_norm_eps,
            batch_count,
        );
        self.recordCategoryWithShape(.normalization, start, .{
            .rows = batch_count,
            .cols = self.model.embedding_length,
        });
    }

    fn runPromptAttentionLayerBatched(
        self: *Session,
        layer: LayerDesc,
        layer_index: usize,
        batch_count: usize,
        base_position: usize,
    ) !void {
        try self.runBatchRmsNorm(layer.attn_norm, self.prompt_prefill_inputs, self.prompt_prefill_normed, batch_count);
        try self.runBatchProjection(layer.attn_q, self.prompt_prefill_normed, self.prompt_prefill_q, batch_count);
        try self.runBatchProjection(layer.attn_k, self.prompt_prefill_normed, self.prompt_prefill_k, batch_count);
        try self.runBatchProjection(layer.attn_v, self.prompt_prefill_normed, self.prompt_prefill_v, batch_count);

        const q_bytes = self.model.embedding_length * @sizeOf(f32);
        const kv_bytes = self.model.kv_dimension * @sizeOf(f32);
        const attn_bytes = self.model.embedding_length * @sizeOf(f32);
        for (0..batch_count) |i| {
            const position = base_position + i;
            try metal_backend.copyBufferRegion(self.backend, self.prompt_prefill_q, i * q_bytes, self.q, 0, q_bytes);
            try metal_backend.copyBufferRegion(self.backend, self.prompt_prefill_k, i * kv_bytes, self.k, 0, kv_bytes);
            try metal_backend.copyBufferRegion(self.backend, self.prompt_prefill_v, i * kv_bytes, self.v, 0, kv_bytes);

            if (self.model.rope_style != 0 or self.model.rope_dimension_count != self.model.head_dimension) {
                return error.InvalidTensorMetadata;
            }
            const q_rope_start = std.time.nanoTimestamp();
            try metal_backend.applyRoPE(
                self.backend,
                self.q,
                self.model.head_count,
                self.model.head_dimension,
                self.model.rope_dimension_count,
                position,
                self.model.rope_freq_base,
                self.model.rope_style,
            );
            self.recordCategoryWithShape(.rope, q_rope_start, .{
                .rows = self.model.head_count,
                .cols = self.model.head_dimension,
                .depth = self.model.rope_dimension_count,
                .extra = position + 1,
            });

            const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
            const kv_offset_elements = layer_base + position * self.model.kv_dimension;
            const kv_start = std.time.nanoTimestamp();
            try metal_backend.packKvHalf(
                self.backend,
                self.k,
                self.v,
                self.k_cache,
                self.v_cache,
                kv_offset_elements,
                self.model.head_count_kv,
                self.model.head_dimension,
                self.model.rope_dimension_count,
                position,
                self.model.rope_freq_base,
                self.model.rope_style,
            );
            self.recordCategoryWithShape(.kv_writes, kv_start, .{
                .rows = 1,
                .cols = self.model.kv_dimension,
                .depth = layer_index,
                .extra = position + 1,
            });

            const attention_start = std.time.nanoTimestamp();
            try metal_backend.attentionFused(
                self.backend,
                self.q,
                self.k_cache,
                self.v_cache,
                self.attn,
                self.model.head_count,
                self.model.head_count_kv,
                self.model.head_dimension,
                self.model.kv_dimension,
                self.model.context_length,
                position,
                layer_base,
                @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(self.model.head_dimension))),
            );
            self.recordCategoryWithShape(.attention, attention_start, .{
                .rows = self.model.head_count,
                .cols = self.model.head_dimension,
                .depth = position + 1,
                .extra = self.model.head_count_kv,
            });
            try metal_backend.copyBufferRegion(self.backend, self.attn, 0, self.prompt_prefill_q, i * attn_bytes, attn_bytes);
        }

        try self.runBatchProjection(layer.attn_output, self.prompt_prefill_q, self.prompt_prefill_normed, batch_count);
        try metal_backend.batchAddInPlaceAll(self.backend, self.prompt_prefill_inputs, self.prompt_prefill_normed, self.model.embedding_length, batch_count);
    }

    fn runPromptFfnLayerBatched(
        self: *Session,
        layer: LayerDesc,
        batch_count: usize,
    ) !void {
        try self.runBatchRmsNorm(layer.ffn_norm, self.prompt_prefill_inputs, self.prompt_prefill_normed, batch_count);
        try self.runBatchProjection(layer.ffn_gate, self.prompt_prefill_normed, self.prompt_prefill_gate, batch_count);
        try self.runBatchProjection(layer.ffn_up, self.prompt_prefill_normed, self.prompt_prefill_up, batch_count);
        const silu_start = std.time.nanoTimestamp();
        try metal_backend.batchSiluMulAll(self.backend, self.prompt_prefill_gate, self.prompt_prefill_up, self.model.feed_forward_length, batch_count);
        self.recordCategoryWithShape(.ffn_activation, silu_start, .{
            .rows = batch_count,
            .cols = self.model.feed_forward_length,
            .depth = 2,
        });
        try self.runBatchProjection(layer.ffn_down, self.prompt_prefill_gate, self.prompt_prefill_normed, batch_count);
        try metal_backend.batchAddInPlaceAll(self.backend, self.prompt_prefill_inputs, self.prompt_prefill_normed, self.model.embedding_length, batch_count);
    }

    fn runBiasAdd(self: *Session, tensor: TensorDesc, target: metal_backend.BufferHandle) !void {
        const start = std.time.nanoTimestamp();
        const bias_weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
        try metal_backend.addBiasF32(self.backend, target, bias_weights, tensor.cols);
        self.recordCategoryWithShape(.bias_add, start, .{
            .rows = 1,
            .cols = tensor.cols,
        });
    }

    fn recordCategoryWithShape(
        self: *Session,
        category: metal_profile.Category,
        start_ns: i128,
        shape: metal_profile.ShapeDesc,
    ) void {
        if (self.profiler) |profiler| {
            profiler.recordWithShape(category, elapsedSince(start_ns), shape);
        }
    }

    fn commitOutputSequence(self: *Session, shape: metal_profile.ShapeDesc) !void {
        if (self.profiler != null) {
            const commit_stats = try metal_backend.commitSequenceTimed(self.backend);
            self.recordCommitWait(shape, commit_stats);
            return;
        }
        try metal_backend.commitSequence(self.backend);
    }

    fn recordCommitWait(self: *Session, shape: metal_profile.ShapeDesc, stats: metal_backend.CommitStats) void {
        if (self.profiler) |profiler| {
            profiler.recordWithShape(.commit_wait, stats.cpu_wait_ns, shape);
            if (stats.gpu_timestamps_valid) profiler.recordCommitWaitGpu(stats.gpu_elapsed_ns);
            profiler.recordCommitWaitNonGpu(stats.non_gpu_wait_ns);
            profiler.recordCommandBuffers(stats.command_buffer_count);
            profiler.recordEncoders(stats.encoder_count);
            profiler.recordDispatches(stats.dispatch_count);
        }
    }

    fn elapsedSince(start_ns: i128) u64 {
        return @intCast(@max(@as(i128, 0), std.time.nanoTimestamp() - start_ns));
    }

    fn percent(numerator: u64, denominator: u64) f64 {
        if (denominator == 0) return 0;
        return (@as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator))) * 100.0;
    }

    fn recordFusedKFallback(self: *Session, reason: FusedKFallbackReason) void {
        self.fused_k_fallback_counts[@intFromEnum(reason)] += 1;
    }

    fn recordFusedQFallback(self: *Session, reason: FusedQFallbackReason) void {
        self.fused_q_fallback_counts[@intFromEnum(reason)] += 1;
    }

    fn recordFusedQkFallback(self: *Session, reason: FusedQkFallbackReason) void {
        self.fused_qk_fallback_counts[@intFromEnum(reason)] += 1;
    }

    fn recordFusedKvFallback(self: *Session, reason: FusedKvFallbackReason) void {
        self.fused_kv_fallback_counts[@intFromEnum(reason)] += 1;
    }

    fn recordFusedKvTensorPair(self: *Session, k_type: u32, v_type: u32) void {
        if (k_type >= ggml_type_count or v_type >= ggml_type_count) return;
        self.fused_kv_tensor_pair_counts[fusedKvTensorPairIndex(k_type, v_type)] += 1;
    }

    fn recordQuantizedLayoutProjection(self: *Session, start_ns: i128, shape: metal_profile.ShapeDesc) void {
        if (shape.layout_path == .raw_q6_k or shape.layout_path == .none) return;
        if (self.profiler) |profiler| {
            profiler.recordMoonQuantProjection(elapsedSince(start_ns), shape);
        }
    }

    fn projectionCategoryFor(tensor_type: u32) metal_profile.Category {
        return switch (tensor_type) {
            8, 12, 14 => .projection_quantized,
            else => .projection_dense,
        };
    }

    fn rmsNormScaleValue(values: []const f32, eps: f32) f32 {
        var sum: f32 = 0;
        for (values) |value| {
            sum += value * value;
        }
        return @as(f32, 1.0) / @sqrt(sum / @as(f32, @floatFromInt(values.len)) + eps);
    }

    fn projectionAddCategoryFor(tensor_type: u32) metal_profile.Category {
        return switch (tensor_type) {
            8, 12, 14 => .projection_add_quantized,
            else => .projection_add_dense,
        };
    }

    fn supportsBatchProjectionTensor(tensor_type: u32) bool {
        return switch (tensor_type) {
            8 => false,
            else => true,
        };
    }

    fn fusedKvTensorPairIndex(k_type: u32, v_type: u32) usize {
        return (@as(usize, @intCast(k_type)) * ggml_type_count) + @as(usize, @intCast(v_type));
    }

    fn fusedKvKernelFor(k_type: u32, v_type: u32) ?FusedKvKernel {
        if (k_type != q4_k_tensor_type) return null;
        return switch (v_type) {
            q4_k_tensor_type => .q4k_q4k,
            q6_k_tensor_type => .q4k_q6k,
            else => null,
        };
    }

    fn tensorTypeName(tensor_type: u32) []const u8 {
        const parsed = std.meta.intToEnum(gguf.TensorType, tensor_type) catch return "unknown";
        return switch (parsed) {
            .f32 => "f32",
            .f16 => "f16",
            .q4_0 => "q4_0",
            .q4_1 => "q4_1",
            .q5_0 => "q5_0",
            .q5_1 => "q5_1",
            .q8_0 => "q8_0",
            .q8_1 => "q8_1",
            .q2_k => "q2_k",
            .q3_k => "q3_k",
            .q4_k => "q4_k",
            .q5_k => "q5_k",
            .q6_k => "q6_k",
            .q8_k => "q8_k",
            .iq2_xxs => "iq2_xxs",
            .iq2_xs => "iq2_xs",
            .iq3_xxs => "iq3_xxs",
            .iq1_s => "iq1_s",
            .iq4_nl => "iq4_nl",
            .iq3_s => "iq3_s",
            .iq2_s => "iq2_s",
            .iq4_xs => "iq4_xs",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .f64 => "f64",
            .iq1_m => "iq1_m",
            .bf16 => "bf16",
            .tq1_0 => "tq1_0",
            .tq2_0 => "tq2_0",
            .mxfp4 => "mxfp4",
            .nvfp4 => "nvfp4",
        };
    }

    pub fn runBatchSpeculativeDecode(
        self: *Session,
        layers: []const LayerDesc,
        draft_tokens: []const u32,
        base_position: usize,
        out_tokens: []u32,
    ) !usize {
        if (draft_tokens.len == 0 or draft_tokens.len > max_draft_len) return 0;
        const batch_count = draft_tokens.len;

        try metal_backend.beginSequence(self.backend);
        for (draft_tokens, 0..) |draft_token, i| {
            const position = base_position + i;

            const emb_rows = self.model.embedding_length;
            const emb_data = self.dense_lookup.getDense(self.model.token_embd_offset) orelse return error.InvalidTensorMetadata;
            const src = emb_data[draft_token * emb_rows ..][0..emb_rows];
            var values: [2048]f32 = undefined;
            if (emb_rows > values.len) return error.InvalidTensorMetadata;
            @memcpy(values[0..emb_rows], src);
            try self.beginTokenWithNormScale(values[0..emb_rows], rmsNormScaleValue(values[0..emb_rows], self.model.rms_norm_eps));

            for (layers, 0..) |layer, layer_index| {
                try self.runAttentionBlock(layer, layer_index, position);
            }

            for (layers) |layer| {
                if (!try self.runFusedFfnFanout(
                    layer.ffn_gate,
                    layer.ffn_up,
                    self.hidden,
                    layer.ffn_norm,
                    self.norm_scale,
                    self.gate,
                    self.up,
                )) {
                    try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);
                    try self.runProjection(layer.ffn_gate, self.normed, self.gate);
                    try self.runProjection(layer.ffn_up, self.normed, self.up);
                }
                try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
                try self.runProjectionAdd(layer.ffn_down, self.gate, self.hidden);
            }

            try self.runRmsNorm(self.model.output_norm, self.hidden, self.normed);
            const logits_offset_bytes = i * self.model.vocab_size * @sizeOf(f32);
            try self.runProjectionToDst(
                self.model.output,
                self.normed,
                self.batch_logits,
                logits_offset_bytes,
            );
        }

        try metal_backend.batchArgmax(
            self.backend,
            self.batch_logits,
            self.batch_tokens,
            self.model.vocab_size,
            batch_count,
        );

        _ = try metal_backend.commitSequenceTimed(self.backend);

        var token_ids: [max_draft_len]u32 = undefined;
        try metal_backend.readBufferU32(self.batch_tokens, token_ids[0..batch_count]);

        var accepted: usize = 0;
        for (draft_tokens, 0..) |draft, i| {
            const predicted = token_ids[i];
            out_tokens[i] = predicted;
            accepted += 1;
            if (predicted != draft) {
                return accepted;
            }
        }

        return accepted;
    }
};

test "fused kv selector accepts q4_k/q4_k and q4_k/q6_k only" {
    try std.testing.expectEqual(FusedKvKernel.q4k_q4k, Session.fusedKvKernelFor(q4_k_tensor_type, q4_k_tensor_type).?);
    try std.testing.expectEqual(FusedKvKernel.q4k_q6k, Session.fusedKvKernelFor(q4_k_tensor_type, q6_k_tensor_type).?);
    try std.testing.expect(Session.fusedKvKernelFor(q6_k_tensor_type, q4_k_tensor_type) == null);
    try std.testing.expect(Session.fusedKvKernelFor(q6_k_tensor_type, q6_k_tensor_type) == null);
}

test "fused kv tensor pair index is stable" {
    try std.testing.expectEqual(
        @as(usize, q4_k_tensor_type * ggml_type_count + q6_k_tensor_type),
        Session.fusedKvTensorPairIndex(q4_k_tensor_type, q6_k_tensor_type),
    );
}
