const std = @import("std");
const backend_api = @import("backend.zig");
const metal_backend = @import("metal_backend.zig");

pub const DenseLookup = struct {
    ctx: ?*const anyopaque,
    get_dense_fn: *const fn (?*const anyopaque, u64) ?[]const f32,
    get_raw_fn: *const fn (?*const anyopaque, u64) ?[]const u8,

    pub fn getDense(self: DenseLookup, offset: u64) ?[]const f32 {
        return self.get_dense_fn(self.ctx, offset);
    }

    pub fn getRaw(self: DenseLookup, offset: u64) ?[]const u8 {
        return self.get_raw_fn(self.ctx, offset);
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
    attn_k: TensorDesc,
    attn_v: TensorDesc,
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
};

pub const Session = struct {
    backend: backend_api.MatVecBackend,
    dense_lookup: DenseLookup,
    model: ModelDesc,
    hidden: metal_backend.BufferHandle,
    normed: metal_backend.BufferHandle,
    q: metal_backend.BufferHandle,
    k: metal_backend.BufferHandle,
    v: metal_backend.BufferHandle,
    attn: metal_backend.BufferHandle,
    gate: metal_backend.BufferHandle,
    up: metal_backend.BufferHandle,
    tmp: metal_backend.BufferHandle,
    k_cache: metal_backend.BufferHandle,
    v_cache: metal_backend.BufferHandle,

    pub fn init(
        backend: backend_api.MatVecBackend,
        dense_lookup: DenseLookup,
        model: ModelDesc,
    ) !Session {
        const max_input = @max(model.embedding_length, model.feed_forward_length);
        const max_vec = @max(model.embedding_length, @max(model.feed_forward_length, model.vocab_size));
        const cache_len = model.block_count * model.context_length * model.kv_dimension;
        const hidden = try metal_backend.createScratchBuffer(backend, max_input);
        errdefer metal_backend.destroyBuffer(hidden);
        const normed = try metal_backend.createScratchBuffer(backend, model.embedding_length);
        errdefer metal_backend.destroyBuffer(normed);
        const q = try metal_backend.createScratchBuffer(backend, model.embedding_length);
        errdefer metal_backend.destroyBuffer(q);
        const k = try metal_backend.createScratchBuffer(backend, model.kv_dimension);
        errdefer metal_backend.destroyBuffer(k);
        const v = try metal_backend.createScratchBuffer(backend, model.kv_dimension);
        errdefer metal_backend.destroyBuffer(v);
        const attn = try metal_backend.createScratchBuffer(backend, model.embedding_length);
        errdefer metal_backend.destroyBuffer(attn);
        const gate = try metal_backend.createScratchBuffer(backend, model.feed_forward_length);
        errdefer metal_backend.destroyBuffer(gate);
        const up = try metal_backend.createScratchBuffer(backend, model.feed_forward_length);
        errdefer metal_backend.destroyBuffer(up);
        const tmp = try metal_backend.createScratchBuffer(backend, max_vec);
        errdefer metal_backend.destroyBuffer(tmp);
        const k_cache = try metal_backend.createScratchBuffer(backend, cache_len);
        errdefer metal_backend.destroyBuffer(k_cache);
        const v_cache = try metal_backend.createScratchBuffer(backend, cache_len);
        errdefer metal_backend.destroyBuffer(v_cache);

        return .{
            .backend = backend,
            .dense_lookup = dense_lookup,
            .model = model,
            .hidden = hidden,
            .normed = normed,
            .q = q,
            .k = k,
            .v = v,
            .attn = attn,
            .gate = gate,
            .up = up,
            .tmp = tmp,
            .k_cache = k_cache,
            .v_cache = v_cache,
        };
    }

    pub fn deinit(self: *Session) void {
        metal_backend.destroyBuffer(self.hidden);
        metal_backend.destroyBuffer(self.normed);
        metal_backend.destroyBuffer(self.q);
        metal_backend.destroyBuffer(self.k);
        metal_backend.destroyBuffer(self.v);
        metal_backend.destroyBuffer(self.attn);
        metal_backend.destroyBuffer(self.gate);
        metal_backend.destroyBuffer(self.up);
        metal_backend.destroyBuffer(self.tmp);
        metal_backend.destroyBuffer(self.k_cache);
        metal_backend.destroyBuffer(self.v_cache);
        self.* = undefined;
    }

    pub fn beginToken(self: *Session, input: []const f32) !void {
        try metal_backend.writeBufferF32(self.hidden, input);
        try metal_backend.beginSequence(self.backend);
    }

    pub fn runAttentionBlock(
        self: *Session,
        layer: LayerDesc,
        layer_index: usize,
        position: usize,
    ) !void {
        try self.runRmsNorm(layer.attn_norm, self.hidden, self.normed);
        try self.runProjection(layer.attn_q, self.normed, self.q);
        try self.runProjection(layer.attn_k, self.normed, self.k);
        try self.runProjection(layer.attn_v, self.normed, self.v);

        try metal_backend.applyRoPE(
            self.backend,
            self.q,
            self.model.head_count,
            self.model.head_dimension,
            self.model.rope_dimension_count,
            position,
            self.model.rope_freq_base,
        );
        try metal_backend.applyRoPE(
            self.backend,
            self.k,
            self.model.head_count_kv,
            self.model.head_dimension,
            self.model.rope_dimension_count,
            position,
            self.model.rope_freq_base,
        );

        const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
        const kv_offset = (layer_base + position * self.model.kv_dimension) * @sizeOf(f32);
        const kv_bytes = self.model.kv_dimension * @sizeOf(f32);
        try metal_backend.copyBufferRegion(self.backend, self.k, 0, self.k_cache, kv_offset, kv_bytes);
        try metal_backend.copyBufferRegion(self.backend, self.v, 0, self.v_cache, kv_offset, kv_bytes);

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
        try self.runProjection(layer.attn_output, self.attn, self.tmp);
        try metal_backend.addInPlace(self.backend, self.hidden, self.tmp, self.model.embedding_length);
    }

    pub fn runFfnBlock(self: *Session, layer: LayerDesc) !void {
        try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);
        try self.runProjection(layer.ffn_gate, self.normed, self.gate);
        try self.runProjection(layer.ffn_up, self.normed, self.up);
        try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
        try self.runProjection(layer.ffn_down, self.gate, self.tmp);
        try metal_backend.addInPlace(self.backend, self.hidden, self.tmp, self.model.embedding_length);
    }

    pub fn runOutput(self: *Session, norm: TensorDesc, tensor: TensorDesc, out: []f32) !void {
        try self.runRmsNorm(norm, self.hidden, self.normed);
        try self.runProjection(tensor, self.normed, self.tmp);
        try metal_backend.commitSequence(self.backend);
        try metal_backend.readBufferF32(self.tmp, out);
    }

    fn runRmsNorm(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
    ) !void {
        const weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
        try metal_backend.rmsNorm(
            self.backend,
            input,
            weights,
            output,
            self.model.embedding_length,
            self.model.rms_norm_eps,
        );
    }

    fn runProjection(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
    ) !void {
        switch (tensor.tensor_type) {
            12 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ4KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
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
    }
};
