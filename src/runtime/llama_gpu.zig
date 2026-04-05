const std = @import("std");
const backend_api = @import("backend.zig");
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
    token_embd_offset: u64,
};

pub const ShortlistEntry = struct {
    token_id: u32,
    logit: f32,
};

pub const max_shortlist_len: usize = 64;
pub const max_draft_len: usize = 4;

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
    sampled_token: metal_backend.BufferHandle,
    sampled_token_packed: metal_backend.BufferHandle,
    shortlist_entries: metal_backend.BufferHandle,
    k_cache: metal_backend.BufferHandle,
    v_cache: metal_backend.BufferHandle,
    batch_logits: metal_backend.BufferHandle,
    batch_tokens: metal_backend.BufferHandle,
    profiler: ?*metal_profile.Profiler = null,

    pub fn init(
        backend: backend_api.MatVecBackend,
        dense_lookup: DenseLookup,
        model: ModelDesc,
        profiler: ?*metal_profile.Profiler,
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
        const sampled_token = try metal_backend.createScratchBuffer(backend, 1);
        errdefer metal_backend.destroyBuffer(sampled_token);
        const sampled_token_packed = try metal_backend.createByteScratchBuffer(backend, 2 * @sizeOf(u32));
        errdefer metal_backend.destroyBuffer(sampled_token_packed);
        const shortlist_entries = try metal_backend.createByteScratchBuffer(backend, max_shortlist_len * @sizeOf(metal_backend.ShortlistEntry));
        errdefer metal_backend.destroyBuffer(shortlist_entries);
        const k_cache = try metal_backend.createByteScratchBuffer(backend, cache_len * @sizeOf(f16));
        errdefer metal_backend.destroyBuffer(k_cache);
        const v_cache = try metal_backend.createByteScratchBuffer(backend, cache_len * @sizeOf(f16));
        errdefer metal_backend.destroyBuffer(v_cache);
        const batch_logits = try metal_backend.createScratchBuffer(backend, max_draft_len * model.vocab_size);
        errdefer metal_backend.destroyBuffer(batch_logits);
        const batch_tokens = try metal_backend.createByteScratchBuffer(backend, max_draft_len * @sizeOf(u32));
        errdefer metal_backend.destroyBuffer(batch_tokens);

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
            .sampled_token = sampled_token,
            .sampled_token_packed = sampled_token_packed,
            .shortlist_entries = shortlist_entries,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .batch_logits = batch_logits,
            .batch_tokens = batch_tokens,
            .profiler = profiler,
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
        metal_backend.destroyBuffer(self.sampled_token);
        metal_backend.destroyBuffer(self.sampled_token_packed);
        metal_backend.destroyBuffer(self.shortlist_entries);
        metal_backend.destroyBuffer(self.k_cache);
        metal_backend.destroyBuffer(self.v_cache);
        metal_backend.destroyBuffer(self.batch_logits);
        metal_backend.destroyBuffer(self.batch_tokens);
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

        const q_rope_start = std.time.nanoTimestamp();
        try metal_backend.applyRoPE(
            self.backend,
            self.q,
            self.model.head_count,
            self.model.head_dimension,
            self.model.rope_dimension_count,
            position,
            self.model.rope_freq_base,
        );
        self.recordCategoryWithShape(.elementwise_ops, q_rope_start, .{
            .rows = self.model.head_count,
            .cols = self.model.head_dimension,
            .depth = self.model.rope_dimension_count,
            .extra = position + 1,
        });

        const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
        const kv_offset_elements = layer_base + position * self.model.kv_dimension;

        const kv_k_start = std.time.nanoTimestamp();
        try self.runProjection(layer.attn_k, self.normed, self.k);
        try metal_backend.applyRoPE(
            self.backend,
            self.k,
            self.model.head_count_kv,
            self.model.head_dimension,
            self.model.rope_dimension_count,
            position,
            self.model.rope_freq_base,
        );
        try metal_backend.storeKvHalf(
            self.backend,
            self.k,
            self.k_cache,
            kv_offset_elements,
            self.model.kv_dimension,
        );
        self.recordCategoryWithShape(.elementwise_ops, kv_k_start, .{
            .rows = self.model.head_count_kv,
            .cols = self.model.head_dimension,
            .depth = self.model.rope_dimension_count,
            .extra = position + 1,
        });
        self.recordCategoryWithShape(.kv_writes, kv_k_start, .{
            .rows = 1,
            .cols = self.model.kv_dimension,
            .depth = layer_index,
            .extra = position + 1,
        });

        const kv_v_start = std.time.nanoTimestamp();
        try self.runProjection(layer.attn_v, self.normed, self.v);
        try metal_backend.storeKvHalf(
            self.backend,
            self.v,
            self.v_cache,
            kv_offset_elements,
            self.model.kv_dimension,
        );
        self.recordCategoryWithShape(.kv_writes, kv_v_start, .{
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
        try self.runProjectionAdd(layer.attn_output, self.attn, self.hidden);
    }

    pub fn runFfnBlock(self: *Session, layer: LayerDesc) !void {
        try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);
        try self.runProjection(layer.ffn_gate, self.normed, self.gate);
        try self.runProjection(layer.ffn_up, self.normed, self.up);

        const tensor = layer.ffn_down;
        var handled_fused = false;

        if (tensor.tensor_type == 12) {
            if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                const start = std.time.nanoTimestamp();
                try metal_backend.runMatVecMoonQuantQ4KSiluDownAddToBuffer(self.backend, matrix, self.gate, self.up, self.hidden, tensor.rows, tensor.cols);
                const shape = metal_profile.ShapeDesc{ .rows = tensor.rows, .cols = tensor.cols, .tensor_type = tensor.tensor_type, .extra = 2 };
                self.recordCategoryWithShape(.projections, start, shape);
                handled_fused = true;
            } else {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                const start = std.time.nanoTimestamp();
                try metal_backend.runMatVecQ4KSiluDownAddToBuffer(self.backend, matrix, self.gate, self.up, self.hidden, tensor.rows, tensor.cols);
                const shape = metal_profile.ShapeDesc{ .rows = tensor.rows, .cols = tensor.cols, .tensor_type = tensor.tensor_type, .extra = 2 };
                self.recordCategoryWithShape(.projections, start, shape);
                handled_fused = true;
            }
        }

        if (!handled_fused) {
            const silu_start = std.time.nanoTimestamp();
            try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
            self.recordCategoryWithShape(.elementwise_ops, silu_start, .{
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
        const shape = metal_profile.ShapeDesc{
            .rows = 1,
            .cols = out.len,
        };
        try self.commitOutputSequence(shape);
        const host_readback_start = std.time.nanoTimestamp();
        try metal_backend.readBufferF32(self.tmp, out);
        self.recordCategoryWithShape(.host_readback, host_readback_start, shape);
    }

    pub fn commitToken(self: *Session) !void {
        try metal_backend.commitSequence(self.backend);
    }

    pub fn runOutputArgmax(self: *Session, norm: TensorDesc, tensor: TensorDesc) !u32 {
        try self.runRmsNorm(norm, self.hidden, self.normed);
        const shape = metal_profile.ShapeDesc{
            .rows = 1,
            .cols = 1,
            .depth = self.model.vocab_size,
            .tensor_type = tensor.tensor_type,
        };
        const output_reduce_start = std.time.nanoTimestamp();
        if (tensor.tensor_type == 14) {
            const initial_state = [_]u32{ 0, std.math.maxInt(u32) };
            try metal_backend.writeBufferU32(self.sampled_token_packed, &initial_state);
            const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
            try metal_backend.runMatVecQ6KArgmaxToBuffer(
                self.backend,
                matrix,
                self.normed,
                self.sampled_token_packed,
                tensor.rows,
                tensor.cols,
            );
        } else {
            try self.runProjection(tensor, self.normed, self.tmp);
            try metal_backend.argmax(self.backend, self.tmp, self.sampled_token, self.model.vocab_size);
        }
        self.recordCategoryWithShape(.output_reduce, output_reduce_start, shape);
        try self.commitOutputSequence(shape);
        const host_readback_start = std.time.nanoTimestamp();
        var token: [1]u32 = .{0};
        if (tensor.tensor_type == 14) {
            var argmax_state: [2]u32 = .{ 0, 0 };
            try metal_backend.readBufferU32(self.sampled_token_packed, &argmax_state);
            token[0] = argmax_state[1];
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

    fn runProjection(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
    ) !void {
        const start = std.time.nanoTimestamp();
        var used_moon_quant = false;
        switch (tensor.tensor_type) {
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    used_moon_quant = true;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                }
            },
            14 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ6KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
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
        };
        self.recordCategoryWithShape(.projections, start, shape);
        if (used_moon_quant) {
            if (self.profiler) |profiler| {
                profiler.recordMoonQuantProjection(elapsedSince(start), shape);
            }
        }
    }

    fn runProjectionToDst(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
        output_offset_bytes: usize,
    ) !void {
        const start = std.time.nanoTimestamp();
        var used_moon_quant = false;
        switch (tensor.tensor_type) {
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                    used_moon_quant = true;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                }
            },
            14 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ6KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
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
        };
        self.recordCategoryWithShape(.projections, start, shape);
        if (used_moon_quant) {
            if (self.profiler) |profiler| {
                profiler.recordMoonQuantProjection(elapsedSince(start), shape);
            }
        }
    }

    fn runProjectionAdd(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
    ) !void {
        const start = std.time.nanoTimestamp();
        var used_moon_quant = false;
        switch (tensor.tensor_type) {
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    used_moon_quant = true;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                }
            },
            14 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ6KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
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
        };
        self.recordCategoryWithShape(.projections, start, shape);
        if (used_moon_quant) {
            if (self.profiler) |profiler| {
                profiler.recordMoonQuantProjection(elapsedSince(start), shape);
            }
        }
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
        }
    }

    fn elapsedSince(start_ns: i128) u64 {
        return @intCast(@max(@as(i128, 0), std.time.nanoTimestamp() - start_ns));
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
            try metal_backend.writeBufferF32(self.hidden, values[0..emb_rows]);

            for (layers, 0..) |layer, layer_index| {
                try self.runRmsNorm(layer.attn_norm, self.hidden, self.normed);
                try self.runProjection(layer.attn_q, self.normed, self.q);

                try metal_backend.applyRoPE(
                    self.backend,
                    self.q,
                    self.model.head_count,
                    self.model.head_dimension,
                    self.model.rope_dimension_count,
                    position,
                    self.model.rope_freq_base,
                );

                const layer_base = layer_index * self.model.context_length * self.model.kv_dimension;
                const kv_offset_elements = layer_base + position * self.model.kv_dimension;

                try self.runProjection(layer.attn_k, self.normed, self.k);
                try metal_backend.applyRoPE(
                    self.backend,
                    self.k,
                    self.model.head_count_kv,
                    self.model.head_dimension,
                    self.model.rope_dimension_count,
                    position,
                    self.model.rope_freq_base,
                );
                try metal_backend.storeKvHalf(
                    self.backend,
                    self.k,
                    self.k_cache,
                    kv_offset_elements,
                    self.model.kv_dimension,
                );

                try self.runProjection(layer.attn_v, self.normed, self.v);
                try metal_backend.storeKvHalf(
                    self.backend,
                    self.v,
                    self.v_cache,
                    kv_offset_elements,
                    self.model.kv_dimension,
                );

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

                try self.runProjectionAdd(layer.attn_output, self.attn, self.hidden);
            }

            for (layers) |layer| {
                try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);
                try self.runProjection(layer.ffn_gate, self.normed, self.gate);
                try self.runProjection(layer.ffn_up, self.normed, self.up);
                try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
                try self.runProjectionAdd(layer.ffn_down, self.gate, self.hidden);
            }

            try self.runRmsNorm(self.model.output_norm, self.hidden, self.normed);
            try self.runProjection(self.model.output, self.normed, self.tmp);

            const logits_offset = i * self.model.vocab_size;
            try metal_backend.copyBufferRegion(
                self.backend,
                self.tmp,
                0,
                self.batch_logits,
                logits_offset * @sizeOf(f32),
                self.model.vocab_size * @sizeOf(f32),
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
