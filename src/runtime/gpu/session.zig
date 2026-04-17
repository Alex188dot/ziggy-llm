const std = @import("std");
const backend_api = @import("../backend.zig");
const metal_backend = @import("../metal_backend.zig");
const metal_profile = @import("../metal_profile.zig");
const gpu_types = @import("types.zig");

pub const DenseLookup = gpu_types.DenseLookup;
pub const TensorDesc = gpu_types.TensorDesc;
pub const LayerDesc = gpu_types.LayerDesc;
pub const LinearAttnDesc = gpu_types.LinearAttnDesc;
pub const ModelDesc = gpu_types.ModelDesc;
pub const ShortlistEntry = gpu_types.ShortlistEntry;
pub const max_shortlist_len = gpu_types.max_shortlist_len;
pub const max_draft_len = gpu_types.max_draft_len;

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
    linear_qkv: metal_backend.BufferHandle,
    linear_z: metal_backend.BufferHandle,
    linear_a: metal_backend.BufferHandle,
    linear_b: metal_backend.BufferHandle,
    linear_g: metal_backend.BufferHandle,
    linear_conv_tmp: metal_backend.BufferHandle,
    linear_conv_state: metal_backend.BufferHandle,
    linear_recurrent_state: metal_backend.BufferHandle,

    pub fn init(
        backend: backend_api.MatVecBackend,
        dense_lookup: DenseLookup,
        model: ModelDesc,
        profiler: ?*metal_profile.Profiler,
    ) !Session {
        const max_input = @max(model.embedding_length, model.feed_forward_length);
        const max_vec = @max(model.embedding_length, @max(model.feed_forward_length, model.vocab_size));
        const cache_len = model.block_count * model.context_length * model.kv_projection_size;
        const hidden = try metal_backend.createScratchBuffer(backend, max_input);
        errdefer metal_backend.destroyBuffer(hidden);
        const normed = try metal_backend.createScratchBuffer(backend, model.embedding_length);
        errdefer metal_backend.destroyBuffer(normed);
        const q = try metal_backend.createScratchBuffer(backend, model.q_projection_size);
        errdefer metal_backend.destroyBuffer(q);
        const k = try metal_backend.createScratchBuffer(backend, model.kv_projection_size);
        errdefer metal_backend.destroyBuffer(k);
        const v = try metal_backend.createScratchBuffer(backend, model.kv_projection_size);
        errdefer metal_backend.destroyBuffer(v);
        const attn = try metal_backend.createScratchBuffer(backend, model.q_projection_size);
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

        const linear_num_v_heads = model.linear_num_value_heads;
        const linear_value_dim_per_head = model.linear_value_head_dim;
        const linear_key_dim_per_head = model.linear_key_head_dim;
        const conv_state_per_layer = linear_num_v_heads * linear_value_dim_per_head * model.linear_conv_kernel_dim;
        const recurrent_state_per_layer = linear_num_v_heads * model.linear_key_head_dim * model.linear_value_head_dim;
        const linear_qkv_dim = 2 * model.linear_num_key_heads * linear_key_dim_per_head + linear_num_v_heads * linear_value_dim_per_head;
        const linear_z_dim = linear_num_v_heads * linear_value_dim_per_head;

        const linear_qkv = try metal_backend.createScratchBuffer(backend, linear_qkv_dim);
        errdefer metal_backend.destroyBuffer(linear_qkv);
        const linear_z = try metal_backend.createScratchBuffer(backend, linear_z_dim);
        errdefer metal_backend.destroyBuffer(linear_z);
        const linear_a = try metal_backend.createScratchBuffer(backend, model.linear_num_key_heads);
        errdefer metal_backend.destroyBuffer(linear_a);
        const linear_b = try metal_backend.createScratchBuffer(backend, model.linear_num_key_heads);
        errdefer metal_backend.destroyBuffer(linear_b);
        const linear_g = try metal_backend.createScratchBuffer(backend, model.linear_num_key_heads);
        errdefer metal_backend.destroyBuffer(linear_g);
        const linear_conv_tmp = try metal_backend.createScratchBuffer(backend, linear_z_dim);
        errdefer metal_backend.destroyBuffer(linear_conv_tmp);
        const linear_conv_state = try metal_backend.createScratchBuffer(backend, model.block_count * conv_state_per_layer);
        errdefer metal_backend.destroyBuffer(linear_conv_state);
        const linear_recurrent_state = try metal_backend.createScratchBuffer(backend, model.block_count * recurrent_state_per_layer);
        errdefer metal_backend.destroyBuffer(linear_recurrent_state);

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
            .linear_qkv = linear_qkv,
            .linear_z = linear_z,
            .linear_a = linear_a,
            .linear_b = linear_b,
            .linear_g = linear_g,
            .linear_conv_tmp = linear_conv_tmp,
            .linear_conv_state = linear_conv_state,
            .linear_recurrent_state = linear_recurrent_state,
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
        metal_backend.destroyBuffer(self.linear_qkv);
        metal_backend.destroyBuffer(self.linear_z);
        metal_backend.destroyBuffer(self.linear_a);
        metal_backend.destroyBuffer(self.linear_b);
        metal_backend.destroyBuffer(self.linear_g);
        metal_backend.destroyBuffer(self.linear_conv_tmp);
        metal_backend.destroyBuffer(self.linear_conv_state);
        metal_backend.destroyBuffer(self.linear_recurrent_state);
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
        if (layer.linear_attn) |la| {
            try self.runLinearAttention(layer, la, layer_index);
            return;
        }
        if (layer.attn_q == null) {
            return;
        }
        try self.runRmsNorm(layer.attn_norm, self.hidden, self.normed);
        try self.runProjection(layer.attn_q.?, self.normed, self.q);
        if (layer.attn_q_bias) |b| try self.runBiasAdd(b, self.q);
        if (layer.attn_q_norm) |n| try self.runRmsNormPerHead(n, self.q, self.q, self.model.head_count, self.model.rope_dimension_count);

        const q_rope_start = std.time.nanoTimestamp();
        try metal_backend.applyRoPE(
            self.backend,
            self.q,
            self.model.head_count,
            self.model.rope_dimension_count,
            self.model.rope_dimension_count,
            position,
            self.model.rope_freq_base,
            self.model.rope_style,
        );
        self.recordCategoryWithShape(.elementwise_ops, q_rope_start, .{
            .rows = self.model.head_count,
            .cols = self.model.rope_dimension_count,
            .depth = self.model.rope_dimension_count,
            .extra = position + 1,
        });

        const layer_base = layer_index * self.model.context_length * self.model.kv_projection_size;
        const kv_offset_elements = layer_base + position * self.model.kv_projection_size;

        const kv_k_start = std.time.nanoTimestamp();
        try self.runProjection(layer.attn_k.?, self.normed, self.k);
        if (layer.attn_k_bias) |b| try self.runBiasAdd(b, self.k);
        if (layer.attn_k_norm) |n| try self.runRmsNormPerHead(n, self.k, self.k, self.model.head_count_kv, self.model.rope_dimension_count);
        try metal_backend.applyRoPE(
            self.backend,
            self.k,
            self.model.head_count_kv,
            self.model.rope_dimension_count,
            self.model.rope_dimension_count,
            position,
            self.model.rope_freq_base,
            self.model.rope_style,
        );
        try metal_backend.storeKvHalf(
            self.backend,
            self.k,
            self.k_cache,
            kv_offset_elements,
            self.model.kv_projection_size,
        );
        self.recordCategoryWithShape(.elementwise_ops, kv_k_start, .{
            .rows = self.model.head_count_kv,
            .cols = self.model.rope_dimension_count,
            .depth = self.model.rope_dimension_count,
            .extra = position + 1,
        });
        self.recordCategoryWithShape(.kv_writes, kv_k_start, .{
            .rows = 1,
            .cols = self.model.kv_projection_size,
            .depth = layer_index,
            .extra = position + 1,
        });

        const kv_v_start = std.time.nanoTimestamp();
        try self.runProjection(layer.attn_v.?, self.normed, self.v);
        if (layer.attn_v_bias) |b| try self.runBiasAdd(b, self.v);
        try metal_backend.storeKvHalf(
            self.backend,
            self.v,
            self.v_cache,
            kv_offset_elements,
            self.model.kv_projection_size,
        );
        self.recordCategoryWithShape(.kv_writes, kv_v_start, .{
            .rows = 1,
            .cols = self.model.kv_projection_size,
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
            self.model.rope_dimension_count,
            self.model.kv_projection_size,
            self.model.context_length,
            position,
            layer_base,
            @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(self.model.rope_dimension_count))),
        );
        self.recordCategoryWithShape(.attention, attention_start, .{
            .rows = self.model.head_count,
            .cols = self.model.rope_dimension_count,
            .depth = position + 1,
            .extra = self.model.head_count_kv,
        });
        try self.runProjectionAdd(layer.attn_output.?, self.attn, self.hidden);
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

    fn runLinearAttention(
        self: *Session,
        layer: LayerDesc,
        la: gpu_types.LinearAttnDesc,
        layer_index: usize,
    ) !void {
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

        const conv_state_per_layer = num_value_heads * value_head_dim * kernel_dim;
        const recurrent_state_per_layer = num_value_heads * key_head_dim * value_head_dim;
        const conv_state_base = layer_index * conv_state_per_layer;
        const recurrent_state_base = layer_index * recurrent_state_per_layer;

        var cpu_qkv: [4096]f32 = undefined;
        var cpu_z: [4096]f32 = undefined;
        var cpu_a: [256]f32 = undefined;
        var cpu_g: [256]f32 = undefined;
        var cpu_conv_tmp: [4096]f32 = undefined;
        var cpu_conv_state: [4096]f32 = undefined;
        var cpu_recurrent_state: [4096]f32 = undefined;

        try metal_backend.readBufferF32(self.linear_qkv, cpu_qkv[0..qkv_dim]);
        try metal_backend.readBufferF32(self.linear_z, cpu_z[0..v_dim]);
        try metal_backend.readBufferF32(self.linear_conv_state, cpu_conv_state[0 .. model.block_count * conv_state_per_layer]);
        try metal_backend.readBufferF32(self.linear_recurrent_state, cpu_recurrent_state[0 .. model.block_count * recurrent_state_per_layer]);

        try self.runRmsNorm(layer.attn_norm, self.hidden, self.normed);

        try self.runProjection(la.in_proj_qkv, self.normed, self.linear_qkv);
        try metal_backend.readBufferF32(self.linear_qkv, cpu_qkv[0..qkv_dim]);

        try self.runProjectionToDst(la.in_proj_z, self.normed, self.linear_z, 0);
        try metal_backend.readBufferF32(self.linear_z, cpu_z[0..v_dim]);

        try self.runProjectionToDst(la.in_proj_a, self.linear_qkv, self.linear_a, 0);
        try metal_backend.readBufferF32(self.linear_a, cpu_a[0..num_heads]);

        var dt_bias_data: [256]f32 = undefined;
        try self.readDenseTensor(la.dt_bias, dt_bias_data[0..num_heads]);
        var a_log_data: [256]f32 = undefined;
        try self.readDenseTensor(la.A_log, a_log_data[0..num_heads]);

        var h: usize = 0;
        while (h < num_heads) : (h += 1) {
            const a_val = cpu_a[h];
            const dt_bias_val = dt_bias_data[h];
            const a_log_val = a_log_data[h];
            const softplus_val = if (a_val + dt_bias_val > 20.0) a_val + dt_bias_val else std.math.log1p(@exp(a_val + dt_bias_val));
            cpu_g[h] = -@exp(a_log_val) * softplus_val;
        }

        @memset(cpu_conv_tmp[0..v_dim], 0);

        var head_idx: usize = 0;
        while (head_idx < num_value_heads) : (head_idx += 1) {
            const v_head_base = head_idx * value_head_dim;
            const conv_head_base = head_idx * value_head_dim;
            const conv_head_out = cpu_conv_tmp[conv_head_base..][0..value_head_dim];
            const v_head = cpu_qkv[q_dim + q_dim + v_head_base ..][0..value_head_dim];

            var offset: usize = 0;
            while (offset < value_head_dim) : (offset += 1) {
                var kernel_idx: usize = 0;
                while (kernel_idx < kernel_dim - 1) : (kernel_idx += 1) {
                    const state_idx = kernel_idx * value_head_dim + offset;
                    const state_offset = conv_state_base + head_idx * value_head_dim * kernel_dim + state_idx;
                    const w_idx = kernel_idx * value_head_dim * value_head_dim + offset * value_head_dim + offset;
                    var w_val: f32 = undefined;
                    try self.readScalarFromTensor(la.conv1d, w_idx, &w_val);
                    const state_val = cpu_conv_state[state_offset];
                    conv_head_out[offset] += state_val * w_val;
                }

                const w_idx = (kernel_dim - 1) * value_head_dim * value_head_dim + offset * value_head_dim + offset;
                var w_val: f32 = undefined;
                try self.readScalarFromTensor(la.conv1d, w_idx, &w_val);
                conv_head_out[offset] += v_head[offset] * w_val;
            }
        }

        var shift_offset: usize = 0;
        while (shift_offset < value_head_dim * (kernel_dim - 1)) : (shift_offset += value_head_dim) {
            const src_offset = conv_state_base + shift_offset;
            const dst_offset = conv_state_base + value_head_dim + shift_offset;
            @memcpy(cpu_conv_state[dst_offset..][0..value_head_dim], cpu_conv_state[src_offset..][0..value_head_dim]);
        }
        @memcpy(cpu_conv_state[conv_state_base..][0..value_head_dim], cpu_qkv[q_dim + q_dim ..][0..value_head_dim]);

        var g_idx: usize = 0;
        while (g_idx < num_heads) : (g_idx += 1) {
            const g_val = cpu_g[g_idx];
            const recurrent_offset = recurrent_state_base + g_idx * key_head_dim * value_head_dim;
            const conv_head_offset = g_idx * value_head_dim;

            var j: usize = 0;
            while (j < key_head_dim * value_head_dim) : (j += 1) {
                const conv_idx = conv_head_offset + j;
                cpu_recurrent_state[recurrent_offset + j] = g_val * cpu_recurrent_state[recurrent_offset + j] + (1 - g_val) * cpu_conv_tmp[conv_idx];
            }
        }

        var norm_weight_data: [4096]f32 = undefined;
        try self.readDenseTensor(la.norm_weight, norm_weight_data[0..v_dim]);
        var i: usize = 0;
        while (i < v_dim) : (i += 1) {
            cpu_z[i] *= norm_weight_data[i];
        }

        try metal_backend.writeBufferF32(self.linear_z, cpu_z[0..v_dim]);
        try metal_backend.writeBufferF32(self.linear_conv_state, cpu_conv_state[0 .. model.block_count * conv_state_per_layer]);
        try metal_backend.writeBufferF32(self.linear_recurrent_state, cpu_recurrent_state[0 .. model.block_count * recurrent_state_per_layer]);

        try self.runProjectionAdd(la.out_proj, self.linear_z, self.hidden);
    }

    fn readDenseTensor(self: *Session, tensor: TensorDesc, out: []f32) !void {
        const weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
        @memcpy(out, weights);
    }

    fn readScalarFromTensor(self: *Session, tensor: TensorDesc, index: usize, out: *f32) !void {
        const weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
        out.* = weights[index];
    }

    fn runBiasAdd(self: *Session, tensor: TensorDesc, target: metal_backend.BufferHandle) !void {
        const start = std.time.nanoTimestamp();
        const bias_weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
        try metal_backend.addBiasF32(self.backend, target, bias_weights, tensor.cols);
        self.recordCategoryWithShape(.elementwise_ops, start, .{
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
                    self.model.rope_dimension_count,
                    self.model.rope_dimension_count,
                    position,
                    self.model.rope_freq_base,
                    self.model.rope_style,
                );

                const layer_base = layer_index * self.model.context_length * self.model.kv_projection_size;
                const kv_offset_elements = layer_base + position * self.model.kv_projection_size;

                try self.runProjection(layer.attn_k, self.normed, self.k);
                try metal_backend.applyRoPE(
                    self.backend,
                    self.k,
                    self.model.head_count_kv,
                    self.model.rope_dimension_count,
                    self.model.rope_dimension_count,
                    position,
                    self.model.rope_freq_base,
                    self.model.rope_style,
                );
                try metal_backend.storeKvHalf(
                    self.backend,
                    self.k,
                    self.k_cache,
                    kv_offset_elements,
                    self.model.kv_projection_size,
                );

                try self.runProjection(layer.attn_v, self.normed, self.v);
                try metal_backend.storeKvHalf(
                    self.backend,
                    self.v,
                    self.v_cache,
                    kv_offset_elements,
                    self.model.kv_projection_size,
                );

                try metal_backend.attentionFused(
                    self.backend,
                    self.q,
                    self.k_cache,
                    self.v_cache,
                    self.attn,
                    self.model.head_count,
                    self.model.head_count_kv,
                    self.model.rope_dimension_count,
                    self.model.kv_projection_size,
                    self.model.context_length,
                    position,
                    layer_base,
                    @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(self.model.rope_dimension_count))),
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
