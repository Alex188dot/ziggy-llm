const std = @import("std");
const backend_api = @import("../backend.zig");
const metal_backend = @import("../metal_backend.zig");
const metal_profile = @import("../metal_profile.zig");
const qwen35_linear_common = @import("../qwen35_linear_common.zig");
const gpu_types = @import("types.zig");
pub const DenseLookup = gpu_types.DenseLookup;
pub const TensorDesc = gpu_types.TensorDesc;
pub const LayerDesc = gpu_types.LayerDesc;
pub const LinearAttnDesc = gpu_types.LinearAttnDesc;
pub const MoeDesc = gpu_types.MoeDesc;
pub const ModelDesc = gpu_types.ModelDesc;
pub const ShortlistEntry = gpu_types.ShortlistEntry;
pub const max_shortlist_len = gpu_types.max_shortlist_len;
pub const max_draft_len = gpu_types.max_draft_len;

const tensor_type_f32: u32 = 0;
const tensor_type_f16: u32 = 1;
const tensor_type_q8_0: u32 = 8;
const tensor_type_q3_k: u32 = 11;
const tensor_type_q4_k: u32 = 12;
const tensor_type_q5_k: u32 = 13;
const tensor_type_q6_k: u32 = 14;
const tensor_type_iq3_xxs: u32 = 18;
const tensor_type_iq4_xs: u32 = 23;

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
        q_gate: metal_backend.BufferHandle,
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
    host_q_values: []f32,
    host_gate_values: []f32,
    host_attn_values: []f32,
    host_linear_qkv: []f32,
    host_linear_z: []f32,
    host_linear_a: []f32,
    host_linear_b: []f32,
    host_linear_g: []f32,
    host_linear_conv_tmp: []f32,
    host_linear_conv_state: []f32,
    host_linear_recurrent_state: []f32,
    host_linear_conv_weights: []f32,

    pub fn init(
        backend: backend_api.MatVecBackend,
        dense_lookup: DenseLookup,
        model: ModelDesc,
        profiler: ?*metal_profile.Profiler,
    ) !Session {
        const allocator = std.heap.page_allocator;
        const max_input = @max(model.embedding_length, model.feed_forward_length);
        const max_vec = @max(model.embedding_length, @max(model.feed_forward_length, model.vocab_size));
        const cache_len = model.block_count * model.context_length * model.kv_projection_size;
        const gate_capacity = @max(model.feed_forward_length, model.q_projection_size);
        const up_capacity = @max(model.feed_forward_length, model.q_projection_size * 2);
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
        const q_gate = try metal_backend.createScratchBuffer(backend, model.q_projection_size);
        errdefer metal_backend.destroyBuffer(q_gate);
        const gate = try metal_backend.createScratchBuffer(backend, gate_capacity);
        errdefer metal_backend.destroyBuffer(gate);
        const up = try metal_backend.createScratchBuffer(backend, up_capacity);
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

        const linear_num_k_heads: usize = @intCast(model.linear_num_key_heads);
        const linear_num_v_heads: usize = @intCast(model.linear_num_value_heads);
        const linear_value_dim_per_head: usize = @intCast(model.linear_value_head_dim);
        const linear_key_dim_per_head: usize = @intCast(model.linear_key_head_dim);
        const linear_qkv_dim = 2 * linear_num_k_heads * linear_key_dim_per_head + linear_num_v_heads * linear_value_dim_per_head;
        const linear_z_dim = linear_num_v_heads * linear_value_dim_per_head;
        const conv_state_per_layer = linear_qkv_dim * (@as(usize, @intCast(model.linear_conv_kernel_dim)) - 1);
        const recurrent_state_per_layer = linear_num_v_heads * linear_key_dim_per_head * linear_value_dim_per_head;

        const linear_qkv = try metal_backend.createScratchBuffer(backend, linear_qkv_dim);
        errdefer metal_backend.destroyBuffer(linear_qkv);
        const linear_z = try metal_backend.createScratchBuffer(backend, linear_z_dim);
        errdefer metal_backend.destroyBuffer(linear_z);
        const linear_a = try metal_backend.createScratchBuffer(backend, linear_num_v_heads);
        errdefer metal_backend.destroyBuffer(linear_a);
        const linear_b = try metal_backend.createScratchBuffer(backend, linear_num_v_heads);
        errdefer metal_backend.destroyBuffer(linear_b);
        const linear_g = try metal_backend.createScratchBuffer(backend, linear_num_v_heads);
        errdefer metal_backend.destroyBuffer(linear_g);
        const linear_conv_tmp = try metal_backend.createScratchBuffer(backend, linear_qkv_dim);
        errdefer metal_backend.destroyBuffer(linear_conv_tmp);
        const linear_conv_state = try metal_backend.createScratchBuffer(backend, model.block_count * conv_state_per_layer);
        errdefer metal_backend.destroyBuffer(linear_conv_state);
        const linear_recurrent_state = try metal_backend.createScratchBuffer(backend, model.block_count * recurrent_state_per_layer);
        errdefer metal_backend.destroyBuffer(linear_recurrent_state);
        const host_q_values = try allocator.alloc(f32, model.q_projection_size);
        errdefer allocator.free(host_q_values);
        const host_gate_values = try allocator.alloc(f32, model.q_projection_size);
        errdefer allocator.free(host_gate_values);
        const host_attn_values = try allocator.alloc(f32, model.q_projection_size);
        errdefer allocator.free(host_attn_values);
        const host_linear_qkv = try allocator.alloc(f32, linear_qkv_dim);
        errdefer allocator.free(host_linear_qkv);
        const host_linear_z = try allocator.alloc(f32, linear_z_dim);
        errdefer allocator.free(host_linear_z);
        const host_linear_a = try allocator.alloc(f32, linear_num_v_heads);
        errdefer allocator.free(host_linear_a);
        const host_linear_b = try allocator.alloc(f32, linear_num_v_heads);
        errdefer allocator.free(host_linear_b);
        const host_linear_g = try allocator.alloc(f32, linear_num_v_heads);
        errdefer allocator.free(host_linear_g);
        const host_linear_conv_tmp = try allocator.alloc(f32, linear_qkv_dim);
        errdefer allocator.free(host_linear_conv_tmp);
        const host_linear_conv_state = try allocator.alloc(f32, model.block_count * conv_state_per_layer);
        errdefer allocator.free(host_linear_conv_state);
        const host_linear_recurrent_state = try allocator.alloc(f32, model.block_count * recurrent_state_per_layer);
        errdefer allocator.free(host_linear_recurrent_state);
        const max_conv_weights = linear_qkv_dim * @as(usize, @intCast(model.linear_conv_kernel_dim));
        const host_linear_conv_weights = try allocator.alloc(f32, max_conv_weights);
        errdefer allocator.free(host_linear_conv_weights);
        @memset(host_linear_conv_state, 0);
        @memset(host_linear_recurrent_state, 0);
        try metal_backend.writeBufferF32(linear_conv_state, host_linear_conv_state);
        try metal_backend.writeBufferF32(linear_recurrent_state, host_linear_recurrent_state);

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
            .q_gate = q_gate,
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
            .host_q_values = host_q_values,
            .host_gate_values = host_gate_values,
            .host_attn_values = host_attn_values,
            .host_linear_qkv = host_linear_qkv,
            .host_linear_z = host_linear_z,
            .host_linear_a = host_linear_a,
            .host_linear_b = host_linear_b,
            .host_linear_g = host_linear_g,
            .host_linear_conv_tmp = host_linear_conv_tmp,
            .host_linear_conv_state = host_linear_conv_state,
            .host_linear_recurrent_state = host_linear_recurrent_state,
            .host_linear_conv_weights = host_linear_conv_weights,
        };
    }

    pub fn deinit(self: *Session) void {
        const allocator = std.heap.page_allocator;
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
        allocator.free(self.host_q_values);
        allocator.free(self.host_gate_values);
        allocator.free(self.host_attn_values);
        allocator.free(self.host_linear_qkv);
        allocator.free(self.host_linear_z);
        allocator.free(self.host_linear_a);
        allocator.free(self.host_linear_b);
        allocator.free(self.host_linear_g);
        allocator.free(self.host_linear_conv_tmp);
        allocator.free(self.host_linear_conv_state);
        allocator.free(self.host_linear_recurrent_state);
        allocator.free(self.host_linear_conv_weights);
        self.* = undefined;
    }

    pub fn beginToken(self: *Session, input: []const f32) !void {
        try metal_backend.writeBufferF32(self.hidden, input);
        try metal_backend.beginSequence(self.backend);
    }

    pub fn canRunAttentionBlock(self: *const Session, layer: LayerDesc) bool {
        if (!self.hasTensor(layer.attn_norm.offset)) return false;
        if (layer.linear_attn) |la| {
            return self.hasTensor(la.in_proj_qkv.offset) and
                self.hasTensor(la.in_proj_z.offset) and
                self.hasTensor(la.in_proj_b.offset) and
                self.hasTensor(la.in_proj_a.offset) and
                self.hasTensor(la.conv1d.offset) and
                self.hasTensor(la.dt_bias.offset) and
                self.hasTensor(la.A_log.offset) and
                self.hasTensor(la.norm_weight.offset) and
                self.hasTensor(la.out_proj.offset);
        }
        const attn_q = layer.attn_q orelse return false;
        const attn_k = layer.attn_k orelse return false;
        const attn_v = layer.attn_v orelse return false;
        return self.hasTensor(attn_q.offset) and
            self.hasOptionalTensor(layer.attn_q_bias) and
            self.hasOptionalTensor(layer.attn_q_norm) and
            self.hasTensor(attn_k.offset) and
            self.hasOptionalTensor(layer.attn_k_bias) and
            self.hasOptionalTensor(layer.attn_k_norm) and
            self.hasTensor(attn_v.offset) and
            self.hasOptionalTensor(layer.attn_v_bias) and
            self.hasOptionalTensor(layer.attn_output) and
            self.hasOptionalTensor(layer.post_attention_norm);
    }

    pub fn canRunMoeFfnBlock(self: *const Session, layer: LayerDesc) bool {
        const moe = layer.moe orelse return false;
        if (layer.post_ffw_norm != null) return false;
        if (self.model.expert_count == 0 or self.model.expert_used_count == 0) return false;
        if (self.model.expert_used_count > max_shortlist_len) return false;
        if (self.model.expert_feed_forward_length == 0) return false;
        if (!self.hasTensor(layer.ffn_norm.offset) or
            !self.hasTensor(layer.ffn_gate.offset) or
            !self.hasTensor(layer.ffn_up.offset) or
            !self.hasTensor(layer.ffn_down.offset) or
            !self.hasTensor(moe.router.offset) or
            !self.hasTensor(moe.gate_exps.offset) or
            !self.hasTensor(moe.up_exps.offset) or
            !self.hasTensor(moe.down_exps.offset) or
            !self.hasOptionalTensor(moe.shared_router_gate))
        {
            return false;
        }
        if (!supportsProjectionTensor(layer.ffn_gate) or
            !supportsProjectionTensor(layer.ffn_up) or
            !supportsProjectionTensor(layer.ffn_down) or
            !supportsProjectionTensor(moe.router))
        {
            return false;
        }
        if (moe.shared_router_gate) |gate_tensor| {
            if (!supportsProjectionTensor(gate_tensor) or gate_tensor.rows != 1) return false;
        }
        if (!supportsIndexedProjection(moe.gate_exps) or
            !supportsIndexedProjection(moe.up_exps) or
            !supportsIndexedProjection(moe.down_exps))
        {
            return false;
        }
        if (moe.gate_exps.tensor_type != moe.up_exps.tensor_type) return false;
        return true;
    }

    pub fn readHidden(self: *Session, out: []f32) !void {
        if (out.len < self.model.embedding_length) return error.InvalidTensorMetadata;
        try self.readBufferF32Committed(self.hidden, out[0..self.model.embedding_length]);
    }

    pub fn writeHidden(self: *Session, values: []const f32) !void {
        if (values.len < self.model.embedding_length) return error.InvalidTensorMetadata;
        try metal_backend.writeBufferF32(self.hidden, values[0..self.model.embedding_length]);
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
        const attn_q = layer.attn_q.?;
        var q_gate: ?metal_backend.BufferHandle = null;
        if (attn_q.rows == self.model.q_projection_size * 2) {
            try self.runProjectionWithOptionalBias(attn_q, layer.attn_q_bias, self.normed, self.up);
            try metal_backend.splitPackedQ(
                self.backend,
                self.up,
                self.q,
                self.q_gate,
                @intCast(self.model.head_count),
                @intCast(self.model.key_head_dimension),
            );
            q_gate = self.q_gate;
        } else {
            try self.runProjectionWithOptionalBias(attn_q, layer.attn_q_bias, self.normed, self.q);
        }
        if (layer.attn_q_norm) |n| try self.runRmsNormPerHead(n, self.q, self.q, self.model.head_count, self.model.key_head_dimension);

        const q_rope_start = std.time.nanoTimestamp();
        try self.applyRoPEBuffer(self.q, self.host_q_values, self.model.head_count, self.model.key_head_dimension, position);
        self.recordCategoryWithShape(.elementwise_ops, q_rope_start, .{
            .rows = self.model.head_count,
            .cols = self.rotaryDimension(),
            .depth = self.rotaryDimension(),
            .extra = position + 1,
        });

        const layer_base = layer_index * self.model.context_length * self.model.kv_projection_size;
        const kv_offset_elements = layer_base + position * self.model.kv_projection_size;

        const kv_k_start = std.time.nanoTimestamp();
        const k_fused = if (layer.attn_k_norm == null)
            try self.tryRunProjectionStoreKv(
                layer.attn_k.?,
                layer.attn_k_bias,
                self.normed,
                self.k_cache,
                kv_offset_elements,
                self.model.head_count_kv,
                self.model.key_head_dimension,
                self.rotaryDimension(),
                position,
                true,
            )
        else
            false;
        if (!k_fused) {
            try self.runProjectionWithOptionalBias(layer.attn_k.?, layer.attn_k_bias, self.normed, self.k);
            if (layer.attn_k_norm) |n| try self.runRmsNormPerHead(n, self.k, self.k, self.model.head_count_kv, self.model.key_head_dimension);
            try self.applyRoPEBuffer(self.k, self.host_attn_values, self.model.head_count_kv, self.model.key_head_dimension, position);
            try metal_backend.storeKvHalf(
                self.backend,
                self.k,
                self.k_cache,
                kv_offset_elements,
                self.model.kv_projection_size,
            );
        }
        self.recordCategoryWithShape(.elementwise_ops, kv_k_start, .{
            .rows = self.model.head_count_kv,
            .cols = self.rotaryDimension(),
            .depth = self.rotaryDimension(),
            .extra = position + 1,
        });
        self.recordCategoryWithShape(.kv_writes, kv_k_start, .{
            .rows = 1,
            .cols = self.model.kv_projection_size,
            .depth = layer_index,
            .extra = position + 1,
        });

        const kv_v_start = std.time.nanoTimestamp();
        const v_fused = try self.tryRunProjectionStoreKv(
            layer.attn_v.?,
            layer.attn_v_bias,
            self.normed,
            self.v_cache,
            kv_offset_elements,
            self.model.head_count_kv,
            self.model.key_head_dimension,
            self.rotaryDimension(),
            position,
            false,
        );
        if (!v_fused) {
            try self.runProjectionWithOptionalBias(layer.attn_v.?, layer.attn_v_bias, self.normed, self.v);
            try metal_backend.storeKvHalf(
                self.backend,
                self.v,
                self.v_cache,
                kv_offset_elements,
                self.model.kv_projection_size,
            );
        }
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
            self.model.key_head_dimension,
            self.model.kv_projection_size,
            self.model.context_length,
            position,
            layer_base,
            self.attentionWindowStart(layer_index, position),
            @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(self.model.key_head_dimension))),
            self.model.attn_logit_softcapping,
        );
        self.recordCategoryWithShape(.attention, attention_start, .{
            .rows = self.model.head_count,
            .cols = self.model.rope_dimension_count,
            .depth = position + 1,
            .extra = self.model.head_count_kv,
        });
        if (q_gate) |gate_buf| {
            try metal_backend.sigmoidMulGate(
                self.backend,
                self.attn,
                gate_buf,
                @intCast(self.model.head_count * self.model.key_head_dimension),
            );
        }
        if (layer.post_attention_norm) |n| {
            try self.runProjection(layer.attn_output.?, self.attn, self.tmp);
            try self.runRmsNorm(n, self.tmp, self.tmp);
            try metal_backend.addInPlace(self.backend, self.hidden, self.tmp, self.model.embedding_length);
        } else {
            try self.runProjectionAdd(layer.attn_output.?, self.attn, self.hidden);
        }
    }

    pub fn runFfnBlock(self: *Session, layer: LayerDesc) !void {
        try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);
        if (!try self.tryRunDualProjection(layer.ffn_gate, layer.ffn_up, self.normed, self.gate, self.up)) {
            try self.runProjection(layer.ffn_gate, self.normed, self.gate);
            try self.runProjection(layer.ffn_up, self.normed, self.up);
        }

        const tensor = layer.ffn_down;
        var handled_fused = false;

        if (tensor.tensor_type == 12 and layer.post_ffw_norm == null and !self.model.use_gelu_ffn) {
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
            if (self.model.use_gelu_ffn) {
                try metal_backend.geluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
            } else {
                try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
            }
            self.recordCategoryWithShape(.elementwise_ops, silu_start, .{
                .rows = 1,
                .cols = self.model.feed_forward_length,
                .depth = 2,
            });
            if (layer.post_ffw_norm) |n| {
                try self.runProjection(layer.ffn_down, self.gate, self.tmp);
                try self.runRmsNorm(n, self.tmp, self.tmp);
                try metal_backend.addInPlace(self.backend, self.hidden, self.tmp, self.model.embedding_length);
            } else {
                try self.runProjectionAdd(layer.ffn_down, self.gate, self.hidden);
            }
        }
    }

    pub fn runMoeFfnBlock(self: *Session, layer: LayerDesc) !void {
        const moe = layer.moe orelse return error.InvalidTensorMetadata;
        if (!self.canRunMoeFfnBlock(layer)) return error.InvalidTensorMetadata;

        try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);

        const routing_start = std.time.nanoTimestamp();
        try self.runProjection(moe.router, self.normed, self.tmp);
        try metal_backend.topKShortlist(
            self.backend,
            self.tmp,
            self.shortlist_entries,
            self.model.expert_count,
            self.model.expert_used_count,
        );
        try metal_backend.normalizeTopKShortlist(
            self.backend,
            self.shortlist_entries,
            self.model.expert_used_count,
            self.model.expert_gating_func == 1 or self.model.expert_gating_func == 2,
            self.model.expert_weights_norm,
            self.model.expert_weights_scale,
        );
        self.recordCategoryWithShape(.output_reduce, routing_start, .{
            .rows = 2,
            .cols = self.model.expert_used_count,
            .depth = self.model.expert_count,
        });

        for (0..self.model.expert_used_count) |slot_idx| {
            try self.runDualIndexedProjection(
                moe.gate_exps,
                moe.up_exps,
                self.normed,
                self.gate,
                self.up,
                slot_idx,
                self.model.expert_feed_forward_length,
            );

            const silu_start = std.time.nanoTimestamp();
            try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.expert_feed_forward_length);
            self.recordCategoryWithShape(.elementwise_ops, silu_start, .{
                .rows = 1,
                .cols = self.model.expert_feed_forward_length,
                .depth = 2,
            });

            const weighted_down_start = std.time.nanoTimestamp();
            try self.runIndexedProjectionAddWeighted(
                moe.down_exps,
                self.gate,
                self.hidden,
                slot_idx,
                self.model.embedding_length,
            );
            self.recordCategoryWithShape(.elementwise_ops, weighted_down_start, .{
                .rows = self.model.embedding_length,
                .cols = moe.down_exps.cols,
                .depth = slot_idx + 1,
                .tensor_type = moe.down_exps.tensor_type,
            });
        }

        if (!try self.tryRunDualProjection(layer.ffn_gate, layer.ffn_up, self.normed, self.gate, self.up)) {
            try self.runProjection(layer.ffn_gate, self.normed, self.gate);
            try self.runProjection(layer.ffn_up, self.normed, self.up);
        }

        const shared_act_start = std.time.nanoTimestamp();
        if (self.model.use_gelu_ffn) {
            try metal_backend.geluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
        } else {
            try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
        }
        self.recordCategoryWithShape(.elementwise_ops, shared_act_start, .{
            .rows = 1,
            .cols = self.model.feed_forward_length,
            .depth = 2,
        });

        try self.runProjection(layer.ffn_down, self.gate, self.tmp);
        if (moe.shared_router_gate) |gate_tensor| {
            try self.runProjection(gate_tensor, self.normed, self.sampled_token);
            const shared_gate_start = std.time.nanoTimestamp();
            try metal_backend.sigmoidScaleAdd(
                self.backend,
                self.hidden,
                self.tmp,
                self.sampled_token,
                self.model.embedding_length,
            );
            self.recordCategoryWithShape(.elementwise_ops, shared_gate_start, .{
                .rows = 1,
                .cols = self.model.embedding_length,
                .depth = 1,
            });
        } else {
            try metal_backend.addInPlace(self.backend, self.hidden, self.tmp, self.model.embedding_length);
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
        if (self.model.final_logit_softcapping) |cap| softcapInPlace(out, cap);
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
        var use_fused_argmax = false;
        switch (tensor.tensor_type) {
            tensor_type_q4_k => {
                if (self.dense_lookup.getMoonQuant(tensor.offset) == null) {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try self.initFusedArgmaxState();
                    try metal_backend.runMatVecQ4KArgmaxToBuffer(self.backend, matrix, self.normed, self.sampled_token_packed, tensor.rows, tensor.cols);
                    use_fused_argmax = true;
                }
            },
            tensor_type_q6_k => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try self.initFusedArgmaxState();
                try metal_backend.runMatVecQ6KArgmaxToBuffer(self.backend, matrix, self.normed, self.sampled_token_packed, tensor.rows, tensor.cols);
                use_fused_argmax = true;
            },
            tensor_type_q8_0 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try self.initFusedArgmaxState();
                try metal_backend.runMatVecQ8_0ArgmaxToBuffer(self.backend, matrix, self.normed, self.sampled_token_packed, tensor.rows, tensor.cols);
                use_fused_argmax = true;
            },
            tensor_type_q3_k => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try self.initFusedArgmaxState();
                try metal_backend.runMatVecQ3KArgmaxToBuffer(self.backend, matrix, self.normed, self.sampled_token_packed, tensor.rows, tensor.cols);
                use_fused_argmax = true;
            },
            else => {},
        }
        if (!use_fused_argmax) {
            try self.runProjection(tensor, self.normed, self.tmp);
            try metal_backend.argmax(self.backend, self.tmp, self.sampled_token, self.model.vocab_size);
        }
        self.recordCategoryWithShape(.output_reduce, output_reduce_start, shape);
        try self.commitOutputSequence(shape);
        const host_readback_start = std.time.nanoTimestamp();
        var token: [1]u32 = .{0};
        if (use_fused_argmax) {
            var packed_state: [2]u32 = .{ 0, 0 };
            try self.readBufferU32Committed(self.sampled_token_packed, &packed_state);
            token[0] = packed_state[1];
        } else {
            try self.readBufferU32Committed(self.sampled_token, &token);
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
        try self.readBufferU32Committed(self.sampled_token, &token);
        self.recordCategoryWithShape(.host_readback, host_readback_start, shape);
        return token[0];
    }

    fn attentionWindowStart(self: *const Session, layer_index: usize, position: usize) usize {
        if (!self.layerUsesSlidingWindow(layer_index)) return 0;
        const window = self.model.sliding_window;
        return (position + 1) - @min(position + 1, window);
    }

    fn rotaryDimension(self: *const Session) usize {
        return @min(
            @as(usize, @intFromFloat(@as(f32, @floatFromInt(self.model.rope_dimension_count)) * self.model.partial_rotary_factor)),
            self.model.key_head_dimension,
        );
    }

    fn layerUsesSlidingWindow(self: *const Session, layer_index: usize) bool {
        if (self.model.sliding_window == 0) return false;
        if (self.model.global_attention_interval == 0) return true;
        return ((layer_index + 1) % self.model.global_attention_interval) != 0;
    }

    fn hasOptionalTensor(self: *const Session, tensor: ?TensorDesc) bool {
        return if (tensor) |value| self.hasTensor(value.offset) else true;
    }

    fn supportsProjectionTensor(tensor: TensorDesc) bool {
        return switch (tensor.tensor_type) {
            tensor_type_f32,
            tensor_type_f16,
            tensor_type_q8_0,
            tensor_type_q3_k,
            tensor_type_q4_k,
            tensor_type_q6_k,
            tensor_type_q5_k,
            => true,
            else => false,
        };
    }

    fn supportsIndexedProjection(tensor: TensorDesc) bool {
        return switch (tensor.tensor_type) {
            tensor_type_q3_k,
            tensor_type_iq3_xxs,
            tensor_type_iq4_xs,
            => true,
            else => false,
        };
    }

    fn hasTensor(self: *const Session, offset: u64) bool {
        return self.dense_lookup.getDense(offset) != null or
            self.dense_lookup.getRaw(offset) != null or
            self.dense_lookup.getMoonQuant(offset) != null;
    }

    fn flushSequenceForHostAccess(self: *Session) !void {
        try metal_backend.commitSequence(self.backend);
    }

    fn readBufferF32Committed(self: *Session, buffer: metal_backend.BufferHandle, out: []f32) !void {
        try self.flushSequenceForHostAccess();
        try metal_backend.readBufferF32(buffer, out);
    }

    fn readBufferU32Committed(self: *Session, buffer: metal_backend.BufferHandle, out: []u32) !void {
        try self.flushSequenceForHostAccess();
        try metal_backend.readBufferU32(buffer, out);
    }

    fn initFusedArgmaxState(self: *Session) !void {
        var packed_state: [2]u32 = .{ orderedFloatBits(-std.math.inf(f32)), 0 };
        try metal_backend.writeBufferU32(self.sampled_token_packed, &packed_state);
    }

    fn applyRoPEBuffer(
        self: *Session,
        buffer: metal_backend.BufferHandle,
        _: []f32,
        head_count: usize,
        head_dim: usize,
        position: usize,
    ) !void {
        const rope_dim = self.rotaryDimension();
        try metal_backend.applyRoPE(
            self.backend,
            buffer,
            head_count,
            rope_dim,
            head_dim,
            position,
            self.model.rope_freq_base,
            self.model.rope_style,
        );
    }

    fn softcapInPlace(values: []f32, cap: f32) void {
        if (!(cap > 0)) return;
        for (values) |*value| value.* = std.math.tanh(value.* / cap) * cap;
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
            self.model.rms_norm_weight_offset,
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
        try self.runRmsNormPerHeadWithOffset(tensor, input, output, head_count, head_dim, self.model.rms_norm_weight_offset);
    }

    fn runRmsNormPerHeadWithOffset(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
        head_count: usize,
        head_dim: usize,
        weight_offset: f32,
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
            weight_offset,
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
            11 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ3KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
            },
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    used_moon_quant = true;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                }
            },
            13 => {
                if (self.dense_lookup.getRaw(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecQ5KToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                } else {
                    const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
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

    fn runProjectionWithOptionalBias(
        self: *Session,
        tensor: TensorDesc,
        bias: ?TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
    ) !void {
        if (bias == null) {
            try self.runProjection(tensor, input, output);
            return;
        }

        const bias_tensor = bias.?;
        const bias_weights = self.dense_lookup.getDense(bias_tensor.offset) orelse return error.InvalidTensorMetadata;
        const start = std.time.nanoTimestamp();

        switch (tensor.tensor_type) {
            tensor_type_f32 => {
                const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecWithBiasToBuffer(self.backend, matrix, input, output, bias_weights, tensor.rows, tensor.cols);
            },
            tensor_type_q3_k => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ3KWithBiasToBuffer(self.backend, matrix, input, output, bias_weights, tensor.rows, tensor.cols);
            },
            tensor_type_q4_k => {
                if (self.dense_lookup.getMoonQuant(tensor.offset) != null) {
                    try self.runProjection(tensor, input, output);
                    try self.runBiasAdd(bias_tensor, output);
                    return;
                }
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ4KWithBiasToBuffer(self.backend, matrix, input, output, bias_weights, tensor.rows, tensor.cols);
            },
            tensor_type_q5_k => {
                if (self.dense_lookup.getRaw(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecQ5KWithBiasToBuffer(self.backend, matrix, input, output, bias_weights, tensor.rows, tensor.cols);
                } else {
                    const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecWithBiasToBuffer(self.backend, matrix, input, output, bias_weights, tensor.rows, tensor.cols);
                }
            },
            tensor_type_q6_k => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ6KWithBiasToBuffer(self.backend, matrix, input, output, bias_weights, tensor.rows, tensor.cols);
            },
            tensor_type_q8_0 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ8_0WithBiasToBuffer(self.backend, matrix, input, output, bias_weights, tensor.rows, tensor.cols);
            },
            else => {
                try self.runProjection(tensor, input, output);
                try self.runBiasAdd(bias_tensor, output);
                return;
            },
        }

        const shape = metal_profile.ShapeDesc{
            .rows = tensor.rows,
            .cols = tensor.cols,
            .tensor_type = tensor.tensor_type,
        };
        self.recordCategoryWithShape(.projections, start, shape);
    }

    fn tryRunDualProjection(
        self: *Session,
        tensor_a: TensorDesc,
        tensor_b: TensorDesc,
        input: metal_backend.BufferHandle,
        output_a: metal_backend.BufferHandle,
        output_b: metal_backend.BufferHandle,
    ) !bool {
        if (tensor_a.rows != tensor_b.rows or tensor_a.cols != tensor_b.cols or tensor_a.tensor_type != tensor_b.tensor_type) return false;

        const start = std.time.nanoTimestamp();
        switch (tensor_a.tensor_type) {
            tensor_type_f32 => {
                const matrix_a = self.dense_lookup.getDense(tensor_a.offset) orelse return error.InvalidTensorMetadata;
                const matrix_b = self.dense_lookup.getDense(tensor_b.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runDualMatVecToBuffers(self.backend, matrix_a, matrix_b, input, output_a, output_b, tensor_a.rows, tensor_a.cols);
            },
            tensor_type_q3_k => {
                const matrix_a = self.dense_lookup.getRaw(tensor_a.offset) orelse return error.InvalidTensorMetadata;
                const matrix_b = self.dense_lookup.getRaw(tensor_b.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runDualMatVecQ3KToBuffers(self.backend, matrix_a, matrix_b, input, output_a, output_b, tensor_a.rows, tensor_a.cols);
            },
            tensor_type_q4_k => {
                if (self.dense_lookup.getMoonQuant(tensor_a.offset) != null or self.dense_lookup.getMoonQuant(tensor_b.offset) != null) return false;
                const matrix_a = self.dense_lookup.getRaw(tensor_a.offset) orelse return error.InvalidTensorMetadata;
                const matrix_b = self.dense_lookup.getRaw(tensor_b.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runDualMatVecQ4KToBuffers(self.backend, matrix_a, matrix_b, input, output_a, output_b, tensor_a.rows, tensor_a.cols);
            },
            tensor_type_q5_k => {
                if (self.dense_lookup.getRaw(tensor_a.offset)) |matrix_a| {
                    const matrix_b = self.dense_lookup.getRaw(tensor_b.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runDualMatVecQ5KToBuffers(self.backend, matrix_a, matrix_b, input, output_a, output_b, tensor_a.rows, tensor_a.cols);
                } else {
                    const matrix_a = self.dense_lookup.getDense(tensor_a.offset) orelse return error.InvalidTensorMetadata;
                    const matrix_b = self.dense_lookup.getDense(tensor_b.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runDualMatVecToBuffers(self.backend, matrix_a, matrix_b, input, output_a, output_b, tensor_a.rows, tensor_a.cols);
                }
            },
            tensor_type_q6_k => {
                const matrix_a = self.dense_lookup.getRaw(tensor_a.offset) orelse return error.InvalidTensorMetadata;
                const matrix_b = self.dense_lookup.getRaw(tensor_b.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runDualMatVecQ6KToBuffers(self.backend, matrix_a, matrix_b, input, output_a, output_b, tensor_a.rows, tensor_a.cols);
            },
            tensor_type_q8_0 => {
                const matrix_a = self.dense_lookup.getRaw(tensor_a.offset) orelse return error.InvalidTensorMetadata;
                const matrix_b = self.dense_lookup.getRaw(tensor_b.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runDualMatVecQ8_0ToBuffers(self.backend, matrix_a, matrix_b, input, output_a, output_b, tensor_a.rows, tensor_a.cols);
            },
            else => return false,
        }

        self.recordCategoryWithShape(.projections, start, .{
            .rows = tensor_a.rows,
            .cols = tensor_a.cols,
            .tensor_type = tensor_a.tensor_type,
            .extra = 2,
        });
        return true;
    }

    fn tryRunProjectionStoreKv(
        self: *Session,
        tensor: TensorDesc,
        bias: ?TensorDesc,
        input: metal_backend.BufferHandle,
        dst: metal_backend.BufferHandle,
        dst_offset_elements: usize,
        head_count: usize,
        head_dim: usize,
        rope_dim: usize,
        position: usize,
        apply_rope: bool,
    ) !bool {
        const bias_weights: ?[]const f32 = if (bias) |b| (self.dense_lookup.getDense(b.offset) orelse return error.InvalidTensorMetadata) else null;
        const start = std.time.nanoTimestamp();

        switch (tensor.tensor_type) {
            tensor_type_f32 => {
                const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecStoreKvHalf(self.backend, matrix, input, bias_weights, dst, dst_offset_elements, tensor.rows, tensor.cols, head_count, head_dim, rope_dim, position, self.model.rope_freq_base, self.model.rope_style, apply_rope);
            },
            tensor_type_q3_k => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ3KStoreKvHalf(self.backend, matrix, input, bias_weights, dst, dst_offset_elements, tensor.rows, tensor.cols, head_count, head_dim, rope_dim, position, self.model.rope_freq_base, self.model.rope_style, apply_rope);
            },
            tensor_type_q4_k => {
                if (self.dense_lookup.getMoonQuant(tensor.offset) != null) return false;
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ4KStoreKvHalf(self.backend, matrix, input, bias_weights, dst, dst_offset_elements, tensor.rows, tensor.cols, head_count, head_dim, rope_dim, position, self.model.rope_freq_base, self.model.rope_style, apply_rope);
            },
            tensor_type_q5_k => {
                if (self.dense_lookup.getRaw(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecQ5KStoreKvHalf(self.backend, matrix, input, bias_weights, dst, dst_offset_elements, tensor.rows, tensor.cols, head_count, head_dim, rope_dim, position, self.model.rope_freq_base, self.model.rope_style, apply_rope);
                } else {
                    const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecStoreKvHalf(self.backend, matrix, input, bias_weights, dst, dst_offset_elements, tensor.rows, tensor.cols, head_count, head_dim, rope_dim, position, self.model.rope_freq_base, self.model.rope_style, apply_rope);
                }
            },
            tensor_type_q6_k => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ6KStoreKvHalf(self.backend, matrix, input, bias_weights, dst, dst_offset_elements, tensor.rows, tensor.cols, head_count, head_dim, rope_dim, position, self.model.rope_freq_base, self.model.rope_style, apply_rope);
            },
            tensor_type_q8_0 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ8_0StoreKvHalf(self.backend, matrix, input, bias_weights, dst, dst_offset_elements, tensor.rows, tensor.cols, head_count, head_dim, rope_dim, position, self.model.rope_freq_base, self.model.rope_style, apply_rope);
            },
            else => return false,
        }

        self.recordCategoryWithShape(.projections, start, .{
            .rows = tensor.rows,
            .cols = tensor.cols,
            .tensor_type = tensor.tensor_type,
            .extra = if (apply_rope) 2 else 1,
        });
        return true;
    }

    fn runIndexedProjection(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
        slot_idx: usize,
        rows_per_expert: usize,
    ) !void {
        const start = std.time.nanoTimestamp();
        const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
        switch (tensor.tensor_type) {
            tensor_type_q3_k => try metal_backend.indexedMatvecQ3K(
                self.backend,
                matrix,
                input,
                output,
                rows_per_expert,
                tensor.cols,
                self.shortlist_entries,
                slot_idx,
                rows_per_expert,
            ),
            tensor_type_iq3_xxs => try metal_backend.indexedMatvecIQ3XXS(
                self.backend,
                matrix,
                input,
                output,
                rows_per_expert,
                tensor.cols,
                self.shortlist_entries,
                slot_idx,
                rows_per_expert,
            ),
            tensor_type_iq4_xs => try metal_backend.indexedMatvecIQ4XS(
                self.backend,
                matrix,
                input,
                output,
                rows_per_expert,
                tensor.cols,
                self.shortlist_entries,
                slot_idx,
                rows_per_expert,
            ),
            else => return error.UnsupportedTensorType,
        }
        self.recordCategoryWithShape(.projections, start, .{
            .rows = rows_per_expert,
            .cols = tensor.cols,
            .tensor_type = tensor.tensor_type,
            .extra = slot_idx,
        });
    }

    fn runDualIndexedProjection(
        self: *Session,
        tensor_a: TensorDesc,
        tensor_b: TensorDesc,
        input: metal_backend.BufferHandle,
        output_a: metal_backend.BufferHandle,
        output_b: metal_backend.BufferHandle,
        slot_idx: usize,
        rows_per_expert: usize,
    ) !void {
        const matrix_a = self.dense_lookup.getRaw(tensor_a.offset) orelse return error.InvalidTensorMetadata;
        const matrix_b = self.dense_lookup.getRaw(tensor_b.offset) orelse return error.InvalidTensorMetadata;
        const start = std.time.nanoTimestamp();
        if (tensor_a.tensor_type == tensor_type_q3_k and tensor_b.tensor_type == tensor_type_q3_k) {
            try metal_backend.dualIndexedMatvecQ3K(
                self.backend,
                matrix_a,
                matrix_b,
                input,
                output_a,
                output_b,
                rows_per_expert,
                tensor_a.cols,
                self.shortlist_entries,
                slot_idx,
                rows_per_expert,
            );
        } else if (tensor_a.tensor_type == tensor_type_iq3_xxs and tensor_b.tensor_type == tensor_type_iq3_xxs) {
            try metal_backend.dualIndexedMatvecIQ3XXS(
                self.backend,
                matrix_a,
                matrix_b,
                input,
                output_a,
                output_b,
                rows_per_expert,
                tensor_a.cols,
                self.shortlist_entries,
                slot_idx,
                rows_per_expert,
            );
        } else {
            return error.UnsupportedTensorType;
        }
        const shape = metal_profile.ShapeDesc{
            .rows = rows_per_expert,
            .cols = tensor_a.cols,
            .tensor_type = tensor_a.tensor_type,
            .extra = slot_idx,
        };
        self.recordCategoryWithShape(.projections, start, shape);
    }

    fn runIndexedProjectionAddWeighted(
        self: *Session,
        tensor: TensorDesc,
        input: metal_backend.BufferHandle,
        output: metal_backend.BufferHandle,
        slot_idx: usize,
        rows_per_expert: usize,
    ) !void {
        const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
        const start = std.time.nanoTimestamp();
        switch (tensor.tensor_type) {
            tensor_type_q3_k => try metal_backend.indexedMatvecQ3KAddWeighted(
                self.backend,
                matrix,
                input,
                output,
                rows_per_expert,
                tensor.cols,
                self.shortlist_entries,
                slot_idx,
                rows_per_expert,
            ),
            tensor_type_iq4_xs => try metal_backend.indexedMatvecIQ4XSAddWeighted(
                self.backend,
                matrix,
                input,
                output,
                rows_per_expert,
                tensor.cols,
                self.shortlist_entries,
                slot_idx,
                rows_per_expert,
            ),
            else => return error.UnsupportedTensorType,
        }
        self.recordCategoryWithShape(.projections, start, .{
            .rows = rows_per_expert,
            .cols = tensor.cols,
            .tensor_type = tensor.tensor_type,
            .extra = slot_idx,
        });
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
            11 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ3KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
            },
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                    used_moon_quant = true;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                }
            },
            13 => {
                if (self.dense_lookup.getRaw(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecQ5KToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
                } else {
                    const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecToDstBuffer(self.backend, matrix, input, output, output_offset_bytes, tensor.rows, tensor.cols);
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
            11 => {
                const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                try metal_backend.runMatVecQ3KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
            },
            12 => {
                if (self.dense_lookup.getMoonQuant(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecMoonQuantQ4KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                    used_moon_quant = true;
                } else {
                    const matrix = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecQ4KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                }
            },
            13 => {
                if (self.dense_lookup.getRaw(tensor.offset)) |matrix| {
                    try metal_backend.runMatVecQ5KAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
                } else {
                    const matrix = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                    try metal_backend.runMatVecAddToBuffer(self.backend, matrix, input, output, tensor.rows, tensor.cols);
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

    fn loadTensorF32(self: *Session, tensor: TensorDesc, out: []f32) !void {
        const len = tensor.rows * tensor.cols;
        if (out.len < len) return error.InvalidTensorMetadata;
        switch (tensor.tensor_type) {
            0 => {
                const weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                if (weights.len < len) return error.InvalidTensorMetadata;
                @memcpy(out[0..len], weights[0..len]);
            },
            1 => {
                const bytes = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                if (bytes.len < len * 2) return error.InvalidTensorMetadata;
                for (0..len) |i| {
                    const raw = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
                    out[i] = @floatCast(@as(f16, @bitCast(raw)));
                }
            },
            else => return error.UnsupportedTensorType,
        }
    }

    fn runLinearAttention(
        self: *Session,
        layer: LayerDesc,
        la: gpu_types.LinearAttnDesc,
        layer_index: usize,
    ) !void {
        const model = self.model;
        const num_key_heads: usize = @intCast(model.linear_num_key_heads);
        const num_value_heads: usize = @intCast(model.linear_num_value_heads);
        const key_head_dim: usize = @intCast(model.linear_key_head_dim);
        const value_head_dim: usize = @intCast(model.linear_value_head_dim);
        const kernel_dim: usize = @intCast(model.linear_conv_kernel_dim);
        const q_dim = num_key_heads * key_head_dim;
        const v_dim = num_value_heads * value_head_dim;
        const qkv_dim = q_dim + q_dim + v_dim;

        try self.runRmsNorm(layer.attn_norm, self.hidden, self.normed);
        try self.runProjection(la.in_proj_qkv, self.normed, self.linear_qkv);
        try self.runProjectionToDst(la.in_proj_z, self.normed, self.linear_z, 0);
        try self.runProjectionToDst(la.in_proj_a, self.normed, self.linear_a, 0);
        try self.runProjectionToDst(la.in_proj_b, self.normed, self.linear_b, 0);

        if (model.is_qwen35_text) {
            // GPU-only path for Qwen3.5 dense: zero CPU readbacks
            const conv_weights = self.dense_lookup.getDense(la.conv1d.offset) orelse return error.InvalidTensorMetadata;
            const dt_bias_weights = self.dense_lookup.getDense(la.dt_bias.offset) orelse return error.InvalidTensorMetadata;
            const a_log_weights = self.dense_lookup.getDense(la.A_log.offset) orelse return error.InvalidTensorMetadata;
            const norm_weights = self.dense_lookup.getDense(la.norm_weight.offset) orelse return error.InvalidTensorMetadata;

            const conv_buffer = try metal_backend.getOrCreateBufferForF32(self.backend, conv_weights);
            const dt_bias_buffer = try metal_backend.getOrCreateBufferForF32(self.backend, dt_bias_weights);
            const a_log_buffer = try metal_backend.getOrCreateBufferForF32(self.backend, a_log_weights);
            const norm_buffer = try metal_backend.getOrCreateBufferForF32(self.backend, norm_weights);

            const scale = @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(key_head_dim)));

            const conv_start = std.time.nanoTimestamp();
            try metal_backend.linearConv1dF32(
                self.backend,
                self.linear_qkv,
                self.linear_conv_state,
                conv_buffer,
                self.linear_conv_tmp,
                @intCast(layer_index),
                @intCast(model.block_count),
                @intCast(kernel_dim),
                @intCast(qkv_dim),
            );
            self.recordCategoryWithShape(.elementwise_ops, conv_start, .{
                .rows = 1,
                .cols = qkv_dim,
                .depth = kernel_dim,
            });

            const recurrent_start = std.time.nanoTimestamp();
            try metal_backend.linearRecurrentNormF32(
                self.backend,
                self.linear_conv_tmp,
                self.linear_recurrent_state,
                self.linear_z,
                self.linear_a,
                self.linear_b,
                dt_bias_buffer,
                a_log_buffer,
                norm_buffer,
                self.linear_qkv,
                @intCast(layer_index),
                @intCast(num_key_heads),
                @intCast(num_value_heads),
                @intCast(key_head_dim),
                @intCast(value_head_dim),
                @intCast(qkv_dim),
                model.rms_norm_eps,
                scale,
            );
            self.recordCategoryWithShape(.elementwise_ops, recurrent_start, .{
                .rows = num_value_heads,
                .cols = value_head_dim,
                .depth = key_head_dim,
            });

            try self.runProjectionAdd(la.out_proj, self.linear_qkv, self.hidden);
            return;
        }

        // CPU fallback path for MoE and other models
        const conv_state_per_layer = qkv_dim * (kernel_dim - 1);
        const recurrent_state_per_layer = num_value_heads * key_head_dim * value_head_dim;
        const conv_state_base = layer_index * conv_state_per_layer;
        const recurrent_state_base = layer_index * recurrent_state_per_layer;
        const conv_state = self.host_linear_conv_state[conv_state_base..][0..conv_state_per_layer];
        const recurrent_state = self.host_linear_recurrent_state[recurrent_state_base..][0..recurrent_state_per_layer];
        const qkv = self.host_linear_qkv[0..qkv_dim];
        const z = self.host_linear_z[0..v_dim];
        const alpha = self.host_linear_a[0..num_value_heads];
        const beta = self.host_linear_b[0..num_value_heads];
        const gate = self.host_linear_g[0..num_value_heads];
        const conv_out = self.host_linear_conv_tmp[0..qkv_dim];

        try self.flushSequenceForHostAccess();
        try metal_backend.readBufferF32(self.linear_qkv, qkv);
        try metal_backend.readBufferF32(self.linear_z, z);
        try metal_backend.readBufferF32(self.linear_a, alpha);
        try metal_backend.readBufferF32(self.linear_b, beta);

        if (num_value_heads > 128) return error.InvalidTensorMetadata;
        var dt_bias_vals: [128]f32 = undefined;
        var a_log_vals: [128]f32 = undefined;
        try self.loadTensorF32(la.dt_bias, dt_bias_vals[0..num_value_heads]);
        try self.loadTensorF32(la.A_log, a_log_vals[0..num_value_heads]);

        const conv1d_len = qkv_dim * kernel_dim;
        const conv1d_vals = self.host_linear_conv_weights[0..conv1d_len];
        try self.loadTensorF32(la.conv1d, conv1d_vals);

        for (0..num_value_heads) |h| {
            gate[h] = a_log_vals[h] * softplusScalar(alpha[h] + dt_bias_vals[h]);
            beta[h] = sigmoidScalar(beta[h]);
        }

        @memset(conv_out, 0);
        for (0..kernel_dim - 1) |kernel_idx| {
            const state_offset = kernel_idx * qkv_dim;
            const weight_offset = kernel_idx * qkv_dim;
            for (0..qkv_dim) |channel| {
                conv_out[channel] += conv_state[state_offset + channel] * conv1d_vals[weight_offset + channel];
            }
        }
        const final_weight_offset = (kernel_dim - 1) * qkv_dim;
        for (0..qkv_dim) |channel| {
            conv_out[channel] += qkv[channel] * conv1d_vals[final_weight_offset + channel];
            conv_out[channel] = siluScalar(conv_out[channel]);
        }
        if (kernel_dim > 1) {
            var state_idx: usize = 0;
            while (state_idx + qkv_dim < conv_state_per_layer) : (state_idx += qkv_dim) {
                const src_offset = state_idx + qkv_dim;
                std.mem.copyForwards(f32, conv_state[state_idx..][0..qkv_dim], conv_state[src_offset..][0..qkv_dim]);
            }
            const last_offset = (kernel_dim - 2) * qkv_dim;
            @memcpy(conv_state[last_offset..][0..qkv_dim], qkv);
        }

        const q_conv = conv_out[0..q_dim];
        const k_conv = conv_out[q_dim .. q_dim * 2];
        const v_conv = conv_out[q_dim * 2 ..][0..v_dim];
        l2NormalizePerHead(q_conv, num_key_heads, key_head_dim, model.rms_norm_eps);
        l2NormalizePerHead(k_conv, num_key_heads, key_head_dim, model.rms_norm_eps);

        const delta_out = self.host_q_values[0..v_dim];
        const scale = @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(key_head_dim)));
        if (value_head_dim > 256) return error.InvalidTensorMetadata;
        for (0..num_value_heads) |head| {
            const recurrent_offset = head * key_head_dim * value_head_dim;
            const qk_head = try qwen35_linear_common.qkHeadIndex(head, num_key_heads, num_value_heads);
            const q_head = q_conv[qk_head * key_head_dim ..][0..key_head_dim];
            const k_head = k_conv[qk_head * key_head_dim ..][0..key_head_dim];
            const v_head = v_conv[head * value_head_dim ..][0..value_head_dim];
            const out_head = delta_out[head * value_head_dim ..][0..value_head_dim];
            const state = recurrent_state[recurrent_offset..][0 .. key_head_dim * value_head_dim];
            const decay = @exp(gate[head]);

            for (0..state.len) |idx| state[idx] *= decay;

            var sk: [256]f32 = undefined;
            var delta_arr: [256]f32 = undefined;
            @memset(sk[0..value_head_dim], 0);

            for (0..key_head_dim) |row| {
                const k_row = k_head[row];
                const row_offset = row * value_head_dim;
                for (0..value_head_dim) |col| {
                    sk[col] += state[row_offset + col] * k_row;
                }
            }

            for (0..value_head_dim) |col| {
                delta_arr[col] = (v_head[col] - sk[col]) * beta[head];
            }

            for (0..key_head_dim) |row| {
                const k_row = k_head[row];
                const row_offset = row * value_head_dim;
                for (0..value_head_dim) |col| {
                    state[row_offset + col] += k_row * delta_arr[col];
                }
            }

            @memset(sk[0..value_head_dim], 0);
            for (0..key_head_dim) |row| {
                const q_row = q_head[row] * scale;
                const row_offset = row * value_head_dim;
                for (0..value_head_dim) |col| {
                    sk[col] += state[row_offset + col] * q_row;
                }
            }

            for (0..value_head_dim) |col| {
                out_head[col] = sk[col];
            }
        }

        const norm_weights = self.dense_lookup.getDense(la.norm_weight.offset) orelse return error.InvalidTensorMetadata;
        if (norm_weights.len < value_head_dim) return error.InvalidTensorMetadata;
        for (0..num_value_heads) |head| {
            const head_data = delta_out[head * value_head_dim ..][0..value_head_dim];
            var sum_sq: f32 = 0;
            for (head_data) |v| sum_sq += v * v;
            const scale_norm = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(value_head_dim)) + model.rms_norm_eps);
            for (head_data, 0..) |*v, i| {
                v.* *= scale_norm * norm_weights[i];
            }
        }
        for (0..v_dim) |idx| {
            delta_out[idx] *= siluScalar(z[idx]);
        }
        try self.writeHostSlice(self.linear_z, delta_out);
        try metal_backend.beginSequence(self.backend);
        try self.runProjectionAdd(la.out_proj, self.linear_z, self.hidden);
    }

    fn writeHostSlice(_: *Session, buffer: metal_backend.BufferHandle, values: []const f32) !void {
        try metal_backend.writeBufferF32(buffer, values);
    }

    fn readTensorValue(self: *Session, tensor: TensorDesc, index: usize) !f32 {
        return switch (tensor.tensor_type) {
            0 => blk: {
                const weights = self.dense_lookup.getDense(tensor.offset) orelse return error.InvalidTensorMetadata;
                if (index >= tensor.rows * tensor.cols) return error.InvalidTensorMetadata;
                const row = index / tensor.cols;
                const col = index % tensor.cols;
                break :blk weights[row + col * tensor.rows];
            },
            1 => blk: {
                const bytes = self.dense_lookup.getRaw(tensor.offset) orelse return error.InvalidTensorMetadata;
                const raw = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
                break :blk @floatCast(@as(f16, @bitCast(raw)));
            },
            else => error.UnsupportedTensorType,
        };
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

    fn orderedFloatBits(value: f32) u32 {
        const bits: u32 = @bitCast(value);
        return if ((bits & 0x8000_0000) != 0) ~bits else (bits | 0x8000_0000);
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
            if (self.model.embedding_scale != 1.0) {
                for (values[0..emb_rows]) |*value| value.* *= self.model.embedding_scale;
            }
            try metal_backend.writeBufferF32(self.hidden, values[0..emb_rows]);

            for (layers, 0..) |layer, layer_index| {
                try self.runRmsNorm(layer.attn_norm, self.hidden, self.normed);
                try self.runProjection(layer.attn_q.?, self.normed, self.q);

                try self.applyRoPEBuffer(self.q, self.host_q_values, self.model.head_count, self.model.key_head_dimension, position);

                const layer_base = layer_index * self.model.context_length * self.model.kv_projection_size;
                const kv_offset_elements = layer_base + position * self.model.kv_projection_size;

                if (!try self.tryRunProjectionStoreKv(layer.attn_k.?, null, self.normed, self.k_cache, kv_offset_elements, self.model.head_count_kv, self.model.key_head_dimension, self.rotaryDimension(), position, true)) {
                    try self.runProjection(layer.attn_k.?, self.normed, self.k);
                    try self.applyRoPEBuffer(self.k, self.host_attn_values, self.model.head_count_kv, self.model.key_head_dimension, position);
                    try metal_backend.storeKvHalf(
                        self.backend,
                        self.k,
                        self.k_cache,
                        kv_offset_elements,
                        self.model.kv_projection_size,
                    );
                }

                if (!try self.tryRunProjectionStoreKv(layer.attn_v.?, null, self.normed, self.v_cache, kv_offset_elements, self.model.head_count_kv, self.model.key_head_dimension, self.rotaryDimension(), position, false)) {
                    try self.runProjection(layer.attn_v.?, self.normed, self.v);
                    try metal_backend.storeKvHalf(
                        self.backend,
                        self.v,
                        self.v_cache,
                        kv_offset_elements,
                        self.model.kv_projection_size,
                    );
                }

                try metal_backend.attentionFused(
                    self.backend,
                    self.q,
                    self.k_cache,
                    self.v_cache,
                    self.attn,
                    self.model.head_count,
                    self.model.head_count_kv,
                    self.model.key_head_dimension,
                    self.model.kv_projection_size,
                    self.model.context_length,
                    position,
                    layer_base,
                    self.attentionWindowStart(layer_index, position),
                    @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(self.model.key_head_dimension))),
                    self.model.attn_logit_softcapping,
                );

                try self.runProjectionAdd(layer.attn_output.?, self.attn, self.hidden);
            }

            for (layers) |layer| {
                try self.runRmsNorm(layer.ffn_norm, self.hidden, self.normed);
                if (!try self.tryRunDualProjection(layer.ffn_gate, layer.ffn_up, self.normed, self.gate, self.up)) {
                    try self.runProjection(layer.ffn_gate, self.normed, self.gate);
                    try self.runProjection(layer.ffn_up, self.normed, self.up);
                }
                if (self.model.use_gelu_ffn) {
                    try metal_backend.geluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
                } else {
                    try metal_backend.siluMul(self.backend, self.gate, self.up, self.model.feed_forward_length);
                }
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

fn siluScalar(value: f32) f32 {
    return value / (1 + @exp(-value));
}

fn sigmoidScalar(value: f32) f32 {
    if (value >= 0) {
        const exp_neg = @exp(-value);
        return 1 / (1 + exp_neg);
    }
    const exp_pos = @exp(value);
    return exp_pos / (1 + exp_pos);
}

fn softplusScalar(value: f32) f32 {
    if (value > 0) return value + std.math.log1p(@exp(-value));
    return std.math.log1p(@exp(value));
}

fn l2NormalizePerHead(values: []f32, head_count: usize, head_dim: usize, eps: f32) void {
    for (0..head_count) |head_index| {
        const head = values[head_index * head_dim ..][0..head_dim];
        var norm_sq: f32 = 0;
        for (head) |value| norm_sq += value * value;
        const norm = @sqrt(norm_sq);
        const scale = @as(f32, 1.0) / @max(norm, eps);
        for (head) |*value| value.* *= scale;
    }
}

test "readTensorValue maps dense tensors from column-major storage to logical row-major order" {
    const weights = [_]f32{
        0, 10,
        1, 11,
        2, 12,
    };
    const DenseLookupTest = struct {
        fn getDense(_: ?*const anyopaque, _: u64) ?[]const f32 {
            return &weights;
        }

        fn getNoneU8(_: ?*const anyopaque, _: u64) ?[]const u8 {
            return null;
        }
    };

    var session: Session = undefined;
    session.dense_lookup = .{
        .ctx = null,
        .get_dense_fn = DenseLookupTest.getDense,
        .get_raw_fn = DenseLookupTest.getNoneU8,
        .get_moon_quant_fn = DenseLookupTest.getNoneU8,
    };

    const tensor = TensorDesc{
        .offset = 1,
        .rows = 2,
        .cols = 3,
        .tensor_type = 0,
    };

    try std.testing.expectEqual(@as(f32, 0), try session.readTensorValue(tensor, 0));
    try std.testing.expectEqual(@as(f32, 1), try session.readTensorValue(tensor, 1));
    try std.testing.expectEqual(@as(f32, 2), try session.readTensorValue(tensor, 2));
    try std.testing.expectEqual(@as(f32, 10), try session.readTensorValue(tensor, 3));
    try std.testing.expectEqual(@as(f32, 11), try session.readTensorValue(tensor, 4));
    try std.testing.expectEqual(@as(f32, 12), try session.readTensorValue(tensor, 5));
}

fn applyRoPEHost(values: []f32, head_count: usize, head_dim: usize, rope_dim: usize, position: usize, freq_base: f32, rope_style: u32, rope_sections: [4]u32) void {
    const n_rot = @min(rope_dim, head_dim);
    const pos_f32 = @as(f32, @floatFromInt(position));
    for (0..head_count) |head_index| {
        const head = values[head_index * head_dim ..][0..head_dim];
        switch (rope_style) {
            0 => applyInterleavedRoPE(head, n_rot, pos_f32, freq_base),
            1 => applyNeoxRoPE(head, n_rot, pos_f32, freq_base),
            2 => applyImrope(head, n_rot, pos_f32, freq_base, rope_sections),
            else => applyNeoxRoPE(head, n_rot, pos_f32, freq_base),
        }
    }
}

fn applyInterleavedRoPE(head: []f32, n_rot: usize, pos_f32: f32, freq_base: f32) void {
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
}

fn applyNeoxRoPE(head: []f32, n_rot: usize, pos_f32: f32, freq_base: f32) void {
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

fn applyImrope(head: []f32, n_rot: usize, pos_f32: f32, freq_base: f32, rope_sections: [4]u32) void {
    const section_count = @as(usize, rope_sections[0] + rope_sections[1] + rope_sections[2] + rope_sections[3]);
    if (section_count == 0) {
        applyNeoxRoPE(head, n_rot, pos_f32, freq_base);
        return;
    }

    const half_rot = n_rot / 2;
    const theta_scale = std.math.pow(f32, freq_base, -2.0 / @as(f32, @floatFromInt(n_rot)));
    const section_t = @as(usize, rope_sections[0]);
    const section_h = @as(usize, rope_sections[1]);
    const section_w = @as(usize, rope_sections[2]);

    var theta_t = pos_f32;
    var theta_h = pos_f32;
    var theta_w = pos_f32;
    var theta_e = pos_f32;

    for (0..half_rot) |i| {
        const sector = i % section_count;
        const theta = if (sector % 3 == 1 and sector < 3 * section_h)
            theta_h
        else if (sector % 3 == 2 and sector < 3 * section_w)
            theta_w
        else if (sector % 3 == 0 and sector < 3 * section_t)
            theta_t
        else
            theta_e;
        const cos_theta = @cos(theta);
        const sin_theta = @sin(theta);
        const x0 = head[i];
        const x1 = head[i + half_rot];
        head[i] = @mulAdd(f32, x0, cos_theta, -x1 * sin_theta);
        head[i + half_rot] = @mulAdd(f32, x0, sin_theta, x1 * cos_theta);

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }
}
