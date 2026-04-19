const std = @import("std");

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

pub const LinearAttnDesc = struct {
    in_proj_qkv: TensorDesc,
    in_proj_z: TensorDesc,
    in_proj_b: TensorDesc,
    in_proj_a: TensorDesc,
    conv1d: TensorDesc,
    dt_bias: TensorDesc,
    A_log: TensorDesc,
    norm_weight: TensorDesc,
    out_proj: TensorDesc,
};

pub const LayerDesc = struct {
    attn_norm: TensorDesc,
    attn_q: ?TensorDesc = null,
    attn_q_bias: ?TensorDesc = null,
    attn_q_norm: ?TensorDesc = null,
    attn_k: ?TensorDesc = null,
    attn_k_bias: ?TensorDesc = null,
    attn_k_norm: ?TensorDesc = null,
    attn_v: ?TensorDesc = null,
    attn_v_bias: ?TensorDesc = null,
    attn_output: ?TensorDesc = null,
    ffn_norm: TensorDesc,
    ffn_gate: TensorDesc,
    ffn_down: TensorDesc,
    ffn_up: TensorDesc,
    post_attention_norm: ?TensorDesc = null,
    post_ffw_norm: ?TensorDesc = null,
    linear_attn: ?LinearAttnDesc = null,
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
    key_head_dimension: usize,
    value_head_dimension: usize,
    q_projection_size: usize,
    kv_projection_size: usize,
    kv_dimension: usize,
    rope_freq_base: f32,
    rope_scaling_factor: f32 = 0.0,
    sliding_window: usize = 0,
    attn_logit_softcapping: ?f32 = null,
    final_logit_softcapping: ?f32 = null,
    global_attention_interval: usize = 0,
    use_gelu_ffn: bool = false,
    embedding_scale: f32 = 1.0,
    rms_norm_weight_offset: f32 = 0.0,
    vocab_size: usize,
    rms_norm_eps: f32,
    token_embd_offset: u64,
    rope_style: u32,
    linear_num_key_heads: u32 = 16,
    linear_num_value_heads: u32 = 16,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
};

pub const ShortlistEntry = struct {
    token_id: u32,
    logit: f32,
};

pub const max_shortlist_len: usize = 64;
pub const max_draft_len: usize = 4;
