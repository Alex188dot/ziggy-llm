const types = @import("../../types.zig");
const llama = @import("../../../model/loader.zig");
const metal_backend = @import("../../metal_backend.zig");

const qwen35_moe_auto_gpu_layers: usize = 8;
const full_offload_safety_margin = 0.8;

pub const OffloadPolicy = struct {
    offloaded_attention_layers: usize,
    cpu_only_moe_ffn: bool,

    pub fn forModel(model: *const llama.Model, gpu_layers: types.GpuLayers, device_info: ?metal_backend.DeviceInfo) OffloadPolicy {
        if (!model.is_qwen35_moe) {
            return .{
                .offloaded_attention_layers = model.layers.len,
                .cpu_only_moe_ffn = false,
            };
        }

        return .{
            .offloaded_attention_layers = resolveQwen35MoeLayers(model, gpu_layers, device_info),
            .cpu_only_moe_ffn = true,
        };
    }

    pub fn offloadsAttention(self: OffloadPolicy, layer_index: usize) bool {
        return layer_index < self.offloaded_attention_layers;
    }

    pub fn offloadsMoeFfn(self: OffloadPolicy, has_moe: bool) bool {
        return has_moe and !self.cpu_only_moe_ffn;
    }
};

fn resolveQwen35MoeLayers(model: *const llama.Model, gpu_layers: types.GpuLayers, device_info: ?metal_backend.DeviceInfo) usize {
    return switch (gpu_layers) {
        .auto => if (canFullyOffloadAttention(model, device_info))
            model.layers.len
        else
            @min(model.layers.len, qwen35_moe_auto_gpu_layers),
        .all => model.layers.len,
        .count => |count| @min(model.layers.len, count),
    };
}

fn canFullyOffloadAttention(model: *const llama.Model, device_info: ?metal_backend.DeviceInfo) bool {
    const info = device_info orelse return false;
    if (!info.has_unified_memory or info.recommended_max_working_set_size == 0) return false;

    const estimated_bytes = estimateFullAttentionBytes(model);
    const budget = @as(f64, @floatFromInt(info.recommended_max_working_set_size)) * full_offload_safety_margin;
    return @as(f64, @floatFromInt(estimated_bytes)) <= budget;
}

fn estimateFullAttentionBytes(model: *const llama.Model) u64 {
    var total: u64 = estimateTensorResidentBytes(model.output);
    total += estimateTensorResidentBytes(model.output_norm);
    for (model.layers) |layer| {
        total += estimateTensorResidentBytes(layer.attn_norm);
        if (layer.attn_q) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.attn_q_bias) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.attn_q_norm) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.attn_k) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.attn_k_bias) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.attn_k_norm) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.attn_v) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.attn_v_bias) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.attn_output) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.post_attention_norm) |tensor| total += estimateTensorResidentBytes(tensor);
        if (layer.linear_attn) |linear_attn| {
            total += estimateTensorResidentBytes(linear_attn.in_proj_qkv);
            total += estimateTensorResidentBytes(linear_attn.in_proj_z);
            total += estimateTensorResidentBytes(linear_attn.in_proj_b);
            total += estimateTensorResidentBytes(linear_attn.in_proj_a);
            total += estimateTensorResidentBytes(linear_attn.conv1d);
            total += estimateTensorResidentBytes(linear_attn.dt_bias);
            total += estimateTensorResidentBytes(linear_attn.A_log);
            total += estimateTensorResidentBytes(linear_attn.norm_weight);
            total += estimateTensorResidentBytes(linear_attn.out_proj);
        }
    }
    return total;
}

fn estimateTensorResidentBytes(tensor: llama.TensorRef) u64 {
    const rows = tensor.rowCount() catch return 0;
    const cols = tensor.rowLen() catch return 0;
    return switch (tensor.tensor_type) {
        .q3_k, .q4_k, .q6_k, .q8_0, .iq3_xxs, .iq4_xs => @as(u64, @intCast(llama.tensorRowByteSize(tensor.tensor_type, cols) catch return 0)) * rows,
        else => @as(u64, @intCast(rows)) * @as(u64, @intCast(cols)) * @sizeOf(f32),
    };
}
