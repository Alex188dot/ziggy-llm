const types = @import("../../types.zig");
const llama = @import("../../../model/loader.zig");
const metal_backend = @import("../../metal_backend.zig");

pub const OffloadPolicy = struct {
    offloaded_attention_layers: usize,
    offloaded_moe_ffn_layers: usize,

    pub fn forModel(model: *const llama.Model, gpu_layers: types.GpuLayers, device_info: ?metal_backend.DeviceInfo) OffloadPolicy {
        if (!model.is_qwen35_moe) {
            return .{
                .offloaded_attention_layers = model.layers.len,
                .offloaded_moe_ffn_layers = model.layers.len,
            };
        }

        const moe_ffn_layers = resolveMoeFfnLayers(model, gpu_layers, device_info);
        return .{
            .offloaded_attention_layers = model.layers.len,
            .offloaded_moe_ffn_layers = moe_ffn_layers,
        };
    }

    pub fn offloadsAttention(self: OffloadPolicy, layer_index: usize) bool {
        return layer_index < self.offloaded_attention_layers;
    }

    pub fn offloadsMoeFfn(self: OffloadPolicy, has_moe: bool, layer_index: usize) bool {
        return has_moe and layer_index < self.offloaded_moe_ffn_layers;
    }
};

fn resolveMoeFfnLayers(model: *const llama.Model, gpu_layers: types.GpuLayers, device_info: ?metal_backend.DeviceInfo) usize {
    return switch (gpu_layers) {
        .auto => blk: {
            const di = device_info orelse break :blk 4;
            if (di.recommended_max_working_set_size == 0) break :blk 4;
            if (!di.has_unified_memory) break :blk 4;

            const layers = model.layers.len;
            if (layers == 0) break :blk 0;

            const runtime_ctx = @min(model.context_length, 8192);
            const kv_bytes: u64 = @intCast(model.block_count * runtime_ctx * model.kv_projection_size * @sizeOf(u16) * 2);
            const scratch: u64 = 512 * 1024 * 1024;
            const fixed = kv_bytes + scratch;
            if (fixed >= di.recommended_max_working_set_size) break :blk 4;

            const available = di.recommended_max_working_set_size - fixed;

            const total_weight_bytes = if (model.mapped_bytes) |m| m.len else model.bytes.len;
            const per_layer = (total_weight_bytes + layers - 1) / layers;
            if (per_layer == 0) break :blk layers;

            const weight_budget = available * 4 / 5;
            break :blk @max(1, @min(layers, weight_budget / per_layer));
        },
        .all => model.layers.len,
        .count => |count| @min(model.layers.len, count),
    };
}
