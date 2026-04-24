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

            const total_gb = di.recommended_max_working_set_size / (1024 * 1024 * 1024);
            if (total_gb >= 96) break :blk model.layers.len;
            if (total_gb >= 48) break :blk @min(model.layers.len, model.layers.len * 3 / 4);
            if (total_gb >= 24) break :blk @min(model.layers.len, model.layers.len / 2);
            if (total_gb >= 16) break :blk @min(model.layers.len, model.layers.len / 3);
            if (total_gb >= 12) break :blk @min(model.layers.len, model.layers.len / 4);
            break :blk @min(model.layers.len, model.layers.len / 5);
        },
        .all => model.layers.len,
        .count => |count| @min(model.layers.len, count),
    };
}
