const types = @import("../../types.zig");
const llama = @import("../../../model/loader.zig");
const metal_backend = @import("../../metal_backend.zig");

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
            .cpu_only_moe_ffn = false,
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
    _ = device_info;
    return switch (gpu_layers) {
        .auto => model.layers.len,
        .all => model.layers.len,
        .count => |count| @min(model.layers.len, count),
    };
}
