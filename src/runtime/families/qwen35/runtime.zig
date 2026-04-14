const std = @import("std");
const families_mod = @import("../mod.zig");

pub const supported_quantization = families_mod.supported_quantization;

const Qwen35NotImplementedError = error{
    Qwen35MoENotYetImplemented,
};

fn qwen35Generate(
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: families_mod.FamilyGenerateOptions,
) !families_mod.FamilyReport {
    _ = ctx;
    _ = allocator;
    _ = model_path;
    _ = prompt;
    _ = options;
    return Qwen35NotImplementedError.Qwen35MoENotYetImplemented;
}

fn qwen35Deinit(ctx: ?*anyopaque) void {
    _ = ctx;
}

pub fn createQwen35FamilyRuntime() families_mod.FamilyRuntime {
    return families_mod.FamilyRuntime{
        .ctx = null,
        .generate_fn = struct {
            fn f(ctx: ?*anyopaque, allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8, options: families_mod.FamilyGenerateOptions) !families_mod.FamilyReport {
                return qwen35Generate(ctx, allocator, model_path, prompt, options);
            }
        }.f,
        .deinit_fn = qwen35Deinit,
    };
}

pub fn createQwen35FamilyHandler() families_mod.FamilyHandler {
    return families_mod.FamilyHandler{
        .family = .qwen35,
        .capabilities = families_mod.FamilyCapabilities{
            .supports_cpu = false,
            .supports_metal = false,
            .supported_quant_types = &.{},
            .max_context_length = 8192,
        },
        .runtime = createQwen35FamilyRuntime(),
    };
}
