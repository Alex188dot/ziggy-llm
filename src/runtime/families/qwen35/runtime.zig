const std = @import("std");
const llama_runtime = @import("../../llama_runtime.zig");
const llama_fixture = @import("../../llama_fixture.zig");
const metal_backend = @import("../../metal_backend.zig");
const types = @import("../../types.zig");
const families_mod = @import("../mod.zig");

pub const supported_quantization = families_mod.supported_quantization;

fn qwen35Generate(
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: families_mod.FamilyGenerateOptions,
) !families_mod.FamilyReport {
    _ = ctx;
    const gen_opts = types.GenerationOptions{
        .max_tokens = options.max_tokens,
        .context_length = options.context_length,
        .seed = options.seed,
        .temperature = options.temperature,
        .repeat_penalty = options.repeat_penalty,
        .top_k = options.top_k,
        .top_p = options.top_p,
        .min_p = options.min_p,
        .backend = @enumFromInt(@intFromEnum(options.backend)),
        .moon_quant = options.moon_quant,
        .metal_profile = options.metal_profile,
        .sampling_strategy = options.sampling_strategy,
    };
    const report = try llama_runtime.generate(allocator, model_path, prompt, gen_opts);
    return families_mod.FamilyReport{
        .generated_text = report.generated_text,
        .prompt_token_count = report.prompt_token_count,
        .reused_prompt_token_count = report.reused_prompt_token_count,
        .generated_token_count = report.generated_token_count,
        .startup_ns = report.startup_ns,
        .prompt_ns = report.prompt_ns,
        .ttft_ns = report.ttft_ns,
        .decode_ns = report.decode_ns,
        .seed = report.seed,
        .temperature = report.temperature,
        .backend = @enumFromInt(@intFromEnum(report.backend)),
        .sampling_strategy = report.sampling_strategy,
        .sampling_path = report.sampling_path,
        .readback_mode = report.readback_mode,
        .startup_breakdown = report.startup_breakdown,
        .metal_profile_summary = report.metal_profile_summary,
    };
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
            .supports_cpu = true,
            .supports_metal = true,
            .supported_quant_types = &.{},
            .max_context_length = 8192,
        },
        .runtime = createQwen35FamilyRuntime(),
    };
}

test "qwen35 family runtime generates on cpu" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixtureWithOptions(std.testing.allocator, .{
        .architecture = "qwen3_moe",
    });
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "qwen35-cpu.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "qwen35-cpu.gguf");
    defer std.testing.allocator.free(path);

    const runtime = createQwen35FamilyRuntime();
    var report = try runtime.generate(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    });
    defer report.deinit(std.testing.allocator);

    try std.testing.expectEqual(families_mod.BackendUsed.cpu, report.backend);
    try std.testing.expectEqualStrings(" b c!", report.generated_text);
    try std.testing.expectEqual(@as(usize, 1), report.prompt_token_count);
    try std.testing.expectEqual(@as(usize, 3), report.generated_token_count);
}

test "qwen35 family runtime supports metal backend when available" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixtureWithOptions(std.testing.allocator, .{
        .architecture = "qwen3_moe",
    });
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "qwen35-metal.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "qwen35-metal.gguf");
    defer std.testing.allocator.free(path);

    const runtime = createQwen35FamilyRuntime();

    var cpu_report = try runtime.generate(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    });
    defer cpu_report.deinit(std.testing.allocator);

    var metal_report = try runtime.generate(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 0,
        .temperature = 0,
        .backend = .metal,
    });
    defer metal_report.deinit(std.testing.allocator);

    try std.testing.expectEqual(families_mod.BackendUsed.metal, metal_report.backend);
    try std.testing.expectEqualStrings(cpu_report.generated_text, metal_report.generated_text);
    try std.testing.expectEqual(cpu_report.prompt_token_count, metal_report.prompt_token_count);
    try std.testing.expectEqual(cpu_report.generated_token_count, metal_report.generated_token_count);
}

test "qwen35 family handler exposes cpu and metal support" {
    const handler = createQwen35FamilyHandler();
    try std.testing.expect(handler.capabilities.supports_cpu);
    try std.testing.expect(handler.capabilities.supports_metal);
}
