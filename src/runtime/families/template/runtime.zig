// Template Family Module
//
// This template provides a starting point for implementing a new model family.
// Copy this directory to src/runtime/families/<family_name>/ and modify as needed.
//
// Steps to add a new family:
// 1. Copy this directory to src/runtime/families/<your_family>/
// 2. Rename files and update contents
// 3. Register the family in src/runtime/mod.zig
// 4. Add family detection in src/runtime/families/mod.zig detectModelFamily()
//
// Key interfaces to implement:
//
// FamilyRuntime Interface:
// - ctx: ?*anyopaque - Context pointer for runtime state
// - generate_fn: fn(?*anyopaque, Allocator, model_path, prompt, FamilyGenerateOptions) anyerror!FamilyReport
// - deinit_fn: fn(?*anyopaque) void
//
// FamilyHandler Struct:
// - family: ModelFamily - The model family enum variant
// - capabilities: FamilyCapabilities - CPU/Metal support, quantizations, context length
// - runtime: FamilyRuntime - The runtime implementation

const std = @import("std");
const llama_runtime = @import("../../llama_runtime.zig");
const types = @import("../../types.zig");
const families_mod = @import("../mod.zig");

pub const supported_quantization = families_mod.supported_quantization;

fn templateGenerate(
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

fn templateDeinit(ctx: ?*anyopaque) void {
    _ = ctx;
}

pub fn createTemplateFamilyRuntime() families_mod.FamilyRuntime {
    return families_mod.FamilyRuntime{
        .ctx = null,
        .generate_fn = struct {
            fn f(ctx: ?*anyopaque, allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8, options: families_mod.FamilyGenerateOptions) !families_mod.FamilyReport {
                return templateGenerate(ctx, allocator, model_path, prompt, options);
            }
        }.f,
        .deinit_fn = templateDeinit,
    };
}

pub fn createTemplateFamilyHandler() families_mod.FamilyHandler {
    return families_mod.FamilyHandler{
        .family = .custom("template"),
        .capabilities = families_mod.FamilyCapabilities{
            .supports_cpu = true,
            .supports_metal = true,
            .supported_quant_types = &.{},
            .max_context_length = 8192,
        },
        .runtime = createTemplateFamilyRuntime(),
    };
}
