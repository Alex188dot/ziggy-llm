const std = @import("std");
const llama_runtime = @import("../../llama_runtime.zig");
const types = @import("../../types.zig");
const families_mod = @import("../mod.zig");

pub const supported_quantization = families_mod.supported_quantization;

fn llamaGenerate(
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
        .exp_block_decode = options.exp_block_decode,
        .exp_block_k = options.exp_block_k,
        .exp_block_confidence_margin = options.exp_block_confidence_margin,
        .exp_block_cooldown_tokens = options.exp_block_cooldown_tokens,
        .exp_block_gpu_verifier = options.exp_block_gpu_verifier,
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
        .exp_block_decode = report.exp_block_decode,
        .exp_block_k = report.exp_block_k,
        .exp_block_confidence_margin = report.exp_block_confidence_margin,
        .exp_block_cooldown_tokens = report.exp_block_cooldown_tokens,
        .exp_block_gpu_verifier = report.exp_block_gpu_verifier,
        .block_accepted_prefix_len = report.block_accepted_prefix_len,
        .block_rollback_count = report.block_rollback_count,
        .block_confidence_gated_count = report.block_confidence_gated_count,
        .block_cooldown_active_count = report.block_cooldown_active_count,
        .block_verify_ns = report.block_verify_ns,
        .block_gpu_backup_ns = report.block_gpu_backup_ns,
        .block_gpu_restore_ns = report.block_gpu_restore_ns,
        .block_gpu_sequence_commits = report.block_gpu_sequence_commits,
        .block_gpu_fallback_count = report.block_gpu_fallback_count,
    };
}

fn llamaDeinit(ctx: ?*anyopaque) void {
    _ = ctx;
}

pub fn createLlamaFamilyRuntime() families_mod.FamilyRuntime {
    return families_mod.FamilyRuntime{
        .ctx = null,
        .generate_fn = struct {
            fn f(ctx: ?*anyopaque, allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8, options: families_mod.FamilyGenerateOptions) !families_mod.FamilyReport {
                return llamaGenerate(ctx, allocator, model_path, prompt, options);
            }
        }.f,
        .deinit_fn = llamaDeinit,
    };
}

pub fn createLlamaFamilyHandler() families_mod.FamilyHandler {
    return families_mod.FamilyHandler{
        .family = .llama,
        .capabilities = families_mod.FamilyCapabilities{
            .supports_cpu = true,
            .supports_metal = true,
            .supported_quant_types = &.{},
            .max_context_length = 8192,
        },
        .runtime = createLlamaFamilyRuntime(),
    };
}
