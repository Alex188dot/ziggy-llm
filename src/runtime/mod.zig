const std = @import("std");
const gguf = @import("../gguf.zig");
const bench_runner = @import("bench_runner.zig");
const llama_runtime = @import("llama_runtime.zig");
const types = @import("types.zig");
const families_mod = @import("families/mod.zig");
const registry_mod = @import("families/registry.zig");
const llama_family = @import("families/llama/mod.zig");
const qwen_family = @import("families/qwen/mod.zig");
const mistral_family = @import("families/mistral/mod.zig");
const mistral3_family = @import("families/ministral3_2512/mod.zig");
const gemma_family = @import("families/gemma/mod.zig");
const qwen35_family = @import("families/qwen35/mod.zig");
const qwen35_text_family = @import("families/qwen35_text/mod.zig");

pub const primary_target = types.primary_target;
pub const fallback_target = types.fallback_target;
pub const native_architecture = types.native_architecture;
pub const supported_architecture = types.supported_architecture;
pub const supported_model_family = types.supported_model_family;
pub const supported_quantization = types.supported_quantization;
pub const default_context_length = types.default_context_length;
pub const RuntimeError = types.RuntimeError;
pub const BackendPreference = types.BackendPreference;
pub const BackendUsed = types.BackendUsed;
pub const MoonQuantMode = types.MoonQuantMode;
pub const SamplingStrategy = types.SamplingStrategy;
pub const EffectiveSamplingPath = types.EffectiveSamplingPath;
pub const ReadbackMode = types.ReadbackMode;
pub const GenerationOptions = types.GenerationOptions;
pub const GenerationReport = types.GenerationReport;
pub const BenchSummary = bench_runner.BenchSummary;
pub const deltaNs = types.deltaNs;
pub const nsToMs = types.nsToMs;

fn getRegistry() *registry_mod.FamilyRegistry {
    const reg = registry_mod.getGlobalRegistry();
    if (reg.count == 0) {
        reg.register(llama_family.FamilyHandler) catch unreachable;
        reg.register(qwen_family.FamilyHandler) catch unreachable;
        reg.register(mistral_family.FamilyHandler) catch unreachable;
        reg.register(mistral3_family.FamilyHandler) catch unreachable;
        reg.register(gemma_family.FamilyHandler) catch unreachable;
        reg.register(qwen35_family.FamilyHandler) catch unreachable;
        reg.register(qwen35_text_family.FamilyHandler) catch unreachable;
    }
    return reg;
}

pub fn generate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: GenerationOptions,
) !GenerationReport {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const gguf_report = try gguf.inspectFile(arena.allocator(), model_path);
    std.debug.print("DEBUG: gguf architecture='{s}'\n", .{gguf_report.architecture});
    const family = families_mod.detectModelFamily(gguf_report.architecture);
    std.debug.print("DEBUG: detected family='{s}'\n", .{family.label()});
    const reg = getRegistry();
    std.debug.print("DEBUG: registry count={d}\n", .{reg.count});

    if (reg.getRuntime(family)) |runtime| {
        std.debug.print("DEBUG: calling runtime.generate\n", .{});
        std.debug.print("DEBUG: about to enter runtime.generate\n", .{});
        const family_options = families_mod.FamilyGenerateOptions{
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
            .exp_block_trace = options.exp_block_trace,
            .exp_block_acceptance_threshold = options.exp_block_acceptance_threshold,
            .exp_block_acceptance_window = options.exp_block_acceptance_window,
            .exp_block_disable_steps = options.exp_block_disable_steps,
            .exp_block_precheck_margin_multiplier = options.exp_block_precheck_margin_multiplier,
        };

        const family_report = try runtime.generate(allocator, model_path, prompt, family_options);

        return types.GenerationReport{
            .generated_text = family_report.generated_text,
            .prompt_token_count = family_report.prompt_token_count,
            .reused_prompt_token_count = family_report.reused_prompt_token_count,
            .generated_token_count = family_report.generated_token_count,
            .startup_ns = family_report.startup_ns,
            .prompt_ns = family_report.prompt_ns,
            .ttft_ns = family_report.ttft_ns,
            .decode_ns = family_report.decode_ns,
            .seed = family_report.seed,
            .temperature = family_report.temperature,
            .backend = @enumFromInt(@intFromEnum(family_report.backend)),
            .sampling_strategy = family_report.sampling_strategy,
            .sampling_path = family_report.sampling_path,
            .readback_mode = family_report.readback_mode,
            .startup_breakdown = family_report.startup_breakdown,
            .metal_profile_summary = family_report.metal_profile_summary,
            .exp_block_decode = family_report.exp_block_decode,
            .exp_block_k = family_report.exp_block_k,
            .exp_block_confidence_margin = family_report.exp_block_confidence_margin,
            .exp_block_cooldown_tokens = family_report.exp_block_cooldown_tokens,
            .exp_block_gpu_verifier = family_report.exp_block_gpu_verifier,
            .exp_block_trace = family_report.exp_block_trace,
            .exp_block_acceptance_threshold = family_report.exp_block_acceptance_threshold,
            .exp_block_acceptance_window = family_report.exp_block_acceptance_window,
            .exp_block_disable_steps = family_report.exp_block_disable_steps,
            .exp_block_precheck_margin_multiplier = family_report.exp_block_precheck_margin_multiplier,
            .block_accepted_prefix_len = family_report.block_accepted_prefix_len,
            .block_rollback_count = family_report.block_rollback_count,
            .block_confidence_gated_count = family_report.block_confidence_gated_count,
            .block_cooldown_active_count = family_report.block_cooldown_active_count,
            .block_quality_gate_active_count = family_report.block_quality_gate_active_count,
            .block_quality_gate_trigger_count = family_report.block_quality_gate_trigger_count,
            .block_precheck_count = family_report.block_precheck_count,
            .block_precheck_fail_count = family_report.block_precheck_fail_count,
            .block_draft_pos1_count = family_report.block_draft_pos1_count,
            .block_draft_pos2_count = family_report.block_draft_pos2_count,
            .block_draft_pos3_count = family_report.block_draft_pos3_count,
            .block_draft_pos4_count = family_report.block_draft_pos4_count,
            .block_accept_pos1_count = family_report.block_accept_pos1_count,
            .block_accept_pos2_count = family_report.block_accept_pos2_count,
            .block_accept_pos3_count = family_report.block_accept_pos3_count,
            .block_accept_pos4_count = family_report.block_accept_pos4_count,
            .block_mismatch_pos0_count = family_report.block_mismatch_pos0_count,
            .block_mismatch_pos1_count = family_report.block_mismatch_pos1_count,
            .block_mismatch_pos2_count = family_report.block_mismatch_pos2_count,
            .block_mismatch_pos3_count = family_report.block_mismatch_pos3_count,
            .block_full_accept_count = family_report.block_full_accept_count,
            .block_verify_ns = family_report.block_verify_ns,
            .block_gpu_backup_ns = family_report.block_gpu_backup_ns,
            .block_gpu_restore_ns = family_report.block_gpu_restore_ns,
            .block_gpu_sequence_commits = family_report.block_gpu_sequence_commits,
            .block_gpu_fallback_count = family_report.block_gpu_fallback_count,
        };
    }

    std.debug.print("Unsupported model family: {s}\n", .{family.label()});
    return error.UnsupportedArchitecture;
}

pub fn runCommand(
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: GenerationOptions,
) !void {
    var report = try generate(allocator, model_path, prompt, options);
    defer report.deinit(allocator);

    try writer.print(
        \\backend: {s}
        \\generated_text: {s}
        \\prompt_tokens: {d}
        \\generated_tokens: {d}
        \\seed: {d}
        \\temperature: {d:.3}
        \\repeat_penalty: {d:.3}
        \\top_k: {d}
        \\top_p: {d:.3}
        \\min_p: {d:.3}
        \\sampling_strategy: {s}
        \\sampling_path: {s}
        \\readback_mode: {s}
        \\startup_ms: {d:.3}
        \\startup.model_load_ms: {d:.3}
        \\startup.tensor_prepare_ms: {d:.3}
        \\startup.backend_init_ms: {d:.3}
        \\startup.metal_prewarm_ms: {d:.3}
        \\startup.session_init_ms: {d:.3}
        \\prompt_ms: {d:.3}
        \\ttft_ms: {d:.3}
        \\first_decode_step_ms: {d:.3}
        \\tps: {d:.3}
        \\decode_tok_s: {d:.3}
        \\
    ,
        .{
            report.backend.label(),
            report.generated_text,
            report.prompt_token_count,
            report.generated_token_count,
            report.seed,
            report.temperature,
            options.repeat_penalty,
            options.top_k,
            options.top_p,
            options.min_p,
            report.sampling_strategy.label(),
            report.sampling_path.label(),
            report.readback_mode.label(),
            nsToMs(report.startup_ns),
            nsToMs(report.startup_breakdown.model_load_ns),
            nsToMs(report.startup_breakdown.tensor_prepare_ns),
            nsToMs(report.startup_breakdown.backend_init_ns),
            nsToMs(report.startup_breakdown.metal_prewarm_ns),
            nsToMs(report.startup_breakdown.session_init_ns),
            nsToMs(report.prompt_ns),
            nsToMs(report.ttft_ns),
            nsToMs(report.startup_breakdown.first_decode_step_ns),
            report.decodeTokensPerSecond(),
            report.decodeTokensPerSecond(),
        },
    );
    try writer.print(
        \\block.exp_enabled: {}
        \\block.k: {d}
        \\block.confidence_margin: {d:.3}
        \\block.cooldown_tokens: {d}
        \\block.gpu_verifier: {}
        \\block.accepted_prefix_len: {d:.3}
        \\block.rollback_count: {d}
        \\block.confidence_gated_count: {d}
        \\block.cooldown_active_count: {d}
        \\block.quality_gate_active_count: {d}
        \\block.quality_gate_trigger_count: {d}
        \\block.precheck_count: {d}
        \\block.precheck_fail_count: {d}
        \\block.draft_pos1_count: {d}
        \\block.draft_pos2_count: {d}
        \\block.draft_pos3_count: {d}
        \\block.draft_pos4_count: {d}
        \\block.accept_pos1_count: {d}
        \\block.accept_pos2_count: {d}
        \\block.accept_pos3_count: {d}
        \\block.accept_pos4_count: {d}
        \\block.mismatch_pos0_count: {d}
        \\block.mismatch_pos1_count: {d}
        \\block.mismatch_pos2_count: {d}
        \\block.mismatch_pos3_count: {d}
        \\block.full_accept_count: {d}
        \\block.verify_ms: {d:.3}
        \\block.gpu_backup_ms: {d:.3}
        \\block.gpu_restore_ms: {d:.3}
        \\block.gpu_sequence_commits: {d}
        \\block.gpu_fallback_count: {d}
        \\block.gpu_commits_per_token: {d:.6}
        \\
    ,
        .{
            report.exp_block_decode,
            report.exp_block_k,
            report.exp_block_confidence_margin,
            report.exp_block_cooldown_tokens,
            report.exp_block_gpu_verifier,
            report.block_accepted_prefix_len,
            report.block_rollback_count,
            report.block_confidence_gated_count,
            report.block_cooldown_active_count,
            report.block_quality_gate_active_count,
            report.block_quality_gate_trigger_count,
            report.block_precheck_count,
            report.block_precheck_fail_count,
            report.block_draft_pos1_count,
            report.block_draft_pos2_count,
            report.block_draft_pos3_count,
            report.block_draft_pos4_count,
            report.block_accept_pos1_count,
            report.block_accept_pos2_count,
            report.block_accept_pos3_count,
            report.block_accept_pos4_count,
            report.block_mismatch_pos0_count,
            report.block_mismatch_pos1_count,
            report.block_mismatch_pos2_count,
            report.block_mismatch_pos3_count,
            report.block_full_accept_count,
            nsToMs(report.block_verify_ns),
            nsToMs(report.block_gpu_backup_ns),
            nsToMs(report.block_gpu_restore_ns),
            report.block_gpu_sequence_commits,
            report.block_gpu_fallback_count,
            if (report.generated_token_count > 0)
                @as(f64, @floatFromInt(report.block_gpu_sequence_commits)) / @as(f64, @floatFromInt(report.generated_token_count))
            else
                0.0,
        },
    );
    if (report.metal_profile_summary) |summary| {
        try writer.print("metal_profile:\n{s}", .{summary});
    }
}

pub fn benchCommand(
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: GenerationOptions,
    bench_runs: usize,
) !void {
    if (bench_runs > 1) {
        var summary = try bench_runner.runWarmBench(allocator, model_path, prompt, options, bench_runs);
        defer summary.deinit(allocator);

        try writer.print(
            \\backend={s}
            \\sampling_strategy={s}
            \\sampling_path={s}
            \\readback_mode={s}
            \\bench_runs={d}
            \\cold.startup_ms={d:.3}
            \\cold.startup.model_load_ms={d:.3}
            \\cold.startup.tensor_prepare_ms={d:.3}
            \\cold.startup.backend_init_ms={d:.3}
            \\cold.startup.metal_prewarm_ms={d:.3}
            \\cold.startup.session_init_ms={d:.3}
            \\cold.prompt_ms={d:.3}
            \\cold.ttft_ms={d:.3}
            \\cold.first_decode_step_ms={d:.3}
            \\cold.decode_ms={d:.3}
            \\cold.prompt_tokens={d}
            \\cold.generated_tokens={d}
            \\cold.tps={d:.3}
            \\cold.decode_tok_s={d:.3}
            \\
        ,
            .{
                summary.cold.backend.label(),
                summary.cold.sampling_strategy.label(),
                summary.cold.sampling_path.label(),
                summary.cold.readback_mode.label(),
                bench_runs,
                nsToMs(summary.cold.startup_ns),
                nsToMs(summary.cold.startup_breakdown.model_load_ns),
                nsToMs(summary.cold.startup_breakdown.tensor_prepare_ns),
                nsToMs(summary.cold.startup_breakdown.backend_init_ns),
                nsToMs(summary.cold.startup_breakdown.metal_prewarm_ns),
                nsToMs(summary.cold.startup_breakdown.session_init_ns),
                nsToMs(summary.cold.prompt_ns),
                nsToMs(summary.cold.ttft_ns),
                nsToMs(summary.cold.startup_breakdown.first_decode_step_ns),
                nsToMs(summary.cold.decode_ns),
                summary.cold.prompt_token_count,
                summary.cold.generated_token_count,
                summary.cold.decodeTokensPerSecond(),
                summary.cold.decodeTokensPerSecond(),
            },
        );
        try writer.print(
            \\cold.block.confidence_margin={d:.3}
            \\cold.block.cooldown_tokens={d}
            \\cold.block.gpu_verifier={}
            \\cold.block.accepted_prefix_len={d:.3}
            \\cold.block.rollback_count={d}
            \\cold.block.confidence_gated_count={d}
            \\cold.block.cooldown_active_count={d}
            \\cold.block.quality_gate_active_count={d}
            \\cold.block.quality_gate_trigger_count={d}
            \\cold.block.precheck_count={d}
            \\cold.block.precheck_fail_count={d}
            \\cold.block.draft_pos1_count={d}
            \\cold.block.draft_pos2_count={d}
            \\cold.block.draft_pos3_count={d}
            \\cold.block.draft_pos4_count={d}
            \\cold.block.accept_pos1_count={d}
            \\cold.block.accept_pos2_count={d}
            \\cold.block.accept_pos3_count={d}
            \\cold.block.accept_pos4_count={d}
            \\cold.block.mismatch_pos0_count={d}
            \\cold.block.mismatch_pos1_count={d}
            \\cold.block.mismatch_pos2_count={d}
            \\cold.block.mismatch_pos3_count={d}
            \\cold.block.full_accept_count={d}
            \\cold.block.verify_ms={d:.3}
            \\cold.block.gpu_backup_ms={d:.3}
            \\cold.block.gpu_restore_ms={d:.3}
            \\cold.block.gpu_sequence_commits={d}
            \\cold.block.gpu_fallback_count={d}
            \\cold.block.gpu_commits_per_token={d:.6}
            \\
        ,
            .{
                summary.cold.exp_block_confidence_margin,
                summary.cold.exp_block_cooldown_tokens,
                summary.cold.exp_block_gpu_verifier,
                summary.cold.block_accepted_prefix_len,
                summary.cold.block_rollback_count,
                summary.cold.block_confidence_gated_count,
                summary.cold.block_cooldown_active_count,
                summary.cold.block_quality_gate_active_count,
                summary.cold.block_quality_gate_trigger_count,
                summary.cold.block_precheck_count,
                summary.cold.block_precheck_fail_count,
                summary.cold.block_draft_pos1_count,
                summary.cold.block_draft_pos2_count,
                summary.cold.block_draft_pos3_count,
                summary.cold.block_draft_pos4_count,
                summary.cold.block_accept_pos1_count,
                summary.cold.block_accept_pos2_count,
                summary.cold.block_accept_pos3_count,
                summary.cold.block_accept_pos4_count,
                summary.cold.block_mismatch_pos0_count,
                summary.cold.block_mismatch_pos1_count,
                summary.cold.block_mismatch_pos2_count,
                summary.cold.block_mismatch_pos3_count,
                summary.cold.block_full_accept_count,
                nsToMs(summary.cold.block_verify_ns),
                nsToMs(summary.cold.block_gpu_backup_ns),
                nsToMs(summary.cold.block_gpu_restore_ns),
                summary.cold.block_gpu_sequence_commits,
                summary.cold.block_gpu_fallback_count,
                if (summary.cold.generated_token_count > 0)
                    @as(f64, @floatFromInt(summary.cold.block_gpu_sequence_commits)) / @as(f64, @floatFromInt(summary.cold.generated_token_count))
                else
                    0.0,
            },
        );
        try writer.print(
            \\warm.runs={d}
            \\warm.startup_ms_avg={d:.3}
            \\warm.startup.model_load_ms_avg={d:.3}
            \\warm.startup.tensor_prepare_ms_avg={d:.3}
            \\warm.startup.backend_init_ms_avg={d:.3}
            \\warm.startup.metal_prewarm_ms_avg={d:.3}
            \\warm.startup.session_init_ms_avg={d:.3}
            \\warm.prompt_ms_avg={d:.3}
            \\warm.ttft_ms_avg={d:.3}
            \\warm.first_decode_step_ms_avg={d:.3}
            \\warm.decode_ms_avg={d:.3}
            \\warm.reused_prompt_tokens_avg={d}
            \\warm.generated_tokens_avg={d}
            \\warm.tps_avg={d:.3}
            \\warm.decode_tok_s_avg={d:.3}
            \\
        ,
            .{
                summary.warm_runs,
                nsToMs(summary.warm_startup_ns_avg),
                nsToMs(summary.warm_startup_breakdown_avg.model_load_ns),
                nsToMs(summary.warm_startup_breakdown_avg.tensor_prepare_ns),
                nsToMs(summary.warm_startup_breakdown_avg.backend_init_ns),
                nsToMs(summary.warm_startup_breakdown_avg.metal_prewarm_ns),
                nsToMs(summary.warm_startup_breakdown_avg.session_init_ns),
                nsToMs(summary.warm_prompt_ns_avg),
                nsToMs(summary.warm_ttft_ns_avg),
                nsToMs(summary.warm_startup_breakdown_avg.first_decode_step_ns),
                nsToMs(summary.warm_decode_ns_avg),
                summary.warm_reused_prompt_token_count_avg,
                summary.warm_generated_token_count_avg,
                summary.warmDecodeTokensPerSecond(),
                summary.warmDecodeTokensPerSecond(),
            },
        );
        try writer.print(
            \\warm.block.accepted_prefix_len_avg={d:.3}
            \\warm.block.rollback_count_avg={d}
            \\warm.block.confidence_gated_count_avg={d}
            \\warm.block.cooldown_active_count_avg={d}
            \\warm.block.quality_gate_active_count_avg={d}
            \\warm.block.quality_gate_trigger_count_avg={d}
            \\warm.block.precheck_count_avg={d}
            \\warm.block.precheck_fail_count_avg={d}
            \\warm.block.draft_pos1_count_avg={d}
            \\warm.block.draft_pos2_count_avg={d}
            \\warm.block.draft_pos3_count_avg={d}
            \\warm.block.draft_pos4_count_avg={d}
            \\warm.block.accept_pos1_count_avg={d}
            \\warm.block.accept_pos2_count_avg={d}
            \\warm.block.accept_pos3_count_avg={d}
            \\warm.block.accept_pos4_count_avg={d}
            \\warm.block.mismatch_pos0_count_avg={d}
            \\warm.block.mismatch_pos1_count_avg={d}
            \\warm.block.mismatch_pos2_count_avg={d}
            \\warm.block.mismatch_pos3_count_avg={d}
            \\warm.block.full_accept_count_avg={d}
            \\warm.block.verify_ms_avg={d:.3}
            \\warm.block.gpu_backup_ms_avg={d:.3}
            \\warm.block.gpu_restore_ms_avg={d:.3}
            \\warm.block.gpu_sequence_commits_avg={d:.3}
            \\warm.block.gpu_fallback_count_avg={d}
            \\warm.block.gpu_commits_per_token_avg={d:.6}
            \\
        ,
            .{
                summary.warm_block_accepted_prefix_len_avg,
                summary.warm_block_rollback_count_avg,
                summary.warm_block_confidence_gated_count_avg,
                summary.warm_block_cooldown_active_count_avg,
                summary.warm_block_quality_gate_active_count_avg,
                summary.warm_block_quality_gate_trigger_count_avg,
                summary.warm_block_precheck_count_avg,
                summary.warm_block_precheck_fail_count_avg,
                summary.warm_block_draft_pos1_count_avg,
                summary.warm_block_draft_pos2_count_avg,
                summary.warm_block_draft_pos3_count_avg,
                summary.warm_block_draft_pos4_count_avg,
                summary.warm_block_accept_pos1_count_avg,
                summary.warm_block_accept_pos2_count_avg,
                summary.warm_block_accept_pos3_count_avg,
                summary.warm_block_accept_pos4_count_avg,
                summary.warm_block_mismatch_pos0_count_avg,
                summary.warm_block_mismatch_pos1_count_avg,
                summary.warm_block_mismatch_pos2_count_avg,
                summary.warm_block_mismatch_pos3_count_avg,
                summary.warm_block_full_accept_count_avg,
                nsToMs(summary.warm_block_verify_ns_avg),
                nsToMs(summary.warm_block_gpu_backup_ns_avg),
                nsToMs(summary.warm_block_gpu_restore_ns_avg),
                summary.warm_block_gpu_sequence_commits_avg,
                summary.warm_block_gpu_fallback_count_avg,
                if (summary.warm_generated_token_count_avg > 0)
                    summary.warm_block_gpu_sequence_commits_avg / @as(f64, @floatFromInt(summary.warm_generated_token_count_avg))
                else
                    0.0,
            },
        );
        if (summary.cold.metal_profile_summary) |summary_text| {
            try writer.print("cold.metal_profile:\n{s}", .{summary_text});
        }
        return;
    }

    var report = try generate(allocator, model_path, prompt, options);
    defer report.deinit(allocator);

    try writer.print(
        \\backend={s}
        \\sampling_strategy={s}
        \\sampling_path={s}
        \\readback_mode={s}
        \\repeat_penalty={d:.3}
        \\top_k={d}
        \\top_p={d:.3}
        \\min_p={d:.3}
        \\startup_ms={d:.3}
        \\startup.model_load_ms={d:.3}
        \\startup.tensor_prepare_ms={d:.3}
        \\startup.backend_init_ms={d:.3}
        \\startup.metal_prewarm_ms={d:.3}
        \\startup.session_init_ms={d:.3}
        \\prompt_ms={d:.3}
        \\ttft_ms={d:.3}
        \\first_decode_step_ms={d:.3}
        \\decode_ms={d:.3}
        \\prompt_tokens={d}
        \\generated_tokens={d}
        \\tps={d:.3}
        \\decode_tok_s={d:.3}
        \\
    ,
        .{
            report.backend.label(),
            report.sampling_strategy.label(),
            report.sampling_path.label(),
            report.readback_mode.label(),
            options.repeat_penalty,
            options.top_k,
            options.top_p,
            options.min_p,
            nsToMs(report.startup_ns),
            nsToMs(report.startup_breakdown.model_load_ns),
            nsToMs(report.startup_breakdown.tensor_prepare_ns),
            nsToMs(report.startup_breakdown.backend_init_ns),
            nsToMs(report.startup_breakdown.metal_prewarm_ns),
            nsToMs(report.startup_breakdown.session_init_ns),
            nsToMs(report.prompt_ns),
            nsToMs(report.ttft_ns),
            nsToMs(report.startup_breakdown.first_decode_step_ns),
            nsToMs(report.decode_ns),
            report.prompt_token_count,
            report.generated_token_count,
            report.decodeTokensPerSecond(),
            report.decodeTokensPerSecond(),
        },
    );
    try writer.print(
        \\block.confidence_margin={d:.3}
        \\block.cooldown_tokens={d}
        \\block.gpu_verifier={}
        \\block.accepted_prefix_len={d:.3}
        \\block.rollback_count={d}
        \\block.confidence_gated_count={d}
        \\block.cooldown_active_count={d}
        \\block.quality_gate_active_count={d}
        \\block.quality_gate_trigger_count={d}
        \\block.precheck_count={d}
        \\block.precheck_fail_count={d}
        \\block.draft_pos1_count={d}
        \\block.draft_pos2_count={d}
        \\block.draft_pos3_count={d}
        \\block.draft_pos4_count={d}
        \\block.accept_pos1_count={d}
        \\block.accept_pos2_count={d}
        \\block.accept_pos3_count={d}
        \\block.accept_pos4_count={d}
        \\block.mismatch_pos0_count={d}
        \\block.mismatch_pos1_count={d}
        \\block.mismatch_pos2_count={d}
        \\block.mismatch_pos3_count={d}
        \\block.full_accept_count={d}
        \\block.verify_ms={d:.3}
        \\block.gpu_backup_ms={d:.3}
        \\block.gpu_restore_ms={d:.3}
        \\block.gpu_sequence_commits={d}
        \\block.gpu_fallback_count={d}
        \\block.gpu_commits_per_token={d:.6}
        \\
    ,
        .{
            report.exp_block_confidence_margin,
            report.exp_block_cooldown_tokens,
            report.exp_block_gpu_verifier,
            report.block_accepted_prefix_len,
            report.block_rollback_count,
            report.block_confidence_gated_count,
            report.block_cooldown_active_count,
            report.block_quality_gate_active_count,
            report.block_quality_gate_trigger_count,
            report.block_precheck_count,
            report.block_precheck_fail_count,
            report.block_draft_pos1_count,
            report.block_draft_pos2_count,
            report.block_draft_pos3_count,
            report.block_draft_pos4_count,
            report.block_accept_pos1_count,
            report.block_accept_pos2_count,
            report.block_accept_pos3_count,
            report.block_accept_pos4_count,
            report.block_mismatch_pos0_count,
            report.block_mismatch_pos1_count,
            report.block_mismatch_pos2_count,
            report.block_mismatch_pos3_count,
            report.block_full_accept_count,
            nsToMs(report.block_verify_ns),
            nsToMs(report.block_gpu_backup_ns),
            nsToMs(report.block_gpu_restore_ns),
            report.block_gpu_sequence_commits,
            report.block_gpu_fallback_count,
            if (report.generated_token_count > 0)
                @as(f64, @floatFromInt(report.block_gpu_sequence_commits)) / @as(f64, @floatFromInt(report.generated_token_count))
            else
                0.0,
        },
    );
    if (report.metal_profile_summary) |summary| {
        try writer.print("{s}", .{summary});
    }
}
