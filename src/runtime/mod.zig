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
const gemma2_family = @import("families/gemma2/mod.zig");
const gemma3_family = @import("families/gemma3/mod.zig");
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
        reg.register(gemma2_family.FamilyHandler) catch unreachable;
        reg.register(gemma3_family.FamilyHandler) catch unreachable;
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
    const family = families_mod.detectModelFamily(gguf_report.architecture);
    const reg = getRegistry();

    if (reg.getRuntime(family)) |runtime| {
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
    if (report.metal_profile_summary) |summary| {
        try writer.print("{s}", .{summary});
    }
}
