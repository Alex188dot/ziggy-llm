const std = @import("std");
const gguf = @import("../gguf.zig");
const bench_runner = @import("bench_runner.zig");
const llama_runtime = @import("llama_runtime.zig");
const types = @import("types.zig");

pub const primary_target = types.primary_target;
pub const fallback_target = types.fallback_target;
pub const native_architecture = types.native_architecture;
pub const supported_architecture = types.supported_architecture;
pub const supported_model_family = types.supported_model_family;
pub const supported_quantization = types.supported_quantization;
pub const RuntimeError = types.RuntimeError;
pub const BackendPreference = types.BackendPreference;
pub const BackendUsed = types.BackendUsed;
pub const GenerationOptions = types.GenerationOptions;
pub const GenerationReport = types.GenerationReport;
pub const BenchSummary = bench_runner.BenchSummary;
pub const deltaNs = types.deltaNs;
pub const nsToMs = types.nsToMs;

pub fn generate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: GenerationOptions,
) !GenerationReport {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const report = try gguf.inspectFile(arena.allocator(), model_path);
    if (!std.mem.eql(u8, report.architecture, native_architecture)) return error.UnsupportedArchitecture;
    return llama_runtime.generate(allocator, model_path, prompt, options);
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
        \\startup_ms: {d:.3}
        \\prompt_ms: {d:.3}
        \\ttft_ms: {d:.3}
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
            nsToMs(report.startup_ns),
            nsToMs(report.prompt_ns),
            nsToMs(report.ttft_ns),
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
            \\bench_runs={d}
            \\cold.startup_ns={d}
            \\cold.prompt_ns={d}
            \\cold.ttft_ns={d}
            \\cold.decode_ns={d}
            \\cold.prompt_tokens={d}
            \\cold.generated_tokens={d}
            \\cold.tps={d:.3}
            \\cold.decode_tok_s={d:.3}
            \\warm.runs={d}
            \\warm.startup_ns_avg={d}
            \\warm.prompt_ns_avg={d}
            \\warm.ttft_ns_avg={d}
            \\warm.decode_ns_avg={d}
            \\warm.generated_tokens_avg={d}
            \\warm.tps_avg={d:.3}
            \\warm.decode_tok_s_avg={d:.3}
            \\
        ,
            .{
                summary.cold.backend.label(),
                bench_runs,
                summary.cold.startup_ns,
                summary.cold.prompt_ns,
                summary.cold.ttft_ns,
                summary.cold.decode_ns,
                summary.cold.prompt_token_count,
                summary.cold.generated_token_count,
                summary.cold.decodeTokensPerSecond(),
                summary.cold.decodeTokensPerSecond(),
                summary.warm_runs,
                summary.warm_startup_ns_avg,
                summary.warm_prompt_ns_avg,
                summary.warm_ttft_ns_avg,
                summary.warm_decode_ns_avg,
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
        \\startup_ns={d}
        \\prompt_ns={d}
        \\ttft_ns={d}
        \\decode_ns={d}
        \\prompt_tokens={d}
        \\generated_tokens={d}
        \\tps={d:.3}
        \\decode_tok_s={d:.3}
        \\
    ,
        .{
            report.backend.label(),
            report.startup_ns,
            report.prompt_ns,
            report.ttft_ns,
            report.decode_ns,
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
