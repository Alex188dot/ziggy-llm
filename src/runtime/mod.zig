const std = @import("std");
const gguf = @import("../gguf.zig");
const llama_cpu = @import("../llama_cpu.zig");
const tiny_runtime = @import("tiny_runtime.zig");
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
    if (std.mem.eql(u8, report.architecture, "llama")) {
        if (options.backend == .metal) return error.UnsupportedBackend;

        const llama_report = try llama_cpu.generate(
            allocator,
            model_path,
            prompt,
            options.max_tokens,
            options.seed,
            options.temperature,
        );
        return .{
            .generated_text = llama_report.generated_text,
            .prompt_token_count = llama_report.prompt_token_count,
            .generated_token_count = llama_report.generated_token_count,
            .startup_ns = llama_report.startup_ns,
            .prompt_ns = llama_report.prompt_ns,
            .ttft_ns = llama_report.ttft_ns,
            .decode_ns = llama_report.decode_ns,
            .seed = options.seed,
            .temperature = options.temperature,
            .backend = .cpu,
        };
    }
    if (!std.mem.eql(u8, report.architecture, native_architecture)) return error.UnsupportedArchitecture;

    return tiny_runtime.generate(allocator, model_path, prompt, options);
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
}

pub fn benchCommand(
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: GenerationOptions,
) !void {
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
}
