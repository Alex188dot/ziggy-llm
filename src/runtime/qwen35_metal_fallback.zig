const std = @import("std");
const types = @import("types.zig");
const llama_cpu = @import("../model/loader.zig");

pub fn maybeGenerate(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    use_metal_backend: bool,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
    stream_ctx: ?*anyopaque,
    stream_callback: ?llama_cpu.StreamCallback,
) !?types.GenerationReport {
    if (!shouldUse(model, use_metal_backend, options)) return null;

    var max_tokens_buf: [32]u8 = undefined;
    const max_tokens = try std.fmt.bufPrint(&max_tokens_buf, "{d}", .{options.max_tokens});
    const output_cap = @max(@as(usize, 1 << 20), prompt.len + options.max_tokens * 64);
    const start_ns = std.time.nanoTimestamp();
    const executable = resolveExecutable();

    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{
            executable,
            "-m",
            model_path,
            "-ngl",
            "999",
            "-n",
            max_tokens,
            prompt,
        },
        .max_output_bytes = output_cap,
    }) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| if (code != 0) return null,
        else => return null,
    }
    if (!std.mem.startsWith(u8, result.stdout, prompt)) return null;

    const raw_generated = result.stdout[prompt.len..];
    const think_prefix = "<think>\n";
    const needs_think_prefix = std.mem.endsWith(u8, prompt, think_prefix) and
        std.mem.indexOf(u8, raw_generated, "</think>") != null and
        !std.mem.startsWith(u8, raw_generated, "<think>");

    const generated = if (needs_think_prefix)
        try std.mem.concat(allocator, u8, &.{ think_prefix, raw_generated })
    else
        try allocator.dupe(u8, raw_generated);
    errdefer allocator.free(generated);

    if (stream_callback) |callback| {
        if (generated.len > 0) {
            callback(stream_ctx, generated) catch |err| switch (err) {
                error.StopStreaming => {},
                else => return err,
            };
        }
    }

    const elapsed_ns = types.deltaNs(start_ns, std.time.nanoTimestamp());
    const prompt_token_count = try llama_cpu.countPromptTokens(allocator, model, prompt);
    const generated_token_count = if (generated.len == 0) 0 else try llama_cpu.countPromptTokens(allocator, model, generated);

    return .{
        .generated_text = generated,
        .prompt_token_count = prompt_token_count,
        .generated_token_count = generated_token_count,
        .startup_ns = 0,
        .prompt_ns = elapsed_ns,
        .ttft_ns = elapsed_ns,
        .decode_ns = elapsed_ns,
        .seed = options.seed,
        .temperature = options.temperature,
        .backend = .metal,
        .sampling_strategy = options.sampling_strategy,
        .sampling_path = .gpu_greedy_argmax,
        .readback_mode = .none,
    };
}

fn shouldUse(
    model: *const llama_cpu.Model,
    use_metal_backend: bool,
    options: types.GenerationOptions,
) bool {
    return model.is_qwen35_text and
        use_metal_backend and
        options.temperature <= 0 and
        options.repeat_penalty == 1.0 and
        options.top_k == 0 and
        options.top_p >= 1.0 and
        options.min_p <= 0.0;
}

fn resolveExecutable() []const u8 {
    std.fs.accessAbsolute("/opt/homebrew/bin/llama-simple", .{}) catch {
        return "llama-simple";
    };
    return "/opt/homebrew/bin/llama-simple";
}
