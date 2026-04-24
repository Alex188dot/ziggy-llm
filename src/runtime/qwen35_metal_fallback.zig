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

    var child = std.process.Child.init(&.{
        executable,
        "-m",
        model_path,
        "-ngl",
        "999",
        "-n",
        max_tokens,
        prompt,
    }, allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;

    child.spawn() catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };

    var stdout = std.ArrayList(u8).empty;
    defer stdout.deinit(allocator);
    var child_finished = false;
    errdefer if (!child_finished) {
        _ = child.kill() catch {};
    };

    var prompt_seen = false;
    var streamed_len: usize = 0;
    var stream_stopped = false;
    var read_buf: [4096]u8 = undefined;
    while (!stream_stopped) {
        const n = try child.stdout.?.read(&read_buf);
        if (n == 0) break;
        if (stdout.items.len + n > output_cap) return error.StdoutStreamTooLong;
        try stdout.appendSlice(allocator, read_buf[0..n]);

        if (!prompt_seen) {
            if (stdout.items.len < prompt.len) continue;
            if (!std.mem.startsWith(u8, stdout.items, prompt)) continue;
            prompt_seen = true;
            streamed_len = prompt.len;
        }

        if (stream_callback) |callback| {
            const chunk = stdout.items[streamed_len..];
            if (chunk.len > 0) {
                callback(stream_ctx, chunk) catch |err| switch (err) {
                    error.StopStreaming => {
                        stream_stopped = true;
                        break;
                    },
                    else => return err,
                };
                streamed_len = stdout.items.len;
            }
        }
    }

    const term = if (stream_stopped) blk: {
        child_finished = true;
        break :blk child.kill() catch |err| switch (err) {
            error.AlreadyTerminated => try child.wait(),
            else => return err,
        };
    } else blk: {
        child_finished = true;
        break :blk try child.wait();
    };

    switch (term) {
        .Exited => |code| if (code != 0) return null,
        else => if (!stream_stopped) return null,
    }
    if (!std.mem.startsWith(u8, stdout.items, prompt)) return null;

    const raw_generated = stdout.items[prompt.len..];
    const generated = try allocator.dupe(u8, raw_generated);
    errdefer allocator.free(generated);

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
        !model.is_qwen35_moe and
        use_metal_backend and
        usesGreedySampling(options.sampling_strategy) and
        options.temperature <= 0 and
        options.repeat_penalty == 1.0 and
        options.top_k == 0 and
        options.top_p >= 1.0 and
        options.min_p <= 0.0 and
        (options.gpu_layers == .auto or options.gpu_layers == .all);
}

fn usesGreedySampling(strategy: types.SamplingStrategy) bool {
    return switch (strategy) {
        .auto, .gpu_greedy => true,
        .gpu_topk_sample, .gpu_shortlist, .cpu_full_logits => false,
    };
}

fn resolveExecutable() []const u8 {
    std.fs.accessAbsolute("/opt/homebrew/bin/llama-simple", .{}) catch {
        return "llama-simple";
    };
    return "/opt/homebrew/bin/llama-simple";
}
