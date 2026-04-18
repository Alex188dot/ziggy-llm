const std = @import("std");
const cli = @import("cli.zig");
const prompt_builder = @import("chat_prompt.zig");
const resident_runtime = @import("runtime/resident_runtime.zig");
const runtime = @import("runtime.zig");

pub fn runChat(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;

    var cache = resident_runtime.ResidentRuntime.init(allocator);
    defer cache.deinit();
    cache.setKeepAliveSeconds(-1);

    var messages = std.ArrayList(prompt_builder.Message).empty;
    defer prompt_builder.deinitMessages(allocator, &messages);

    var stdin_buf: [4096]u8 = undefined;
    var stdin = std.fs.File.stdin().reader(&stdin_buf);
    const context_window = try prompt_builder.contextWindow(&cache, model_path, config.backend, config.context_length);
    const max_tokens = effectiveChatMaxTokens(config, context_window);

    try writer.print("chat_ready: true\nmodel: {s}\ncontext_window: {d}\nhistory_window_messages: {d}\ncommands: /help /clear /unload /bye\n\n", .{
        model_path,
        context_window,
        prompt_builder.history_window_messages,
    });

    if (config.prompt) |prompt| {
        try handleUserTurn(writer, allocator, &cache, model_path, config, max_tokens, &messages, prompt);
    }

    while (true) {
        try writer.print("\x1b[36m>>> \x1b[0m", .{});
        try writer.flush();

        const line = (try stdin.interface.takeDelimiter('\n')) orelse break;
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "/bye") or std.mem.eql(u8, trimmed, "/exit")) break;
        if (std.mem.eql(u8, trimmed, "/clear")) {
            prompt_builder.deinitMessages(allocator, &messages);
            messages = .empty;
            try writer.print("history_cleared: true\n\n", .{});
            continue;
        }
        if (std.mem.eql(u8, trimmed, "/unload")) {
            cache.unload();
            try writer.print("model_unloaded: true\n\n", .{});
            continue;
        }
        if (std.mem.eql(u8, trimmed, "/help")) {
            try writer.print("/help /clear /unload /bye\n\n", .{});
            continue;
        }

        try handleUserTurn(writer, allocator, &cache, model_path, config, max_tokens, &messages, trimmed);
    }
}

fn handleUserTurn(
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    config: cli.Config,
    max_tokens: usize,
    messages: *std.ArrayList(prompt_builder.Message),
    user_text: []const u8,
) !void {
    try prompt_builder.appendMessage(allocator, messages, .user, user_text);
    const prompt = try prompt_builder.buildPrompt(allocator, cache, model_path, config.backend, config.context_length, max_tokens, messages.items);
    defer allocator.free(prompt);

    const is_qwen = std.mem.indexOf(u8, model_path, "Qwen") != null or std.mem.indexOf(u8, model_path, "qwen") != null;
    if (!is_qwen) {
        try writer.print("\n", .{});
        try writer.flush();
    }

    var stream_state = StreamState.init(allocator, writer);
    defer stream_state.deinit();

    var report = try cache.generateStreaming(model_path, prompt, generationOptions(config, max_tokens), &stream_state, streamChunk);
    defer report.deinit(allocator);
    const trimmed = prompt_builder.trimAssistantReply(report.generated_text);
    try prompt_builder.appendMessage(allocator, messages, .assistant, trimmed);
    try stream_state.flushFinal(trimmed);

    try printTurnTimings(writer, &report);
    try writer.print("\n", .{});
}

fn generationOptions(config: cli.Config, max_tokens: usize) runtime.GenerationOptions {
    return .{
        .max_tokens = max_tokens,
        .context_length = config.context_length,
        .seed = config.seed,
        .temperature = config.temperature,
        .repeat_penalty = config.repeat_penalty,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .min_p = config.min_p,
        .backend = config.backend,
        .metal_profile = config.metal_profile,
        .sampling_strategy = config.sampling_strategy,
        .exp_block_decode = config.exp_block_decode,
        .exp_block_k = config.exp_block_k,
        .exp_block_confidence_margin = config.exp_block_confidence_margin,
        .exp_block_cooldown_tokens = config.exp_block_cooldown_tokens,
    };
}

fn effectiveChatMaxTokens(config: cli.Config, context_window: usize) usize {
    const cli_default: usize = 16;
    if (config.max_tokens != cli_default) return config.max_tokens;
    return @min(@as(usize, 2048), context_window / 2);
}

const StreamState = struct {
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    buffer: std.ArrayList(u8) = .empty,
    emitted_len: usize = 0,
    stopped: bool = false,
    in_think: bool = false,

    fn init(allocator: std.mem.Allocator, writer: *std.Io.Writer) StreamState {
        return .{
            .allocator = allocator,
            .writer = writer,
        };
    }

    fn deinit(self: *StreamState) void {
        self.buffer.deinit(self.allocator);
        self.* = undefined;
    }

    fn flushSafePrefix(self: *StreamState) !void {
        const safe = prompt_builder.trimAssistantReply(self.buffer.items);
        var process_end = safe.len;

        const think_tag = "<think>";
        const end_think_tag = "</think>";
        const stop_markers = [_][]const u8{ "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<user|>", "<assistant|>", "<system|>", "<|im_end|>", "<|im_start|>" };

        var partial_len: usize = 0;
        for (1..think_tag.len) |i| {
            if (safe.len >= i and std.mem.endsWith(u8, safe, think_tag[0..i])) {
                partial_len = @max(partial_len, i);
            }
        }
        for (1..end_think_tag.len) |i| {
            if (safe.len >= i and std.mem.endsWith(u8, safe, end_think_tag[0..i])) {
                partial_len = @max(partial_len, i);
            }
        }
        for (stop_markers) |marker| {
            for (1..marker.len) |i| {
                if (safe.len >= i and std.mem.endsWith(u8, safe, marker[0..i])) {
                    partial_len = @max(partial_len, i);
                }
            }
        }

        process_end -= partial_len;

        if (process_end <= self.emitted_len) return;

        const chunk = safe[self.emitted_len..process_end];
        try self.printChunk(chunk);

        try self.writer.flush();
        self.emitted_len = process_end;
    }

    fn flushFinal(self: *StreamState, final_trimmed: []const u8) !void {
        if (final_trimmed.len <= self.emitted_len) {
            if (self.in_think) {
                try self.writer.print("\x1b[0m", .{});
                try self.writer.flush();
            }
            return;
        }

        const chunk = final_trimmed[self.emitted_len..];
        try self.printChunk(chunk);

        if (self.in_think) {
            try self.writer.print("\x1b[0m", .{});
        }
        try self.writer.flush();
        self.emitted_len = final_trimmed.len;
    }

    fn printChunk(self: *StreamState, chunk: []const u8) !void {
        const think_tag = "<think>";
        const end_think_tag = "</think>";
        var i: usize = 0;

        while (i < chunk.len) {
            if (!self.in_think) {
                if (std.mem.startsWith(u8, chunk[i..], think_tag)) {
                    self.in_think = true;
                    try self.writer.print("\x1b[3m\x1b[90m", .{});
                    i += think_tag.len;
                    if (i < chunk.len and chunk[i] == '\n') i += 1;
                } else {
                    try self.writer.writeByte(chunk[i]);
                    i += 1;
                }
            } else {
                if (std.mem.startsWith(u8, chunk[i..], end_think_tag)) {
                    self.in_think = false;
                    try self.writer.print("\x1b[0m", .{});
                    i += end_think_tag.len;
                    if (i < chunk.len and chunk[i] == '\n') i += 1;
                } else {
                    try self.writer.writeByte(chunk[i]);
                    i += 1;
                }
            }
        }
    }
};

fn streamChunk(ctx: ?*anyopaque, chunk: []const u8) anyerror!void {
    const state: *StreamState = @ptrCast(@alignCast(ctx.?));
    try state.buffer.appendSlice(state.allocator, chunk);
    try state.flushSafePrefix();
    if (prompt_builder.hasCompletedAssistantReply(state.buffer.items)) {
        state.stopped = true;
        return error.StopStreaming;
    }
}

fn printTurnTimings(writer: *std.Io.Writer, report: *const runtime.GenerationReport) !void {
    try writer.print(
        \\
        \\
        \\{s}Chat Turn Metrics:{s}
        \\prompt_tokens: {d}
        \\reused_prompt_tokens: {d}
        \\prompt_ms: {d:.3}
        \\ttft_ms: {d:.3}
        \\{s}tps: {d:.3}{s}
        \\
    ,
        .{
            "\x1b[32m",
            "\x1b[0m",
            report.prompt_token_count,
            report.reused_prompt_token_count,
            runtime.nsToMs(report.prompt_ns),
            runtime.nsToMs(report.ttft_ns),
            "\x1b[95m",
            report.decodeTokensPerSecond(),
            "\x1b[0m",
        },
    );

    if (report.startup_breakdown.model_load_ns != 0 or
        report.startup_breakdown.tensor_prepare_ns != 0 or
        report.startup_breakdown.backend_init_ns != 0 or
        report.startup_breakdown.metal_prewarm_ns != 0 or
        report.startup_breakdown.session_init_ns != 0)
    {
        try writer.print(
            \\{s}Startup Metrics:{s}
            \\startup_ms: {d:.3}
            \\startup.model_load_ms: {d:.3}
            \\startup.tensor_prepare_ms: {d:.3}
            \\startup.backend_init_ms: {d:.3}
            \\startup.metal_prewarm_ms: {d:.3}
            \\startup.session_init_ms: {d:.3}
            \\
        ,
            .{
                "\x1b[33m",
                "\x1b[0m",
                runtime.nsToMs(report.startup_ns),
                runtime.nsToMs(report.startup_breakdown.model_load_ns),
                runtime.nsToMs(report.startup_breakdown.tensor_prepare_ns),
                runtime.nsToMs(report.startup_breakdown.backend_init_ns),
                runtime.nsToMs(report.startup_breakdown.metal_prewarm_ns),
                runtime.nsToMs(report.startup_breakdown.session_init_ns),
            },
        );
    }
}
