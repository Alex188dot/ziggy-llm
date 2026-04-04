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
        try writer.print("> ", .{});
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
    };
}

fn effectiveChatMaxTokens(config: cli.Config, context_window: usize) usize {
    const cli_default: usize = 16;
    if (config.max_tokens != cli_default) return config.max_tokens;
    return @min(@as(usize, 256), context_window / 4);
}

const StreamState = struct {
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    buffer: std.ArrayList(u8) = .empty,
    emitted_len: usize = 0,
    stopped: bool = false,

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
        if (safe.len <= self.emitted_len) return;
        try self.writer.print("{s}", .{safe[self.emitted_len..]});
        try self.writer.flush();
        self.emitted_len = safe.len;
    }

    fn flushFinal(self: *StreamState, final_trimmed: []const u8) !void {
        if (final_trimmed.len <= self.emitted_len) return;
        try self.writer.print("{s}", .{final_trimmed[self.emitted_len..]});
        try self.writer.flush();
        self.emitted_len = final_trimmed.len;
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
        \\chat_turn_metrics:
        \\prompt_tokens: {d}
        \\reused_prompt_tokens: {d}
        \\prompt_ms: {d:.3}
        \\ttft_ms: {d:.3}
        \\tps: {d:.3}
        \\
    ,
        .{
            report.prompt_token_count,
            report.reused_prompt_token_count,
            runtime.nsToMs(report.prompt_ns),
            runtime.nsToMs(report.ttft_ns),
            report.decodeTokensPerSecond(),
        },
    );

    if (report.startup_breakdown.model_load_ns != 0 or
        report.startup_breakdown.tensor_prepare_ns != 0 or
        report.startup_breakdown.backend_init_ns != 0 or
        report.startup_breakdown.metal_prewarm_ns != 0 or
        report.startup_breakdown.session_init_ns != 0)
    {
        try writer.print(
            \\startup_ms: {d:.3}
            \\startup.model_load_ms: {d:.3}
            \\startup.tensor_prepare_ms: {d:.3}
            \\startup.backend_init_ms: {d:.3}
            \\startup.metal_prewarm_ms: {d:.3}
            \\startup.session_init_ms: {d:.3}
            \\
        ,
            .{
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
