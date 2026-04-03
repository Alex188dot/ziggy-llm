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
    const context_window = try prompt_builder.contextWindow(&cache, model_path, config.backend);

    try writer.print("chat_ready: true\nmodel: {s}\ncontext_window: {d}\nhistory_window_messages: {d}\ncommands: /help /clear /unload /bye\n\n", .{
        model_path,
        context_window,
        prompt_builder.history_window_messages,
    });

    if (config.prompt) |prompt| {
        try handleUserTurn(writer, allocator, &cache, model_path, config, &messages, prompt);
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

        try handleUserTurn(writer, allocator, &cache, model_path, config, &messages, trimmed);
    }
}

fn handleUserTurn(
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    config: cli.Config,
    messages: *std.ArrayList(prompt_builder.Message),
    user_text: []const u8,
) !void {
    try prompt_builder.appendMessage(allocator, messages, .user, user_text);
    const prompt = try prompt_builder.buildPrompt(allocator, cache, model_path, config.backend, config.max_tokens, messages.items);
    defer allocator.free(prompt);

    var stream_state = StreamState.init(allocator, writer);
    defer stream_state.deinit();

    var report = try cache.generateStreaming(model_path, prompt, generationOptions(config), &stream_state, streamChunk);
    defer report.deinit(allocator);
    const trimmed = prompt_builder.trimAssistantReply(report.generated_text);
    try prompt_builder.appendMessage(allocator, messages, .assistant, trimmed);
    try stream_state.flushFinal(trimmed);

    try writer.print("\n\n", .{});
}

fn generationOptions(config: cli.Config) runtime.GenerationOptions {
    return .{
        .max_tokens = config.max_tokens,
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

const StreamState = struct {
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    buffer: std.ArrayList(u8) = .empty,
    emitted_len: usize = 0,

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
}
