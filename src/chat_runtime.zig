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

    try writer.print("chat_ready: true\nmodel: {s}\ncommands: /help /clear /unload /bye\n\n", .{model_path});

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

    var report = try cache.generate(model_path, prompt, generationOptions(config));
    defer report.deinit(allocator);
    const trimmed = prompt_builder.trimAssistantReply(report.generated_text);
    try prompt_builder.appendMessage(allocator, messages, .assistant, trimmed);

    try writer.print("{s}\n\n", .{trimmed});
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
    };
}
