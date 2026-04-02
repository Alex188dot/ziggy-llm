const std = @import("std");
const resident_runtime = @import("runtime/resident_runtime.zig");
const runtime = @import("runtime.zig");

pub const Role = enum { user, assistant };

pub const Message = struct {
    role: Role,
    content: []u8,
};

const system_message =
    \\You are in an interactive terminal chat.
    \\Reply only as the assistant.
    \\Never emit role labels such as User, Assistant, Customer, Human, or System.
    \\Do not invent the user's next message.
    \\Keep answers concise unless the user asks for detail.
;

pub const history_window_messages: usize = 8;

pub fn deinitMessages(allocator: std.mem.Allocator, messages: *std.ArrayList(Message)) void {
    for (messages.items) |message| allocator.free(message.content);
    messages.deinit(allocator);
}

pub fn appendMessage(allocator: std.mem.Allocator, messages: *std.ArrayList(Message), role: Role, content: []const u8) !void {
    try messages.append(allocator, .{
        .role = role,
        .content = try allocator.dupe(u8, std.mem.trim(u8, content, " \n\r\t")),
    });
}

pub fn buildPrompt(
    allocator: std.mem.Allocator,
    runtime_cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    backend: runtime.BackendPreference,
    max_tokens: usize,
    messages: []const Message,
) ![]u8 {
    const context_length = runtime_cache.contextLength() orelse blk: {
        _ = try runtime_cache.promptTokenCount(model_path, "<|user|>\nhi</s>\n<|assistant|>\n", backend);
        break :blk runtime_cache.contextLength().?;
    };
    const token_budget = context_length -| (max_tokens + 64);

    var start_index: usize = if (messages.len > history_window_messages) messages.len - history_window_messages else 0;
    if (start_index % 2 != 0) start_index -= 1;
    while (true) {
        const prompt = try renderMessages(allocator, messages[start_index..]);
        errdefer allocator.free(prompt);
        const count = runtime_cache.promptTokenCount(model_path, prompt, backend) catch |err| switch (err) {
            error.ContextOverflow => context_length + 1,
            else => return err,
        };
        if (count <= token_budget or start_index == 0) return prompt;
        allocator.free(prompt);
        start_index -|= 2;
    }
}

pub fn trimAssistantReply(reply: []const u8) []const u8 {
    var end = reply.len;
    var lines = std.mem.splitScalar(u8, reply, '\n');
    var offset: usize = 0;
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (offset > 0 and isDialogueBoundary(trimmed)) {
            end = @min(end, offset);
            break;
        }
        offset += line.len + 1;
    }

    for ([_][]const u8{ "</s>", "</s", "</", "<|user|>", "<|assistant|>", "<|system|>", "<user|>", "<assistant|>", "<system|>", "<|", "<user|", "<assistant|", "<system|" }) |marker| {
        if (std.mem.indexOf(u8, reply[0..end], marker)) |index| {
            end = @min(end, index);
        }
    }
    return std.mem.trim(u8, reply[0..end], " \n\r\t");
}

fn isDialogueBoundary(line: []const u8) bool {
    if (line.len == 0) return false;

    const explicit = [_][]const u8{
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "User:",
        "User",
        "Assistant:",
        "Assistant",
        "System:",
        "Human:",
        "Customer:",
        "Question:",
        "Answer:",
    };
    for (explicit) |marker| {
        if (std.mem.startsWith(u8, line, marker)) return true;
    }

    const colon_index = std.mem.indexOfScalar(u8, line, ':') orelse return false;
    if (colon_index == 0 or colon_index > 24) return false;
    for (line[0..colon_index]) |char| {
        if (!(std.ascii.isAlphabetic(char) or char == ' ' or char == '_' or char == '-')) return false;
    }
    return true;
}

fn renderMessages(allocator: std.mem.Allocator, messages: []const Message) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);
    try writer.print("<|system|>\n{s}\n</s>\n", .{system_message});
    for (messages) |message| {
        const tag = switch (message.role) {
            .user => "user",
            .assistant => "assistant",
        };
        try writer.print("<|{s}|>\n{s}\n</s>\n", .{ tag, message.content });
    }
    try writer.print("<|assistant|>\n", .{});
    return buf.toOwnedSlice(allocator);
}

pub fn contextWindow(runtime_cache: *resident_runtime.ResidentRuntime, model_path: []const u8, backend: runtime.BackendPreference) !usize {
    if (runtime_cache.contextLength()) |length| return length;
    _ = try runtime_cache.promptTokenCount(model_path, "<|user|>\nhi</s>\n<|assistant|>\n", backend);
    return runtime_cache.contextLength().?;
}

test "trimAssistantReply stops at generic dialogue boundary" {
    const reply =
        \\Hi there! My name is Asa.
        \\
        \\Customer: I want to place an order.
    ;
    try std.testing.expectEqualStrings("Hi there! My name is Asa.", trimAssistantReply(reply));
}

test "trimAssistantReply stops at chat template marker" {
    const reply =
        \\Hello Alessio.
        \\<|user|>
        \\what is your name
    ;
    try std.testing.expectEqualStrings("Hello Alessio.", trimAssistantReply(reply));
}

test "trimAssistantReply stops at malformed emitted role marker" {
    const reply = "Hello there. </s>\n<user|>\nquestion";
    try std.testing.expectEqualStrings("Hello there.", trimAssistantReply(reply));
}

test "trimAssistantReply stops at partial stop marker" {
    const reply = "Hello there. </s";
    try std.testing.expectEqualStrings("Hello there.", trimAssistantReply(reply));
}

test "trimAssistantReply stops at shortest stop prefix" {
    const reply = "Hello there. </";
    try std.testing.expectEqualStrings("Hello there.", trimAssistantReply(reply));
}
