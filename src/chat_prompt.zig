const std = @import("std");
const resident_runtime = @import("runtime/resident_runtime.zig");
const runtime = @import("runtime.zig");

pub const Role = enum { user, assistant };

pub const Message = struct {
    role: Role,
    content: []u8,
    token_count: usize = 0,
};

const system_message =
    \\You are a helpful AI assistant.
;

const retained_recent_messages: usize = 2;
const token_safety_margin: usize = 128;

pub const history_window_messages: usize = retained_recent_messages;

pub fn deinitMessages(allocator: std.mem.Allocator, messages: *std.ArrayList(Message)) void {
    for (messages.items) |message| allocator.free(message.content);
    messages.deinit(allocator);
}

pub fn appendMessage(allocator: std.mem.Allocator, messages: *std.ArrayList(Message), role: Role, content: []const u8) !void {
    try messages.append(allocator, .{
        .role = role,
        .content = try allocator.dupe(u8, std.mem.trim(u8, content, " \n\r\t")),
        .token_count = 0,
    });
}

pub fn buildPrompt(
    allocator: std.mem.Allocator,
    runtime_cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    backend: runtime.BackendPreference,
    max_tokens: usize,
    messages: []Message,
) ![]u8 {
    const context_length = runtime_cache.contextLength() orelse blk: {
        _ = try runtime_cache.promptTokenCount(model_path, "System: hi\nUser: hi\nAssistant:", backend);
        break :blk runtime_cache.contextLength().?;
    };
    const token_budget = context_length -| (max_tokens + token_safety_margin);

    const base_token_count = try promptBaseTokenCount(runtime_cache, model_path, backend);
    var estimated_tokens = base_token_count;

    for (messages) |*message| {
        if (message.token_count == 0) {
            message.token_count = try countRenderedMessageTokens(runtime_cache, model_path, backend, message.role, message.content);
        }
    }

    var start_index: usize = if (messages.len > retained_recent_messages) messages.len - retained_recent_messages else 0;
    if (start_index % 2 != 0) start_index -= 1;

    for (messages[start_index..]) |message| estimated_tokens += message.token_count;

    while (estimated_tokens > token_budget and start_index < messages.len) {
        estimated_tokens -|= messages[start_index].token_count;
        start_index += 1;
        if (start_index < messages.len) {
            estimated_tokens -|= messages[start_index].token_count;
            start_index += 1;
        }
    }

    return renderConversation(allocator, messages[start_index..]);
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

pub fn hasCompletedAssistantReply(reply: []const u8) bool {
    var lines = std.mem.splitScalar(u8, reply, '\n');
    var offset: usize = 0;
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (offset > 0 and isDialogueBoundary(trimmed)) return true;
        offset += line.len + 1;
    }

    for ([_][]const u8{ "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<user|>", "<assistant|>", "<system|" }) |marker| {
        if (std.mem.indexOf(u8, reply, marker) != null) return true;
    }
    return false;
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
        "U=",
        "A=",
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

fn renderConversation(allocator: std.mem.Allocator, messages: []const Message) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);
    try writer.print("System: {s}\n", .{system_message});
    for (messages) |message| {
        const tag = switch (message.role) {
            .user => "User",
            .assistant => "Assistant",
        };
        try writer.print("{s}: {s}\n", .{ tag, message.content });
    }
    try writer.print("Assistant:", .{});
    return buf.toOwnedSlice(allocator);
}

pub fn contextWindow(runtime_cache: *resident_runtime.ResidentRuntime, model_path: []const u8, backend: runtime.BackendPreference) !usize {
    if (runtime_cache.contextLength()) |length| return length;
    _ = try runtime_cache.promptTokenCount(model_path, "System: hi\nUser: hi\nAssistant:", backend);
    return runtime_cache.contextLength().?;
}

fn promptBaseTokenCount(
    runtime_cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    backend: runtime.BackendPreference,
) !usize {
    return runtime_cache.promptTokenCount(model_path, "System: " ++ system_message ++ "\nAssistant:", backend);
}

fn countRenderedMessageTokens(
    runtime_cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    backend: runtime.BackendPreference,
    role: Role,
    content: []const u8,
) !usize {
    const tag = switch (role) {
        .user => "User",
        .assistant => "Assistant",
    };
    const snippet = try std.fmt.allocPrint(std.heap.page_allocator, "{s}: {s}\n", .{ tag, content });
    defer std.heap.page_allocator.free(snippet);
    return runtime_cache.promptTokenCount(model_path, snippet, backend);
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
