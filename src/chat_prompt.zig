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
    \\You are a helpful assistant
;

const summary_intro = "Context: ";
const summary_separator = " | ";
const compact_summary_chars: usize = 192;
const retained_recent_messages: usize = 4;
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

    var summary: ?[]u8 = null;
    errdefer if (summary) |value| allocator.free(value);
    if (start_index > 0) {
        summary = try summarizeMessages(allocator, messages[0..start_index]);
        if (summary) |value| estimated_tokens += try countSummaryTokens(runtime_cache, model_path, backend, value);
    }

    for (messages[start_index..]) |message| estimated_tokens += message.token_count;

    while (estimated_tokens > token_budget and start_index < messages.len) {
        estimated_tokens -|= messages[start_index].token_count;
        start_index += 1;
        if (start_index < messages.len) {
            estimated_tokens -|= messages[start_index].token_count;
            start_index += 1;
        }
        if (summary) |value| {
            allocator.free(value);
            summary = null;
        }
    }

    return renderConversation(allocator, summary, messages[start_index..]);
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

fn renderConversation(allocator: std.mem.Allocator, summary: ?[]const u8, messages: []const Message) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);
    try writer.print("System: {s}\n", .{system_message});
    if (summary) |value| {
        try writer.print("{s}{s}\n", .{ summary_intro, value });
    }
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

fn countSummaryTokens(
    runtime_cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    backend: runtime.BackendPreference,
    summary: []const u8,
) !usize {
    const snippet = try std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}\n", .{ summary_intro, summary });
    defer std.heap.page_allocator.free(snippet);
    return runtime_cache.promptTokenCount(model_path, snippet, backend);
}

fn summarizeMessages(allocator: std.mem.Allocator, messages: []const Message) !?[]u8 {
    if (messages.len == 0) return null;

    var summary = std.ArrayList(u8).empty;
    errdefer summary.deinit(allocator);

    const summary_start = messages.len - @min(messages.len, @as(usize, 4));
    for (messages[summary_start..], summary_start..) |message, index| {
        if (summary.items.len > 0) try summary.appendSlice(allocator, summary_separator);
        const role_prefix = switch (message.role) {
            .user => "U=",
            .assistant => "A=",
        };
        try summary.appendSlice(allocator, role_prefix);
        try appendCompactText(allocator, &summary, message.content);
        if (summary.items.len >= compact_summary_chars) break;
        if (index + 1 == messages.len) break;
    }

    if (summary.items.len == 0) return null;
    if (summary.items.len > compact_summary_chars) summary.items.len = compact_summary_chars;
    return @as(?[]u8, try summary.toOwnedSlice(allocator));
}

fn appendCompactText(allocator: std.mem.Allocator, out: *std.ArrayList(u8), text: []const u8) !void {
    var pending_space = false;
    for (text) |char| {
        if (char == '\n' or char == '\r' or char == '\t' or char == ' ') {
            pending_space = out.items.len > 0;
            continue;
        }
        if (pending_space and out.items.len < compact_summary_chars) {
            try out.append(allocator, ' ');
        }
        pending_space = false;
        if (out.items.len >= compact_summary_chars) break;
        try out.append(allocator, char);
    }
    if (out.items.len >= compact_summary_chars and out.items.len >= 3) {
        out.items[out.items.len - 3] = '.';
        out.items[out.items.len - 2] = '.';
        out.items[out.items.len - 1] = '.';
    }
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
