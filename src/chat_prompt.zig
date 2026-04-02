const std = @import("std");
const resident_runtime = @import("runtime/resident_runtime.zig");
const runtime = @import("runtime.zig");

pub const Role = enum { user, assistant };

pub const Message = struct {
    role: Role,
    content: []u8,
};

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
        _ = try runtime_cache.promptTokenCount(model_path, "User: hi\nAssistant:", backend);
        break :blk runtime_cache.contextLength().?;
    };
    const token_budget = context_length -| (max_tokens + 32);

    var start_index: usize = if (messages.len > 0) messages.len - 1 else 0;
    while (true) {
        const prompt = try renderMessages(allocator, messages[start_index..]);
        errdefer allocator.free(prompt);
        const count = runtime_cache.promptTokenCount(model_path, prompt, backend) catch |err| switch (err) {
            error.ContextOverflow => context_length + 1,
            else => return err,
        };
        if (count <= token_budget or start_index == 0) return prompt;
        allocator.free(prompt);
        start_index -= 1;
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
    return std.mem.trim(u8, reply[0..end], " \n\r\t");
}

fn isDialogueBoundary(line: []const u8) bool {
    if (line.len == 0) return false;

    const explicit = [_][]const u8{
        "User:",
        "Assistant:",
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
    for (messages) |message| {
        try writer.print("{s}: {s}\n\n", .{
            switch (message.role) {
                .user => "User",
                .assistant => "Assistant",
            },
            message.content,
        });
    }
    try writer.print("Assistant:", .{});
    return buf.toOwnedSlice(allocator);
}

test "trimAssistantReply stops at generic dialogue boundary" {
    const reply =
        \\Hi there! My name is Asa.
        \\
        \\Customer: I want to place an order.
    ;
    try std.testing.expectEqualStrings("Hi there! My name is Asa.", trimAssistantReply(reply));
}
