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
    const markers = [_][]const u8{ "\nUser:", "\nAssistant:" };
    var end = reply.len;
    for (markers) |marker| {
        if (std.mem.indexOf(u8, reply, marker)) |idx| end = @min(end, idx);
    }
    return std.mem.trim(u8, reply[0..end], " \n\r\t");
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
