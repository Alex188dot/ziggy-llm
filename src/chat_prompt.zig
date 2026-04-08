const std = @import("std");
const gguf = @import("gguf.zig");
const resident_runtime = @import("runtime/resident_runtime.zig");
const runtime = @import("runtime.zig");

pub const Role = enum { system, user, assistant };

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
    context_length_limit: usize,
    max_tokens: usize,
    messages: []Message,
) ![]u8 {
    const template_style = try runtime_cache.chatTemplateStyle(model_path, backend, context_length_limit);
    const context_length = try contextWindow(runtime_cache, model_path, backend, context_length_limit);
    const token_budget = context_length -| (max_tokens + token_safety_margin);

    const system_prefix_len = countLeadingSystemMessages(messages);
    const include_default_system = shouldInjectDefaultSystem(template_style, system_prefix_len);
    const base_token_count = try promptBaseTokenCount(runtime_cache, model_path, backend, template_style, include_default_system);
    var estimated_tokens = base_token_count;

    for (messages) |*message| {
        if (message.token_count == 0) {
            message.token_count = try countRenderedMessageTokens(runtime_cache, model_path, backend, template_style, message.role, message.content);
        }
    }

    for (messages[0..system_prefix_len]) |message| estimated_tokens += message.token_count;

    var start_index = system_prefix_len;
    const conversational_count = messages.len - system_prefix_len;
    if (conversational_count > retained_recent_messages) {
        start_index = messages.len - retained_recent_messages;
        if ((start_index - system_prefix_len) % 2 != 0) start_index -= 1;
    }

    for (messages[start_index..]) |message| estimated_tokens += message.token_count;

    while (estimated_tokens > token_budget and start_index < messages.len) {
        estimated_tokens -|= messages[start_index].token_count;
        start_index += 1;
        if (start_index < messages.len) {
            estimated_tokens -|= messages[start_index].token_count;
            start_index += 1;
        }
    }

    return renderConversation(allocator, template_style, messages[0..system_prefix_len], messages[start_index..], include_default_system);
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

    // Check for EOS markers and cut at first occurrence
    const eos_markers = [_][]const u8{ "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<user|>", "<assistant|>", "<system|>", "\xe0\xb8\xad\xe0\xb9\x87\xe0\xb8\xad\xe0\xb8\xaa", "<|im_start|>" };
    for (eos_markers) |marker| {
        if (std.mem.indexOf(u8, reply[0..end], marker)) |index| {
            end = @min(end, index);
        }
    }

    // FIXED: stronger end-of-reply detection
    if (std.mem.indexOf(u8, reply[0..end], "\xe0\xb8\xad\xe0\xb9\x87\xe0\xb8\xad\xe0\xb8\xaa") != null or
        std.mem.indexOf(u8, reply[0..end], "</s>") != null or
        std.mem.indexOf(u8, reply[0..end], "<|assistant|>") != null) {
        // Already handled by the loop above, but ensures we detect these early
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

    for ([_][]const u8{ "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<user|>", "<assistant|>", "<system|>", "\xe0\xb8\xad\xe0\xb9\x87\xe0\xb8\xad\xe0\xb8\xaa", "<|im_start|>" }) |marker| {
        if (std.mem.indexOf(u8, reply, marker) != null) return true;
    }
    return false;
}

fn isDialogueBoundary(line: []const u8) bool {
    if (line.len == 0) return false;

    const explicit = [_][]const u8{
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_start|>system",
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

fn countLeadingSystemMessages(messages: []const Message) usize {
    var count: usize = 0;
    while (count < messages.len and messages[count].role == .system) : (count += 1) {}
    return count;
}

fn shouldInjectDefaultSystem(template_style: gguf.ChatTemplateStyle, system_prefix_len: usize) bool {
    return template_style == .generic and system_prefix_len == 0;
}

fn renderConversation(
    allocator: std.mem.Allocator,
    template_style: gguf.ChatTemplateStyle,
    system_messages: []const Message,
    messages: []const Message,
    include_default_system: bool,
) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);

    switch (template_style) {
        .generic => {
            if (include_default_system) {
                try writer.print("System: {s}\n", .{system_message});
            } else {
                for (system_messages) |message| {
                    try writer.print("System: {s}\n", .{message.content});
                }
            }
            for (messages) |message| {
                const tag = switch (message.role) {
                    .system => "System",
                    .user => "User",
                    .assistant => "Assistant",
                };
                try writer.print("{s}: {s}\n", .{ tag, message.content });
            }
            try writer.print("Assistant:", .{});
        },
        .chatml => {
            if (include_default_system) {
                try writer.print("<|system|>\n{s}</s>\n", .{system_message});
            } else {
                for (system_messages) |message| {
                    try writer.print("<|system|>\n{s}</s>\n", .{message.content});
                }
            }
            for (messages) |message| {
                const tag = switch (message.role) {
                    .system => "system",
                    .user => "user",
                    .assistant => "assistant",
                };
                try writer.print("<|{s}|>\n{s}</s>\n", .{ tag, message.content });
            }
            try writer.print("<|assistant|>\n", .{});
        },
        .qwen => {
            if (include_default_system) {
                try writer.print("<|im_start|>system\n{s}\xe0\xb8\xad\xe0\xb9\x87\xe0\xb8\xad\xe0\xb8\xaa\n", .{system_message});
            } else {
                for (system_messages) |message| {
                    try writer.print("<|im_start|>system\n{s}\xe0\xb8\xad\xe0\xb9\x87\xe0\xb8\xad\xe0\xb8\xaa\n", .{message.content});
                }
            }
            for (messages) |message| {
                const tag = switch (message.role) {
                    .system => "system",
                    .user => "user",
                    .assistant => "assistant",
                };
                try writer.print("<|im_start|>{s}\n{s}\xe0\xb8\xad\xe0\xb9\x87\xe0\xb8\xad\xe0\xb8\xaa\n", .{ tag, message.content });
            }
            try writer.print("<|im_start|>assistant\n", .{});
        },
    }
    return buf.toOwnedSlice(allocator);
}

pub fn contextWindow(runtime_cache: *resident_runtime.ResidentRuntime, model_path: []const u8, backend: runtime.BackendPreference, context_length_limit: usize) !usize {
    if (runtime_cache.contextLength(context_length_limit)) |length| return length;
    _ = try runtime_cache.chatTemplateStyle(model_path, backend, context_length_limit);
    return runtime_cache.contextLength(context_length_limit).?;
}

fn promptBaseTokenCount(
    runtime_cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    backend: runtime.BackendPreference,
    template_style: gguf.ChatTemplateStyle,
    include_default_system: bool,
) !usize {
    const scaffold = promptScaffold(template_style, include_default_system);
    return runtime_cache.promptTokenCount(model_path, scaffold, backend);
}

fn countRenderedMessageTokens(
    runtime_cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    backend: runtime.BackendPreference,
    template_style: gguf.ChatTemplateStyle,
    role: Role,
    content: []const u8,
) !usize {
    const snippet = switch (template_style) {
        .generic => blk: {
            const tag = switch (role) {
                .system => "System",
                .user => "User",
                .assistant => "Assistant",
            };
            break :blk try std.fmt.allocPrint(std.heap.page_allocator, "{s}: {s}\n", .{ tag, content });
        },
        .chatml => blk: {
            const tag = switch (role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            };
            break :blk try std.fmt.allocPrint(std.heap.page_allocator, "<|{s}|>\n{s}</s>\n", .{ tag, content });
        },
        .qwen => blk: {
            const tag = switch (role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            };
            break :blk try std.fmt.allocPrint(std.heap.page_allocator, "<|im_start|>{s}\n{s}\xe0\xb8\xad\xe0\xb9\x87\xe0\xb8\xad\xe0\xb8\xaa\n", .{ tag, content });
        },
    };
    defer std.heap.page_allocator.free(snippet);
    return runtime_cache.promptTokenCount(model_path, snippet, backend);
}

fn promptScaffold(template_style: gguf.ChatTemplateStyle, include_default_system: bool) []const u8 {
    return switch (template_style) {
        .generic => if (include_default_system) "System: " ++ system_message ++ "\nAssistant:" else "Assistant:",
        .chatml => if (include_default_system) "<|system|>\n" ++ system_message ++ "</s>\n<|assistant|>\n" else "<|assistant|>\n",
        .qwen => if (include_default_system) "<|im_start|>system\n" ++ system_message ++ "\xe0\xb8\xad\xe0\xb9\x87\xe0\xb8\xad\xe0\xb8\xaa\n<|im_start|>assistant\n" else "<|im_start|>assistant\n",
    };
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

test "trimAssistantReply does not stop at partial stop marker" {
    const reply = "Hello there. </s";
    try std.testing.expectEqualStrings("Hello there. </s", trimAssistantReply(reply));
}

test "trimAssistantReply does not stop at shortest stop prefix" {
    const reply = "Hello there. </";
    try std.testing.expectEqualStrings("Hello there. </", trimAssistantReply(reply));
}

test "renderConversation uses chatml markers when GGUF template requests it" {
    const messages = [_]Message{
        .{ .role = .user, .content = @constCast("my name is alessio") },
    };
    const rendered = try renderConversation(std.testing.allocator, .chatml, &.{}, &messages, false);
    defer std.testing.allocator.free(rendered);

    try std.testing.expectEqualStrings(
        "<|user|>\nmy name is alessio</s>\n<|assistant|>\n",
        rendered,
    );
}
