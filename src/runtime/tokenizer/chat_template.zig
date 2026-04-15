const std = @import("std");

pub const MessageRole = enum {
    system,
    user,
    assistant,
    tool,
};

pub const ContentPartTag = enum {
    text,
    image_url,
};

pub const ContentPart = union(ContentPartTag) {
    text: []const u8,
    image_url: []const u8,
};

pub const Message = struct {
    role: MessageRole,
    content: []const ContentPart,
};

pub const ToolSchema = struct {
    name: []const u8,
    description: []const u8,
    json_schema: []const u8,
};

pub const ResponseFormat = enum {
    text,
    json_object,
};

pub const ChatTemplateOptions = struct {
    bos_token: []const u8 = "<s>",
    eos_token: []const u8 = "</s>",
    add_bos: bool = true,
    add_eos: bool = false,
};

pub fn applyMistral3ChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    tools: []const ToolSchema,
    response_format: ResponseFormat,
    opts: ChatTemplateOptions,
) ![]u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    errdefer buffer.deinit();

    if (opts.add_bos) {
        try buffer.appendSlice(opts.bos_token);
    }

    const default_system = "You are Ministral-3-3B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI.";

    if (messages.len > 0 and messages[0].role == .system) {
        try buffer.appendSlice("[SYSTEM_PROMPT]");
        for (messages[0].content) |part| {
            switch (part) {
                .text => |t| try buffer.appendSlice(t),
                .image_url => return error.UnsupportedContentType,
            }
        }
        try buffer.appendSlice("[/SYSTEM_PROMPT]");
    } else {
        try buffer.appendSlice("[SYSTEM_PROMPT]");
        try buffer.appendSlice(default_system);
        try buffer.appendSlice("[/SYSTEM_PROMPT]");
    }

    if (tools.len > 0) {
        try buffer.appendSlice("[AVAILABLE_TOOLS]");
        try buffer.appendSlice("[\n");
        for (tools, 0..) |tool, i| {
            try buffer.appendSlice("  {\n");
            try buffer.appendSlice("    \"name\": \"");
            try buffer.appendSlice(tool.name);
            try buffer.appendSlice("\",\n    \"description\": \"");
            try buffer.appendSlice(tool.description);
            try buffer.appendSlice("\",\n    \"parameters\": ");
            try buffer.appendSlice(tool.json_schema);
            try buffer.appendSlice("\n  }");
            if (i < tools.len - 1) try buffer.appendSlice(",");
            try buffer.appendSlice("\n");
        }
        try buffer.appendSlice("]\n");
        try buffer.appendSlice("[/AVAILABLE_TOOLS]");
    }

    for (messages) |message| {
        switch (message.role) {
            .system => {
                if (messages[0].role != .system or message.content.ptr != messages[0].content.ptr) {
                    try buffer.appendSlice("[SYSTEM_PROMPT]");
                    for (message.content) |part| {
                        switch (part) {
                            .text => |t| try buffer.appendSlice(t),
                            .image_url => return error.UnsupportedContentType,
                        }
                    }
                    try buffer.appendSlice("[/SYSTEM_PROMPT]");
                }
            },
            .user => {
                try buffer.appendSlice("[INST]");
                for (message.content) |part| {
                    switch (part) {
                        .text => |t| try buffer.appendSlice(t),
                        .image_url => try buffer.appendSlice("[IMG]"),
                    }
                }
                try buffer.appendSlice("[/INST]");
            },
            .assistant => {
                for (message.content) |part| {
                    switch (part) {
                        .text => |t| try buffer.appendSlice(t),
                        .image_url => return error.UnsupportedContentType,
                    }
                }
                if (opts.add_eos) {
                    try buffer.appendSlice(opts.eos_token);
                }
            },
            .tool => {
                try buffer.appendSlice("[TOOL_RESULTS]");
                for (message.content) |part| {
                    switch (part) {
                        .text => |t| try buffer.appendSlice(t),
                        .image_url => return error.UnsupportedContentType,
                    }
                }
                try buffer.appendSlice("[/TOOL_RESULTS]");
            },
        }
    }

    if (response_format == .json_object) {
        try buffer.appendSlice("\nOutput JSON:");
    }

    return try buffer.toOwnedSlice(allocator);
}
