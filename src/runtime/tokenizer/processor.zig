const std = @import("std");
const tekken = @import("tekken.zig");
const chat_template = @import("chat_template.zig");

pub const Processor = struct {
    tokenizer: tekken.TekkenTokenizer,
    has_vision: bool = true,
    config_path: []const u8,

    pub fn deinit(self: *Processor, allocator: std.mem.Allocator) void {
        self.tokenizer.deinit(allocator);
        allocator.free(self.config_path);
        self.* = undefined;
    }

    pub fn applyChatTemplate(
        self: *Processor,
        allocator: std.mem.Allocator,
        messages: []const chat_template.Message,
        tools: []const chat_template.ToolSchema,
        response_format: chat_template.ResponseFormat,
    ) ![]u8 {
        return chat_template.applyMistral3ChatTemplate(
            allocator,
            messages,
            tools,
            response_format,
            .{
                .bos_token = "<s>",
                .eos_token = "</s>",
                .add_bos = true,
                .add_eos = false,
            },
        );
    }

    pub fn preprocessImages(
        self: *Processor,
        allocator: std.mem.Allocator,
        messages: []const chat_template.Message,
    ) !?struct {
        pixel_values: []u8,
        image_sizes: []const [2]u32,
    } {
        _ = allocator;
        _ = self;
        for (messages) |message| {
            for (message.content) |part| {
                switch (part) {
                    .image_url => return error.ImagesNotYetSupported,
                    .text => {},
                }
            }
        }
        return null;
    }
};

pub fn loadProcessor(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
) !Processor {
    const tekken_path = try std.fmt.allocPrint(allocator, "{s}/tekken.json", .{model_dir});
    defer allocator.free(tekken_path);

    var tokenizer = try tekken.loadTekkenTokenizer(allocator, tekken_path);
    errdefer tokenizer.deinit(allocator);

    tokenizer.bos_token_id = 1;
    tokenizer.eos_token_id = 2;
    tokenizer.pad_token_id = 11;
    tokenizer.unk_token_id = 0;

    const config_path = try allocator.dupe(u8, model_dir);

    return .{
        .tokenizer = tokenizer,
        .has_vision = true,
        .config_path = config_path,
    };
}
