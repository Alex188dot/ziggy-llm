const std = @import("std");
const cli = @import("cli.zig");
const prompt_builder = @import("chat_prompt.zig");
const resident_runtime = @import("runtime/resident_runtime.zig");
const runtime = @import("runtime.zig");

const max_body_bytes = 1 << 20;

pub fn serve(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;
    var cache = resident_runtime.ResidentRuntime.init(allocator);
    defer cache.deinit();
    cache.setKeepAliveSeconds(-1);

    const address = try std.net.Address.parseIp4("127.0.0.1", config.port);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    try writer.print("server_listening: http://127.0.0.1:{d}\nmodel: {s}\n", .{ config.port, model_path });

    while (true) {
        const connection = try server.accept();
        defer connection.stream.close();

        var recv_buffer: [8192]u8 = undefined;
        var send_buffer: [8192]u8 = undefined;
        var conn_reader = connection.stream.reader(&recv_buffer);
        var conn_writer = connection.stream.writer(&send_buffer);
        var http_server = std.http.Server.init(conn_reader.interface(), &conn_writer.interface);

        while (http_server.reader.state == .ready) {
            var request = http_server.receiveHead() catch |err| switch (err) {
                error.HttpConnectionClosing => break,
                else => return err,
            };
            try handleRequest(allocator, &cache, model_path, config, &request);
        }
    }
}

fn handleRequest(
    allocator: std.mem.Allocator,
    cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    config: cli.Config,
    request: *std.http.Server.Request,
) !void {
    if (std.mem.eql(u8, request.head.target, "/health")) {
        return respondJson(request, "{\"status\":\"ok\"}");
    }
    if (std.mem.eql(u8, request.head.target, "/v1/unload")) {
        cache.unload();
        return respondJson(request, "{\"unloaded\":true}");
    }
    if (!request.head.method.requestHasBody()) {
        return respondText(request, .not_found, "not found");
    }

    var body_buffer: [4096]u8 = undefined;
    const body_reader = request.readerExpectNone(&body_buffer);
    var body_list = std.ArrayList(u8).empty;
    defer body_list.deinit(allocator);
    try body_reader.appendRemainingUnlimited(allocator, &body_list);
    if (body_list.items.len > max_body_bytes) return error.Overflow;
    const body = body_list.items;

    if (std.mem.eql(u8, request.head.target, "/v1/completions")) {
        return handleCompletion(allocator, cache, model_path, config, request, body);
    }
    if (std.mem.eql(u8, request.head.target, "/v1/chat/completions")) {
        return handleChatCompletion(allocator, cache, model_path, config, request, body);
    }
    return respondText(request, .not_found, "not found");
}

fn handleCompletion(
    allocator: std.mem.Allocator,
    cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    config: cli.Config,
    request: *std.http.Server.Request,
    body: []const u8,
) !void {
    const Payload = struct {
        prompt: []const u8,
        max_tokens: ?usize = null,
        seed: ?u64 = null,
        temperature: ?f32 = null,
        repeat_penalty: ?f32 = null,
        top_k: ?usize = null,
        top_p: ?f32 = null,
        min_p: ?f32 = null,
        keep_alive_seconds: ?i64 = null,
    };

    var parsed = try std.json.parseFromSlice(Payload, allocator, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const payload = parsed.value;
    if (payload.keep_alive_seconds) |seconds| cache.setKeepAliveSeconds(seconds);

    var report = try cache.generate(model_path, payload.prompt, mergeOptions(config, payload));
    defer report.deinit(allocator);
    if (payload.keep_alive_seconds != null and payload.keep_alive_seconds.? == 0) cache.unload();

    const response = try std.fmt.allocPrint(allocator, "{{\"text\":{s},\"backend\":\"{s}\",\"prompt_tokens\":{d},\"generated_tokens\":{d}}}", .{ try jsonStringify(allocator, report.generated_text), report.backend.label(), report.prompt_token_count, report.generated_token_count });
    defer allocator.free(response);
    return respondJson(request, response);
}

fn handleChatCompletion(
    allocator: std.mem.Allocator,
    cache: *resident_runtime.ResidentRuntime,
    model_path: []const u8,
    config: cli.Config,
    request: *std.http.Server.Request,
    body: []const u8,
) !void {
    const MessagePayload = struct { role: []const u8, content: []const u8 };
    const Payload = struct {
        messages: []MessagePayload,
        max_tokens: ?usize = null,
        seed: ?u64 = null,
        temperature: ?f32 = null,
        repeat_penalty: ?f32 = null,
        top_k: ?usize = null,
        top_p: ?f32 = null,
        min_p: ?f32 = null,
        keep_alive_seconds: ?i64 = null,
    };

    var parsed = try std.json.parseFromSlice(Payload, allocator, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const payload = parsed.value;
    if (payload.keep_alive_seconds) |seconds| cache.setKeepAliveSeconds(seconds);

    var messages = std.ArrayList(prompt_builder.Message).empty;
    defer prompt_builder.deinitMessages(allocator, &messages);
    for (payload.messages) |message| {
        const role: prompt_builder.Role = if (std.mem.eql(u8, message.role, "system"))
            .system
        else if (std.mem.eql(u8, message.role, "assistant"))
            .assistant
        else
            .user;
        try prompt_builder.appendMessage(allocator, &messages, role, message.content);
    }

    const prompt = try prompt_builder.buildPrompt(allocator, cache, model_path, config.backend, payload.max_tokens orelse config.max_tokens, messages.items);
    defer allocator.free(prompt);
    var report = try cache.generate(model_path, prompt, mergeOptions(config, payload));
    defer report.deinit(allocator);
    if (payload.keep_alive_seconds != null and payload.keep_alive_seconds.? == 0) cache.unload();

    const reply = prompt_builder.trimAssistantReply(report.generated_text);
    const reply_json = try jsonStringify(allocator, reply);
    defer allocator.free(reply_json);
    const response = try std.fmt.allocPrint(allocator, "{{\"message\":{{\"role\":\"assistant\",\"content\":{s}}},\"backend\":\"{s}\"}}", .{ reply_json, report.backend.label() });
    defer allocator.free(response);
    return respondJson(request, response);
}

fn mergeOptions(config: cli.Config, payload: anytype) runtime.GenerationOptions {
    return .{
        .max_tokens = payload.max_tokens orelse config.max_tokens,
        .seed = payload.seed orelse config.seed,
        .temperature = payload.temperature orelse config.temperature,
        .repeat_penalty = payload.repeat_penalty orelse config.repeat_penalty,
        .top_k = payload.top_k orelse config.top_k,
        .top_p = payload.top_p orelse config.top_p,
        .min_p = payload.min_p orelse config.min_p,
        .backend = config.backend,
        .metal_profile = config.metal_profile,
        .sampling_strategy = config.sampling_strategy,
    };
}

fn jsonStringify(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);
    try writer.writeByte('"');
    for (value) |byte| switch (byte) {
        '\\' => try writer.writeAll("\\\\"),
        '"' => try writer.writeAll("\\\""),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
    return buf.toOwnedSlice(allocator);
}

fn respondJson(request: *std.http.Server.Request, body: []const u8) !void {
    try request.respond(body, .{
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

fn respondText(request: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
    try request.respond(body, .{
        .status = status,
        .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
    });
}
