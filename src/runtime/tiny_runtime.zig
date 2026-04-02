const std = @import("std");
const terminal = @import("../terminal.zig");
const backend_api = @import("backend.zig");
const cpu_backend = @import("cpu_backend.zig");
const metal_backend = @import("metal_backend.zig");
const tiny_fixture = @import("tiny_fixture.zig");
const tiny_model = @import("tiny_model.zig");
const types = @import("types.zig");

const Session = struct {
    model: *const tiny_model.Model,
    backend: backend_api.MatVecBackend,
    token_buffer: []u32,
    k_cache: []f32,
    v_cache: []f32,
    hidden: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    ff_hidden: []f32,
    logits: []f32,
    scores: []f32,
    position: usize = 0,

    fn init(
        allocator: std.mem.Allocator,
        model: *const tiny_model.Model,
        backend: backend_api.MatVecBackend,
        token_capacity: usize,
    ) !Session {
        return .{
            .model = model,
            .backend = backend,
            .token_buffer = try allocator.alloc(u32, token_capacity),
            .k_cache = try allocator.alloc(f32, model.context_length * model.embedding_length),
            .v_cache = try allocator.alloc(f32, model.context_length * model.embedding_length),
            .hidden = try allocator.alloc(f32, model.embedding_length),
            .q = try allocator.alloc(f32, model.embedding_length),
            .k = try allocator.alloc(f32, model.embedding_length),
            .v = try allocator.alloc(f32, model.embedding_length),
            .attn = try allocator.alloc(f32, model.embedding_length),
            .ff_hidden = try allocator.alloc(f32, model.feed_forward_length),
            .logits = try allocator.alloc(f32, model.tokenizer.tokens.len),
            .scores = try allocator.alloc(f32, model.context_length),
        };
    }

    fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        allocator.free(self.token_buffer);
        allocator.free(self.k_cache);
        allocator.free(self.v_cache);
        allocator.free(self.hidden);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.attn);
        allocator.free(self.ff_hidden);
        allocator.free(self.logits);
        allocator.free(self.scores);
        self.* = undefined;
    }

    fn runPrompt(self: *Session, prompt_tokens: []const u32) !void {
        if (prompt_tokens.len == 0) return error.EmptyPrompt;
        for (prompt_tokens) |token| _ = try self.step(token);
    }

    fn step(self: *Session, token_id: u32) ![]const f32 {
        if (self.position >= self.token_buffer.len or self.position >= self.model.context_length) {
            return error.ContextOverflow;
        }

        copyEmbedding(self.hidden, self.model.token_embd, self.model.embedding_length, token_id);

        try self.backend.matVec(self.q, self.model.attn_q, self.hidden, self.model.embedding_length, self.model.embedding_length);
        try self.backend.matVec(self.k, self.model.attn_k, self.hidden, self.model.embedding_length, self.model.embedding_length);
        try self.backend.matVec(self.v, self.model.attn_v, self.hidden, self.model.embedding_length, self.model.embedding_length);

        const cache_offset = self.position * self.model.embedding_length;
        @memcpy(self.k_cache[cache_offset .. cache_offset + self.model.embedding_length], self.k);
        @memcpy(self.v_cache[cache_offset .. cache_offset + self.model.embedding_length], self.v);

        const scale = @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(self.model.embedding_length)));
        for (0..self.position + 1) |index| {
            const other_k = self.k_cache[index * self.model.embedding_length ..][0..self.model.embedding_length];
            self.scores[index] = dot(self.q, other_k) * scale;
        }
        softmaxInPlace(self.scores[0 .. self.position + 1]);

        @memset(self.attn, 0);
        for (0..self.position + 1) |index| {
            const weight = self.scores[index];
            const other_v = self.v_cache[index * self.model.embedding_length ..][0..self.model.embedding_length];
            for (self.attn, other_v) |*dst, src| dst.* += weight * src;
        }

        try self.backend.matVec(self.q, self.model.attn_out, self.attn, self.model.embedding_length, self.model.embedding_length);
        for (self.hidden, self.q) |*dst, src| dst.* += src;

        try self.backend.matVec(self.ff_hidden, self.model.ffn_up, self.hidden, self.model.feed_forward_length, self.model.embedding_length);
        for (self.ff_hidden) |*value| value.* = @max(value.*, 0);
        try self.backend.matVec(self.q, self.model.ffn_down, self.ff_hidden, self.model.embedding_length, self.model.feed_forward_length);
        for (self.hidden, self.q) |*dst, src| dst.* += src;

        try self.backend.matVec(self.logits, self.model.output, self.hidden, self.model.tokenizer.tokens.len, self.model.embedding_length);

        self.token_buffer[self.position] = token_id;
        self.position += 1;
        return self.logits;
    }
};

pub fn generate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !types.GenerationReport {
    const startup_begin = std.time.nanoTimestamp();
    var spinner = terminal.Spinner{};
    try spinner.start();
    errdefer spinner.stop();

    var model = try tiny_model.loadModel(allocator, model_path);
    defer model.deinit(allocator);

    const matvec_backend = try selectBackend(allocator, options.backend, &model);
    defer matvec_backend.deinit(allocator);

    const prompt_capacity = prompt.len + options.max_tokens + 4;
    var session = try Session.init(allocator, &model, matvec_backend, @min(model.context_length, prompt_capacity));
    defer session.deinit(allocator);

    const startup_end = std.time.nanoTimestamp();
    spinner.stop();

    const prompt_begin = std.time.nanoTimestamp();
    const prompt_token_count = try model.tokenizer.encodeInto(prompt, session.token_buffer);
    if (prompt_token_count == 0) return error.EmptyPrompt;
    try session.runPrompt(session.token_buffer[0..prompt_token_count]);
    const prompt_end = std.time.nanoTimestamp();

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(options.seed);
    const random = rng.random();
    var generated_token_count: usize = 0;

    const decode_begin = std.time.nanoTimestamp();
    while (generated_token_count < options.max_tokens) {
        const next_token = sampleToken(session.logits, options.temperature, random);
        if (model.tokenizer.eos_token_id != null and next_token == model.tokenizer.eos_token_id.?) break;

        const token_text = try model.tokenizer.tokenString(next_token);
        if (!(model.tokenizer.bos_token_id != null and next_token == model.tokenizer.bos_token_id.?)) {
            try output.appendSlice(allocator, token_text);
        }
        _ = try session.step(next_token);
        generated_token_count += 1;
    }
    const decode_end = std.time.nanoTimestamp();

    return .{
        .generated_text = try output.toOwnedSlice(allocator),
        .prompt_token_count = prompt_token_count,
        .generated_token_count = generated_token_count,
        .startup_ns = types.deltaNs(startup_begin, startup_end),
        .prompt_ns = types.deltaNs(prompt_begin, prompt_end),
        .decode_ns = types.deltaNs(decode_begin, decode_end),
        .seed = options.seed,
        .temperature = options.temperature,
        .backend = matvec_backend.label,
    };
}

fn selectBackend(
    allocator: std.mem.Allocator,
    preference: types.BackendPreference,
    model: *const tiny_model.Model,
) !backend_api.MatVecBackend {
    switch (preference) {
        .cpu => return cpu_backend.create(),
        .metal => {
            const explicit_backend = try metal_backend.create(allocator);
            errdefer explicit_backend.deinit(allocator);
            try metal_backend.prewarm(explicit_backend, model);
            return explicit_backend;
        },
        .auto => {
            const metal = metal_backend.create(allocator) catch |err| {
                if (isRecoverableMetalError(err)) return cpu_backend.create();
                return err;
            };
            errdefer metal.deinit(allocator);
            metal_backend.prewarm(metal, model) catch |err| {
                if (isRecoverableMetalError(err)) return cpu_backend.create();
                return err;
            };
            return metal;
        },
    }
}

fn isRecoverableMetalError(err: anyerror) bool {
    return switch (err) {
        error.MetalDisabled,
        error.MetalUnavailable,
        error.MetalInitializationFailed,
        error.MetalCompilationFailed,
        error.MetalBufferError,
        => true,
        else => false,
    };
}

fn copyEmbedding(out: []f32, embedding: []const f32, embedding_length: usize, token_id: u32) void {
    const start = token_id * embedding_length;
    @memcpy(out, embedding[start .. start + embedding_length]);
}

fn dot(a: []const f32, b: []const f32) f32 {
    var total: f32 = 0;
    for (a, b) |lhs, rhs| total += lhs * rhs;
    return total;
}

fn softmaxInPlace(values: []f32) void {
    var max_value = values[0];
    for (values[1..]) |value| max_value = @max(max_value, value);

    var sum: f32 = 0;
    for (values) |*value| {
        value.* = @exp(value.* - max_value);
        sum += value.*;
    }
    for (values) |*value| value.* /= sum;
}

fn sampleToken(logits: []const f32, temperature: f32, random: std.Random) u32 {
    if (temperature <= 0) return argmax(logits);

    var max_logit = logits[0];
    for (logits[1..]) |value| max_logit = @max(max_logit, value);

    var sum: f32 = 0;
    for (logits) |value| sum += @exp((value - max_logit) / temperature);

    var threshold = random.float(f32) * sum;
    for (logits, 0..) |value, index| {
        threshold -= @exp((value - max_logit) / temperature);
        if (threshold <= 0) return @intCast(index);
    }
    return @intCast(logits.len - 1);
}

fn argmax(values: []const f32) u32 {
    var best_index: usize = 0;
    var best_value = values[0];
    for (values[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return @intCast(best_index);
}

test "ziggy-tiny model runs end to end deterministically" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try tiny_fixture.makeTinyModelFixture(std.testing.allocator, .{});
    defer std.testing.allocator.free(fixture);
    try tiny_fixture.writeFixtureFile(tmp.dir, "tiny.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "tiny.gguf");
    defer std.testing.allocator.free(path);

    var report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 123,
        .temperature = 0,
        .backend = .cpu,
    });
    defer report.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("bc!", report.generated_text);
    try std.testing.expectEqual(@as(types.BackendUsed, .cpu), report.backend);
    try std.testing.expectEqual(@as(usize, 1), report.prompt_token_count);
    try std.testing.expectEqual(@as(usize, 3), report.generated_token_count);
}

test "seeded sampling stays reproducible" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try tiny_fixture.makeTinyModelFixture(std.testing.allocator, .{
        .ambiguous_a = true,
    });
    defer std.testing.allocator.free(fixture);
    try tiny_fixture.writeFixtureFile(tmp.dir, "seeded.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "seeded.gguf");
    defer std.testing.allocator.free(path);

    var first = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 2,
        .seed = 77,
        .temperature = 1.0,
        .backend = .cpu,
    });
    defer first.deinit(std.testing.allocator);

    var second = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 2,
        .seed = 77,
        .temperature = 1.0,
        .backend = .cpu,
    });
    defer second.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings(first.generated_text, second.generated_text);
}

test "unsupported quantization fails clearly" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try tiny_fixture.makeTinyModelFixture(std.testing.allocator, .{
        .file_type = 7,
    });
    defer std.testing.allocator.free(fixture);
    try tiny_fixture.writeFixtureFile(tmp.dir, "bad-file-type.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "bad-file-type.gguf");
    defer std.testing.allocator.free(path);

    try std.testing.expectError(error.UnsupportedFileType, generate(std.testing.allocator, path, "a", .{
        .backend = .cpu,
    }));
}

test "metal backend matches cpu reference when available" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try tiny_fixture.makeTinyModelFixture(std.testing.allocator, .{
        .ambiguous_a = true,
    });
    defer std.testing.allocator.free(fixture);
    try tiny_fixture.writeFixtureFile(tmp.dir, "metal.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "metal.gguf");
    defer std.testing.allocator.free(path);

    var cpu_report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 2,
        .seed = 77,
        .temperature = 1.0,
        .backend = .cpu,
    });
    defer cpu_report.deinit(std.testing.allocator);

    var metal_report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 2,
        .seed = 77,
        .temperature = 1.0,
        .backend = .metal,
    });
    defer metal_report.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(types.BackendUsed, .metal), metal_report.backend);
    try std.testing.expectEqualStrings(cpu_report.generated_text, metal_report.generated_text);
    try std.testing.expectEqual(cpu_report.prompt_token_count, metal_report.prompt_token_count);
    try std.testing.expectEqual(cpu_report.generated_token_count, metal_report.generated_token_count);
}
