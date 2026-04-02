const std = @import("std");
const llama_loaded_runtime = @import("llama_loaded_runtime.zig");
const llama_fixture = @import("llama_fixture.zig");
const llama_speculative_runtime = @import("llama_speculative_runtime.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");

pub fn generate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !types.GenerationReport {
    var runtime = try llama_loaded_runtime.LoadedRuntime.init(allocator, model_path, options.backend, options.moon_quant, options.metal_profile);
    defer runtime.deinit();
    if (options.speculative.enabled()) {
        const draft_model_path = options.speculative.draft_model_path orelse return error.InvalidSpeculativeConfig;
        var draft_runtime = try llama_loaded_runtime.LoadedRuntime.init(allocator, draft_model_path, options.backend, options.moon_quant, false);
        defer draft_runtime.deinit();
        return llama_speculative_runtime.generateWithDraftRuntime(
            allocator,
            &runtime,
            &draft_runtime,
            prompt,
            options,
            llama_speculative_runtime.coldStartupBreakdown(&runtime, &draft_runtime),
        );
    }
    var report = try runtime.generate(prompt, options, runtime.model_load_ns);
    report.startup_breakdown.model_load_ns = runtime.model_load_ns;
    return report;
}

test "llama fixture runs deterministically on cpu" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "llama.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "llama.gguf");
    defer std.testing.allocator.free(path);

    var report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    });
    defer report.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(types.BackendUsed, .cpu), report.backend);
    try std.testing.expectEqualStrings(" b c!", report.generated_text);
    try std.testing.expectEqual(@as(usize, 1), report.prompt_token_count);
    try std.testing.expectEqual(@as(usize, 3), report.generated_token_count);
}

test "llama metal backend matches cpu reference when available" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "llama-metal.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "llama-metal.gguf");
    defer std.testing.allocator.free(path);

    var cpu_report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    });
    defer cpu_report.deinit(std.testing.allocator);

    var metal_report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 0,
        .temperature = 0,
        .backend = .metal,
    });
    defer metal_report.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(types.BackendUsed, .metal), metal_report.backend);
    try std.testing.expectEqualStrings(cpu_report.generated_text, metal_report.generated_text);
    try std.testing.expectEqual(cpu_report.prompt_token_count, metal_report.prompt_token_count);
    try std.testing.expectEqual(cpu_report.generated_token_count, metal_report.generated_token_count);
}

test "llama q4_k MoonQuant metal path matches cpu and generic q4 metal execution" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaQ4KFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "llama-q4k.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "llama-q4k.gguf");
    defer std.testing.allocator.free(path);

    const options = types.GenerationOptions{
        .max_tokens = 6,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    };

    var cpu_report = try generate(std.testing.allocator, path, "a", options);
    defer cpu_report.deinit(std.testing.allocator);

    var moon_quant_report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = options.max_tokens,
        .seed = options.seed,
        .temperature = options.temperature,
        .backend = .metal,
        .moon_quant = .enabled,
    });
    defer moon_quant_report.deinit(std.testing.allocator);

    var generic_q4_report = try generate(std.testing.allocator, path, "a", .{
        .max_tokens = options.max_tokens,
        .seed = options.seed,
        .temperature = options.temperature,
        .backend = .metal,
        .moon_quant = .disabled,
    });
    defer generic_q4_report.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(types.BackendUsed, .metal), moon_quant_report.backend);
    try std.testing.expectEqual(@as(types.BackendUsed, .metal), generic_q4_report.backend);
    try std.testing.expectEqualStrings(cpu_report.generated_text, moon_quant_report.generated_text);
    try std.testing.expectEqualStrings(cpu_report.generated_text, generic_q4_report.generated_text);
    try std.testing.expectEqualStrings(moon_quant_report.generated_text, generic_q4_report.generated_text);
    try std.testing.expectEqual(cpu_report.prompt_token_count, moon_quant_report.prompt_token_count);
    try std.testing.expectEqual(cpu_report.generated_token_count, moon_quant_report.generated_token_count);
    try std.testing.expectEqual(cpu_report.prompt_token_count, generic_q4_report.prompt_token_count);
    try std.testing.expectEqual(cpu_report.generated_token_count, generic_q4_report.generated_token_count);
}
