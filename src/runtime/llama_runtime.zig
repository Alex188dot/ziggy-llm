const std = @import("std");
const llama_cpu = @import("../model/loader.zig");
const backend_api = @import("backend.zig");
const llama_fixture = @import("llama_fixture.zig");
const llama_metal = @import("gpu/metal/tensor_store.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");

const ExecutionResources = struct {
    backend: ?backend_api.MatVecBackend = null,
    dense_tensors: ?llama_metal.DenseTensorStore = null,
    startup_breakdown: types.StartupBreakdown = .{},
    startup_profile_summary: ?[]u8 = null,

    fn deinit(self: *ExecutionResources, allocator: std.mem.Allocator) void {
        if (self.backend) |backend| backend.deinit(allocator);
        if (self.dense_tensors) |*dense_tensors| dense_tensors.deinit();
        if (self.startup_profile_summary) |summary| allocator.free(summary);
        self.* = undefined;
    }
};

pub fn generate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !types.GenerationReport {
    const model_load_begin = std.time.nanoTimestamp();
    var model = try llama_cpu.loadModel(allocator, model_path);
    const model_load_ns = types.deltaNs(model_load_begin, std.time.nanoTimestamp());
    var execution = try selectExecution(allocator, &model, options.backend, options.moon_quant, options.metal_profile);
    defer execution.deinit(allocator);

    const lookup = if (execution.dense_tensors) |*dense_tensors|
        llama_cpu.DenseTensorLookup{
            .ctx = dense_tensors,
            .get_fn = lookupDenseTensor,
            .get_by_offset_fn = lookupDenseTensorByOffset,
            .get_raw_by_offset_fn = lookupRawTensorByOffset,
            .get_moon_quant_by_offset_fn = lookupMoonQuantTensorByOffset,
        }
    else
        null;
    var llama_report = try llama_cpu.generateLoaded(
        allocator,
        &model,
        prompt,
        options,
        execution.backend,
        lookup,
    );
    llama_report.startup_ns += model_load_ns + execution.startup_breakdown.tensor_prepare_ns + execution.startup_breakdown.backend_init_ns + execution.startup_breakdown.metal_prewarm_ns;
    llama_report.ttft_ns += model_load_ns + execution.startup_breakdown.tensor_prepare_ns + execution.startup_breakdown.backend_init_ns + execution.startup_breakdown.metal_prewarm_ns;
    llama_report.startup_breakdown.model_load_ns = model_load_ns;
    llama_report.startup_breakdown.tensor_prepare_ns = execution.startup_breakdown.tensor_prepare_ns;
    llama_report.startup_breakdown.backend_init_ns = execution.startup_breakdown.backend_init_ns;
    llama_report.startup_breakdown.metal_prewarm_ns = execution.startup_breakdown.metal_prewarm_ns;

    const combined_profile_summary = try combineProfileSummaries(
        allocator,
        execution.startup_profile_summary,
        llama_report.metal_profile_summary,
    );
    if (llama_report.metal_profile_summary) |summary| allocator.free(summary);

    return .{
        .generated_text = llama_report.generated_text,
        .prompt_token_count = llama_report.prompt_token_count,
        .reused_prompt_token_count = llama_report.reused_prompt_token_count,
        .generated_token_count = llama_report.generated_token_count,
        .startup_ns = llama_report.startup_ns,
        .prompt_ns = llama_report.prompt_ns,
        .ttft_ns = llama_report.ttft_ns,
        .decode_ns = llama_report.decode_ns,
        .seed = options.seed,
        .temperature = options.temperature,
        .backend = llama_report.backend,
        .sampling_strategy = llama_report.sampling_strategy,
        .sampling_path = llama_report.sampling_path,
        .readback_mode = llama_report.readback_mode,
        .startup_breakdown = llama_report.startup_breakdown,
        .metal_profile_summary = combined_profile_summary,
    };
}

fn selectExecution(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    preference: types.BackendPreference,
    moon_quant_mode: types.MoonQuantMode,
    startup_profile_enabled: bool,
) !ExecutionResources {
    return switch (preference) {
        .cpu => .{},
        .metal => try createMetalExecution(allocator, model, moon_quant_mode, startup_profile_enabled),
        .auto => createMetalExecution(allocator, model, moon_quant_mode, startup_profile_enabled) catch |err| {
            if (isRecoverableMetalError(err)) return .{};
            return err;
        },
    };
}

fn createMetalExecution(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    moon_quant_mode: types.MoonQuantMode,
    startup_profile_enabled: bool,
) !ExecutionResources {
    var dense_tensors = llama_metal.DenseTensorStore.init(allocator);
    errdefer dense_tensors.deinit();
    var startup_profiler = llama_metal.StartupProfiler{ .enabled = startup_profile_enabled };
    const tensor_prepare_begin = std.time.nanoTimestamp();
    try dense_tensors.populate(model, moon_quant_mode, if (startup_profiler.enabled) &startup_profiler else null);
    const tensor_prepare_ns = types.deltaNs(tensor_prepare_begin, std.time.nanoTimestamp());

    const backend_init_begin = std.time.nanoTimestamp();
    const backend = try metal_backend.create(allocator);
    const backend_init_ns = types.deltaNs(backend_init_begin, std.time.nanoTimestamp());
    errdefer backend.deinit(allocator);
    const metal_prewarm_begin = std.time.nanoTimestamp();
    try dense_tensors.prewarm(backend, if (startup_profiler.enabled) &startup_profiler else null);
    const metal_prewarm_ns = types.deltaNs(metal_prewarm_begin, std.time.nanoTimestamp());
    const startup_profile_summary = if (startup_profiler.enabled) try startup_profiler.renderSummary(allocator) else null;

    return .{
        .backend = backend,
        .dense_tensors = dense_tensors,
        .startup_breakdown = .{
            .tensor_prepare_ns = tensor_prepare_ns,
            .backend_init_ns = backend_init_ns,
            .metal_prewarm_ns = metal_prewarm_ns,
        },
        .startup_profile_summary = startup_profile_summary,
    };
}

fn combineProfileSummaries(
    allocator: std.mem.Allocator,
    startup_summary: ?[]const u8,
    decode_summary: ?[]const u8,
) !?[]u8 {
    const startup = startup_summary orelse "";
    const decode = decode_summary orelse "";
    if (startup.len == 0 and decode.len == 0) return null;

    var buffer = std.ArrayList(u8).empty;
    errdefer buffer.deinit(allocator);
    if (startup.len > 0) try buffer.appendSlice(allocator, startup);
    if (startup.len > 0 and decode.len > 0 and startup[startup.len - 1] != '\n') {
        try buffer.append(allocator, '\n');
    }
    if (decode.len > 0) try buffer.appendSlice(allocator, decode);
    return try buffer.toOwnedSlice(allocator);
}

fn lookupDenseTensor(ctx: ?*const anyopaque, tensor: llama_cpu.TensorRef) ?[]const f32 {
    const dense_tensors: *const llama_metal.DenseTensorStore = @ptrCast(@alignCast(ctx orelse return null));
    return dense_tensors.get(tensor);
}

fn lookupDenseTensorByOffset(ctx: ?*const anyopaque, offset: u64) ?[]const f32 {
    const dense_tensors: *const llama_metal.DenseTensorStore = @ptrCast(@alignCast(ctx orelse return null));
    return dense_tensors.getByOffset(offset);
}

fn lookupRawTensorByOffset(ctx: ?*const anyopaque, offset: u64) ?[]const u8 {
    const dense_tensors: *const llama_metal.DenseTensorStore = @ptrCast(@alignCast(ctx orelse return null));
    return dense_tensors.getRawByOffset(offset);
}

fn lookupMoonQuantTensorByOffset(ctx: ?*const anyopaque, offset: u64) ?[]const u8 {
    const dense_tensors: *const llama_metal.DenseTensorStore = @ptrCast(@alignCast(ctx orelse return null));
    return dense_tensors.getMoonQuantBytesByOffset(offset);
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
