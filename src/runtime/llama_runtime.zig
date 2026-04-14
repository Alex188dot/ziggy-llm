const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");
const backend_api = @import("backend.zig");
const llama_gpu = @import("llama_gpu.zig");
const llama_fixture = @import("llama_fixture.zig");
const llama_metal = @import("llama_metal.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");
const ziggy_format = @import("../ziggy_format.zig");
const gguf = @import("../gguf.zig");

pub fn loadModelWithDispatch(allocator: std.mem.Allocator, path: []const u8) !llama_cpu.Model {
    if (std.mem.endsWith(u8, path, ".ziggy")) {
        return loadRuntimeModelForZiggy(allocator, path);
    }

    std.debug.print("→ GGUF fallback (consider running conversion for best speed)\n", .{});
    return try llama_cpu.loadModel(allocator, path);
}

fn loadRuntimeModelForZiggy(allocator: std.mem.Allocator, ziggy_path: []const u8) !llama_cpu.Model {
    const gguf_path = try ziggy_format.deriveSourceGgufPath(allocator, ziggy_path);
    defer allocator.free(gguf_path);

    if (std.fs.accessAbsolute(gguf_path, .{})) {
        std.debug.print("→ using GGUF metadata/runtime source alongside compiled MoonQuant tensors: {s}\n", .{gguf_path});
        return try llama_cpu.loadModel(allocator, gguf_path);
    } else |_| {
        std.debug.print("→ using compiled MoonQuant format without GGUF sibling: {s}\n", .{ziggy_path});
        return try ziggy_format.loadExecutionModel(allocator, ziggy_path);
    }
}

const ExecutionResources = struct {
    backend: ?backend_api.MatVecBackend = null,
    dense_tensors: ?llama_metal.DenseTensorStore = null,
    gated_ffn_policies: []llama_gpu.GatedFfnLayerPolicy = &.{},
    startup_breakdown: types.StartupBreakdown = .{},
    startup_profile_summary: ?[]u8 = null,

    fn deinit(self: *ExecutionResources, allocator: std.mem.Allocator) void {
        if (self.backend) |backend| backend.deinit(allocator);
        if (self.dense_tensors) |*dense_tensors| dense_tensors.deinit();
        if (self.gated_ffn_policies.len > 0) allocator.free(self.gated_ffn_policies);
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
    var compiled_model: ?ziggy_format.CompiledModel = null;
    defer if (compiled_model) |*model| model.deinit();

    const model_load_begin = std.time.nanoTimestamp();
    var model = try loadModelWithDispatch(allocator, model_path);
    if (std.mem.endsWith(u8, model_path, ".gguf")) {
        const ziggy_path = try ziggy_format.deriveCompiledPath(allocator, model_path);
        defer allocator.free(ziggy_path);
        if (std.fs.accessAbsolute(ziggy_path, .{})) {
            compiled_model = try ziggy_format.loadCompiledModel(allocator, ziggy_path);
        } else |_| {}
    }
    const model_load_ns = types.deltaNs(model_load_begin, std.time.nanoTimestamp());
    defer model.deinit(allocator);

    var execution = try selectExecution(
        allocator,
        &model,
        if (compiled_model) |*compiled| compiled else null,
        options.backend,
        options.moon_quant,
        options.metal_profile,
        .off,
    );
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
        execution.gated_ffn_policies,
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

pub fn generateZiggy(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !types.GenerationReport {
    var compiled_model = try ziggy_format.loadCompiledModel(allocator, model_path);
    defer compiled_model.deinit();

    const model_load_begin = std.time.nanoTimestamp();
    var model = try loadRuntimeModelForZiggy(allocator, model_path);
    const model_load_ns = types.deltaNs(model_load_begin, std.time.nanoTimestamp());
    defer model.deinit(allocator);

    var execution = try selectExecution(
        allocator,
        &model,
        &compiled_model,
        options.backend,
        options.moon_quant,
        options.metal_profile,
        .off,
    );
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
        execution.gated_ffn_policies,
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

pub fn promptPerplexity(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !f64 {
    if (std.mem.endsWith(u8, model_path, ".ziggy")) {
        return promptPerplexityZiggy(allocator, model_path, prompt, options);
    }

    var compiled_model: ?ziggy_format.CompiledModel = null;
    defer if (compiled_model) |*model| model.deinit();
    var model = try loadModelWithDispatch(allocator, model_path);
    defer model.deinit(allocator);

    if (std.mem.endsWith(u8, model_path, ".gguf")) {
        const ziggy_path = try ziggy_format.deriveCompiledPath(allocator, model_path);
        defer allocator.free(ziggy_path);
        if (std.fs.accessAbsolute(ziggy_path, .{})) {
            compiled_model = try ziggy_format.loadCompiledModel(allocator, ziggy_path);
        } else |_| {}
    }

    var execution = try selectExecution(
        allocator,
        &model,
        if (compiled_model) |*compiled| compiled else null,
        options.backend,
        options.moon_quant,
        false,
        .off,
    );
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

    return llama_cpu.measurePromptPerplexity(
        allocator,
        &model,
        prompt,
        options,
        execution.backend,
        lookup,
        execution.gated_ffn_policies,
    );
}

fn promptPerplexityZiggy(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !f64 {
    var compiled_model = try ziggy_format.loadCompiledModel(allocator, model_path);
    defer compiled_model.deinit();
    var model = try loadRuntimeModelForZiggy(allocator, model_path);
    defer model.deinit(allocator);

    var execution = try selectExecution(
        allocator,
        &model,
        &compiled_model,
        options.backend,
        options.moon_quant,
        false,
        .off,
    );
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

    return llama_cpu.measurePromptPerplexity(
        allocator,
        &model,
        prompt,
        options,
        execution.backend,
        lookup,
        execution.gated_ffn_policies,
    );
}

fn selectExecution(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    compiled_model: ?*const ziggy_format.CompiledModel,
    preference: types.BackendPreference,
    moon_quant_mode: types.MoonQuantMode,
    startup_profile_enabled: bool,
    metal_prewarm_mode: types.MetalPrewarmMode,
) !ExecutionResources {
    return switch (preference) {
        .cpu => .{},
        .metal => try createMetalExecution(allocator, model, compiled_model, moon_quant_mode, startup_profile_enabled, metal_prewarm_mode),
        .auto => createMetalExecution(allocator, model, compiled_model, moon_quant_mode, startup_profile_enabled, metal_prewarm_mode) catch |err| {
            if (isRecoverableMetalError(err)) return .{};
            return err;
        },
    };
}

fn createMetalExecution(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    compiled_model: ?*const ziggy_format.CompiledModel,
    moon_quant_mode: types.MoonQuantMode,
    startup_profile_enabled: bool,
    metal_prewarm_mode: types.MetalPrewarmMode,
) !ExecutionResources {
    var dense_tensors = llama_metal.DenseTensorStore.init(allocator);
    errdefer dense_tensors.deinit();
    const disable_compiled_for_arch =
        std.mem.startsWith(u8, model.architecture, "llama") or
        std.mem.startsWith(u8, model.architecture, "tinyllama");

    const effective_compiled_model = if (disable_compiled_for_arch) null else compiled_model;
    const force_raw_metal_for_arch =
        std.mem.startsWith(u8, model.architecture, "llama") or
        std.mem.startsWith(u8, model.architecture, "tinyllama");
    const effective_moon_quant_mode: types.MoonQuantMode = if (force_raw_metal_for_arch) .disabled else moon_quant_mode;
    const gated_ffn_policies = try buildGatedFfnPolicies(allocator, model, effective_compiled_model);
    errdefer allocator.free(gated_ffn_policies);
    var startup_profiler = llama_metal.StartupProfiler{ .enabled = startup_profile_enabled };
    const tensor_prepare_begin = std.time.nanoTimestamp();
    if (effective_compiled_model) |compiled| {
        try dense_tensors.populateFromCompiled(model, compiled, effective_moon_quant_mode, if (startup_profiler.enabled) &startup_profiler else null);
    } else {
        try dense_tensors.populate(model, effective_moon_quant_mode, if (startup_profiler.enabled) &startup_profiler else null);
    }
    const tensor_prepare_ns = types.deltaNs(tensor_prepare_begin, std.time.nanoTimestamp());

    const backend_init_begin = std.time.nanoTimestamp();
    const backend = try metal_backend.create(allocator);
    const backend_init_ns = types.deltaNs(backend_init_begin, std.time.nanoTimestamp());
    errdefer backend.deinit(allocator);
    const metal_prewarm_ns = if (metal_prewarm_mode == .on) blk: {
        const metal_prewarm_begin = std.time.nanoTimestamp();
        try dense_tensors.prewarm(backend, if (startup_profiler.enabled) &startup_profiler else null);
        break :blk types.deltaNs(metal_prewarm_begin, std.time.nanoTimestamp());
    } else 0;
    const startup_profile_summary = if (startup_profiler.enabled) try startup_profiler.renderSummary(allocator) else null;

    return .{
        .backend = backend,
        .dense_tensors = dense_tensors,
        .gated_ffn_policies = gated_ffn_policies,
        .startup_breakdown = .{
            .tensor_prepare_ns = tensor_prepare_ns,
            .backend_init_ns = backend_init_ns,
            .metal_prewarm_ns = metal_prewarm_ns,
        },
        .startup_profile_summary = startup_profile_summary,
    };
}

fn buildGatedFfnPolicies(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    compiled_model: ?*const ziggy_format.CompiledModel,
) ![]llama_gpu.GatedFfnLayerPolicy {
    var policies = try allocator.alloc(llama_gpu.GatedFfnLayerPolicy, model.block_count);
    @memset(policies, .{});
    if (compiled_model == null or compiled_model.?.compiled_metadata_blob.len == 0) return policies;

    const parsed = try ziggy_format.parseGatedFfnMetadata(allocator, compiled_model.?.compiled_metadata_blob);
    defer allocator.free(parsed);
    for (parsed) |entry| {
        if (entry.layer_index >= policies.len) continue;
        policies[entry.layer_index] = .{
            .threshold = entry.threshold,
            .active_block_ratio = entry.active_block_ratio,
            .avg_active_blocks = entry.avg_active_blocks,
            .avg_total_blocks = entry.avg_total_blocks,
        };
    }
    return policies;
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
