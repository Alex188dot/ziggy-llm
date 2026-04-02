const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");
const backend_api = @import("backend.zig");
const llama_metal = @import("llama_metal.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");

pub const LoadedRuntime = struct {
    allocator: std.mem.Allocator,
    model: llama_cpu.Model,
    execution: ExecutionResources,
    model_load_ns: u64,

    pub fn init(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        preference: types.BackendPreference,
        moon_quant_mode: types.MoonQuantMode,
        startup_profile_enabled: bool,
    ) !LoadedRuntime {
        const model_load_begin = std.time.nanoTimestamp();
        var model = try llama_cpu.loadModel(allocator, model_path);
        const model_load_ns = types.deltaNs(model_load_begin, std.time.nanoTimestamp());
        errdefer model.deinit(allocator);

        var execution = try selectExecution(allocator, &model, preference, moon_quant_mode, startup_profile_enabled);
        errdefer execution.deinit(allocator);

        return .{
            .allocator = allocator,
            .model = model,
            .execution = execution,
            .model_load_ns = model_load_ns,
        };
    }

    pub fn deinit(self: *LoadedRuntime) void {
        self.execution.deinit(self.allocator);
        self.model.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn generate(self: *LoadedRuntime, prompt: []const u8, options: types.GenerationOptions, setup_ns: u64) !types.GenerationReport {
        const lookup = if (self.execution.dense_tensors) |*dense_tensors|
            llama_cpu.DenseTensorLookup{
                .ctx = dense_tensors,
                .get_fn = lookupDenseTensor,
                .get_by_offset_fn = lookupDenseTensorByOffset,
                .get_raw_by_offset_fn = lookupRawTensorByOffset,
                .get_moon_quant_by_offset_fn = lookupMoonQuantTensorByOffset,
            }
        else
            null;

        var report = try llama_cpu.generateLoaded(
            self.allocator,
            &self.model,
            prompt,
            options,
            self.execution.backend,
            lookup,
        );
        report.startup_ns += setup_ns;
        report.ttft_ns += setup_ns;
        report.startup_breakdown.model_load_ns += setup_ns;
        report.startup_breakdown.tensor_prepare_ns = self.execution.startup_breakdown.tensor_prepare_ns;
        report.startup_breakdown.backend_init_ns = self.execution.startup_breakdown.backend_init_ns;
        report.startup_breakdown.metal_prewarm_ns = self.execution.startup_breakdown.metal_prewarm_ns;
        const combined_profile_summary = try combineProfileSummaries(
            self.allocator,
            self.execution.startup_profile_summary,
            report.metal_profile_summary,
        );
        if (report.metal_profile_summary) |summary| self.allocator.free(summary);

        return .{
            .generated_text = report.generated_text,
            .prompt_token_count = report.prompt_token_count,
            .reused_prompt_token_count = report.reused_prompt_token_count,
            .generated_token_count = report.generated_token_count,
            .startup_ns = report.startup_ns,
            .prompt_ns = report.prompt_ns,
            .ttft_ns = report.ttft_ns,
            .decode_ns = report.decode_ns,
            .seed = options.seed,
            .temperature = options.temperature,
            .backend = report.backend,
            .startup_breakdown = report.startup_breakdown,
            .metal_profile_summary = combined_profile_summary,
        };
    }

    pub fn denseLookup(self: *LoadedRuntime) ?llama_cpu.DenseTensorLookup {
        if (self.execution.dense_tensors) |*dense_tensors| {
            return .{
                .ctx = dense_tensors,
                .get_fn = lookupDenseTensor,
                .get_by_offset_fn = lookupDenseTensorByOffset,
                .get_raw_by_offset_fn = lookupRawTensorByOffset,
                .get_moon_quant_by_offset_fn = lookupMoonQuantTensorByOffset,
            };
        }
        return null;
    }
};

pub const ExecutionResources = struct {
    backend: ?backend_api.MatVecBackend = null,
    dense_tensors: ?llama_metal.DenseTensorStore = null,
    startup_breakdown: types.StartupBreakdown = .{},
    startup_profile_summary: ?[]u8 = null,

    pub fn deinit(self: *ExecutionResources, allocator: std.mem.Allocator) void {
        if (self.backend) |backend| backend.deinit(allocator);
        if (self.dense_tensors) |*dense_tensors| dense_tensors.deinit();
        if (self.startup_profile_summary) |summary| allocator.free(summary);
        self.* = undefined;
    }
};

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
