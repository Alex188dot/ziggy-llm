const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");
const llama_fixture = @import("llama_fixture.zig");
const backend_api = @import("backend.zig");
const llama_metal = @import("llama_metal.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");

pub const ResidentRuntime = struct {
    allocator: std.mem.Allocator,
    loaded: ?LoadedModel = null,
    keep_alive_seconds: i64 = 300,

    pub fn init(allocator: std.mem.Allocator) ResidentRuntime {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ResidentRuntime) void {
        self.unload();
        self.* = undefined;
    }

    pub fn setKeepAliveSeconds(self: *ResidentRuntime, seconds: i64) void {
        self.keep_alive_seconds = seconds;
    }

    pub fn isLoaded(self: *const ResidentRuntime) bool {
        return self.loaded != null;
    }

    pub fn unload(self: *ResidentRuntime) void {
        if (self.loaded) |*loaded| loaded.deinit(self.allocator);
        self.loaded = null;
    }

    pub fn unloadIfExpired(self: *ResidentRuntime) void {
        if (self.keep_alive_seconds < 0) return;
        if (self.loaded) |loaded| {
            const now = std.time.timestamp();
            if (now - loaded.last_used_ts >= self.keep_alive_seconds) self.unload();
        }
    }

    pub fn contextLength(self: *const ResidentRuntime) ?usize {
        return if (self.loaded) |loaded| loaded.model.context_length else null;
    }

    pub fn generate(
        self: *ResidentRuntime,
        model_path: []const u8,
        prompt: []const u8,
        options: types.GenerationOptions,
    ) !types.GenerationReport {
        return self.generateStreaming(model_path, prompt, options, null, null);
    }

    pub fn generateStreaming(
        self: *ResidentRuntime,
        model_path: []const u8,
        prompt: []const u8,
        options: types.GenerationOptions,
        stream_ctx: ?*anyopaque,
        stream_callback: ?llama_cpu.StreamCallback,
    ) !types.GenerationReport {
        self.unloadIfExpired();
        try self.ensureLoaded(model_path, options.backend);
        const loaded = &self.loaded.?;
        loaded.last_used_ts = std.time.timestamp();

        const lookup = if (loaded.execution.dense_tensors) |*dense_tensors|
            llama_cpu.DenseTensorLookup{
                .ctx = dense_tensors,
                .get_fn = lookupDenseTensor,
                .get_by_offset_fn = lookupDenseTensorByOffset,
                .get_raw_by_offset_fn = lookupRawTensorByOffset,
                .get_moon_quant_by_offset_fn = lookupMoonQuantTensorByOffset,
            }
        else
            null;

        var report = if (options.metal_profile)
            try llama_cpu.generateLoadedStreaming(
                self.allocator,
                &loaded.model,
                prompt,
                options,
                loaded.execution.backend,
                lookup,
                stream_ctx,
                stream_callback,
            )
        else
            llama_cpu.generateLoadedStreamingCached(
                self.allocator,
                &loaded.model,
                prompt,
                options,
                loaded.execution.backend,
                lookup,
                &loaded.reusable_session,
                stream_ctx,
                stream_callback,
            ) catch |err| {
                loaded.reusable_session.reset();
                return err;
            };
        report.startup_breakdown.model_load_ns = loaded.startup_breakdown.model_load_ns;
        report.startup_breakdown.tensor_prepare_ns = loaded.startup_breakdown.tensor_prepare_ns;
        report.startup_breakdown.backend_init_ns = loaded.startup_breakdown.backend_init_ns;
        report.startup_breakdown.metal_prewarm_ns = loaded.startup_breakdown.metal_prewarm_ns;
        report.startup_ns += loaded.startup_breakdown.model_load_ns +
            loaded.startup_breakdown.tensor_prepare_ns +
            loaded.startup_breakdown.backend_init_ns +
            loaded.startup_breakdown.metal_prewarm_ns;
        report.ttft_ns += loaded.startup_breakdown.model_load_ns +
            loaded.startup_breakdown.tensor_prepare_ns +
            loaded.startup_breakdown.backend_init_ns +
            loaded.startup_breakdown.metal_prewarm_ns;

        const was_warm = loaded.warm;
        loaded.warm = true;
        if (was_warm) {
            report.startup_ns -= loaded.startup_breakdown.model_load_ns +
                loaded.startup_breakdown.tensor_prepare_ns +
                loaded.startup_breakdown.backend_init_ns +
                loaded.startup_breakdown.metal_prewarm_ns;
            report.ttft_ns -= loaded.startup_breakdown.model_load_ns +
                loaded.startup_breakdown.tensor_prepare_ns +
                loaded.startup_breakdown.backend_init_ns +
                loaded.startup_breakdown.metal_prewarm_ns;
            report.startup_breakdown.model_load_ns = 0;
            report.startup_breakdown.tensor_prepare_ns = 0;
            report.startup_breakdown.backend_init_ns = 0;
            report.startup_breakdown.metal_prewarm_ns = 0;
        }

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
            .metal_profile_summary = report.metal_profile_summary,
        };
    }

    pub fn promptTokenCount(self: *ResidentRuntime, model_path: []const u8, prompt: []const u8, backend: types.BackendPreference) !usize {
        try self.ensureLoaded(model_path, backend);
        const loaded = &self.loaded.?;
        return llama_cpu.countPromptTokens(self.allocator, &loaded.model, prompt);
    }

    fn ensureLoaded(self: *ResidentRuntime, model_path: []const u8, backend_pref: types.BackendPreference) !void {
        if (self.loaded) |loaded| {
            if (std.mem.eql(u8, loaded.model_path, model_path) and loaded.backend_pref == backend_pref) return;
            self.unload();
        }

        const model_load_begin = std.time.nanoTimestamp();
        var model = try llama_cpu.loadModel(self.allocator, model_path);
        const model_load_ns = types.deltaNs(model_load_begin, std.time.nanoTimestamp());
        errdefer model.deinit(self.allocator);

        var execution = try selectExecution(self.allocator, &model, backend_pref);
        errdefer execution.deinit(self.allocator);

        const owned_model_path = try self.allocator.dupe(u8, model_path);
        errdefer self.allocator.free(owned_model_path);

        var reusable_session = try llama_cpu.ReusableSession.init(
            self.allocator,
            &model,
            execution.backend,
            if (execution.dense_tensors) |*dense_tensors|
                llama_cpu.DenseTensorLookup{
                    .ctx = dense_tensors,
                    .get_fn = lookupDenseTensor,
                    .get_by_offset_fn = lookupDenseTensorByOffset,
                    .get_raw_by_offset_fn = lookupRawTensorByOffset,
                    .get_moon_quant_by_offset_fn = lookupMoonQuantTensorByOffset,
                }
            else
                null,
        );
        errdefer reusable_session.deinit(self.allocator);

        self.loaded = .{
            .model_path = owned_model_path,
            .backend_pref = backend_pref,
            .model = model,
            .execution = execution,
            .reusable_session = reusable_session,
            .last_used_ts = std.time.timestamp(),
            .startup_breakdown = .{
                .model_load_ns = model_load_ns,
                .tensor_prepare_ns = execution.startup_breakdown.tensor_prepare_ns,
                .backend_init_ns = execution.startup_breakdown.backend_init_ns,
                .metal_prewarm_ns = execution.startup_breakdown.metal_prewarm_ns,
            },
        };

        const loaded = &self.loaded.?;
        loaded.reusable_session.session.model = &loaded.model;
        if (loaded.execution.dense_tensors) |*dense_tensors| {
            loaded.reusable_session.session.dense_tensors = llama_cpu.DenseTensorLookup{
                .ctx = dense_tensors,
                .get_fn = lookupDenseTensor,
                .get_by_offset_fn = lookupDenseTensorByOffset,
                .get_raw_by_offset_fn = lookupRawTensorByOffset,
                .get_moon_quant_by_offset_fn = lookupMoonQuantTensorByOffset,
            };
            if (loaded.reusable_session.session.gpu_session) |*gpu_session| {
                gpu_session.dense_lookup.ctx = dense_tensors;
            }
        } else {
            loaded.reusable_session.session.dense_tensors = null;
        }
    }
};

const LoadedModel = struct {
    model_path: []u8,
    backend_pref: types.BackendPreference,
    model: llama_cpu.Model,
    execution: ExecutionResources,
    reusable_session: llama_cpu.ReusableSession,
    startup_breakdown: types.StartupBreakdown = .{},
    last_used_ts: i64,
    warm: bool = false,

    fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        allocator.free(self.model_path);
        self.reusable_session.deinit(allocator);
        self.execution.deinit(allocator);
        self.model.deinit(allocator);
        self.* = undefined;
    }
};

const ExecutionResources = struct {
    backend: ?backend_api.MatVecBackend = null,
    dense_tensors: ?llama_metal.DenseTensorStore = null,
    startup_breakdown: types.StartupBreakdown = .{},

    fn deinit(self: *ExecutionResources, allocator: std.mem.Allocator) void {
        if (self.backend) |backend| backend.deinit(allocator);
        if (self.dense_tensors) |*dense_tensors| dense_tensors.deinit();
        self.* = undefined;
    }
};

fn selectExecution(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    preference: types.BackendPreference,
) !ExecutionResources {
    return switch (preference) {
        .cpu => .{},
        .metal => try createMetalExecution(allocator, model),
        .auto => createMetalExecution(allocator, model) catch |err| {
            if (isRecoverableMetalError(err)) return .{};
            return err;
        },
    };
}

fn createMetalExecution(allocator: std.mem.Allocator, model: *const llama_cpu.Model) !ExecutionResources {
    var dense_tensors = llama_metal.DenseTensorStore.init(allocator);
    errdefer dense_tensors.deinit();
    const tensor_prepare_begin = std.time.nanoTimestamp();
    try dense_tensors.populate(model);
    const tensor_prepare_ns = types.deltaNs(tensor_prepare_begin, std.time.nanoTimestamp());

    const backend_init_begin = std.time.nanoTimestamp();
    const backend = try metal_backend.create(allocator);
    const backend_init_ns = types.deltaNs(backend_init_begin, std.time.nanoTimestamp());
    errdefer backend.deinit(allocator);
    const metal_prewarm_begin = std.time.nanoTimestamp();
    try dense_tensors.prewarm(backend);
    const metal_prewarm_ns = types.deltaNs(metal_prewarm_begin, std.time.nanoTimestamp());

    return .{
        .backend = backend,
        .dense_tensors = dense_tensors,
        .startup_breakdown = .{
            .tensor_prepare_ns = tensor_prepare_ns,
            .backend_init_ns = backend_init_ns,
            .metal_prewarm_ns = metal_prewarm_ns,
        },
    };
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

test "resident runtime reuses cached prompt prefix across turns" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "llama-cache.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "llama-cache.gguf");
    defer std.testing.allocator.free(path);

    var runtime = ResidentRuntime.init(std.testing.allocator);
    defer runtime.deinit();
    runtime.setKeepAliveSeconds(-1);

    var first = try runtime.generate(path, "a", .{
        .max_tokens = 1,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    });
    defer first.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings(" b", first.generated_text);
    try std.testing.expectEqual(@as(usize, 0), first.reused_prompt_token_count);

    var second = try runtime.generate(path, "a b", .{
        .max_tokens = 2,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    });
    defer second.deinit(std.testing.allocator);

    try std.testing.expect(second.reused_prompt_token_count > 0);
    try std.testing.expectEqualStrings(" c!", second.generated_text);
}
