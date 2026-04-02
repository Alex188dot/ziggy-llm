const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");
const llama_fixture = @import("llama_fixture.zig");
const llama_metal = @import("llama_metal.zig");
const backend_api = @import("backend.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");

pub const BenchSummary = struct {
    cold: types.GenerationReport,
    warm_runs: usize = 0,
    warm_startup_ns_avg: u64 = 0,
    warm_prompt_ns_avg: u64 = 0,
    warm_ttft_ns_avg: u64 = 0,
    warm_decode_ns_avg: u64 = 0,
    warm_generated_token_count_avg: usize = 0,

    pub fn deinit(self: *BenchSummary, allocator: std.mem.Allocator) void {
        self.cold.deinit(allocator);
        self.* = undefined;
    }

    pub fn warmDecodeTokensPerSecond(self: BenchSummary) f64 {
        if (self.warm_runs == 0 or self.warm_generated_token_count_avg == 0 or self.warm_decode_ns_avg == 0) return 0;
        return @as(f64, @floatFromInt(self.warm_generated_token_count_avg)) /
            (@as(f64, @floatFromInt(self.warm_decode_ns_avg)) / std.time.ns_per_s);
    }
};

const LoadedRuntime = struct {
    allocator: std.mem.Allocator,
    model: llama_cpu.Model,
    execution: ExecutionResources,

    fn init(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        preference: types.BackendPreference,
    ) !LoadedRuntime {
        var model = try llama_cpu.loadModel(allocator, model_path);
        errdefer model.deinit(allocator);

        var execution = try selectExecution(allocator, &model, preference);
        errdefer execution.deinit(allocator);

        return .{
            .allocator = allocator,
            .model = model,
            .execution = execution,
        };
    }

    fn deinit(self: *LoadedRuntime) void {
        self.execution.deinit(self.allocator);
        self.model.deinit(self.allocator);
        self.* = undefined;
    }

    fn generate(self: *LoadedRuntime, prompt: []const u8, options: types.GenerationOptions, setup_ns: u64) !types.GenerationReport {
        const lookup = if (self.execution.dense_tensors) |*dense_tensors|
            llama_cpu.DenseTensorLookup{
                .ctx = dense_tensors,
                .get_fn = lookupDenseTensor,
                .get_by_offset_fn = lookupDenseTensorByOffset,
                .get_raw_by_offset_fn = lookupRawTensorByOffset,
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

        return .{
            .generated_text = report.generated_text,
            .prompt_token_count = report.prompt_token_count,
            .generated_token_count = report.generated_token_count,
            .startup_ns = report.startup_ns,
            .prompt_ns = report.prompt_ns,
            .ttft_ns = report.ttft_ns,
            .decode_ns = report.decode_ns,
            .seed = options.seed,
            .temperature = options.temperature,
            .backend = report.backend,
            .metal_profile_summary = report.metal_profile_summary,
        };
    }
};

const ExecutionResources = struct {
    backend: ?backend_api.MatVecBackend = null,
    dense_tensors: ?llama_metal.DenseTensorStore = null,

    fn deinit(self: *ExecutionResources, allocator: std.mem.Allocator) void {
        if (self.backend) |backend| backend.deinit(allocator);
        if (self.dense_tensors) |*dense_tensors| dense_tensors.deinit();
        self.* = undefined;
    }
};

pub fn runWarmBench(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
    bench_runs: usize,
) !BenchSummary {
    std.debug.assert(bench_runs > 0);

    const setup_begin = std.time.nanoTimestamp();
    var runtime = try LoadedRuntime.init(allocator, model_path, options.backend);
    defer runtime.deinit();
    const setup_ns = types.deltaNs(setup_begin, std.time.nanoTimestamp());

    var cold = try runtime.generate(prompt, options, setup_ns);
    errdefer cold.deinit(allocator);

    if (bench_runs == 1) {
        return .{ .cold = cold };
    }

    var warm_startup_total: u128 = 0;
    var warm_prompt_total: u128 = 0;
    var warm_ttft_total: u128 = 0;
    var warm_decode_total: u128 = 0;
    var warm_generated_token_total: u128 = 0;

    for (1..bench_runs) |_| {
        var warm = try runtime.generate(prompt, options, 0);
        warm_startup_total += warm.startup_ns;
        warm_prompt_total += warm.prompt_ns;
        warm_ttft_total += warm.ttft_ns;
        warm_decode_total += warm.decode_ns;
        warm_generated_token_total += warm.generated_token_count;
        warm.deinit(allocator);
    }

    const warm_runs = bench_runs - 1;
    return .{
        .cold = cold,
        .warm_runs = warm_runs,
        .warm_startup_ns_avg = @intCast(warm_startup_total / warm_runs),
        .warm_prompt_ns_avg = @intCast(warm_prompt_total / warm_runs),
        .warm_ttft_ns_avg = @intCast(warm_ttft_total / warm_runs),
        .warm_decode_ns_avg = @intCast(warm_decode_total / warm_runs),
        .warm_generated_token_count_avg = @intCast(warm_generated_token_total / warm_runs),
    };
}

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
    try dense_tensors.populate(model);

    const backend = try metal_backend.create(allocator);
    errdefer backend.deinit(allocator);
    try dense_tensors.prewarm(backend);

    return .{
        .backend = backend,
        .dense_tensors = dense_tensors,
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

test "warm bench reuses loaded cpu runtime and reports warm averages" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "bench-llama.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "bench-llama.gguf");
    defer std.testing.allocator.free(path);

    var summary = try runWarmBench(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    }, 3);
    defer summary.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), summary.warm_runs);
    try std.testing.expectEqual(@as(types.BackendUsed, .cpu), summary.cold.backend);
    try std.testing.expect(summary.cold.ttft_ns > 0);
    try std.testing.expect(summary.warm_ttft_ns_avg > 0);
    try std.testing.expect(summary.warm_generated_token_count_avg > 0);
}
