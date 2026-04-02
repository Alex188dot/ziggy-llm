const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");
const backend_api = @import("backend.zig");
const llama_fixture = @import("llama_fixture.zig");
const llama_metal = @import("llama_metal.zig");
const metal_backend = @import("metal_backend.zig");
const types = @import("types.zig");

const ExecutionResources = struct {
    backend: ?backend_api.MatVecBackend = null,
    dense_tensors: ?llama_metal.DenseTensorStore = null,

    fn deinit(self: *ExecutionResources, allocator: std.mem.Allocator) void {
        if (self.backend) |backend| backend.deinit(allocator);
        if (self.dense_tensors) |*dense_tensors| dense_tensors.deinit();
        self.* = undefined;
    }
};

pub fn generate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !types.GenerationReport {
    const startup_begin = std.time.nanoTimestamp();
    var model = try llama_cpu.loadModel(allocator, model_path);
    defer model.deinit(allocator);

    var execution = try selectExecution(allocator, &model, options.backend);
    defer execution.deinit(allocator);

    const lookup = if (execution.dense_tensors) |*dense_tensors|
        llama_cpu.DenseTensorLookup{
            .ctx = dense_tensors,
            .get_fn = lookupDenseTensor,
        }
    else
        null;

    var llama_report = try llama_cpu.generateLoaded(
        allocator,
        &model,
        prompt,
        options.max_tokens,
        options.seed,
        options.temperature,
        execution.backend,
        lookup,
    );
    const setup_ns = types.deltaNs(startup_begin, std.time.nanoTimestamp()) -| llama_report.startup_ns;
    llama_report.startup_ns += setup_ns;
    llama_report.ttft_ns += setup_ns;

    return .{
        .generated_text = llama_report.generated_text,
        .prompt_token_count = llama_report.prompt_token_count,
        .generated_token_count = llama_report.generated_token_count,
        .startup_ns = llama_report.startup_ns,
        .prompt_ns = llama_report.prompt_ns,
        .ttft_ns = llama_report.ttft_ns,
        .decode_ns = llama_report.decode_ns,
        .seed = options.seed,
        .temperature = options.temperature,
        .backend = llama_report.backend,
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
