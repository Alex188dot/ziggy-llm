const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");
const backend_api = @import("backend.zig");
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
