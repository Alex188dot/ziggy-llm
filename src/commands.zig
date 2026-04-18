const std = @import("std");
const cli = @import("cli.zig");
const runtime = @import("runtime.zig");
const gguf = @import("gguf.zig");
const moon_quant = @import("moon_quant.zig");
const chat_runtime = @import("chat_runtime.zig");
const server = @import("server.zig");
const server_runtime = @import("server_runtime.zig");
const build_options = @import("build_options");
const update = @import("update.zig");

pub fn dispatch(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    switch (config.command) {
        .help => try cli.printHelp(writer),
        .version => try writer.print("ziggy-llm {s}\n", .{build_options.version}),
        .run => try runModel(writer, allocator, config),
        .chat => try chat_runtime.runChat(writer, allocator, config),
        .bench => try benchModel(writer, allocator, config),
        .inspect => try printInspect(writer, config),
        .serve => try server_runtime.serve(writer, allocator, config),
        .update => try update.runUpdateCommand(writer, allocator),
    }
}

fn runModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;
    const prompt = config.prompt orelse return error.MissingPrompt;

    try runtime.runCommand(writer, allocator, model_path, prompt, .{
        .max_tokens = config.max_tokens,
        .context_length = config.context_length,
        .seed = config.seed,
        .temperature = config.temperature,
        .repeat_penalty = config.repeat_penalty,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .min_p = config.min_p,
        .backend = config.backend,
        .moon_quant = config.moon_quant,
        .metal_profile = config.metal_profile,
        .sampling_strategy = config.sampling_strategy,
        .exp_block_decode = config.exp_block_decode,
        .exp_block_k = config.exp_block_k,
    });
}

fn benchModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;
    const prompt = config.prompt orelse return error.MissingPrompt;

    try runtime.benchCommand(writer, allocator, model_path, prompt, .{
        .max_tokens = config.max_tokens,
        .context_length = config.context_length,
        .seed = config.seed,
        .temperature = config.temperature,
        .repeat_penalty = config.repeat_penalty,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .min_p = config.min_p,
        .backend = config.backend,
        .moon_quant = config.moon_quant,
        .metal_profile = config.metal_profile,
        .sampling_strategy = config.sampling_strategy,
        .exp_block_decode = config.exp_block_decode,
        .exp_block_k = config.exp_block_k,
    }, config.bench_runs);
}

fn printInspect(writer: *std.Io.Writer, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const report = try gguf.inspectFile(arena.allocator(), model_path);
    try gguf.printInspectReport(writer, model_path, report);
    try moon_quant.printInspectSummary(writer, report);
}
