const std = @import("std");
const cli = @import("cli.zig");
const runtime = @import("runtime.zig");
const gguf = @import("gguf.zig");
const server = @import("server.zig");
const build_options = @import("build_options");

pub fn dispatch(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    switch (config.command) {
        .help => try cli.printHelp(writer),
        .version => try writer.print("ziggy-llm {s}\n", .{build_options.version}),
        .run => try runModel(writer, allocator, config),
        .chat => try printRuntimeStub(writer, "chat", config),
        .bench => try benchModel(writer, allocator, config),
        .inspect => try printInspect(writer, config),
        .serve => try printServerStub(writer, config),
    }
}

fn printRuntimeStub(writer: *std.Io.Writer, name: []const u8, config: cli.Config) !void {
    try writer.print(
        \\[{s}] scaffold command
        \\version: {s}
        \\runtime_target: {s}
        \\fallback_target: {s}
        \\metal_enabled: {s}
        \\model: {s}
        \\prompt: {s}
        \\supported_cpu_architecture: {s}
        \\supported_quantization: {s}
        \\
        \\{s}
        \\
    ,
        .{
            name,
            build_options.version,
            runtime.primary_target,
            runtime.fallback_target,
            if (build_options.enable_metal) "yes" else "no",
            config.model_path orelse "<unset>",
            config.prompt orelse "<unset>",
            runtime.supported_architecture,
            runtime.supported_quantization,
            "Interactive chat is still scaffold-only. Use `run` or `bench` with a ziggy-tiny fixture or a native llama-family GGUF model for the current implemented paths.",
        },
    );
}

fn runModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;
    const prompt = config.prompt orelse return error.MissingPrompt;

    try runtime.runCommand(writer, allocator, model_path, prompt, .{
        .max_tokens = config.max_tokens,
        .seed = config.seed,
        .temperature = config.temperature,
        .backend = config.backend,
    });
}

fn benchModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;
    const prompt = config.prompt orelse return error.MissingPrompt;

    try runtime.benchCommand(writer, allocator, model_path, prompt, .{
        .max_tokens = config.max_tokens,
        .seed = config.seed,
        .temperature = config.temperature,
        .backend = config.backend,
    });
}

fn printInspect(writer: *std.Io.Writer, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const report = try gguf.inspectFile(arena.allocator(), model_path);
    try gguf.printInspectReport(writer, model_path, report);
}

fn printServerStub(writer: *std.Io.Writer, config: cli.Config) !void {
    try writer.print(
        \\[serve] scaffold command
        \\version: {s}
        \\port: {d}
        \\api_surface: {s}
        \\compatibility_note: {s}
        \\
        \\{s}
        \\
    ,
        .{
            build_options.version,
            config.port,
            server.api_surface,
            server.compatibility_note,
            "The HTTP surface is still scaffold-only. The current implemented path is local CPU generation through `run` and `bench`.",
        },
    );
}
