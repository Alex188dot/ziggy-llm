const std = @import("std");
const cli = @import("cli.zig");
const runtime = @import("runtime.zig");
const gguf = @import("gguf.zig");
const server = @import("server.zig");
const build_options = @import("build_options");

pub fn dispatch(writer: *std.Io.Writer, config: cli.Config) !void {
    switch (config.command) {
        .help => try cli.printHelp(writer),
        .version => try writer.print("ziggy-llm {s}\n", .{build_options.version}),
        .run => try printRuntimeStub(writer, "run", config),
        .chat => try printRuntimeStub(writer, "chat", config),
        .bench => try printRuntimeStub(writer, "bench", config),
        .inspect => try printInspectStub(writer, config),
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
            runtime.scaffold_status,
        },
    );
}

fn printInspectStub(writer: *std.Io.Writer, config: cli.Config) !void {
    const summary = gguf.inspectSummary(config.model_path);
    try writer.print(
        \\[inspect] scaffold command
        \\version: {s}
        \\model: {s}
        \\gguf_status: {s}
        \\supported_versions: {s}
        \\
        \\{s}
        \\
    ,
        .{
            build_options.version,
            config.model_path orelse "<unset>",
            summary.status,
            gguf.supported_versions,
            summary.message,
        },
    );
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
            runtime.scaffold_status,
        },
    );
}
