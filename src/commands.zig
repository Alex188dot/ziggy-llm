const std = @import("std");
const cli = @import("cli.zig");
const runtime = @import("runtime.zig");
const gguf = @import("gguf.zig");
const moon_quant = @import("moon_quant.zig");
const ziggy_format = @import("ziggy_format.zig");
const chat_runtime = @import("chat_runtime.zig");
const server = @import("server.zig");
const server_runtime = @import("server_runtime.zig");
const build_options = @import("build_options");
const update = @import("update.zig");
const terminal = @import("terminal.zig");

pub fn dispatch(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    switch (config.command) {
        .help => try cli.printHelp(writer),
        .version => try writer.print("ziggy-llm {s}\n", .{build_options.version}),
        .run => try runModel(writer, allocator, try configWithPreparedModel(writer, allocator, config)),
        .chat => try chat_runtime.runChat(writer, allocator, try configWithPreparedModel(writer, allocator, config)),
        .serve => try server_runtime.serve(writer, allocator, try configWithPreparedModel(writer, allocator, config)),
        .compile => try compileModel(writer, allocator, config),
        .convert => try convertModel(writer, allocator, config), // NEW: convert command
        .bench => try benchModel(writer, allocator, try configWithPreparedModel(writer, allocator, config)),
        .inspect => try printInspect(writer, config),
        .update => try update.runUpdateCommand(writer, allocator),
    }
}

fn configWithPreparedModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !cli.Config {
    const model_path = config.model_path orelse return config;
    var modified_config = config;

    if (std.mem.endsWith(u8, model_path, ".gguf")) {
        try ensureZiggyCache(writer, allocator, model_path);
        return modified_config;
    }

    if (std.mem.endsWith(u8, model_path, ".ziggy")) {
        const gguf_path = try ziggy_format.deriveSourceGgufPath(allocator, model_path);
        errdefer allocator.free(gguf_path);

        if (std.fs.accessAbsolute(gguf_path, .{})) {
            modified_config.model_path = gguf_path;
        } else |_| {
            allocator.free(gguf_path);
        }
        return modified_config;
    }

    return modified_config;
}

fn ensureZiggyCache(writer: *std.Io.Writer, allocator: std.mem.Allocator, model_path: []const u8) !void {
    const ziggy_path = try ziggy_format.deriveCompiledPath(allocator, model_path);
    defer allocator.free(ziggy_path);

    const need_compile = blk: {
        const gguf_stat = std.fs.cwd().statFile(model_path) catch break :blk true;
        const ziggy_stat = std.fs.cwd().statFile(ziggy_path) catch break :blk true;
        if (ziggy_stat.mtime < gguf_stat.mtime) break :blk true;
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        _ = ziggy_format.inspectFile(arena.allocator(), ziggy_path) catch break :blk true;
        break :blk false;
    };

    if (!need_compile) return;

    try writer.print("Building .ziggy file, this could take a while...", .{});
    try writer.flush();

    var spinner = terminal.Spinner{};
    try spinner.start();
    defer spinner.stop();

    try ziggy_format.compileFromGGUF(allocator, model_path, ziggy_path, .{
        .repack_q4_k = true,
        .repack_q6_k = true,
        .repack_q8_0 = true,
        .keep_raw_for_all = false,
    });

    try writer.print("\rBuilding .ziggy file, this could take a while... done\n", .{});
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
    }, config.bench_runs);
}

fn compileModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    _ = writer;
    const model_path = config.model_path orelse return error.MissingModelPath;
    const output_path = config.output_path orelse return error.MissingOutputPath;

    try ziggy_format.compileFromGGUF(allocator, model_path, output_path, .{
        .repack_q4_k = true,
        .repack_q6_k = true,
        .repack_q8_0 = true,
        .keep_raw_for_all = false,
    });
}

fn convertModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;

    // Derive output path if not provided
    var output_path: []const u8 = undefined;
    var owned_output = false;
    if (config.output_path) |op| {
        output_path = op;
    } else if (std.mem.endsWith(u8, model_path, ".gguf")) {
        output_path = try ziggy_format.deriveCompiledPath(allocator, model_path);
        owned_output = true;
    } else {
        return error.MissingOutputPath;
    }
    defer if (owned_output) allocator.free(output_path);

    try writer.print("Converting {s} → {s}\n", .{ model_path, output_path });
    try ziggy_format.compileFromGGUF(allocator, model_path, output_path, .{
        .repack_q4_k = true,
        .repack_q6_k = true,
        .repack_q8_0 = true,
        .keep_raw_for_all = false,
    });
    try writer.print("✅ Conversion complete. You can now use:\n", .{});
    try writer.print("   ziggy-llm chat --model {s}\n", .{output_path});
}

fn printInspect(writer: *std.Io.Writer, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    if (std.mem.endsWith(u8, model_path, ".ziggy")) {
        const report = try ziggy_format.inspectFile(arena.allocator(), model_path);
        try ziggy_format.printInspectReport(writer, report);
    } else {
        const report = try gguf.inspectFile(arena.allocator(), model_path);
        try gguf.printInspectReport(writer, model_path, report);
        try moon_quant.printInspectSummary(writer, report);
    }
}
