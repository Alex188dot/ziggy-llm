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
        .run => try runModel(writer, allocator, config),
        .chat => try chat_runtime.runChat(writer, allocator, try configWithZiggyCache(writer, allocator, config)),
        .compile => try compileModel(writer, allocator, config),
        .bench => try benchModel(writer, allocator, config),
        .inspect => try printInspect(writer, config),
        .serve => try server_runtime.serve(writer, allocator, try configWithZiggyCache(writer, allocator, config)),
        .update => try update.runUpdateCommand(writer, allocator),
    }
}

fn configWithZiggyCache(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !cli.Config {
    const model_path = config.model_path orelse return config;

    // Only process .gguf files
    if (!std.mem.endsWith(u8, model_path, ".gguf")) return config;

    // Build .ziggy path (same directory, append .ziggy)
    const ziggy_path = try std.fmt.allocPrint(allocator, "{s}.ziggy", .{model_path});
    defer allocator.free(ziggy_path);

    // Check if .ziggy exists, is up to date, and is valid
    const need_compile = blk: {
        const gguf_stat = std.fs.cwd().statFile(model_path) catch break :blk true;
        const ziggy_stat = std.fs.cwd().statFile(ziggy_path) catch break :blk true;
        // Compile if ziggy is older than gguf
        if (ziggy_stat.mtime < gguf_stat.mtime) break :blk true;
        // Validate header is readable
        const file = std.fs.cwd().openFile(ziggy_path, .{}) catch break :blk true;
        defer file.close();
        var header_buf: [@sizeOf(ziggy_format.ZiggyHeader)]u8 = undefined;
        const read = file.readAll(&header_buf) catch break :blk true;
        if (read != header_buf.len) break :blk true;
        const header = std.mem.bytesToValue(ziggy_format.ZiggyHeader, &header_buf);
        header.validate() catch break :blk true;
        break :blk false;
    };

    if (need_compile) {
        try writer.print("Building .ziggy file, this could take a while...", .{});
        try writer.flush();

        var spinner = terminal.Spinner{};
        try spinner.start();

        try ziggy_format.compileFromGGUF(allocator, model_path, ziggy_path, .{
            .repack_q4_k = true,
            .keep_raw_for_all = false,
        });

        spinner.stop();
        try writer.print("\rBuilding .ziggy file, this could take a while... done\n", .{});
    }

    // NOTE: Return original GGUF path for now
    // .ziggy execution requires Phase 2/3 implementation
    // The .ziggy file is compiled as a cache for future use
    return config;
}

fn runModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const effective_config = try configWithZiggyCache(writer, allocator, config);
    const model_path = effective_config.model_path orelse return error.MissingModelPath;
    const prompt = effective_config.prompt orelse return error.MissingPrompt;

    try runtime.runCommand(writer, allocator, model_path, prompt, .{
        .max_tokens = effective_config.max_tokens,
        .context_length = effective_config.context_length,
        .seed = effective_config.seed,
        .temperature = effective_config.temperature,
        .repeat_penalty = effective_config.repeat_penalty,
        .top_k = effective_config.top_k,
        .top_p = effective_config.top_p,
        .min_p = effective_config.min_p,
        .backend = effective_config.backend,
        .moon_quant = effective_config.moon_quant,
        .metal_profile = effective_config.metal_profile,
        .sampling_strategy = effective_config.sampling_strategy,
    });
}

fn benchModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const effective_config = try configWithZiggyCache(writer, allocator, config);
    const model_path = effective_config.model_path orelse return error.MissingModelPath;
    const prompt = effective_config.prompt orelse return error.MissingPrompt;

    try runtime.benchCommand(writer, allocator, model_path, prompt, .{
        .max_tokens = effective_config.max_tokens,
        .context_length = effective_config.context_length,
        .seed = effective_config.seed,
        .temperature = effective_config.temperature,
        .repeat_penalty = effective_config.repeat_penalty,
        .top_k = effective_config.top_k,
        .top_p = effective_config.top_p,
        .min_p = effective_config.min_p,
        .backend = effective_config.backend,
        .moon_quant = effective_config.moon_quant,
        .metal_profile = effective_config.metal_profile,
        .sampling_strategy = effective_config.sampling_strategy,
    }, effective_config.bench_runs);
}

fn compileModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    _ = writer;
    const model_path = config.model_path orelse return error.MissingModelPath;
    const output_path = config.output_path orelse return error.MissingOutputPath;

    try ziggy_format.compileFromGGUF(allocator, model_path, output_path, .{
        .repack_q4_k = true,
        .keep_raw_for_all = false,
    });
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
