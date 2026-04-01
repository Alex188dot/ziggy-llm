const std = @import("std");
const build_options = @import("build_options");

const Command = enum {
    run,
    chat,
    inspect,
    bench,
    serve,
    help,
    version,
};

const Config = struct {
    command: Command = .help,
    model_path: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    port: u16 = 8080,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const config = try parseArgs(args);
    try dispatch(config);
}

fn parseArgs(args: []const []const u8) !Config {
    var config = Config{};

    if (args.len <= 1) return config;

    const first = args[1];
    if (std.mem.eql(u8, first, "--help") or std.mem.eql(u8, first, "-h") or std.mem.eql(u8, first, "help")) {
        config.command = .help;
        return config;
    }
    if (std.mem.eql(u8, first, "--version") or std.mem.eql(u8, first, "version")) {
        config.command = .version;
        return config;
    }

    config.command = parseCommand(first) orelse return error.UnknownCommand;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config.model_path = args[i];
            continue;
        }
        if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config.prompt = args[i];
            continue;
        }
        if (std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config.port = try std.fmt.parseUnsigned(u16, args[i], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            config.command = .help;
            return config;
        }

        return error.UnknownFlag;
    }

    return config;
}

fn parseCommand(name: []const u8) ?Command {
    if (std.mem.eql(u8, name, "run")) return .run;
    if (std.mem.eql(u8, name, "chat")) return .chat;
    if (std.mem.eql(u8, name, "inspect")) return .inspect;
    if (std.mem.eql(u8, name, "bench")) return .bench;
    if (std.mem.eql(u8, name, "serve")) return .serve;
    return null;
}

fn dispatch(config: Config) !void {
    const stdout = std.io.getStdOut().writer();

    switch (config.command) {
        .help => try printHelp(stdout),
        .version => try stdout.print("ziggy-llm {s}\n", .{build_options.version}),
        .run => try printStub(stdout, "run", config),
        .chat => try printStub(stdout, "chat", config),
        .inspect => try printStub(stdout, "inspect", config),
        .bench => try printStub(stdout, "bench", config),
        .serve => try printStub(stdout, "serve", config),
    }
}

fn printHelp(writer: anytype) !void {
    try writer.print(
        \\ziggy-llm {s}
        \\
        \\Mac-first, Zig-native GGUF inference for Apple Silicon.
        \\
        \\Usage:
        \\  ziggy-llm <command> [options]
        \\
        \\Commands:
        \\  run       Execute a single prompt against a model
        \\  chat      Start an interactive chat session
        \\  inspect   Inspect GGUF metadata and support status
        \\  bench     Run benchmark routines
        \\  serve     Start the tiny HTTP server
        \\  help      Print this help
        \\  version   Print the build version
        \\
        \\Options:
        \\  -m, --model <path>    Path to a GGUF model
        \\  -p, --prompt <text>   Prompt text for one-shot generation
        \\      --port <port>     Port for server mode (default: 8080)
        \\
        \\Build:
        \\  Metal enabled by default on macOS/aarch64: {s}
        \\
        \\Status:
        \\  This is the initial scaffold. The command surface is in place,
        \\  but the GGUF runtime and Metal backend are not implemented yet.
        \\
    ,
        .{
            build_options.version,
            if (build_options.enable_metal) "yes" else "no",
        },
    );
}

fn printStub(writer: anytype, name: []const u8, config: Config) !void {
    try writer.print(
        \\[{s}] scaffold command
        \\version: {s}
        \\metal_enabled: {s}
        \\model: {s}
        \\prompt: {s}
        \\port: {d}
        \\
        \\This command is part of the initial scaffold only.
        \\Implementation will arrive in roadmap order:
        \\GGUF inspect -> CPU correctness -> Metal runtime -> tiny server.
        \\
    ,
        .{
            name,
            build_options.version,
            if (build_options.enable_metal) "yes" else "no",
            config.model_path orelse "<unset>",
            config.prompt orelse "<unset>",
            config.port,
        },
    );
}

test "known command parsing works" {
    const config = try parseArgs(&.{ "ziggy-llm", "inspect", "-m", "demo.gguf" });
    try std.testing.expectEqual(Command.inspect, config.command);
    try std.testing.expectEqualStrings("demo.gguf", config.model_path.?);
}

test "version flag parsing works" {
    const config = try parseArgs(&.{ "ziggy-llm", "--version" });
    try std.testing.expectEqual(Command.version, config.command);
}
