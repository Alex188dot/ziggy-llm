const std = @import("std");
const runtime = @import("runtime.zig");
const server = @import("server.zig");
const build_options = @import("build_options");

pub const Command = enum {
    run,
    chat,
    inspect,
    bench,
    serve,
    help,
    version,
};

pub const Config = struct {
    command: Command = .help,
    model_path: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    port: u16 = server.default_port,
    max_tokens: usize = 16,
    bench_runs: usize = 1,
    seed: u64 = 0,
    temperature: f32 = 0,
    backend: runtime.BackendPreference = .auto,
    metal_profile: bool = false,
};

pub const ParseError = error{
    UnknownCommand,
    UnknownFlag,
    MissingFlagValue,
    InvalidPort,
    InvalidMaxTokens,
    InvalidSeed,
    InvalidTemperature,
    InvalidBackend,
};

pub fn parseArgs(args: []const []const u8) ParseError!Config {
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
            config.port = std.fmt.parseUnsigned(u16, args[i], 10) catch return error.InvalidPort;
            continue;
        }
        if (std.mem.eql(u8, arg, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config.max_tokens = std.fmt.parseUnsigned(usize, args[i], 10) catch return error.InvalidMaxTokens;
            continue;
        }
        if (std.mem.eql(u8, arg, "--bench-runs")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config.bench_runs = std.fmt.parseUnsigned(usize, args[i], 10) catch return error.InvalidMaxTokens;
            continue;
        }
        if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config.seed = std.fmt.parseUnsigned(u64, args[i], 10) catch return error.InvalidSeed;
            continue;
        }
        if (std.mem.eql(u8, arg, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config.temperature = std.fmt.parseFloat(f32, args[i]) catch return error.InvalidTemperature;
            continue;
        }
        if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            config.backend = runtime.BackendPreference.parse(args[i]) orelse return error.InvalidBackend;
            continue;
        }
        if (std.mem.eql(u8, arg, "--metal-profile")) {
            config.metal_profile = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            config.command = .help;
            return config;
        }
        if (config.command == .inspect and config.model_path == null and arg.len > 0 and arg[0] != '-') {
            config.model_path = arg;
            continue;
        }

        return error.UnknownFlag;
    }

    return config;
}

pub fn parseCommand(name: []const u8) ?Command {
    if (std.mem.eql(u8, name, "run")) return .run;
    if (std.mem.eql(u8, name, "chat")) return .chat;
    if (std.mem.eql(u8, name, "inspect")) return .inspect;
    if (std.mem.eql(u8, name, "bench")) return .bench;
    if (std.mem.eql(u8, name, "serve")) return .serve;
    return null;
}

pub fn printHelp(writer: *std.Io.Writer) !void {
    try writer.print(
        \\ziggy-llm {s}
        \\
        \\A Mac-first, Zig-native GGUF inference engine with first-class Apple Metal support.
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
        \\      --max-tokens <n>  Maximum generated tokens for run/bench (default: {d})
        \\      --bench-runs <n>  Bench only: one cold run plus warm averages over reused runtime (default: {d})
        \\      --seed <n>        Seed for deterministic sampling (default: {d})
        \\      --temperature <f> Sampling temperature; 0 means argmax (default: {d:.1})
        \\      --backend <name>  Backend preference: auto, cpu, metal (default: {s})
        \\      --metal-profile   Print aggregated Metal decode timing and dominant shape data
        \\      --port <port>     Port for server mode (default: {d})
        \\
        \\Build:
        \\  Metal enabled by default on macOS/aarch64: {s}
        \\
        \\Status:
        \\  LLaMA-family GGUF execution supports the native CPU runtime and Metal on Apple Silicon.
        \\  Metal acceleration is focused on the implemented llama-family runtime.
        \\
    ,
        .{
            build_options.version,
            configDefaults.max_tokens,
            configDefaults.bench_runs,
            configDefaults.seed,
            configDefaults.temperature,
            configDefaults.backend.label(),
            server.default_port,
            if (build_options.enable_metal) "yes" else "no",
        },
    );
}

const configDefaults = Config{};

test "known command parsing works" {
    const config = try parseArgs(&.{ "ziggy-llm", "inspect", "-m", "demo.gguf" });
    try std.testing.expectEqual(Command.inspect, config.command);
    try std.testing.expectEqualStrings("demo.gguf", config.model_path.?);
}

test "inspect accepts positional model path" {
    const config = try parseArgs(&.{ "ziggy-llm", "inspect", "demo.gguf" });
    try std.testing.expectEqual(Command.inspect, config.command);
    try std.testing.expectEqualStrings("demo.gguf", config.model_path.?);
}

test "version flag parsing works" {
    const config = try parseArgs(&.{ "ziggy-llm", "--version" });
    try std.testing.expectEqual(Command.version, config.command);
}

test "runtime flags parse correctly" {
    const config = try parseArgs(&.{ "ziggy-llm", "bench", "-m", "demo.gguf", "-p", "hi", "--max-tokens", "4", "--bench-runs", "3", "--seed", "9", "--temperature", "0.5", "--backend", "metal", "--metal-profile" });
    try std.testing.expectEqual(@as(usize, 4), config.max_tokens);
    try std.testing.expectEqual(@as(usize, 3), config.bench_runs);
    try std.testing.expectEqual(@as(u64, 9), config.seed);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), config.temperature, 0.0001);
    try std.testing.expectEqual(runtime.BackendPreference.metal, config.backend);
    try std.testing.expect(config.metal_profile);
}
