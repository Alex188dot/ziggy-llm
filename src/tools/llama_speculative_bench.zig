const std = @import("std");
const runtime = @import("ziggy_runtime");

const Config = struct {
    model_path: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    max_tokens: usize = 64,
    draft_tokens: usize = 4,
    bench_runs: usize = 3,
    backend: runtime.BackendPreference = .auto,
    moon_quant: runtime.MoonQuantMode = .enabled,
    mismatch_mode: runtime.llama_speculative.DraftMismatchMode = .exact,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const config = try parseArgs(args);
    const model_path = config.model_path orelse return error.MissingModelPath;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdout_buffer);

    const spec_case = runtime.llama_speculative.BenchCase{
        .name = "llama-first",
        .prompt = config.prompt orelse runtime.llama_speculative.default_cases[0].prompt,
        .max_tokens = config.max_tokens,
        .draft_tokens = config.draft_tokens,
        .mismatch_mode = config.mismatch_mode,
    };
    const result = try runtime.llama_speculative.runCase(
        allocator,
        model_path,
        .{
            .backend = config.backend,
            .moon_quant = config.moon_quant,
            .max_tokens = config.max_tokens,
            .temperature = 0,
        },
        spec_case,
        config.bench_runs,
    );
    try stdout.interface.print(
        "speculative.scope=llama-first\nspeculative.temperature=0.000\nspeculative.single_user=true\n",
        .{},
    );
    try runtime.llama_speculative.printCaseResult(&stdout.interface, result);
    try stdout.interface.flush();
}

fn parseArgs(args: []const []const u8) !Config {
    var config = Config{};
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            config.model_path = args[index];
            continue;
        }
        if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            config.prompt = args[index];
            continue;
        }
        if (std.mem.eql(u8, arg, "--max-tokens")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            config.max_tokens = try std.fmt.parseUnsigned(usize, args[index], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--draft-tokens")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            config.draft_tokens = try std.fmt.parseUnsigned(usize, args[index], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--bench-runs")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            config.bench_runs = try std.fmt.parseUnsigned(usize, args[index], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            config.backend = runtime.BackendPreference.parse(args[index]) orelse return error.InvalidBackend;
            continue;
        }
        if (std.mem.eql(u8, arg, "--moon-quant")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            config.moon_quant = runtime.MoonQuantMode.parse(args[index]) orelse return error.InvalidMoonQuant;
            continue;
        }
        if (std.mem.eql(u8, arg, "--mismatch-mode")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            config.mismatch_mode = parseMismatchMode(args[index]) orelse return error.InvalidMismatchMode;
            continue;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printHelp();
            std.process.exit(0);
        }
        return error.UnknownFlag;
    }

    if (config.draft_tokens == 0) return error.InvalidDraftTokens;
    if (config.bench_runs == 0) return error.InvalidBenchRuns;
    return config;
}

fn parseMismatchMode(value: []const u8) ?runtime.llama_speculative.DraftMismatchMode {
    if (std.mem.eql(u8, value, "exact")) return .exact;
    if (std.mem.eql(u8, value, "reject-last-token-every-round")) return .reject_last_token_every_round;
    if (std.mem.eql(u8, value, "reject-last-token-every-other-round")) return .reject_last_token_every_other_round;
    return null;
}

fn printHelp() !void {
    var buffer: [2048]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&buffer);
    try stdout.interface.print(
        \\Usage:
        \\  zig build llama-spec-bench -- --model <path> [options]
        \\
        \\Options:
        \\  -m, --model <path>     Path to a llama GGUF model
        \\  -p, --prompt <text>    Prompt text for the benchmark case
        \\      --max-tokens <n>   Number of decode tokens to target (default: 64)
        \\      --draft-tokens <n> Draft proposal width per round (default: 4)
        \\      --bench-runs <n>   Number of repeated warm runs to average (default: 3)
        \\      --backend <name>   Backend preference: auto, cpu, metal
        \\      --moon-quant <m>   MoonQuant mode: enabled or disabled
        \\      --mismatch-mode    exact | reject-last-token-every-round | reject-last-token-every-other-round
        \\
    , .{});
    try stdout.interface.flush();
}
