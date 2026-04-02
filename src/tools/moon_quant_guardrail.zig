const std = @import("std");
const runtime = @import("ziggy_runtime");
const moon_quant_bench = runtime.moon_quant_bench;

const Config = struct {
    model_path: ?[]const u8 = null,
    bench_runs: usize = 5,
    backend: runtime.BackendPreference = .metal,
    min_warm_decode_speedup_pct: f64 = 0.0,
    max_warm_ttft_regression_pct: f64 = 25.0,
    seed: u64 = 7,
    temperature: f32 = 0.0,
    repeat_penalty: f32 = 1.0,
    top_k: usize = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
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

    const results = try moon_quant_bench.runSuite(
        allocator,
        model_path,
        config.backend,
        config.bench_runs,
        .{
            .seed = config.seed,
            .temperature = config.temperature,
            .repeat_penalty = config.repeat_penalty,
            .top_k = config.top_k,
            .top_p = config.top_p,
            .min_p = config.min_p,
            .backend = config.backend,
            .moon_quant = .enabled,
        },
        &moon_quant_bench.default_cases,
    );
    defer {
        for (results) |*result| result.deinit(allocator);
        allocator.free(results);
    }

    const guardrail = moon_quant_bench.GuardrailConfig{
        .min_warm_decode_speedup_pct = config.min_warm_decode_speedup_pct,
        .max_warm_ttft_regression_pct = config.max_warm_ttft_regression_pct,
    };
    try moon_quant_bench.printSuiteReport(&stdout.interface, results, guardrail);

    if (moon_quant_bench.findViolation(results, guardrail)) |violation| {
        try stdout.interface.print(
            "moon_quant.guardrail=FAIL\nmoon_quant.guardrail.case={s}\nmoon_quant.guardrail.reason={s}\n",
            .{ violation.case_name, violation.message },
        );
        try stdout.interface.flush();
        return error.MoonQuantGuardrailFailed;
    }

    try stdout.interface.print("moon_quant.guardrail=PASS\n", .{});
    try stdout.interface.flush();
}

fn parseArgs(args: []const []const u8) !Config {
    var config = Config{};
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            index += 1;
            if (index >= args.len) return error.MissingModelPath;
            config.model_path = args[index];
            continue;
        }
        if (std.mem.eql(u8, arg, "--bench-runs")) {
            index += 1;
            if (index >= args.len) return error.InvalidBenchRuns;
            config.bench_runs = try std.fmt.parseUnsigned(usize, args[index], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--backend")) {
            index += 1;
            if (index >= args.len) return error.InvalidBackend;
            config.backend = runtime.BackendPreference.parse(args[index]) orelse return error.InvalidBackend;
            continue;
        }
        if (std.mem.eql(u8, arg, "--min-warm-decode-speedup-pct")) {
            index += 1;
            if (index >= args.len) return error.InvalidWarmDecodeSpeedupPct;
            config.min_warm_decode_speedup_pct = try std.fmt.parseFloat(f64, args[index]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--max-warm-ttft-regression-pct")) {
            index += 1;
            if (index >= args.len) return error.InvalidWarmTtftRegressionPct;
            config.max_warm_ttft_regression_pct = try std.fmt.parseFloat(f64, args[index]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--seed")) {
            index += 1;
            if (index >= args.len) return error.InvalidSeed;
            config.seed = try std.fmt.parseUnsigned(u64, args[index], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--temperature")) {
            index += 1;
            if (index >= args.len) return error.InvalidTemperature;
            config.temperature = @floatCast(try std.fmt.parseFloat(f64, args[index]));
            continue;
        }
        if (std.mem.eql(u8, arg, "--repeat-penalty")) {
            index += 1;
            if (index >= args.len) return error.InvalidRepeatPenalty;
            config.repeat_penalty = @floatCast(try std.fmt.parseFloat(f64, args[index]));
            continue;
        }
        if (std.mem.eql(u8, arg, "--top-k")) {
            index += 1;
            if (index >= args.len) return error.InvalidTopK;
            config.top_k = try std.fmt.parseUnsigned(usize, args[index], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--top-p")) {
            index += 1;
            if (index >= args.len) return error.InvalidTopP;
            config.top_p = @floatCast(try std.fmt.parseFloat(f64, args[index]));
            continue;
        }
        if (std.mem.eql(u8, arg, "--min-p")) {
            index += 1;
            if (index >= args.len) return error.InvalidMinP;
            config.min_p = @floatCast(try std.fmt.parseFloat(f64, args[index]));
            continue;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printHelp();
            std.process.exit(0);
        }
        return error.UnknownArgument;
    }
    return config;
}

fn printHelp() !void {
    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdout_buffer);
    try stdout.interface.print(
        \\Usage:
        \\  moon-quant-guardrail --model <path> [options]
        \\
        \\Options:
        \\  --bench-runs <n>                     One cold run plus warm averages over reused runtime (default: 5)
        \\  --backend <auto|cpu|metal>          Backend to compare (default: metal)
        \\  --min-warm-decode-speedup-pct <n>   Required enabled-vs-disabled warm decode speedup floor (default: 0.0)
        \\  --max-warm-ttft-regression-pct <n>  Allowed enabled-vs-disabled warm TTFT regression ceiling (default: 25.0)
        \\  --seed <n>                          Sampling seed (default: 7)
        \\  --temperature <n>                   Sampling temperature (default: 0.0)
        \\  --repeat-penalty <n>                Repeat penalty (default: 1.0)
        \\  --top-k <n>                         Top-k (default: 0)
        \\  --top-p <n>                         Top-p (default: 1.0)
        \\  --min-p <n>                         Min-p (default: 0.0)
        \\
    , .{});
    try stdout.interface.flush();
}
