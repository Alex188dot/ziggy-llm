const std = @import("std");
const bench_runner = @import("bench_runner.zig");
const metal_profile = @import("metal_profile.zig");
const types = @import("types.zig");

pub const BenchCase = struct {
    name: []const u8,
    prompt: []const u8,
    max_tokens: usize,
};

pub const GuardrailConfig = struct {
    min_warm_decode_speedup_pct: f64 = 0.0,
    max_warm_ttft_regression_pct: f64 = 25.0,
};

pub const CaseResult = struct {
    case: BenchCase,
    enabled: bench_runner.BenchSummary,
    disabled: bench_runner.BenchSummary,
    warm_decode_speedup_pct: f64,
    warm_ttft_regression_pct: f64,
    warm_profile_delta: ?WarmProfileDelta = null,

    pub fn deinit(self: *CaseResult, allocator: std.mem.Allocator) void {
        self.enabled.deinit(allocator);
        self.disabled.deinit(allocator);
        self.* = undefined;
    }
};

pub const CategoryDelta = struct {
    label: []const u8,
    enabled_ns: u64,
    disabled_ns: u64,
    delta_ns: i64,
};

pub const WarmProfileDelta = struct {
    categories: [metal_profile.categoryCount()]CategoryDelta,
    moon_quant_total_enabled_ns: u64,
    moon_quant_total_disabled_ns: u64,
    moon_quant_total_delta_ns: i64,
};

pub const GuardrailViolation = struct {
    case_name: []const u8,
    message: []const u8,
};

pub const default_cases = [_]BenchCase{
    .{
        .name = "short-rewrite-32",
        .prompt = "Rewrite this sentence with sharper wording: MoonQuant keeps Apple-first benchmarks honest.",
        .max_tokens = 32,
    },
    .{
        .name = "reasoning-96",
        .prompt = "List three practical reasons benchmark guardrails matter for local inference runtimes, and keep each reason concise but concrete.",
        .max_tokens = 96,
    },
    .{
        .name = "chat-160",
        .prompt = "User: Summarize what MoonQuant is trying to optimize on Apple Silicon.\nAssistant:",
        .max_tokens = 160,
    },
};

pub fn runSuite(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    backend: types.BackendPreference,
    bench_runs: usize,
    base_options: types.GenerationOptions,
    cases: []const BenchCase,
) ![]CaseResult {
    var results = try allocator.alloc(CaseResult, cases.len);
    var initialized: usize = 0;
    errdefer {
        for (results[0..initialized]) |*result| result.deinit(allocator);
        allocator.free(results);
    }

    for (cases, 0..) |case, index| {
        var enabled_options = base_options;
        enabled_options.backend = backend;
        enabled_options.max_tokens = case.max_tokens;
        enabled_options.moon_quant = .enabled;

        var enabled = try bench_runner.runWarmBench(
            allocator,
            model_path,
            case.prompt,
            enabled_options,
            bench_runs,
        );
        errdefer enabled.deinit(allocator);

        var disabled_options = enabled_options;
        disabled_options.moon_quant = .disabled;

        var disabled = try bench_runner.runWarmBench(
            allocator,
            model_path,
            case.prompt,
            disabled_options,
            bench_runs,
        );
        errdefer disabled.deinit(allocator);

        results[index] = .{
            .case = case,
            .enabled = enabled,
            .disabled = disabled,
            .warm_decode_speedup_pct = percentDelta(
                disabled.warmDecodeTokensPerSecond(),
                enabled.warmDecodeTokensPerSecond(),
            ),
            .warm_ttft_regression_pct = percentDelta(
                disabled.warm_ttft_ns_avg,
                enabled.warm_ttft_ns_avg,
            ),
            .warm_profile_delta = try collectWarmProfileDelta(
                allocator,
                model_path,
                case.prompt,
                enabled_options,
                disabled_options,
            ),
        };
        initialized += 1;
    }

    return results;
}

pub fn findViolation(results: []const CaseResult, guardrail: GuardrailConfig) ?GuardrailViolation {
    for (results) |result| {
        if (findCaseViolation(result.warm_decode_speedup_pct, result.warm_ttft_regression_pct, guardrail)) |message| {
            return .{
                .case_name = result.case.name,
                .message = message,
            };
        }
    }
    return null;
}

pub fn printSuiteReport(
    writer: *std.Io.Writer,
    results: []const CaseResult,
    guardrail: GuardrailConfig,
) !void {
    try writer.print(
        \\moon_quant.guardrail.min_warm_decode_speedup_pct={d:.3}
        \\moon_quant.guardrail.max_warm_ttft_regression_pct={d:.3}
        \\moon_quant.case_count={d}
        \\
    ,
        .{
            guardrail.min_warm_decode_speedup_pct,
            guardrail.max_warm_ttft_regression_pct,
            results.len,
        },
    );

    for (results) |result| {
        try writer.print(
            \\case={s}
            \\prompt={s}
            \\max_tokens={d}
            \\enabled.backend={s}
            \\enabled.warm.decode_tok_s={d:.3}
            \\enabled.warm.ttft_ms={d:.3}
            \\disabled.backend={s}
            \\disabled.warm.decode_tok_s={d:.3}
            \\disabled.warm.ttft_ms={d:.3}
            \\warm.decode_speedup_pct={d:.3}
            \\warm.ttft_regression_pct={d:.3}
            \\
        ,
            .{
                result.case.name,
                result.case.prompt,
                result.case.max_tokens,
                result.enabled.cold.backend.label(),
                result.enabled.warmDecodeTokensPerSecond(),
                types.nsToMs(result.enabled.warm_ttft_ns_avg),
                result.disabled.cold.backend.label(),
                result.disabled.warmDecodeTokensPerSecond(),
                types.nsToMs(result.disabled.warm_ttft_ns_avg),
                result.warm_decode_speedup_pct,
                result.warm_ttft_regression_pct,
            },
        );
        if (result.warm_profile_delta) |profile_delta| {
            try writer.print(
                "warm.profile.moon_quant_total.enabled_ns={d}\nwarm.profile.moon_quant_total.disabled_ns={d}\nwarm.profile.moon_quant_total.delta_ns={d}\n",
                .{
                    profile_delta.moon_quant_total_enabled_ns,
                    profile_delta.moon_quant_total_disabled_ns,
                    profile_delta.moon_quant_total_delta_ns,
                },
            );
            for (profile_delta.categories) |delta| {
                try writer.print(
                    "warm.profile.delta.{s}.enabled_ns={d}\nwarm.profile.delta.{s}.disabled_ns={d}\nwarm.profile.delta.{s}.delta_ns={d}\n",
                    .{
                        delta.label,
                        delta.enabled_ns,
                        delta.label,
                        delta.disabled_ns,
                        delta.label,
                        delta.delta_ns,
                    },
                );
            }
        }
    }
}

fn collectWarmProfileDelta(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    enabled_options: types.GenerationOptions,
    disabled_options: types.GenerationOptions,
) !?WarmProfileDelta {
    if (enabled_options.backend != .metal) return null;

    var profiled_enabled_options = enabled_options;
    profiled_enabled_options.metal_profile = true;
    var enabled_profiled = try bench_runner.runWarmBench(
        allocator,
        model_path,
        prompt,
        profiled_enabled_options,
        2,
    );
    defer enabled_profiled.deinit(allocator);

    var profiled_disabled_options = disabled_options;
    profiled_disabled_options.metal_profile = true;
    var disabled_profiled = try bench_runner.runWarmBench(
        allocator,
        model_path,
        prompt,
        profiled_disabled_options,
        2,
    );
    defer disabled_profiled.deinit(allocator);

    const enabled_summary = enabled_profiled.warm_metal_profile_summary orelse return null;
    const disabled_summary = disabled_profiled.warm_metal_profile_summary orelse return null;

    var categories: [metal_profile.categoryCount()]CategoryDelta = undefined;
    for (std.enums.values(metal_profile.Category), 0..) |category, index| {
        const enabled_ns = parseMetricU64ForLabel(enabled_summary, "profile.", category.label(), ".ns=");
        const disabled_ns = parseMetricU64ForLabel(disabled_summary, "profile.", category.label(), ".ns=");
        categories[index] = .{
            .label = category.label(),
            .enabled_ns = enabled_ns,
            .disabled_ns = disabled_ns,
            .delta_ns = deltaSigned(enabled_ns, disabled_ns),
        };
    }

    const moon_quant_enabled_ns = parseMetricU64(enabled_summary, "moon_quant.profile.total_ns=");
    const moon_quant_disabled_ns = parseMetricU64(disabled_summary, "moon_quant.profile.total_ns=");
    return .{
        .categories = categories,
        .moon_quant_total_enabled_ns = moon_quant_enabled_ns,
        .moon_quant_total_disabled_ns = moon_quant_disabled_ns,
        .moon_quant_total_delta_ns = deltaSigned(moon_quant_enabled_ns, moon_quant_disabled_ns),
    };
}

fn parseMetricU64ForLabel(
    summary: []const u8,
    prefix: []const u8,
    label: []const u8,
    suffix: []const u8,
) u64 {
    var key_buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "{s}{s}{s}", .{ prefix, label, suffix }) catch return 0;
    return parseMetricU64(summary, key);
}

fn parseMetricU64(summary: []const u8, key: []const u8) u64 {
    var lines = std.mem.splitScalar(u8, summary, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        return std.fmt.parseUnsigned(u64, line[key.len..], 10) catch 0;
    }
    return 0;
}

fn deltaSigned(enabled: u64, disabled: u64) i64 {
    return @as(i64, @intCast(enabled)) - @as(i64, @intCast(disabled));
}

fn percentDelta(baseline: anytype, candidate: anytype) f64 {
    const baseline_f64 = switch (@TypeOf(baseline)) {
        f64 => baseline,
        else => @as(f64, @floatFromInt(baseline)),
    };
    if (baseline_f64 == 0) return 0;
    const candidate_f64 = switch (@TypeOf(candidate)) {
        f64 => candidate,
        else => @as(f64, @floatFromInt(candidate)),
    };
    return ((candidate_f64 - baseline_f64) / baseline_f64) * 100.0;
}

fn findCaseViolation(
    warm_decode_speedup_pct: f64,
    warm_ttft_regression_pct: f64,
    guardrail: GuardrailConfig,
) ?[]const u8 {
    if (warm_decode_speedup_pct < guardrail.min_warm_decode_speedup_pct) {
        return "warm decode speedup fell below the configured floor";
    }
    if (warm_ttft_regression_pct > guardrail.max_warm_ttft_regression_pct) {
        return "warm TTFT regression exceeded the configured ceiling";
    }
    return null;
}

test "default MoonQuant benchmark suite covers multiple prompt and token pairs" {
    try std.testing.expect(default_cases.len > 1);
    try std.testing.expect(default_cases[0].max_tokens != default_cases[1].max_tokens);
    try std.testing.expect(!std.mem.eql(u8, default_cases[0].prompt, default_cases[1].prompt));
}

test "guardrail violation catches decode regressions" {
    const violation = findCaseViolation(-7.5, 12.0, .{
        .min_warm_decode_speedup_pct = 0.0,
        .max_warm_ttft_regression_pct = 25.0,
    });
    try std.testing.expect(violation != null);
    try std.testing.expectEqualStrings("warm decode speedup fell below the configured floor", violation.?);
}

test "guardrail violation catches ttft regressions" {
    const violation = findCaseViolation(4.0, 30.0, .{
        .min_warm_decode_speedup_pct = 0.0,
        .max_warm_ttft_regression_pct = 25.0,
    });
    try std.testing.expect(violation != null);
    try std.testing.expectEqualStrings("warm TTFT regression exceeded the configured ceiling", violation.?);
}

test "profile metric parser extracts category and moon quant totals" {
    const summary =
        \\profile.projection_quantized.ns=120
        \\profile.attention.ns=40
        \\moon_quant.profile.total_ns=70
        \\
    ;
    try std.testing.expectEqual(@as(u64, 120), parseMetricU64ForLabel(summary, "profile.", "projection_quantized", ".ns="));
    try std.testing.expectEqual(@as(u64, 70), parseMetricU64(summary, "moon_quant.profile.total_ns="));
}
