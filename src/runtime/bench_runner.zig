const std = @import("std");
const llama_fixture = @import("llama_fixture.zig");
const metal_profile = @import("metal_profile.zig");
const resident_runtime = @import("resident_runtime.zig");
const types = @import("types.zig");

pub const BenchSummary = struct {
    cold: types.GenerationReport,
    warm_runs: usize = 0,
    warm_startup_ns_avg: u64 = 0,
    warm_prompt_ns_avg: u64 = 0,
    warm_ttft_ns_avg: u64 = 0,
    warm_decode_ns_avg: u64 = 0,
    warm_generated_token_count_avg: usize = 0,
    warm_reused_prompt_token_count_avg: usize = 0,
    warm_startup_breakdown_avg: types.StartupBreakdown = .{},
    warm_metal_profile_stats: metal_profile.SummaryStats = .{},
    warm_metal_profile_summary: ?[]u8 = null,

    pub fn deinit(self: *BenchSummary, allocator: std.mem.Allocator) void {
        self.cold.deinit(allocator);
        if (self.warm_metal_profile_summary) |summary| allocator.free(summary);
        self.* = undefined;
    }

    pub fn warmDecodeTokensPerSecond(self: BenchSummary) f64 {
        if (self.warm_runs == 0 or self.warm_generated_token_count_avg == 0 or self.warm_decode_ns_avg == 0) return 0;
        return @as(f64, @floatFromInt(self.warm_generated_token_count_avg)) /
            (@as(f64, @floatFromInt(self.warm_decode_ns_avg)) / std.time.ns_per_s);
    }
};

pub fn runWarmBench(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
    bench_runs: usize,
) !BenchSummary {
    std.debug.assert(bench_runs > 0);

    var runtime = resident_runtime.ResidentRuntime.init(allocator);
    defer runtime.deinit();
    runtime.setKeepAliveSeconds(-1);

    var cold = try runtime.generate(model_path, prompt, options);
    errdefer cold.deinit(allocator);

    if (bench_runs == 1) {
        return .{ .cold = cold };
    }

    var warm_startup_total: u128 = 0;
    var warm_prompt_total: u128 = 0;
    var warm_ttft_total: u128 = 0;
    var warm_decode_total: u128 = 0;
    var warm_generated_token_total: u128 = 0;
    var warm_reused_prompt_token_total: u128 = 0;
    var warm_startup_breakdown_total = types.StartupBreakdown{};
    var warm_metal_profile_totals = metal_profile.SummaryStats{};
    var warm_metal_profile_summary: ?[]u8 = null;

    for (1..bench_runs) |_| {
        var warm = try runtime.generate(model_path, prompt, options);
        warm_startup_total += warm.startup_ns;
        warm_prompt_total += warm.prompt_ns;
        warm_ttft_total += warm.ttft_ns;
        warm_decode_total += warm.decode_ns;
        warm_generated_token_total += warm.generated_token_count;
        warm_reused_prompt_token_total += warm.reused_prompt_token_count;
        warm_startup_breakdown_total.model_load_ns += warm.startup_breakdown.model_load_ns;
        warm_startup_breakdown_total.tensor_prepare_ns += warm.startup_breakdown.tensor_prepare_ns;
        warm_startup_breakdown_total.backend_init_ns += warm.startup_breakdown.backend_init_ns;
        warm_startup_breakdown_total.metal_prewarm_ns += warm.startup_breakdown.metal_prewarm_ns;
        warm_startup_breakdown_total.session_init_ns += warm.startup_breakdown.session_init_ns;
        warm_startup_breakdown_total.first_decode_step_ns += warm.startup_breakdown.first_decode_step_ns;
        warm_metal_profile_totals.add(metal_profile.parseSummary(warm.metal_profile_summary));
        if (warm.metal_profile_summary) |summary| {
            if (warm_metal_profile_summary == null) {
                warm_metal_profile_summary = summary;
                warm.metal_profile_summary = null;
            }
        }
        warm.deinit(allocator);
    }

    const warm_runs = bench_runs - 1;
    return .{
        .cold = cold,
        .warm_runs = warm_runs,
        .warm_startup_ns_avg = @intCast(warm_startup_total / warm_runs),
        .warm_prompt_ns_avg = @intCast(warm_prompt_total / warm_runs),
        .warm_ttft_ns_avg = @intCast(warm_ttft_total / warm_runs),
        .warm_decode_ns_avg = @intCast(warm_decode_total / warm_runs),
        .warm_generated_token_count_avg = @intCast(warm_generated_token_total / warm_runs),
        .warm_reused_prompt_token_count_avg = @intCast(warm_reused_prompt_token_total / warm_runs),
        .warm_startup_breakdown_avg = .{
            .model_load_ns = @intCast(warm_startup_breakdown_total.model_load_ns / warm_runs),
            .tensor_prepare_ns = @intCast(warm_startup_breakdown_total.tensor_prepare_ns / warm_runs),
            .backend_init_ns = @intCast(warm_startup_breakdown_total.backend_init_ns / warm_runs),
            .metal_prewarm_ns = @intCast(warm_startup_breakdown_total.metal_prewarm_ns / warm_runs),
            .session_init_ns = @intCast(warm_startup_breakdown_total.session_init_ns / warm_runs),
            .first_decode_step_ns = @intCast(warm_startup_breakdown_total.first_decode_step_ns / warm_runs),
        },
        .warm_metal_profile_stats = warm_metal_profile_totals.average(warm_runs),
        .warm_metal_profile_summary = warm_metal_profile_summary,
    };
}

test "warm bench reuses resident cpu runtime and reports warm averages" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "bench-llama.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "bench-llama.gguf");
    defer std.testing.allocator.free(path);

    var summary = try runWarmBench(std.testing.allocator, path, "a", .{
        .max_tokens = 3,
        .seed = 0,
        .temperature = 0,
        .backend = .cpu,
    }, 3);
    defer summary.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), summary.warm_runs);
    try std.testing.expectEqual(@as(types.BackendUsed, .cpu), summary.cold.backend);
    try std.testing.expect(summary.cold.ttft_ns > 0);
    try std.testing.expect(summary.warm_ttft_ns_avg > 0);
    try std.testing.expect(summary.warm_generated_token_count_avg > 0);
    try std.testing.expectEqual(@as(u64, 0), summary.warm_startup_breakdown_avg.model_load_ns);
    try std.testing.expectEqual(@as(u64, 0), summary.warm_startup_breakdown_avg.session_init_ns);
}
