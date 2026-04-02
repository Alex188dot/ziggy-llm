const std = @import("std");
const llama_fixture = @import("llama_fixture.zig");
const llama_loaded_runtime = @import("llama_loaded_runtime.zig");
const llama_speculative_runtime = @import("llama_speculative_runtime.zig");
const types = @import("types.zig");

pub const BenchSummary = struct {
    cold: types.GenerationReport,
    warm_runs: usize = 0,
    warm_startup_ns_avg: u64 = 0,
    warm_prompt_ns_avg: u64 = 0,
    warm_ttft_ns_avg: u64 = 0,
    warm_decode_ns_avg: u64 = 0,
    warm_generated_token_count_avg: usize = 0,
    warm_startup_breakdown_avg: types.StartupBreakdown = .{},
    warm_speculative_avg: ?types.SpeculativeStats = null,
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

    var runtime = try llama_loaded_runtime.LoadedRuntime.init(allocator, model_path, options.backend, options.moon_quant, options.metal_profile);
    defer runtime.deinit();
    var draft_runtime = if (options.speculative.enabled())
        try llama_loaded_runtime.LoadedRuntime.init(
            allocator,
            options.speculative.draft_model_path orelse return error.InvalidSpeculativeConfig,
            options.backend,
            options.moon_quant,
            false,
        )
    else
        null;
    defer if (draft_runtime) |*loaded| loaded.deinit();

    var cold = try generateWithOptionalSpec(&runtime, if (draft_runtime) |*loaded| loaded else null, prompt, options, runtime.model_load_ns);
    errdefer cold.deinit(allocator);

    if (bench_runs == 1) {
        return .{ .cold = cold };
    }

    var warm_startup_total: u128 = 0;
    var warm_prompt_total: u128 = 0;
    var warm_ttft_total: u128 = 0;
    var warm_decode_total: u128 = 0;
    var warm_generated_token_total: u128 = 0;
    var warm_startup_breakdown_total = types.StartupBreakdown{};
    var warm_speculative_total: ?types.SpeculativeStats = if (cold.speculative != null)
        .{
            .draft_tokens = cold.speculative.?.draft_tokens,
            .rounds = 0,
            .accepted_tokens = 0,
            .rejected_tokens = 0,
            .draft_decode_ns = 0,
            .verifier_decode_ns = 0,
            .resync_count = 0,
        }
    else
        null;
    var warm_metal_profile_summary: ?[]u8 = null;

    for (1..bench_runs) |_| {
        var warm = try generateWithOptionalSpec(&runtime, if (draft_runtime) |*loaded| loaded else null, prompt, options, 0);
        warm_startup_total += warm.startup_ns;
        warm_prompt_total += warm.prompt_ns;
        warm_ttft_total += warm.ttft_ns;
        warm_decode_total += warm.decode_ns;
        warm_generated_token_total += warm.generated_token_count;
        warm_startup_breakdown_total.model_load_ns += warm.startup_breakdown.model_load_ns;
        warm_startup_breakdown_total.tensor_prepare_ns += warm.startup_breakdown.tensor_prepare_ns;
        warm_startup_breakdown_total.backend_init_ns += warm.startup_breakdown.backend_init_ns;
        warm_startup_breakdown_total.metal_prewarm_ns += warm.startup_breakdown.metal_prewarm_ns;
        warm_startup_breakdown_total.session_init_ns += warm.startup_breakdown.session_init_ns;
        warm_startup_breakdown_total.first_decode_step_ns += warm.startup_breakdown.first_decode_step_ns;
        if (warm_speculative_total != null and warm.speculative != null) {
            warm_speculative_total.?.rounds += warm.speculative.?.rounds;
            warm_speculative_total.?.accepted_tokens += warm.speculative.?.accepted_tokens;
            warm_speculative_total.?.rejected_tokens += warm.speculative.?.rejected_tokens;
            warm_speculative_total.?.draft_decode_ns += warm.speculative.?.draft_decode_ns;
            warm_speculative_total.?.verifier_decode_ns += warm.speculative.?.verifier_decode_ns;
            warm_speculative_total.?.resync_count += warm.speculative.?.resync_count;
        }
        if (warm_metal_profile_summary == null and warm.metal_profile_summary != null) {
            warm_metal_profile_summary = warm.metal_profile_summary;
            warm.metal_profile_summary = null;
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
        .warm_startup_breakdown_avg = .{
            .model_load_ns = @intCast(warm_startup_breakdown_total.model_load_ns / warm_runs),
            .tensor_prepare_ns = @intCast(warm_startup_breakdown_total.tensor_prepare_ns / warm_runs),
            .backend_init_ns = @intCast(warm_startup_breakdown_total.backend_init_ns / warm_runs),
            .metal_prewarm_ns = @intCast(warm_startup_breakdown_total.metal_prewarm_ns / warm_runs),
            .session_init_ns = @intCast(warm_startup_breakdown_total.session_init_ns / warm_runs),
            .first_decode_step_ns = @intCast(warm_startup_breakdown_total.first_decode_step_ns / warm_runs),
        },
        .warm_speculative_avg = if (warm_speculative_total) |total|
            .{
                .draft_tokens = total.draft_tokens,
                .rounds = @intCast(total.rounds / warm_runs),
                .accepted_tokens = @intCast(total.accepted_tokens / warm_runs),
                .rejected_tokens = @intCast(total.rejected_tokens / warm_runs),
                .draft_decode_ns = @intCast(total.draft_decode_ns / warm_runs),
                .verifier_decode_ns = @intCast(total.verifier_decode_ns / warm_runs),
                .resync_count = @intCast(total.resync_count / warm_runs),
            }
        else
            null,
        .warm_metal_profile_summary = warm_metal_profile_summary,
    };
}

fn generateWithOptionalSpec(
    runtime: *llama_loaded_runtime.LoadedRuntime,
    draft_runtime: ?*llama_loaded_runtime.LoadedRuntime,
    prompt: []const u8,
    options: types.GenerationOptions,
    setup_ns: u64,
) !types.GenerationReport {
    if (draft_runtime) |loaded| {
        const startup_breakdown = if (setup_ns > 0)
            llama_speculative_runtime.coldStartupBreakdown(runtime, loaded)
        else
            llama_speculative_runtime.warmStartupBreakdown(runtime, loaded);
        return llama_speculative_runtime.generateWithDraftRuntime(
            runtime.allocator,
            runtime,
            loaded,
            prompt,
            options,
            startup_breakdown,
        );
    }
    var report = try runtime.generate(prompt, options, setup_ns);
    report.startup_breakdown.model_load_ns = runtime.model_load_ns;
    return report;
}

test "warm bench reuses loaded cpu runtime and reports warm averages" {
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
}
