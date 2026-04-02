const std = @import("std");
const llama_cpu = @import("../llama_cpu.zig");
const llama_loaded_runtime = @import("llama_loaded_runtime.zig");
const types = @import("types.zig");

pub const DraftMismatchMode = enum {
    exact,
    reject_last_token_every_round,
    reject_last_token_every_other_round,

    pub fn label(self: DraftMismatchMode) []const u8 {
        return switch (self) {
            .exact => "exact",
            .reject_last_token_every_round => "reject-last-token-every-round",
            .reject_last_token_every_other_round => "reject-last-token-every-other-round",
        };
    }
};

pub const BenchCase = struct {
    name: []const u8,
    prompt: []const u8,
    max_tokens: usize,
    draft_tokens: usize,
    mismatch_mode: DraftMismatchMode = .exact,
};

pub const default_cases = [_]BenchCase{
    .{
        .name = "llama-exact-64",
        .prompt = "Summarize why a narrow llama-first speculative path is easier to benchmark honestly.",
        .max_tokens = 64,
        .draft_tokens = 4,
        .mismatch_mode = .exact,
    },
    .{
        .name = "llama-reject-tail-64",
        .prompt = "Give two short reasons verifier overhead matters when speculative decoding acceptance drops.",
        .max_tokens = 64,
        .draft_tokens = 4,
        .mismatch_mode = .reject_last_token_every_round,
    },
};

pub const RunMetrics = struct {
    prompt_token_count: usize,
    generated_token_count: usize,
    rounds: usize,
    accepted_token_count: usize,
    rejected_token_count: usize,
    baseline_decode_ns: u64,
    proposer_decode_ns: u64,
    verifier_decode_ns: u64,
    speculative_decode_ns: u64,

    pub fn acceptanceRate(self: RunMetrics) f64 {
        const attempted = self.accepted_token_count + self.rejected_token_count;
        if (attempted == 0) return 0;
        return ratio(self.accepted_token_count, attempted);
    }

    pub fn verifierOverheadPct(self: RunMetrics) f64 {
        if (self.baseline_decode_ns == 0) return 0;
        return percentDelta(
            @as(f64, @floatFromInt(self.baseline_decode_ns)),
            @as(f64, @floatFromInt(self.verifier_decode_ns)),
        );
    }

    pub fn speculativeTokensPerSecond(self: RunMetrics) f64 {
        if (self.generated_token_count == 0 or self.speculative_decode_ns == 0) return 0;
        return tokPerSecond(self.generated_token_count, self.speculative_decode_ns);
    }

    pub fn baselineTokensPerSecond(self: RunMetrics) f64 {
        if (self.generated_token_count == 0 or self.baseline_decode_ns == 0) return 0;
        return tokPerSecond(self.generated_token_count, self.baseline_decode_ns);
    }
};

pub const CaseResult = struct {
    case: BenchCase,
    bench_runs: usize,
    prompt_token_count_avg: usize,
    generated_token_count_avg: usize,
    rounds_avg: usize,
    acceptance_rate_avg: f64,
    verifier_overhead_pct_avg: f64,
    baseline_tok_s_avg: f64,
    speculative_tok_s_avg: f64,
    speculative_speedup_pct_avg: f64,
};

pub fn runCase(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    options: types.GenerationOptions,
    spec_case: BenchCase,
    bench_runs: usize,
) !CaseResult {
    std.debug.assert(bench_runs > 0);

    var runtime = try llama_loaded_runtime.LoadedRuntime.init(allocator, model_path, options.backend, options.moon_quant, false);
    defer runtime.deinit();

    var baseline_session = try llama_cpu.ReusableSession.init(
        allocator,
        &runtime.model,
        runtime.execution.backend,
        runtime.denseLookup(),
    );
    defer baseline_session.deinit(allocator);

    var draft_session = try llama_cpu.ReusableSession.init(
        allocator,
        &runtime.model,
        runtime.execution.backend,
        runtime.denseLookup(),
    );
    defer draft_session.deinit(allocator);

    var verifier_session = try llama_cpu.ReusableSession.init(
        allocator,
        &runtime.model,
        runtime.execution.backend,
        runtime.denseLookup(),
    );
    defer verifier_session.deinit(allocator);

    const prompt_capacity = runtime.model.context_length;
    var prompt_tokens = try allocator.alloc(u32, prompt_capacity);
    defer allocator.free(prompt_tokens);
    const prompt_token_count = try llama_cpu.encodePromptInto(allocator, &runtime.model, spec_case.prompt, prompt_tokens);
    if (prompt_token_count == 0) return error.EmptyPrompt;

    const generated_tokens = try allocator.alloc(u32, spec_case.max_tokens);
    defer allocator.free(generated_tokens);
    const proposed_tokens = try allocator.alloc(u32, spec_case.draft_tokens);
    defer allocator.free(proposed_tokens);

    var prompt_token_total: u128 = 0;
    var generated_token_total: u128 = 0;
    var rounds_total: u128 = 0;
    var acceptance_total: f64 = 0;
    var verifier_overhead_total: f64 = 0;
    var baseline_tok_s_total: f64 = 0;
    var speculative_tok_s_total: f64 = 0;
    var speedup_total: f64 = 0;

    for (0..bench_runs) |_| {
        const metrics = try runSingle(
            allocator,
            &runtime.model,
            &baseline_session,
            &draft_session,
            &verifier_session,
            prompt_tokens[0..prompt_token_count],
            generated_tokens,
            proposed_tokens,
            spec_case,
        );

        prompt_token_total += metrics.prompt_token_count;
        generated_token_total += metrics.generated_token_count;
        rounds_total += metrics.rounds;
        acceptance_total += metrics.acceptanceRate();
        verifier_overhead_total += metrics.verifierOverheadPct();
        baseline_tok_s_total += metrics.baselineTokensPerSecond();
        speculative_tok_s_total += metrics.speculativeTokensPerSecond();
        speedup_total += percentDelta(metrics.baselineTokensPerSecond(), metrics.speculativeTokensPerSecond());
    }

    return .{
        .case = spec_case,
        .bench_runs = bench_runs,
        .prompt_token_count_avg = @intCast(prompt_token_total / bench_runs),
        .generated_token_count_avg = @intCast(generated_token_total / bench_runs),
        .rounds_avg = @intCast(rounds_total / bench_runs),
        .acceptance_rate_avg = acceptance_total / @as(f64, @floatFromInt(bench_runs)),
        .verifier_overhead_pct_avg = verifier_overhead_total / @as(f64, @floatFromInt(bench_runs)),
        .baseline_tok_s_avg = baseline_tok_s_total / @as(f64, @floatFromInt(bench_runs)),
        .speculative_tok_s_avg = speculative_tok_s_total / @as(f64, @floatFromInt(bench_runs)),
        .speculative_speedup_pct_avg = speedup_total / @as(f64, @floatFromInt(bench_runs)),
    };
}

pub fn printCaseResult(writer: *std.Io.Writer, result: CaseResult) !void {
    try writer.print(
        \\case={s}
        \\draft_tokens={d}
        \\mismatch_mode={s}
        \\bench_runs={d}
        \\prompt_tokens_avg={d}
        \\generated_tokens_avg={d}
        \\rounds_avg={d}
        \\acceptance_rate_avg={d:.3}
        \\verifier_overhead_pct_avg={d:.3}
        \\baseline.decode_tok_s_avg={d:.3}
        \\speculative.decode_tok_s_avg={d:.3}
        \\speculative.speedup_pct_avg={d:.3}
        \\
    ,
        .{
            result.case.name,
            result.case.draft_tokens,
            result.case.mismatch_mode.label(),
            result.bench_runs,
            result.prompt_token_count_avg,
            result.generated_token_count_avg,
            result.rounds_avg,
            result.acceptance_rate_avg,
            result.verifier_overhead_pct_avg,
            result.baseline_tok_s_avg,
            result.speculative_tok_s_avg,
            result.speculative_speedup_pct_avg,
        },
    );
}

fn runSingle(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    baseline_session: *llama_cpu.ReusableSession,
    draft_session: *llama_cpu.ReusableSession,
    verifier_session: *llama_cpu.ReusableSession,
    prompt_tokens: []const u32,
    generated_tokens: []u32,
    proposed_tokens: []u32,
    spec_case: BenchCase,
) !RunMetrics {
    const baseline_decode_ns = try runBaseline(model, baseline_session, prompt_tokens, generated_tokens[0..spec_case.max_tokens]);
    const speculative = try runSpeculative(
        allocator,
        model,
        draft_session,
        verifier_session,
        prompt_tokens,
        generated_tokens[0..spec_case.max_tokens],
        proposed_tokens[0..spec_case.draft_tokens],
        spec_case,
    );

    return .{
        .prompt_token_count = prompt_tokens.len,
        .generated_token_count = speculative.generated_token_count,
        .rounds = speculative.rounds,
        .accepted_token_count = speculative.accepted_token_count,
        .rejected_token_count = speculative.rejected_token_count,
        .baseline_decode_ns = baseline_decode_ns,
        .proposer_decode_ns = speculative.proposer_decode_ns,
        .verifier_decode_ns = speculative.verifier_decode_ns,
        .speculative_decode_ns = speculative.total_decode_ns,
    };
}

fn runBaseline(
    model: *const llama_cpu.Model,
    session: *llama_cpu.ReusableSession,
    prompt_tokens: []const u32,
    out_tokens: []u32,
) !u64 {
    try llama_cpu.reusableSessionReplay(session, prompt_tokens);
    const eos_token = llama_cpu.eosTokenId(model);
    var generated_count: usize = 0;
    const decode_begin = std.time.nanoTimestamp();
    while (generated_count < out_tokens.len) : (generated_count += 1) {
        const token = llama_cpu.reusableSessionNextTokenArgmax(session);
        if (eos_token != null and token == eos_token.?) break;
        out_tokens[generated_count] = token;
        try llama_cpu.reusableSessionAcceptToken(session, token);
    }
    return types.deltaNs(decode_begin, std.time.nanoTimestamp());
}

const SpeculativeRun = struct {
    generated_token_count: usize,
    rounds: usize,
    accepted_token_count: usize,
    rejected_token_count: usize,
    proposer_decode_ns: u64,
    verifier_decode_ns: u64,
    total_decode_ns: u64,
};

fn runSpeculative(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    draft_session: *llama_cpu.ReusableSession,
    verifier_session: *llama_cpu.ReusableSession,
    prompt_tokens: []const u32,
    generated_tokens: []u32,
    proposed_tokens: []u32,
    spec_case: BenchCase,
) !SpeculativeRun {
    _ = allocator;
    try llama_cpu.reusableSessionReplay(draft_session, prompt_tokens);
    try llama_cpu.reusableSessionReplay(verifier_session, prompt_tokens);
    const eos_token = llama_cpu.eosTokenId(model);

    var generated_count: usize = 0;
    var rounds: usize = 0;
    var accepted_count: usize = 0;
    var rejected_count: usize = 0;
    var proposer_total: u64 = 0;
    var verifier_total: u64 = 0;

    const decode_begin = std.time.nanoTimestamp();
    while (generated_count < spec_case.max_tokens) {
        rounds += 1;
        const proposal_count, const proposer_ns = try proposeRound(
            draft_session,
            generated_count,
            proposed_tokens,
            spec_case,
        );
        proposer_total += proposer_ns;

        var mismatch = false;
        const verify_begin = std.time.nanoTimestamp();
        for (proposed_tokens[0..proposal_count], 0..) |proposed, proposal_index| {
            const expected = llama_cpu.reusableSessionNextTokenArgmax(verifier_session);
            if (eos_token != null and expected == eos_token.?) {
                mismatch = true;
                break;
            }
            if (proposed != expected) {
                rejected_count += 1;
                generated_tokens[generated_count] = expected;
                generated_count += 1;
                try llama_cpu.reusableSessionAcceptToken(verifier_session, expected);
                try replayDraftFromAccepted(draft_session, prompt_tokens, generated_tokens[0..generated_count]);
                mismatch = true;
                _ = proposal_index;
                break;
            }

            accepted_count += 1;
            generated_tokens[generated_count] = proposed;
            generated_count += 1;
            try llama_cpu.reusableSessionAcceptToken(verifier_session, proposed);
            if (generated_count >= spec_case.max_tokens) break;
        }
        verifier_total += types.deltaNs(verify_begin, std.time.nanoTimestamp());
        if (!mismatch and generated_count >= spec_case.max_tokens) break;
        if (eos_token != null and generated_count > 0 and generated_tokens[generated_count - 1] == eos_token.?) break;
    }

    return .{
        .generated_token_count = generated_count,
        .rounds = rounds,
        .accepted_token_count = accepted_count,
        .rejected_token_count = rejected_count,
        .proposer_decode_ns = proposer_total,
        .verifier_decode_ns = verifier_total,
        .total_decode_ns = types.deltaNs(decode_begin, std.time.nanoTimestamp()),
    };
}

fn proposeRound(
    draft_session: *llama_cpu.ReusableSession,
    generated_count: usize,
    proposed_tokens: []u32,
    spec_case: BenchCase,
) !struct { usize, u64 } {
    const begin = std.time.nanoTimestamp();
    var proposal_count: usize = 0;
    while (proposal_count < proposed_tokens.len) {
        const expected = llama_cpu.reusableSessionNextTokenArgmax(draft_session);
        const proposal = maybeCorruptToken(draft_session, expected, proposal_count, generated_count, spec_case);
        proposed_tokens[proposal_count] = proposal;
        proposal_count += 1;

        if (proposal == expected) {
            try llama_cpu.reusableSessionAcceptToken(draft_session, proposal);
        } else {
            break;
        }
    }
    return .{ proposal_count, types.deltaNs(begin, std.time.nanoTimestamp()) };
}

fn maybeCorruptToken(
    draft_session: *const llama_cpu.ReusableSession,
    token: u32,
    proposal_index: usize,
    generated_count: usize,
    spec_case: BenchCase,
) u32 {
    const should_reject = switch (spec_case.mismatch_mode) {
        .exact => false,
        .reject_last_token_every_round => proposal_index + 1 == spec_case.draft_tokens,
        .reject_last_token_every_other_round => (generated_count / spec_case.draft_tokens) % 2 == 0 and proposal_index + 1 == spec_case.draft_tokens,
    };
    if (!should_reject) return token;

    const vocab_size: u32 = @intCast(llama_cpu.reusableSessionVocabSize(draft_session));
    return (token + 1) % vocab_size;
}

fn replayDraftFromAccepted(
    draft_session: *llama_cpu.ReusableSession,
    prompt_tokens: []const u32,
    accepted_tokens: []const u32,
) !void {
    try llama_cpu.reusableSessionReplay(draft_session, prompt_tokens);
    for (accepted_tokens) |token| {
        try llama_cpu.reusableSessionAcceptToken(draft_session, token);
    }
}

fn tokPerSecond(token_count: usize, decode_ns: u64) f64 {
    return @as(f64, @floatFromInt(token_count)) / (@as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_s);
}

fn ratio(numerator: usize, denominator: usize) f64 {
    return @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator));
}

fn percentDelta(baseline: f64, measured: f64) f64 {
    const baseline_f = baseline;
    const measured_f = measured;
    if (baseline_f == 0) return 0;
    return ((measured_f - baseline_f) / baseline_f) * 100.0;
}

test "llama speculative benchmark reports full acceptance on exact draft path" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try @import("llama_fixture.zig").makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try @import("llama_fixture.zig").writeFixtureFile(tmp.dir, "speculative.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "speculative.gguf");
    defer std.testing.allocator.free(path);

    const result = try runCase(std.testing.allocator, path, .{
        .backend = .cpu,
        .max_tokens = 6,
        .temperature = 0,
    }, .{
        .name = "exact",
        .prompt = "a",
        .max_tokens = 6,
        .draft_tokens = 3,
        .mismatch_mode = .exact,
    }, 1);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.acceptance_rate_avg, 0.0001);
}

test "llama speculative benchmark lowers acceptance when rejecting round tail" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try @import("llama_fixture.zig").makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try @import("llama_fixture.zig").writeFixtureFile(tmp.dir, "speculative-reject.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "speculative-reject.gguf");
    defer std.testing.allocator.free(path);

    const result = try runCase(std.testing.allocator, path, .{
        .backend = .cpu,
        .max_tokens = 6,
        .temperature = 0,
    }, .{
        .name = "reject-tail",
        .prompt = "a",
        .max_tokens = 6,
        .draft_tokens = 2,
        .mismatch_mode = .reject_last_token_every_round,
    }, 1);

    try std.testing.expect(result.acceptance_rate_avg < 1.0);
}
