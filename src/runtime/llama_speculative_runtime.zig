const std = @import("std");
const terminal = @import("../terminal.zig");
const llama_cpu = @import("../llama_cpu.zig");
const llama_loaded_runtime = @import("llama_loaded_runtime.zig");
const types = @import("types.zig");

pub fn generateWithDraftRuntime(
    allocator: std.mem.Allocator,
    target_runtime: *llama_loaded_runtime.LoadedRuntime,
    draft_runtime: *llama_loaded_runtime.LoadedRuntime,
    prompt: []const u8,
    options: types.GenerationOptions,
    startup_base: types.StartupBreakdown,
) !types.GenerationReport {
    try validateSpeculativeConfig(&target_runtime.model, &draft_runtime.model, options);

    var spinner = terminal.Spinner{};
    try spinner.start();
    errdefer spinner.stop();

    const session_init_begin = std.time.nanoTimestamp();
    var target_session = try llama_cpu.ReusableSession.init(
        allocator,
        &target_runtime.model,
        target_runtime.execution.backend,
        target_runtime.denseLookup(),
    );
    defer target_session.deinit(allocator);

    var draft_session = try llama_cpu.ReusableSession.init(
        allocator,
        &draft_runtime.model,
        draft_runtime.execution.backend,
        draft_runtime.denseLookup(),
    );
    defer draft_session.deinit(allocator);
    const session_init_ns = types.deltaNs(session_init_begin, std.time.nanoTimestamp());
    spinner.stop();

    const prompt_capacity = target_runtime.model.context_length;
    var prompt_tokens = try allocator.alloc(u32, prompt_capacity);
    defer allocator.free(prompt_tokens);

    const prompt_begin = std.time.nanoTimestamp();
    const prompt_token_count = try llama_cpu.encodePromptInto(allocator, &target_runtime.model, prompt, prompt_tokens);
    if (prompt_token_count == 0) return error.EmptyPrompt;
    if (prompt_token_count > draft_runtime.model.context_length) return error.ContextOverflow;

    try llama_cpu.reusableSessionReplay(&target_session, prompt_tokens[0..prompt_token_count]);
    try llama_cpu.reusableSessionReplay(&draft_session, prompt_tokens[0..prompt_token_count]);
    const prompt_end = std.time.nanoTimestamp();

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);

    const generated_tokens = try allocator.alloc(u32, options.max_tokens);
    defer allocator.free(generated_tokens);
    const proposed_tokens = try allocator.alloc(u32, options.speculative.draft_tokens);
    defer allocator.free(proposed_tokens);

    const startup_ns = setupNs(startup_base) + session_init_ns;
    var ttft_ns = startup_ns + types.deltaNs(prompt_begin, prompt_end);
    var first_decode_step_ns: u64 = 0;
    var rounds: usize = 0;
    var generated_count: usize = 0;
    var accepted_tokens: usize = 0;
    var rejected_tokens: usize = 0;
    var draft_decode_ns: u64 = 0;
    var verifier_decode_ns: u64 = 0;
    var resync_count: usize = 0;
    const eos_token = llama_cpu.eosTokenId(&target_runtime.model);

    const decode_begin = std.time.nanoTimestamp();
    outer: while (generated_count < options.max_tokens) {
        rounds += 1;
        const max_round_tokens = @min(options.speculative.draft_tokens, options.max_tokens - generated_count);
        const proposal_begin = std.time.nanoTimestamp();
        const proposal_count = try proposeTokens(&draft_session, eos_token, proposed_tokens[0..max_round_tokens]);
        draft_decode_ns += types.deltaNs(proposal_begin, std.time.nanoTimestamp());
        if (proposal_count == 0) break;

        var all_accepted = true;
        for (proposed_tokens[0..proposal_count]) |proposed| {
            const verify_begin = std.time.nanoTimestamp();
            const expected = llama_cpu.reusableSessionNextTokenArgmax(&target_session);
            if (expected == proposed) {
                if (eos_token != null and expected == eos_token.?) {
                    verifier_decode_ns += types.deltaNs(verify_begin, std.time.nanoTimestamp());
                    break :outer;
                }
                try appendGeneratedToken(
                    allocator,
                    &target_runtime.model,
                    &output,
                    generated_tokens,
                    generated_count,
                    expected,
                );
                generated_count += 1;
                const target_step_begin = std.time.nanoTimestamp();
                try llama_cpu.reusableSessionAcceptToken(&target_session, expected);
                if (first_decode_step_ns == 0) first_decode_step_ns = types.deltaNs(target_step_begin, std.time.nanoTimestamp());
                accepted_tokens += 1;
                verifier_decode_ns += types.deltaNs(verify_begin, std.time.nanoTimestamp());
                if (generated_count >= options.max_tokens) break :outer;
                continue;
            }

            all_accepted = false;
            rejected_tokens += 1;
            if (eos_token != null and expected == eos_token.?) {
                verifier_decode_ns += types.deltaNs(verify_begin, std.time.nanoTimestamp());
                break :outer;
            }
            try appendGeneratedToken(
                allocator,
                &target_runtime.model,
                &output,
                generated_tokens,
                generated_count,
                expected,
            );
            generated_count += 1;
            const target_step_begin = std.time.nanoTimestamp();
            try llama_cpu.reusableSessionAcceptToken(&target_session, expected);
            if (first_decode_step_ns == 0) first_decode_step_ns = types.deltaNs(target_step_begin, std.time.nanoTimestamp());
            verifier_decode_ns += types.deltaNs(verify_begin, std.time.nanoTimestamp());
            resync_count += 1;
            try replayDraftSession(&draft_session, prompt_tokens[0..prompt_token_count], generated_tokens[0..generated_count]);
            if (ttft_ns == startup_ns + types.deltaNs(prompt_begin, prompt_end)) {
                ttft_ns = startup_ns + types.deltaNs(prompt_begin, std.time.nanoTimestamp());
            }
            if (generated_count >= options.max_tokens) break :outer;
            continue :outer;
        }

        if (generated_count > 0 and ttft_ns == startup_ns + types.deltaNs(prompt_begin, prompt_end)) {
            ttft_ns = startup_ns + types.deltaNs(prompt_begin, std.time.nanoTimestamp());
        }
        if (!all_accepted) continue;
    }
    const decode_ns = types.deltaNs(decode_begin, std.time.nanoTimestamp());

    return .{
        .generated_text = try output.toOwnedSlice(allocator),
        .prompt_token_count = prompt_token_count,
        .reused_prompt_token_count = 0,
        .generated_token_count = generated_count,
        .startup_ns = startup_ns,
        .prompt_ns = types.deltaNs(prompt_begin, prompt_end),
        .ttft_ns = ttft_ns,
        .decode_ns = decode_ns,
        .seed = options.seed,
        .temperature = options.temperature,
        .backend = if (target_runtime.execution.backend == null) .cpu else .metal,
        .startup_breakdown = .{
            .model_load_ns = startup_base.model_load_ns,
            .tensor_prepare_ns = startup_base.tensor_prepare_ns,
            .backend_init_ns = startup_base.backend_init_ns,
            .metal_prewarm_ns = startup_base.metal_prewarm_ns,
            .session_init_ns = session_init_ns,
            .first_decode_step_ns = first_decode_step_ns,
        },
        .speculative = .{
            .draft_tokens = options.speculative.draft_tokens,
            .rounds = rounds,
            .accepted_tokens = accepted_tokens,
            .rejected_tokens = rejected_tokens,
            .draft_decode_ns = draft_decode_ns,
            .verifier_decode_ns = verifier_decode_ns,
            .resync_count = resync_count,
        },
        .metal_profile_summary = null,
    };
}

fn validateSpeculativeConfig(
    target_model: *const llama_cpu.Model,
    draft_model: *const llama_cpu.Model,
    options: types.GenerationOptions,
) !void {
    if (!options.speculative.enabled()) return error.InvalidSpeculativeConfig;
    if (options.speculative.draft_tokens == 0) return error.InvalidSpeculativeConfig;
    if (options.temperature != 0 or options.repeat_penalty != 1.0 or options.top_k != 0 or options.top_p != 1.0 or options.min_p != 0.0) {
        return error.UnsupportedSpeculativeSampling;
    }
    if (!llama_cpu.tokenizersMatch(target_model, draft_model)) return error.SpeculativeDraftTokenizerMismatch;
}

fn proposeTokens(session: *llama_cpu.ReusableSession, eos_token: ?u32, proposed_tokens: []u32) !usize {
    var proposal_count: usize = 0;
    while (proposal_count < proposed_tokens.len) : (proposal_count += 1) {
        const token = llama_cpu.reusableSessionNextTokenArgmax(session);
        proposed_tokens[proposal_count] = token;
        if (eos_token != null and token == eos_token.?) {
            proposal_count += 1;
            break;
        }
        try llama_cpu.reusableSessionAcceptToken(session, token);
    }
    return proposal_count;
}

fn appendGeneratedToken(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    output: *std.ArrayList(u8),
    generated_tokens: []u32,
    generated_count: usize,
    token: u32,
) !void {
    generated_tokens[generated_count] = token;
    try llama_cpu.appendDecodedToken(model, output, allocator, token);
}

fn replayDraftSession(
    draft_session: *llama_cpu.ReusableSession,
    prompt_tokens: []const u32,
    generated_tokens: []const u32,
) !void {
    try llama_cpu.reusableSessionReplay(draft_session, prompt_tokens);
    for (generated_tokens) |token| {
        try llama_cpu.reusableSessionAcceptToken(draft_session, token);
    }
}

fn combinedStartupBreakdown(
    target_runtime: *const llama_loaded_runtime.LoadedRuntime,
    draft_runtime: *const llama_loaded_runtime.LoadedRuntime,
) types.StartupBreakdown {
    return .{
        .model_load_ns = target_runtime.model_load_ns + draft_runtime.model_load_ns,
        .tensor_prepare_ns = target_runtime.execution.startup_breakdown.tensor_prepare_ns + draft_runtime.execution.startup_breakdown.tensor_prepare_ns,
        .backend_init_ns = target_runtime.execution.startup_breakdown.backend_init_ns + draft_runtime.execution.startup_breakdown.backend_init_ns,
        .metal_prewarm_ns = target_runtime.execution.startup_breakdown.metal_prewarm_ns + draft_runtime.execution.startup_breakdown.metal_prewarm_ns,
    };
}

pub fn coldStartupBreakdown(
    target_runtime: *const llama_loaded_runtime.LoadedRuntime,
    draft_runtime: *const llama_loaded_runtime.LoadedRuntime,
) types.StartupBreakdown {
    return combinedStartupBreakdown(target_runtime, draft_runtime);
}

pub fn warmStartupBreakdown(
    target_runtime: *const llama_loaded_runtime.LoadedRuntime,
    draft_runtime: *const llama_loaded_runtime.LoadedRuntime,
) types.StartupBreakdown {
    var breakdown = combinedStartupBreakdown(target_runtime, draft_runtime);
    breakdown.model_load_ns = 0;
    return breakdown;
}

fn setupNs(breakdown: types.StartupBreakdown) u64 {
    return breakdown.model_load_ns +
        breakdown.tensor_prepare_ns +
        breakdown.backend_init_ns +
        breakdown.metal_prewarm_ns;
}

test "speculative runtime matches baseline text with identical draft model" {
    const llama_runtime = @import("llama_runtime.zig");
    const llama_fixture = @import("llama_fixture.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "target.gguf", fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "draft.gguf", fixture);

    const target_path = try tmp.dir.realpathAlloc(std.testing.allocator, "target.gguf");
    defer std.testing.allocator.free(target_path);
    const draft_path = try tmp.dir.realpathAlloc(std.testing.allocator, "draft.gguf");
    defer std.testing.allocator.free(draft_path);

    var baseline = try llama_runtime.generate(std.testing.allocator, target_path, "a", .{
        .max_tokens = 4,
        .temperature = 0,
        .backend = .cpu,
    });
    defer baseline.deinit(std.testing.allocator);

    var speculative = try llama_runtime.generate(std.testing.allocator, target_path, "a", .{
        .max_tokens = 4,
        .temperature = 0,
        .backend = .cpu,
        .speculative = .{
            .draft_model_path = draft_path,
            .draft_tokens = 2,
        },
    });
    defer speculative.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings(baseline.generated_text, speculative.generated_text);
    try std.testing.expectEqual(baseline.generated_token_count, speculative.generated_token_count);
    try std.testing.expect(speculative.speculative != null);
    try std.testing.expectEqual(@as(usize, speculative.generated_token_count), speculative.speculative.?.accepted_tokens);
    try std.testing.expectEqual(@as(usize, 0), speculative.speculative.?.rejected_tokens);
}
