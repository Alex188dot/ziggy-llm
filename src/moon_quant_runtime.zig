const std = @import("std");
const llama_cpu = @import("model/loader.zig");
const calibration = @import("moon_quant_calibration.zig");
const llama_fixture = @import("runtime/llama_fixture.zig");

pub fn calibratePrompt(
    allocator: std.mem.Allocator,
    model: *const llama_cpu.Model,
    prompt: []const u8,
) !calibration.Plan {
    var calibrator = calibration.Calibrator.init(allocator);
    defer calibrator.deinit();

    const token_capacity = @min(model.context_length, prompt.len * 4 + 8);
    var session = try llama_cpu.initCalibrationSession(allocator, model, token_capacity, &calibrator);
    defer session.deinit(allocator);

    const prompt_token_count = try llama_cpu.encodePromptInto(allocator, model, prompt, session.tokenBuffer());
    try session.runPrompt(session.tokenBuffer()[0..prompt_token_count]);
    return calibrator.buildPlan(allocator);
}

test "prompt calibration returns a non-uniform per-layer plan" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaModelFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "llama-calibration.gguf", fixture);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "llama-calibration.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    var plan = try calibratePrompt(std.testing.allocator, &model, "a b c");
    defer plan.deinit(std.testing.allocator);

    try std.testing.expect(plan.entries.len > 0);
    var has_q4 = false;
    var has_q5_or_better = false;
    for (plan.entries) |entry| {
        if (entry.target_format == .q4_k_m) has_q4 = true;
        if (entry.target_format == .q5_k_m or entry.target_format == .q6_k or entry.target_format == .q8_0) {
            has_q5_or_better = true;
        }
    }
    try std.testing.expect(has_q4);
    try std.testing.expect(has_q5_or_better);
}
