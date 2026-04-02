const std = @import("std");
const fixture = @import("src/runtime/llama_fixture.zig");
const runtime = @import("src/runtime/llama_runtime.zig");
pub fn main() !void {
    var tmpdir = std.testing.tmpDir(.{});
    defer tmpdir.cleanup();
    const bytes = try fixture.makeLlamaModelFixture(std.heap.page_allocator);
    defer std.heap.page_allocator.free(bytes);
    try fixture.writeFixtureFile(tmpdir.dir, "llama.gguf", bytes);
    const path = try tmpdir.dir.realpathAlloc(std.heap.page_allocator, "llama.gguf");
    defer std.heap.page_allocator.free(path);
    var report = try runtime.generate(std.heap.page_allocator, path, "a", .{ .max_tokens = 3, .seed = 0, .temperature = 0, .backend = .cpu });
    defer report.deinit(std.heap.page_allocator);
    std.debug.print("backend={s} prompt={} generated={} text=<{s}>\n", .{ report.backend.label(), report.prompt_token_count, report.generated_token_count, report.generated_text });
}
