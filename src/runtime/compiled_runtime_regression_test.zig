const std = @import("std");
const llama_fixture = @import("llama_fixture.zig");
const llama_runtime = @import("llama_runtime.zig");
const ziggy_format = @import("../ziggy_format.zig");

test "compiled ziggy path reuses gguf runtime source when sibling exists" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const fixture = try llama_fixture.makeLlamaQ4KFixture(std.testing.allocator);
    defer std.testing.allocator.free(fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "fixture.gguf", fixture);

    const gguf_path = try tmp.dir.realpathAlloc(std.testing.allocator, "fixture.gguf");
    defer std.testing.allocator.free(gguf_path);
    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const ziggy_path = try std.fmt.allocPrint(std.testing.allocator, "{s}/fixture.ziggy", .{root_path});
    defer std.testing.allocator.free(ziggy_path);

    try ziggy_format.compileFromGGUF(std.testing.allocator, gguf_path, ziggy_path, .{});

    var model = try llama_runtime.loadModelWithDispatch(std.testing.allocator, ziggy_path);
    defer model.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("GGUF", model.bytes[0..4]);
}
