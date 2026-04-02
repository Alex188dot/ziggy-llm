const std = @import("std");
const tiny_fixture = @import("runtime/tiny_fixture.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var stderr_buffer: [256]u8 = undefined;
    var stderr = std.fs.File.stderr().writer(&stderr_buffer);

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var loop_output = false;
    var output_path: ?[]const u8 = null;

    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--loop")) {
            loop_output = true;
            continue;
        }
        if (output_path == null) {
            output_path = arg;
            continue;
        }
        try stderr.interface.print("usage: make-tiny-fixture [--loop] <output.gguf>\n", .{});
        try stderr.interface.flush();
        return error.InvalidArguments;
    }

    if (output_path == null) {
        try stderr.interface.print("usage: make-tiny-fixture [--loop] <output.gguf>\n", .{});
        try stderr.interface.flush();
        return error.InvalidArguments;
    }

    const fixture = try tiny_fixture.makeTinyModelFixture(allocator, .{
        .ambiguous_a = true,
        .loop_output = loop_output,
        .context_length = if (loop_output) 1024 else 16,
    });
    defer allocator.free(fixture);

    const file = try std.fs.cwd().createFile(output_path.?, .{ .truncate = true });
    defer file.close();
    try file.writeAll(fixture);
}
