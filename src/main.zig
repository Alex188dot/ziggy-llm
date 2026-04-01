const std = @import("std");
const cli = @import("cli.zig");
const commands = @import("commands.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const config = try cli.parseArgs(args);
    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdout_buffer);
    try commands.dispatch(&stdout.interface, config);
    try stdout.interface.flush();
}

test {
    std.testing.refAllDecls(@import("cli.zig"));
    std.testing.refAllDecls(@import("commands.zig"));
    std.testing.refAllDecls(@import("runtime.zig"));
    std.testing.refAllDecls(@import("gguf.zig"));
    std.testing.refAllDecls(@import("server.zig"));
}
