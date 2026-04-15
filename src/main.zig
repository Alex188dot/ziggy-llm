const std = @import("std");
const cli = @import("cli.zig");
const commands = @import("commands.zig");
const update = @import("update.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const config = try cli.parseArgs(args);

    if (config.command != .help and config.command != .version and config.command != .update) {
        // update.checkForUpdates(allocator);
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdout_buffer);
    try commands.dispatch(&stdout.interface, allocator, config);
    try stdout.interface.flush();
}

test {
    std.testing.refAllDecls(@import("cli.zig"));
    std.testing.refAllDecls(@import("commands.zig"));
    std.testing.refAllDecls(@import("runtime.zig"));
    std.testing.refAllDecls(@import("runtime/metal_backend_test.zig"));
    std.testing.refAllDecls(@import("llama_cpu.zig"));
    std.testing.refAllDecls(@import("gguf.zig"));
    std.testing.refAllDecls(@import("moon_quant.zig"));
    std.testing.refAllDecls(@import("moon_quant_calibration.zig"));
    std.testing.refAllDecls(@import("moon_quant_runtime.zig"));
    std.testing.refAllDecls(@import("runtime/moon_quant_bench.zig"));
    std.testing.refAllDecls(@import("server.zig"));
    std.testing.refAllDecls(@import("terminal.zig"));
}
