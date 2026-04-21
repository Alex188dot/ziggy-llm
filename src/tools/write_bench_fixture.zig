const std = @import("std");
const ziggy = @import("ziggy_lib");
const llama_cpu = ziggy.llama_cpu;
const llama_fixture = ziggy.llama_fixture;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) std.debug.panic("memory leak detected", .{});
    }
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var format: llama_cpu.TensorType = .q4_k;
    var output_path: ?[]const u8 = null;

    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--format")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            format = parseFormat(args[index]) orelse return error.InvalidFormat;
            continue;
        }
        if (std.mem.eql(u8, arg, "--output")) {
            index += 1;
            if (index >= args.len) return error.MissingFlagValue;
            output_path = args[index];
            continue;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printHelp();
            return;
        }
        return error.UnknownFlag;
    }

    const path = output_path orelse {
        try printHelp();
        return error.MissingOutputPath;
    };

    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(allocator, format);
    defer allocator.free(fixture);

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writeAll(fixture);

    var stdout_buffer: [256]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdout_buffer);
    try stdout.interface.print("wrote {s} benchmark fixture to {s}\n", .{ labelFor(format), path });
    try stdout.interface.flush();
}

fn parseFormat(value: []const u8) ?llama_cpu.TensorType {
    if (std.mem.eql(u8, value, "q4_k")) return .q4_k;
    if (std.mem.eql(u8, value, "q6_k")) return .q6_k;
    return null;
}

fn labelFor(format: llama_cpu.TensorType) []const u8 {
    return switch (format) {
        .q4_k => "Q4_K",
        .q6_k => "Q6_K",
        else => unreachable,
    };
}

fn printHelp() !void {
    var stdout_buffer: [512]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdout_buffer);
    try stdout.interface.print(
        \\Usage:
        \\  zig build bench-fixture -- --format <q4_k|q6_k> --output <path>
        \\
        \\Creates a synthetic llama GGUF benchmark fixture with TinyLlama-like decode shapes.
        \\
    , .{});
    try stdout.interface.flush();
}
