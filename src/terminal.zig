const std = @import("std");
const builtin = @import("builtin");

pub const Spinner = struct {
    active: bool = false,
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    thread: ?std.Thread = null,

    pub fn start(self: *Spinner) !void {
        if (!shouldRenderSpinner()) return;

        self.running.store(true, .seq_cst);
        self.thread = try std.Thread.spawn(.{}, renderLoop, .{self});
        self.active = true;
    }

    pub fn stop(self: *Spinner) void {
        if (!self.active) return;

        self.running.store(false, .seq_cst);
        if (self.thread) |thread| thread.join();
        self.thread = null;
        self.active = false;
        clearLine();
    }
};

fn shouldRenderSpinner() bool {
    if (builtin.os.tag == .windows) return false;
    return std.posix.isatty(std.fs.File.stderr().handle);
}

fn renderLoop(spinner: *Spinner) void {
    const frames = [_][]const u8{ "|", "/", "-", "\\" };
    var frame_index: usize = 0;

    while (spinner.running.load(.seq_cst)) {
        drawFrame(frames[frame_index]);
        frame_index = (frame_index + 1) % frames.len;
        std.Thread.sleep(100 * std.time.ns_per_ms);
    }
}

fn drawFrame(frame: []const u8) void {
    var buffer: [64]u8 = undefined;
    const line = std.fmt.bufPrint(&buffer, "\rloading model {s}", .{frame}) catch return;
    const stderr = std.fs.File.stderr();
    stderr.writeAll(line) catch return;
}

fn clearLine() void {
    const stderr = std.fs.File.stderr();
    stderr.writeAll("\r                 \r") catch return;
}
