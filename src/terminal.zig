const std = @import("std");
const builtin = @import("builtin");

pub const Spinner = struct {
    active: bool = false,
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    thread: ?std.Thread = null,

    pub fn start(self: *Spinner) !void {
        if (!shouldRender()) return;

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

pub const LoadingBar = struct {
    enabled: bool,
    phase_index: std.atomic.Value(u8),
    phase_count: u8,
    phase_labels: [5][]const u8,

    pub fn init(phase_labels: [5][]const u8) LoadingBar {
        return .{
            .enabled = shouldRender(),
            .phase_index = std.atomic.Value(u8).init(0),
            .phase_count = 5,
            .phase_labels = phase_labels,
        };
    }

    pub fn setPhase(self: *LoadingBar, index: u8) void {
        if (!self.enabled) return;
        self.phase_index.store(index, .seq_cst);
        self.draw();
    }

    fn draw(self: *const LoadingBar) void {
        if (!self.enabled) return;
        const idx = self.phase_index.load(.seq_cst);
        if (idx >= self.phase_count) return;

        var buf: [128]u8 = undefined;
        const label = self.phase_labels[idx];
        const width: usize = 30;
        const filled = (width * idx) / self.phase_count;

        var bar_buf: [32]u8 = undefined;
        var bar_idx: usize = 0;
        var i: usize = 0;
        while (i < filled) : (i += 1) {
            if (bar_idx < bar_buf.len - 1) {
                bar_buf[bar_idx] = '=';
                bar_idx += 1;
            }
        }
        if (idx < self.phase_count and filled < width) {
            if (bar_idx < bar_buf.len - 1) {
                bar_buf[bar_idx] = '>';
                bar_idx += 1;
            }
        }
        i = filled + 1;
        while (i < width) : (i += 1) {
            if (bar_idx < bar_buf.len - 1) {
                bar_buf[bar_idx] = ' ';
                bar_idx += 1;
            }
        }
        if (bar_idx < bar_buf.len) bar_buf[bar_idx] = 0;

        const pct = @as(u32, @intCast((idx * 100) / self.phase_count));
        const line = std.fmt.bufPrint(
            &buf,
            "\r  [{s}] {d: >3}%  {s}   ",
            .{ bar_buf[0..bar_idx :0], pct, label },
        ) catch return;

        const stderr = std.fs.File.stderr();
        stderr.writeAll(line) catch return;
    }

    pub fn clear(self: *LoadingBar) void {
        if (!self.enabled) return;
        clearProgressLineImpl();
        const stderr = std.fs.File.stderr();
        stderr.writeAll("\n") catch return;
    }
};

fn shouldRender() bool {
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

fn clearProgressLineImpl() void {
    const stderr = std.fs.File.stderr();
    stderr.writeAll("\r                                                                                \r") catch return;
}

fn clearLine() void {
    clearProgressLineImpl();
}

pub fn drawBenchRunProgress(current_run: usize, total_runs: usize) void {
    if (!shouldRender()) return;

    const bar_width: usize = 30;
    const filled = if (total_runs > 0) (bar_width * current_run) / total_runs else @as(usize, 0);
    const pct = if (total_runs > 0) @as(u32, @intCast((current_run * 100) / total_runs)) else @as(u32, 0);

    var bar: [32]u8 = undefined;
    var bar_idx: usize = 0;
    var i: usize = 0;
    while (i < filled) : (i += 1) {
        if (bar_idx < bar.len - 1) {
            bar[bar_idx] = '=';
            bar_idx += 1;
        }
    }
    if (filled < bar_width and current_run > 0) {
        if (bar_idx < bar.len - 1) {
            bar[bar_idx] = '>';
            bar_idx += 1;
        }
    }
    i = filled + 1;
    while (i < bar_width) : (i += 1) {
        if (bar_idx < bar.len - 1) {
            bar[bar_idx] = ' ';
            bar_idx += 1;
        }
    }
    if (bar_idx < bar.len) bar[bar_idx] = 0;

    var buf: [128]u8 = undefined;
    const line = std.fmt.bufPrint(
        &buf,
        "\r  [{s}] {d: >3}%  run {d}/{d}   ",
        .{ bar[0..bar_idx :0], pct, current_run, total_runs },
    ) catch return;

    const stderr = std.fs.File.stderr();
    stderr.writeAll(line) catch return;
}

pub fn clearProgressLine() void {
    if (!shouldRender()) return;
    clearProgressLineImpl();
    const stderr = std.fs.File.stderr();
    stderr.writeAll("\n") catch return;
}
