const std = @import("std");
const backend = @import("backend.zig");

pub fn create() backend.MatVecBackend {
    return .{
        .ctx = null,
        .label = .cpu,
        .mat_vec_fn = cpuMatVec,
        .deinit_fn = noopDeinit,
    };
}

fn cpuMatVec(
    _: ?*anyopaque,
    out: []f32,
    matrix: []const f32,
    input: []const f32,
    rows: usize,
    cols: usize,
) !void {
    std.debug.assert(out.len >= rows);
    std.debug.assert(input.len >= cols);
    std.debug.assert(matrix.len >= rows * cols);

    for (0..rows) |row| {
        var sum: f32 = 0;
        for (0..cols) |col| {
            sum += matrix[row + col * rows] * input[col];
        }
        out[row] = sum;
    }
}

fn noopDeinit(_: ?*anyopaque, _: std.mem.Allocator) void {}
