const std = @import("std");
const types = @import("types.zig");

pub const MatVecBackend = struct {
    ctx: ?*anyopaque,
    label: types.BackendUsed,
    mat_vec_fn: *const fn (?*anyopaque, []f32, []const f32, []const f32, usize, usize) anyerror!void,
    deinit_fn: *const fn (?*anyopaque, std.mem.Allocator) void,

    pub fn matVec(
        self: MatVecBackend,
        out: []f32,
        matrix: []const f32,
        input: []const f32,
        rows: usize,
        cols: usize,
    ) !void {
        try self.mat_vec_fn(self.ctx, out, matrix, input, rows, cols);
    }

    pub fn deinit(self: MatVecBackend, allocator: std.mem.Allocator) void {
        self.deinit_fn(self.ctx, allocator);
    }
};
