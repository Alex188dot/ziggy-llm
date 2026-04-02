const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const backend_api = @import("backend.zig");
const tiny_model = @import("tiny_model.zig");

const shader_source = @embedFile("metal/matvec.metal");
const err_buf_len: usize = 256;
const build_enabled_value = build_options.enable_metal and builtin.os.tag == .macos and builtin.cpu.arch == .aarch64;
const c = if (build_enabled_value) @cImport({
    @cInclude("bridge.h");
}) else struct {};

const MatrixKey = struct {
    address: usize,
    len: usize,
};

const MetalBuffer = if (build_enabled_value) struct {
    raw: *c.ZiggyMetalBuffer,
    byte_len: usize,
} else struct {};

const State = if (build_enabled_value) struct {
    allocator: std.mem.Allocator,
    context: *c.ZiggyMetalContext,
    input_buffer: ?MetalBuffer = null,
    output_buffer: ?MetalBuffer = null,
    matrix_buffers: std.AutoHashMap(MatrixKey, MetalBuffer),

    fn init(allocator: std.mem.Allocator) !*State {
        var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
        var raw_context: ?*c.ZiggyMetalContext = null;
        const status = c.ziggy_metal_create_context(
            shader_source.ptr,
            shader_source.len,
            &raw_context,
            null,
            &error_buf,
            error_buf.len,
        );
        try mapStatus(status);

        const state = try allocator.create(State);
        state.* = .{
            .allocator = allocator,
            .context = raw_context.?,
            .matrix_buffers = std.AutoHashMap(MatrixKey, MetalBuffer).init(allocator),
        };
        return state;
    }

    fn deinit(self: *State) void {
        if (self.input_buffer) |buffer| c.ziggy_metal_destroy_buffer(buffer.raw);
        if (self.output_buffer) |buffer| c.ziggy_metal_destroy_buffer(buffer.raw);

        var iterator = self.matrix_buffers.valueIterator();
        while (iterator.next()) |buffer| c.ziggy_metal_destroy_buffer(buffer.raw);
        self.matrix_buffers.deinit();
        c.ziggy_metal_destroy_context(self.context);
        self.allocator.destroy(self);
    }

    fn prewarm(self: *State, model: *const tiny_model.Model) !void {
        _ = try self.matrixBuffer(model.attn_q);
        _ = try self.matrixBuffer(model.attn_k);
        _ = try self.matrixBuffer(model.attn_v);
        _ = try self.matrixBuffer(model.attn_out);
        _ = try self.matrixBuffer(model.ffn_up);
        _ = try self.matrixBuffer(model.ffn_down);
        _ = try self.matrixBuffer(model.output);
    }

    fn matVec(
        self: *State,
        out: []f32,
        matrix: []const f32,
        input: []const f32,
        rows: usize,
        cols: usize,
    ) !void {
        if (rows == 0 or cols == 0) return error.InvalidTensorMetadata;
        if (matrix.len < rows * cols or input.len < cols or out.len < rows) return error.InvalidTensorMetadata;

        const matrix_buffer = try self.matrixBuffer(matrix[0 .. rows * cols]);
        const input_buffer = try self.ensureScratchBuffer(&self.input_buffer, cols);
        const output_buffer = try self.ensureScratchBuffer(&self.output_buffer, rows);

        try writeBuffer(input_buffer.raw, input[0..cols]);
        try mapStatus(c.ziggy_metal_run_matvec_f32(
            self.context,
            matrix_buffer.raw,
            input_buffer.raw,
            output_buffer.raw,
            @intCast(rows),
            @intCast(cols),
            null,
            0,
        ));
        try readBuffer(output_buffer.raw, out[0..rows]);
    }

    fn matrixBuffer(self: *State, matrix: []const f32) !MetalBuffer {
        const key = MatrixKey{
            .address = @intFromPtr(matrix.ptr),
            .len = matrix.len,
        };
        if (self.matrix_buffers.get(key)) |buffer| return buffer;

        const raw = try createBuffer(self.context, matrix);
        const buffer = MetalBuffer{
            .raw = raw,
            .byte_len = matrix.len * @sizeOf(f32),
        };
        try self.matrix_buffers.put(key, buffer);
        return buffer;
    }

    fn ensureScratchBuffer(self: *State, slot: *?MetalBuffer, elements: usize) !MetalBuffer {
        const required_len = elements * @sizeOf(f32);
        if (slot.*) |existing| {
            if (existing.byte_len >= required_len) return existing;
            c.ziggy_metal_destroy_buffer(existing.raw);
            slot.* = null;
        }

        const raw = try createEmptyBuffer(self.context, required_len);
        const buffer = MetalBuffer{
            .raw = raw,
            .byte_len = required_len,
        };
        slot.* = buffer;
        return buffer;
    }
} else struct {};

pub fn buildEnabled() bool {
    return build_enabled_value;
}

pub fn canInitialize(allocator: std.mem.Allocator) !bool {
    if (!build_enabled_value) return false;
    const metal = create(allocator) catch |err| switch (err) {
        error.MetalUnavailable,
        error.MetalInitializationFailed,
        error.MetalCompilationFailed,
        => return false,
        else => return err,
    };
    metal.deinit(allocator);
    return true;
}

pub fn create(allocator: std.mem.Allocator) !backend_api.MatVecBackend {
    if (!build_enabled_value) return error.MetalDisabled;

    const state = try State.init(allocator);
    return .{
        .ctx = @ptrCast(state),
        .label = .metal,
        .mat_vec_fn = metalMatVec,
        .deinit_fn = metalDeinit,
    };
}

pub fn prewarm(backend: backend_api.MatVecBackend, model: *const tiny_model.Model) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    try state.prewarm(model);
}

pub fn cacheMatrix(backend: backend_api.MatVecBackend, matrix: []const f32) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    _ = try state.matrixBuffer(matrix);
}

fn metalMatVec(
    ctx: ?*anyopaque,
    out: []f32,
    matrix: []const f32,
    input: []const f32,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(ctx);
    try state.matVec(out, matrix, input, rows, cols);
}

fn metalDeinit(ctx: ?*anyopaque, _: std.mem.Allocator) void {
    if (!build_enabled_value or ctx == null) return;
    const state = stateFromCtx(ctx);
    state.deinit();
}

fn stateFromCtx(ctx: ?*anyopaque) *State {
    return @ptrCast(@alignCast(ctx.?));
}

fn createBuffer(context: *c.ZiggyMetalContext, values: []const f32) !*c.ZiggyMetalBuffer {
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    var raw: ?*c.ZiggyMetalBuffer = null;
    const status = c.ziggy_metal_create_buffer(
        context,
        values.ptr,
        values.len * @sizeOf(f32),
        &raw,
        &error_buf,
        error_buf.len,
    );
    try mapStatus(status);
    return raw.?;
}

fn createEmptyBuffer(context: *c.ZiggyMetalContext, byte_len: usize) !*c.ZiggyMetalBuffer {
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    var raw: ?*c.ZiggyMetalBuffer = null;
    const status = c.ziggy_metal_create_empty_buffer(
        context,
        byte_len,
        &raw,
        &error_buf,
        error_buf.len,
    );
    try mapStatus(status);
    return raw.?;
}

fn writeBuffer(buffer: *c.ZiggyMetalBuffer, values: []const f32) !void {
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    const status = c.ziggy_metal_write_buffer(
        buffer,
        values.ptr,
        values.len * @sizeOf(f32),
        0,
        &error_buf,
        error_buf.len,
    );
    try mapStatus(status);
}

fn readBuffer(buffer: *c.ZiggyMetalBuffer, out: []f32) !void {
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    const status = c.ziggy_metal_read_buffer(
        buffer,
        out.ptr,
        out.len * @sizeOf(f32),
        0,
        &error_buf,
        error_buf.len,
    );
    try mapStatus(status);
}

fn mapStatus(status: c_int) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    switch (status) {
        c.ZIGGY_METAL_OK => return,
        c.ZIGGY_METAL_UNAVAILABLE => return error.MetalUnavailable,
        c.ZIGGY_METAL_INITIALIZATION_FAILED => return error.MetalInitializationFailed,
        c.ZIGGY_METAL_COMPILATION_FAILED => return error.MetalCompilationFailed,
        c.ZIGGY_METAL_BUFFER_FAILED => return error.MetalBufferError,
        c.ZIGGY_METAL_EXECUTION_FAILED => return error.MetalExecutionFailed,
        else => return error.MetalInitializationFailed,
    }
}
