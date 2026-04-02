const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const backend_api = @import("backend.zig");

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

pub const BufferHandle = if (build_enabled_value) struct {
    raw: *c.ZiggyMetalBuffer,
    byte_len: usize,
} else struct {};

const State = if (build_enabled_value) struct {
    allocator: std.mem.Allocator,
    context: *c.ZiggyMetalContext,
    input_buffer: ?BufferHandle = null,
    output_buffer: ?BufferHandle = null,
    matrix_buffers: std.AutoHashMap(MatrixKey, BufferHandle),
    raw_buffers: std.AutoHashMap(MatrixKey, BufferHandle),

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
            .matrix_buffers = std.AutoHashMap(MatrixKey, BufferHandle).init(allocator),
            .raw_buffers = std.AutoHashMap(MatrixKey, BufferHandle).init(allocator),
        };
        return state;
    }

    fn deinit(self: *State) void {
        if (self.input_buffer) |buffer| c.ziggy_metal_destroy_buffer(buffer.raw);
        if (self.output_buffer) |buffer| c.ziggy_metal_destroy_buffer(buffer.raw);

        var iterator = self.matrix_buffers.valueIterator();
        while (iterator.next()) |buffer| c.ziggy_metal_destroy_buffer(buffer.raw);
        self.matrix_buffers.deinit();
        iterator = self.raw_buffers.valueIterator();
        while (iterator.next()) |buffer| c.ziggy_metal_destroy_buffer(buffer.raw);
        self.raw_buffers.deinit();
        c.ziggy_metal_destroy_context(self.context);
        self.allocator.destroy(self);
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

    fn matrixBuffer(self: *State, matrix: []const f32) !BufferHandle {
        const key = MatrixKey{
            .address = @intFromPtr(matrix.ptr),
            .len = matrix.len,
        };
        if (self.matrix_buffers.get(key)) |buffer| return buffer;

        const raw = try createBuffer(self.context, matrix);
        const buffer = BufferHandle{
            .raw = raw,
            .byte_len = matrix.len * @sizeOf(f32),
        };
        try self.matrix_buffers.put(key, buffer);
        return buffer;
    }

    fn ensureScratchBuffer(self: *State, slot: *?BufferHandle, elements: usize) !BufferHandle {
        const required_len = elements * @sizeOf(f32);
        if (slot.*) |existing| {
            if (existing.byte_len >= required_len) return existing;
            c.ziggy_metal_destroy_buffer(existing.raw);
            slot.* = null;
        }

        const raw = try createEmptyBuffer(self.context, required_len);
        const buffer = BufferHandle{
            .raw = raw,
            .byte_len = required_len,
        };
        slot.* = buffer;
        return buffer;
    }

    fn rawBuffer(self: *State, bytes: []const u8) !BufferHandle {
        const key = MatrixKey{
            .address = @intFromPtr(bytes.ptr),
            .len = bytes.len,
        };
        if (self.raw_buffers.get(key)) |buffer| return buffer;

        const raw = try createRawBuffer(self.context, bytes);
        const buffer = BufferHandle{
            .raw = raw,
            .byte_len = bytes.len,
        };
        try self.raw_buffers.put(key, buffer);
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

pub fn cacheMatrix(backend: backend_api.MatVecBackend, matrix: []const f32) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    _ = try state.matrixBuffer(matrix);
}

pub fn cacheRawMatrix(backend: backend_api.MatVecBackend, bytes: []const u8) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    _ = try state.rawBuffer(bytes);
}

pub fn createScratchBuffer(backend: backend_api.MatVecBackend, elements: usize) !BufferHandle {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    const raw = try createEmptyBuffer(state.context, elements * @sizeOf(f32));
    return .{
        .raw = raw,
        .byte_len = elements * @sizeOf(f32),
    };
}

pub fn destroyBuffer(buffer: BufferHandle) void {
    if (!build_enabled_value) return;
    c.ziggy_metal_destroy_buffer(buffer.raw);
}

pub fn writeBufferF32(buffer: BufferHandle, values: []const f32) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (values.len * @sizeOf(f32) > buffer.byte_len) return error.MetalBufferError;
    try writeBuffer(buffer.raw, values);
}

pub fn readBufferF32(buffer: BufferHandle, out: []f32) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (out.len * @sizeOf(f32) > buffer.byte_len) return error.MetalBufferError;
    try readBuffer(buffer.raw, out);
}

pub fn beginSequence(backend: backend_api.MatVecBackend) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    try mapStatus(c.ziggy_metal_begin_sequence(
        state.context,
        null,
        0,
    ));
}

pub fn commitSequence(backend: backend_api.MatVecBackend) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    try mapStatus(c.ziggy_metal_commit_sequence(
        state.context,
        null,
        0,
    ));
}

pub fn runMatVecToBuffer(
    backend: backend_api.MatVecBackend,
    matrix: []const f32,
    input: BufferHandle,
    output: BufferHandle,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < cols * @sizeOf(f32) or output.byte_len < rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.matrixBuffer(matrix[0 .. rows * cols]);
    try mapStatus(c.ziggy_metal_run_matvec_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        null,
        0,
    ));
}

pub fn runMatVecQ4KToBuffer(
    backend: backend_api.MatVecBackend,
    matrix_bytes: []const u8,
    input: BufferHandle,
    output: BufferHandle,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < cols * @sizeOf(f32) or output.byte_len < rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.rawBuffer(matrix_bytes);
    try mapStatus(c.ziggy_metal_run_matvec_q4k_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        null,
        0,
    ));
}

pub fn copyBufferRegion(
    backend: backend_api.MatVecBackend,
    src: BufferHandle,
    src_offset_bytes: usize,
    dst: BufferHandle,
    dst_offset_bytes: usize,
    length_bytes: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    try mapStatus(c.ziggy_metal_copy_buffer_region(
        state.context,
        src.raw,
        src_offset_bytes,
        dst.raw,
        dst_offset_bytes,
        length_bytes,
        null,
        0,
    ));
}

pub fn applyRoPE(
    backend: backend_api.MatVecBackend,
    vector: BufferHandle,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    freq_base: f32,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    try mapStatus(c.ziggy_metal_apply_rope_f32(
        state.context,
        vector.raw,
        @intCast(head_count),
        @intCast(head_dim),
        @intCast(rope_dim),
        @intCast(position),
        freq_base,
        null,
        0,
    ));
}

pub fn attentionFused(
    backend: backend_api.MatVecBackend,
    q: BufferHandle,
    k_cache: BufferHandle,
    v_cache: BufferHandle,
    output: BufferHandle,
    head_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    kv_dim: usize,
    context_length: usize,
    position: usize,
    layer_base: usize,
    scale: f32,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    try mapStatus(c.ziggy_metal_attention_fused_f32(
        state.context,
        q.raw,
        k_cache.raw,
        v_cache.raw,
        output.raw,
        @intCast(head_count),
        @intCast(head_count_kv),
        @intCast(head_dim),
        @intCast(kv_dim),
        @intCast(context_length),
        @intCast(position),
        @intCast(layer_base),
        scale,
        null,
        0,
    ));
}

pub fn siluMul(
    backend: backend_api.MatVecBackend,
    gate: BufferHandle,
    up: BufferHandle,
    count: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    try mapStatus(c.ziggy_metal_silu_mul_f32(
        state.context,
        gate.raw,
        up.raw,
        @intCast(count),
        null,
        0,
    ));
}

pub fn addInPlace(
    backend: backend_api.MatVecBackend,
    dst: BufferHandle,
    src: BufferHandle,
    count: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    try mapStatus(c.ziggy_metal_add_in_place_f32(
        state.context,
        dst.raw,
        src.raw,
        @intCast(count),
        null,
        0,
    ));
}

pub fn rmsNorm(
    backend: backend_api.MatVecBackend,
    input: BufferHandle,
    weights: []const f32,
    output: BufferHandle,
    count: usize,
    eps: f32,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    const weights_buffer = try state.matrixBuffer(weights[0..count]);
    try mapStatus(c.ziggy_metal_rms_norm_f32(
        state.context,
        input.raw,
        weights_buffer.raw,
        output.raw,
        @intCast(count),
        eps,
        null,
        0,
    ));
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

fn createRawBuffer(context: *c.ZiggyMetalContext, bytes: []const u8) !*c.ZiggyMetalBuffer {
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    var raw: ?*c.ZiggyMetalBuffer = null;
    const status = c.ziggy_metal_create_buffer(
        context,
        bytes.ptr,
        bytes.len,
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
