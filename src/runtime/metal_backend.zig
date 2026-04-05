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

pub const ShortlistEntry = extern struct {
    token_id: u32,
    score: f32,
};

pub const CommitStats = struct {
    cpu_wait_ns: u64 = 0,
    gpu_elapsed_ns: u64 = 0,
    gpu_timestamps_valid: bool = false,
};

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
        var device_info: c.ZiggyMetalDeviceInfo = std.mem.zeroes(c.ZiggyMetalDeviceInfo);
        const status = c.ziggy_metal_create_context(
            shader_source.ptr,
            shader_source.len,
            &raw_context,
            &device_info,
            &error_buf,
            error_buf.len,
        );
        try mapStatus(status, &error_buf);
        if (!device_info.has_unified_memory) {
            c.ziggy_metal_destroy_context(raw_context.?);
            return error.MetalUnavailable;
        }

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
        var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
        try mapStatus(c.ziggy_metal_run_matvec_f32(
            self.context,
            matrix_buffer.raw,
            input_buffer.raw,
            output_buffer.raw,
            @intCast(rows),
            @intCast(cols),
            &error_buf,
            error_buf.len,
        ), &error_buf);
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

pub fn createByteScratchBuffer(backend: backend_api.MatVecBackend, byte_len: usize) !BufferHandle {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    const raw = try createEmptyBuffer(state.context, byte_len);
    return .{
        .raw = raw,
        .byte_len = byte_len,
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

pub fn writeBufferF16(buffer: BufferHandle, values: []const f16) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (values.len * @sizeOf(f16) > buffer.byte_len) return error.MetalBufferError;
    try writeBufferBytes(buffer.raw, std.mem.sliceAsBytes(values));
}

pub fn storeKvHalf(
    backend: backend_api.MatVecBackend,
    src: BufferHandle,
    dst: BufferHandle,
    dst_offset_elements: usize,
    count: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_store_kv_half(
        state.context,
        src.raw,
        dst.raw,
        @intCast(dst_offset_elements),
        @intCast(count),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn readBufferF32(buffer: BufferHandle, out: []f32) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (out.len * @sizeOf(f32) > buffer.byte_len) return error.MetalBufferError;
    try readBuffer(buffer.raw, out);
}

pub fn readBufferU32(buffer: BufferHandle, out: []u32) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (out.len * @sizeOf(u32) > buffer.byte_len) return error.MetalBufferError;
    try readBufferBytes(buffer.raw, std.mem.sliceAsBytes(out));
}

pub fn writeBufferU32(buffer: BufferHandle, values: []const u32) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (values.len * @sizeOf(u32) > buffer.byte_len) return error.MetalBufferError;
    try writeBufferBytes(buffer.raw, std.mem.sliceAsBytes(values));
}

pub fn writeBufferU64(buffer: BufferHandle, values: []const u64) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (values.len * @sizeOf(u64) > buffer.byte_len) return error.MetalBufferError;
    try writeBufferBytes(buffer.raw, std.mem.sliceAsBytes(values));
}

pub fn readBufferU64(buffer: BufferHandle, out: []u64) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (out.len * @sizeOf(u64) > buffer.byte_len) return error.MetalBufferError;
    try readBufferBytes(buffer.raw, std.mem.sliceAsBytes(out));
}

pub fn beginSequence(backend: backend_api.MatVecBackend) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_begin_sequence(
        state.context,
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn commitSequence(backend: backend_api.MatVecBackend) !void {
    _ = try commitSequenceTimed(backend);
}

pub fn commitSequenceTimed(backend: backend_api.MatVecBackend) !CommitStats {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    var raw_stats: c.ZiggyMetalCommitStats = std.mem.zeroes(c.ZiggyMetalCommitStats);
    try mapStatus(c.ziggy_metal_commit_sequence_timed(
        state.context,
        &raw_stats,
        &error_buf,
        error_buf.len,
    ), &error_buf);
    return .{
        .cpu_wait_ns = raw_stats.cpu_wait_ns,
        .gpu_elapsed_ns = raw_stats.gpu_elapsed_ns,
        .gpu_timestamps_valid = raw_stats.gpu_timestamps_valid,
    };
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecToDstBuffer(
    backend: backend_api.MatVecBackend,
    matrix: []const f32,
    input: BufferHandle,
    output: BufferHandle,
    output_offset_bytes: usize,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < cols * @sizeOf(f32) or output.byte_len < output_offset_bytes + rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.matrixBuffer(matrix[0 .. rows * cols]);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_f32_to_dst(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        output_offset_bytes,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecAddToBuffer(
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_add_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q4k_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ4KToDstBuffer(
    backend: backend_api.MatVecBackend,
    matrix_bytes: []const u8,
    input: BufferHandle,
    output: BufferHandle,
    output_offset_bytes: usize,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < cols * @sizeOf(f32) or output.byte_len < output_offset_bytes + rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.rawBuffer(matrix_bytes);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q4k_f32_to_dst(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        output_offset_bytes,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ4KAddToBuffer(
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q4k_add_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ6KToBuffer(
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q6k_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ6KToDstBuffer(
    backend: backend_api.MatVecBackend,
    matrix_bytes: []const u8,
    input: BufferHandle,
    output: BufferHandle,
    output_offset_bytes: usize,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < cols * @sizeOf(f32) or output.byte_len < output_offset_bytes + rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.rawBuffer(matrix_bytes);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q6k_f32_to_dst(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        output_offset_bytes,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ6KAddToBuffer(
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q6k_add_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ6KArgmaxToBuffer(
    backend: backend_api.MatVecBackend,
    matrix_bytes: []const u8,
    input: BufferHandle,
    output_packed: BufferHandle,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < cols * @sizeOf(f32) or output_packed.byte_len < @sizeOf(u64)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.rawBuffer(matrix_bytes);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q6k_argmax_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output_packed.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ8_0ToBuffer(
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q8_0_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ8_0ToDstBuffer(
    backend: backend_api.MatVecBackend,
    matrix_bytes: []const u8,
    input: BufferHandle,
    output: BufferHandle,
    output_offset_bytes: usize,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < cols * @sizeOf(f32) or output.byte_len < output_offset_bytes + rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.rawBuffer(matrix_bytes);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q8_0_f32_to_dst(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        output_offset_bytes,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecQ8_0AddToBuffer(
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_q8_0_add_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecMoonQuantQ4KToBuffer(
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_moonq_q4k_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecMoonQuantQ4KToDstBuffer(
    backend: backend_api.MatVecBackend,
    matrix_bytes: []const u8,
    input: BufferHandle,
    output: BufferHandle,
    output_offset_bytes: usize,
    rows: usize,
    cols: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < cols * @sizeOf(f32) or output.byte_len < output_offset_bytes + rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.rawBuffer(matrix_bytes);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_moonq_q4k_f32_to_dst(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        output_offset_bytes,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn runMatVecMoonQuantQ4KAddToBuffer(
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_run_matvec_moonq_q4k_add_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        &error_buf,
        error_buf.len,
    ), &error_buf);
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_copy_buffer_region(
        state.context,
        src.raw,
        src_offset_bytes,
        dst.raw,
        dst_offset_bytes,
        length_bytes,
        &error_buf,
        error_buf.len,
    ), &error_buf);
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_apply_rope_f32(
        state.context,
        vector.raw,
        @intCast(head_count),
        @intCast(head_dim),
        @intCast(rope_dim),
        @intCast(position),
        freq_base,
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn applyRoPEAtOffset(
    backend: backend_api.MatVecBackend,
    vector: BufferHandle,
    vector_offset_bytes: usize,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    freq_base: f32,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_apply_rope_at_offset_f32(
        state.context,
        vector.raw,
        vector_offset_bytes,
        @intCast(head_count),
        @intCast(head_dim),
        @intCast(rope_dim),
        @intCast(position),
        freq_base,
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn applyRoPEToDst(
    backend: backend_api.MatVecBackend,
    src: BufferHandle,
    dst: BufferHandle,
    dst_offset_bytes: usize,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    freq_base: f32,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_apply_rope_to_dst_f32(
        state.context,
        src.raw,
        dst.raw,
        dst_offset_bytes,
        @intCast(head_count),
        @intCast(head_dim),
        @intCast(rope_dim),
        @intCast(position),
        freq_base,
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn attentionFused(
    backend: backend_api.MatVecBackend,
    q: BufferHandle,
    k_cache: BufferHandle,
    v_cache: BufferHandle,
    attn_scores: BufferHandle,
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_attention_fused_f32(
        state.context,
        q.raw,
        k_cache.raw,
        v_cache.raw,
        attn_scores.raw,
        output.raw,
        @intCast(head_count),
        @intCast(head_count_kv),
        @intCast(head_dim),
        @intCast(kv_dim),
        @intCast(context_length),
        @intCast(position),
        @intCast(layer_base),
        scale,
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn siluMul(
    backend: backend_api.MatVecBackend,
    gate: BufferHandle,
    up: BufferHandle,
    count: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_silu_mul_f32(
        state.context,
        gate.raw,
        up.raw,
        @intCast(count),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn addInPlace(
    backend: backend_api.MatVecBackend,
    dst: BufferHandle,
    src: BufferHandle,
    count: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_add_in_place_f32(
        state.context,
        dst.raw,
        src.raw,
        @intCast(count),
        &error_buf,
        error_buf.len,
    ), &error_buf);
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
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_rms_norm_f32(
        state.context,
        input.raw,
        weights_buffer.raw,
        output.raw,
        @intCast(count),
        eps,
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn argmax(
    backend: backend_api.MatVecBackend,
    input: BufferHandle,
    output_token: BufferHandle,
    count: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_argmax_f32(
        state.context,
        input.raw,
        output_token.raw,
        @intCast(count),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn topKShortlist(
    backend: backend_api.MatVecBackend,
    input: BufferHandle,
    output_entries: BufferHandle,
    count: usize,
    top_k: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (top_k == 0 or top_k > 64) return error.InvalidTensorMetadata;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_topk_f32(
        state.context,
        input.raw,
        output_entries.raw,
        @intCast(count),
        @intCast(top_k),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn sampleTopK(
    backend: backend_api.MatVecBackend,
    input: BufferHandle,
    output_token: BufferHandle,
    count: usize,
    top_k: usize,
    temperature: f32,
    random_uniform: f32,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (top_k == 0 or top_k > 64 or !(temperature > 0)) return error.InvalidTensorMetadata;
    if (output_token.byte_len < @sizeOf(u32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_sample_topk_f32(
        state.context,
        input.raw,
        output_token.raw,
        @intCast(count),
        @intCast(top_k),
        temperature,
        random_uniform,
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn readShortlistEntries(buffer: BufferHandle, out: []ShortlistEntry) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (out.len * @sizeOf(ShortlistEntry) > buffer.byte_len) return error.MetalBufferError;
    try readBufferBytes(buffer.raw, std.mem.sliceAsBytes(out));
}

pub fn batchArgmax(
    backend: backend_api.MatVecBackend,
    input: BufferHandle,
    output_tokens: BufferHandle,
    vocab_size: usize,
    batch_count: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (batch_count == 0 or batch_count > 8) return error.InvalidTensorMetadata;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_batch_argmax_f32(
        state.context,
        input.raw,
        output_tokens.raw,
        @intCast(vocab_size),
        @intCast(batch_count),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn batchMatvecAdd(
    backend: backend_api.MatVecBackend,
    matrix: []const f32,
    input: BufferHandle,
    output: BufferHandle,
    rows: usize,
    cols: usize,
    batch_idx: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < (batch_idx + 1) * cols * @sizeOf(f32) or output.byte_len < (batch_idx + 1) * rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.matrixBuffer(matrix[0 .. rows * cols]);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_batch_matvec_add_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        @intCast(batch_idx),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn batchMatvecQ4KAdd(
    backend: backend_api.MatVecBackend,
    matrix_bytes: []const u8,
    input: BufferHandle,
    output: BufferHandle,
    rows: usize,
    cols: usize,
    batch_idx: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < (batch_idx + 1) * cols * @sizeOf(f32) or output.byte_len < (batch_idx + 1) * rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.rawBuffer(matrix_bytes);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_batch_matvec_q4k_add_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        @intCast(batch_idx),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn batchSiluMul(
    backend: backend_api.MatVecBackend,
    gate: BufferHandle,
    up: BufferHandle,
    count: usize,
    batch_idx: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (gate.byte_len < (batch_idx + 1) * count * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_batch_silu_mul_f32(
        state.context,
        gate.raw,
        up.raw,
        @intCast(count),
        @intCast(batch_idx),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn batchAddInPlace(
    backend: backend_api.MatVecBackend,
    dst: BufferHandle,
    src: BufferHandle,
    count: usize,
    batch_idx: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (dst.byte_len < (batch_idx + 1) * count * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_batch_add_in_place_f32(
        state.context,
        dst.raw,
        src.raw,
        @intCast(count),
        @intCast(batch_idx),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn batchRmsNorm(
    backend: backend_api.MatVecBackend,
    input: BufferHandle,
    weights: []const f32,
    output: BufferHandle,
    count: usize,
    eps: f32,
    batch_idx: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < (batch_idx + 1) * count * @sizeOf(f32) or output.byte_len < (batch_idx + 1) * count * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const weights_buffer = try state.matrixBuffer(weights[0..count]);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_batch_rms_norm_f32(
        state.context,
        input.raw,
        weights_buffer.raw,
        output.raw,
        @intCast(count),
        eps,
        @intCast(batch_idx),
        &error_buf,
        error_buf.len,
    ), &error_buf);
}

pub fn batchMatvecQ4K(
    backend: backend_api.MatVecBackend,
    matrix_bytes: []const u8,
    input: BufferHandle,
    output: BufferHandle,
    rows: usize,
    cols: usize,
    batch_idx: usize,
) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (input.byte_len < (batch_idx + 1) * cols * @sizeOf(f32) or output.byte_len < (batch_idx + 1) * rows * @sizeOf(f32)) return error.MetalBufferError;
    const state = stateFromCtx(backend.ctx);
    const matrix_buffer = try state.rawBuffer(matrix_bytes);
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    try mapStatus(c.ziggy_metal_batch_matvec_q4k_f32(
        state.context,
        matrix_buffer.raw,
        input.raw,
        output.raw,
        @intCast(rows),
        @intCast(cols),
        @intCast(batch_idx),
        &error_buf,
        error_buf.len,
    ), &error_buf);
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
    try mapStatus(status, &error_buf);
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
    try mapStatus(status, &error_buf);
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
    try mapStatus(status, &error_buf);
    return raw.?;
}

fn writeBuffer(buffer: *c.ZiggyMetalBuffer, values: []const f32) !void {
    try writeBufferBytes(buffer, std.mem.sliceAsBytes(values));
}

fn writeBufferBytes(buffer: *c.ZiggyMetalBuffer, values: []const u8) !void {
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    const status = c.ziggy_metal_write_buffer(
        buffer,
        values.ptr,
        values.len,
        0,
        &error_buf,
        error_buf.len,
    );
    try mapStatus(status, &error_buf);
}

fn readBuffer(buffer: *c.ZiggyMetalBuffer, out: []f32) !void {
    try readBufferBytes(buffer, std.mem.sliceAsBytes(out));
}

fn readBufferBytes(buffer: *c.ZiggyMetalBuffer, out: []u8) !void {
    var error_buf: [err_buf_len]u8 = std.mem.zeroes([err_buf_len]u8);
    const status = c.ziggy_metal_read_buffer(
        buffer,
        out.ptr,
        out.len,
        0,
        &error_buf,
        error_buf.len,
    );
    try mapStatus(status, &error_buf);
}

fn mapStatus(status: c_int, error_buf: *const [err_buf_len]u8) !void {
    if (!build_enabled_value) return error.MetalDisabled;
    if (status != c.ZIGGY_METAL_OK) maybePrintMetalError(error_buf);
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

fn maybePrintMetalError(error_buf: *const [err_buf_len]u8) void {
    const message = std.mem.sliceTo(error_buf, 0);
    if (message.len == 0) return;
    var stderr_buffer: [512]u8 = undefined;
    var stderr = std.fs.File.stderr().writer(&stderr_buffer);
    stderr.interface.print("metal error: {s}\n", .{message}) catch {};
    stderr.interface.flush() catch {};
}
