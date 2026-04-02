const std = @import("std");
const metal_backend = @import("metal_backend.zig");
const llama_cpu = @import("../llama_cpu.zig");

test "metal q4k matvec matches cpu dequantized reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 512;
    var matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&matrix, rows, cols);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 17)) - 8)) * 0.125) + 0.1;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.runMatVecQ4KToBuffer(backend, &matrix, input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = undefined;
    var dequantized_row: [cols]f32 = undefined;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q4_k, row_bytes, cols);
        expected[row] = dot(&dequantized_row, &input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.01);
    }
}

test "metal dense matvec matches cpu reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 96;
    const cols = 160;
    var matrix: [rows * cols]f32 = undefined;
    var input: [cols]f32 = undefined;
    var expected: [rows]f32 = undefined;
    var actual: [rows]f32 = undefined;

    for (&input, 0..) |*value, col| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast(col % 23)) - 11)) * 0.05;
    }
    for (0..cols) |col| {
        for (0..rows) |row| {
            matrix[row + col * rows] = (@as(f32, @floatFromInt((row * 7 + col * 3) % 29)) - 14.0) * 0.025;
        }
    }
    for (0..rows) |row| {
        var sum: f32 = 0.0;
        for (0..cols) |col| sum += matrix[row + col * rows] * input[col];
        expected[row] = sum;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);
    try backend.matVec(&actual, &matrix, &input, rows, cols);

    for (0..rows) |row| {
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.0005);
    }
}

test "metal fused attention matches cpu reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 3;
    const head_count_kv = 1;
    const head_dim = 96;
    const kv_dim = head_dim * head_count_kv;
    const context_length = 5;
    const position = 4;
    const layer_base = 0;
    const scale: f32 = 0.125;

    var q: [head_count * head_dim]f32 = undefined;
    var k_cache: [context_length * kv_dim]f32 = undefined;
    var v_cache: [context_length * kv_dim]f32 = undefined;
    var expected: [head_count * head_dim]f32 = undefined;
    var actual: [head_count * head_dim]f32 = undefined;

    for (&q, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.07) + 0.02;
    }
    for (&k_cache, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 17)) - 8)) * 0.06) - 0.01;
    }
    for (&v_cache, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 13)) - 6)) * 0.08) + 0.03;
    }

    attentionReference(&expected, &q, &k_cache, &v_cache, head_count, head_count_kv, head_dim, kv_dim, position, layer_base, scale);

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const q_buffer = try metal_backend.createScratchBuffer(backend, q.len);
    defer metal_backend.destroyBuffer(q_buffer);
    const k_buffer = try metal_backend.createScratchBuffer(backend, k_cache.len);
    defer metal_backend.destroyBuffer(k_buffer);
    const v_buffer = try metal_backend.createScratchBuffer(backend, v_cache.len);
    defer metal_backend.destroyBuffer(v_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, actual.len);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(q_buffer, &q);
    try metal_backend.writeBufferF32(k_buffer, &k_cache);
    try metal_backend.writeBufferF32(v_buffer, &v_cache);
    try metal_backend.attentionFused(
        backend,
        q_buffer,
        k_buffer,
        v_buffer,
        output_buffer,
        head_count,
        head_count_kv,
        head_dim,
        kv_dim,
        context_length,
        position,
        layer_base,
        scale,
    );
    try metal_backend.readBufferF32(output_buffer, &actual);

    for (0..actual.len) |index| {
        try std.testing.expectApproxEqAbs(expected[index], actual[index], 0.001);
    }
}

fn fillQ4KMatrix(buffer: []u8, rows: usize, cols: usize) void {
    const row_size = cols / 256 * 144;
    for (0..rows) |row| {
        for (0..cols / 256) |block| {
            const base = row * row_size + block * 144;
            writeHalf(buffer[base..][0..2], 0.125 * @as(f32, @floatFromInt(row + block + 1)));
            writeHalf(buffer[base + 2 ..][0..2], 0.0625 * @as(f32, @floatFromInt(block + 1)));
            writeScales(buffer[base + 4 ..][0..12], row, block);
            for (0..128) |index| {
                const low = @as(u8, @intCast((index + row + block) % 16));
                const high = @as(u8, @intCast((index * 3 + row + block + 5) % 16));
                buffer[base + 16 + index] = low | (high << 4);
            }
        }
    }
}

fn writeHalf(bytes: []u8, value: f32) void {
    const encoded: u16 = @bitCast(@as(f16, @floatCast(value)));
    std.mem.writeInt(u16, bytes[0..2], encoded, .little);
}

fn writeScales(bytes: []u8, row: usize, block: usize) void {
    for (0..4) |index| {
        bytes[index] = @as(u8, @intCast((1 + row + block + index) & 63));
        bytes[4 + index] = @as(u8, @intCast((index + block) & 63));
        bytes[8 + index] =
            @as(u8, @intCast((1 + row + index) & 0x0F)) |
            (@as(u8, @intCast((2 + block + index) & 0x0F)) << 4);
    }
}

fn dot(lhs: []const f32, rhs: []const f32) f32 {
    var sum: f32 = 0;
    for (lhs, rhs) |l, r| sum += l * r;
    return sum;
}

fn attentionReference(
    output: []f32,
    q: []const f32,
    k_cache: []const f32,
    v_cache: []const f32,
    head_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    kv_dim: usize,
    position: usize,
    layer_base: usize,
    scale: f32,
) void {
    var scores: [5]f32 = undefined;
    for (0..head_count) |head| {
        const kv_group = head_count / head_count_kv;
        const kv_head = head / kv_group;
        const kv_offset = kv_head * head_dim;
        const q_head = q[head * head_dim ..][0..head_dim];

        var max_value = -std.math.inf(f32);
        for (0..position + 1) |token| {
            const k_head = k_cache[layer_base + token * kv_dim + kv_offset ..][0..head_dim];
            const value = dot(q_head, k_head) * scale;
            scores[token] = value;
            max_value = @max(max_value, value);
        }

        var denom: f32 = 0.0;
        for (0..position + 1) |token| {
            const shifted = @exp(scores[token] - max_value);
            scores[token] = shifted;
            denom += shifted;
        }

        for (0..head_dim) |dim| {
            var sum: f32 = 0.0;
            for (0..position + 1) |token| {
                const weight = scores[token] / denom;
                sum += weight * v_cache[layer_base + token * kv_dim + kv_offset + dim];
            }
            output[head * head_dim + dim] = sum;
        }
    }
}
