const std = @import("std");
const metal_backend = @import("metal_backend.zig");
const llama_cpu = @import("../llama_cpu.zig");
const moon_quant = @import("../moon_quant.zig");
const llama_fixture = @import("llama_fixture.zig");

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

test "metal top-k shortlist returns descending logits with stable ties" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const logits = [_]f32{ 0.2, 1.5, -2.0, 1.0, 3.0, 0.4, 1.5, -1.0 };
    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, logits.len);
    defer metal_backend.destroyBuffer(input_buffer);
    const shortlist_buffer = try metal_backend.createByteScratchBuffer(backend, 3 * @sizeOf(metal_backend.ShortlistEntry));
    defer metal_backend.destroyBuffer(shortlist_buffer);

    try metal_backend.writeBufferF32(input_buffer, &logits);
    try metal_backend.topKShortlist(backend, input_buffer, shortlist_buffer, logits.len, 3);

    var entries: [3]metal_backend.ShortlistEntry = undefined;
    try metal_backend.readShortlistEntries(shortlist_buffer, &entries);

    try std.testing.expectEqualSlices(u32, &.{ 4, 1, 6 }, &.{ entries[0].token_id, entries[1].token_id, entries[2].token_id });
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), entries[0].score, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), entries[1].score, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), entries[2].score, 0.0001);
}

test "metal sample top-k returns one of the weighted winners" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const logits = [_]f32{ 5.0, 4.0, 0.1, -1.0 };
    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, logits.len);
    defer metal_backend.destroyBuffer(input_buffer);
    const token_buffer = try metal_backend.createScratchBuffer(backend, 1);
    defer metal_backend.destroyBuffer(token_buffer);

    try metal_backend.writeBufferF32(input_buffer, &logits);

    try metal_backend.sampleTopK(backend, input_buffer, token_buffer, logits.len, 2, 1.0, 0.0);
    var sampled_low: [1]u32 = undefined;
    try metal_backend.readBufferU32(token_buffer, &sampled_low);
    try std.testing.expectEqual(@as(u32, 0), sampled_low[0]);

    try metal_backend.sampleTopK(backend, input_buffer, token_buffer, logits.len, 2, 1.0, 0.999);
    var sampled_high: [1]u32 = undefined;
    try metal_backend.readBufferU32(token_buffer, &sampled_high);
    try std.testing.expect(sampled_high[0] == 0 or sampled_high[0] == 1);
}

test "metal q4k fused add matches cpu dequantized reference for dominant llama shape" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q4_k);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q4k-add.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q4k-add.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const rows = 3;
    const cols = model.feed_forward_length;
    const tensor = model.layers[0].ffn_down;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);
    const matrix = try llama_cpu.tensorBytes(&model, tensor);

    const input = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(input);
    for (input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 31)) - 15)) * 0.03125) + 0.02;
    }

    var base: [rows]f32 = .{ 0.25, -0.75, 1.5 };
    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, input);
    try metal_backend.writeBufferF32(output_buffer, &base);
    try metal_backend.runMatVecQ4KAddToBuffer(backend, matrix[0 .. rows * row_size], input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = base;
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, row_bytes, cols);
        expected[row] += dot(dequantized_row, input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.05);
    }
}

test "metal q4k dual matvec matches cpu dequantized references for dominant llama shape" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q4_k);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q4k-dual.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q4k-dual.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const rows = model.feed_forward_length;
    const cols = model.embedding_length;
    const gate_tensor = model.layers[0].ffn_gate;
    const up_tensor = model.layers[0].ffn_up;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);
    const gate_matrix = try llama_cpu.tensorBytes(&model, gate_tensor);
    const up_matrix = try llama_cpu.tensorBytes(&model, up_tensor);

    const input = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(input);
    for (input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 31)) - 15)) * 0.03125) + 0.02;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const gate_output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(gate_output_buffer);
    const up_output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(up_output_buffer);

    try metal_backend.writeBufferF32(input_buffer, input);
    try metal_backend.runMatVecQ4KDualToBuffers(
        backend,
        gate_matrix,
        up_matrix,
        input_buffer,
        gate_output_buffer,
        up_output_buffer,
        rows,
        cols,
    );

    const gate_actual = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(gate_actual);
    const up_actual = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(up_actual);
    try metal_backend.readBufferF32(gate_output_buffer, gate_actual);
    try metal_backend.readBufferF32(up_output_buffer, up_actual);

    const gate_expected = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(gate_expected);
    const up_expected = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(up_expected);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);

    for (0..rows) |row| {
        const gate_row_bytes = gate_matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, gate_row_bytes, cols);
        gate_expected[row] = dot(dequantized_row, input);

        const up_row_bytes = up_matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, up_row_bytes, cols);
        up_expected[row] = dot(dequantized_row, input);

        try std.testing.expectApproxEqAbs(gate_expected[row], gate_actual[row], 0.05);
        try std.testing.expectApproxEqAbs(up_expected[row], up_actual[row], 0.05);
    }
}

test "metal MoonQuant q4k matvec matches cpu dequantized reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 512;
    var raw_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&raw_matrix, rows, cols);
    var packed_matrix = try moon_quant.packQ4KTensor(std.testing.allocator, &raw_matrix, rows, cols);
    defer packed_matrix.deinit(std.testing.allocator);

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
    try metal_backend.runMatVecMoonQuantQ4KToBuffer(backend, packed_matrix.bytes, input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = undefined;
    var dequantized_row: [cols]f32 = undefined;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);
    for (0..rows) |row| {
        const row_bytes = raw_matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q4_k, row_bytes, cols);
        expected[row] = dot(&dequantized_row, &input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.01);
    }
}

test "metal MoonQuant q4k fused add matches cpu reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 512;
    var raw_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&raw_matrix, rows, cols);
    var packed_matrix = try moon_quant.packQ4KTensor(std.testing.allocator, &raw_matrix, rows, cols);
    defer packed_matrix.deinit(std.testing.allocator);

    var input: [cols]f32 = undefined;
    var base: [rows]f32 = .{ 0.5, -1.25, 2.0 };
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
    try metal_backend.writeBufferF32(output_buffer, &base);
    try metal_backend.runMatVecMoonQuantQ4KAddToBuffer(backend, packed_matrix.bytes, input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = base;
    var dequantized_row: [cols]f32 = undefined;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);
    for (0..rows) |row| {
        const row_bytes = raw_matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q4_k, row_bytes, cols);
        expected[row] += dot(&dequantized_row, &input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.01);
    }
}

test "metal MoonQuant q4k matvec writes to dst offset" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 512;
    var raw_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&raw_matrix, rows, cols);
    var packed_matrix = try moon_quant.packQ4KTensor(std.testing.allocator, &raw_matrix, rows, cols);
    defer packed_matrix.deinit(std.testing.allocator);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 17)) - 8)) * 0.125) + 0.1;
    }

    var expected = [_]f32{ -7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0 };
    var actual = expected;

    var dequantized_row: [cols]f32 = undefined;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);
    for (0..rows) |row| {
        const row_bytes = raw_matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q4_k, row_bytes, cols);
        expected[2 + row] = dot(&dequantized_row, &input);
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);
    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, actual.len);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.writeBufferF32(output_buffer, &actual);
    try metal_backend.runMatVecMoonQuantQ4KToDstBuffer(backend, packed_matrix.bytes, input_buffer, output_buffer, 2 * @sizeOf(f32), rows, cols);
    try metal_backend.readBufferF32(output_buffer, &actual);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.01);
    }
}

test "metal q6k matvec matches cpu dequantized reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 512;
    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q6_k);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q6k-matvec.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q6k-matvec.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const tensor = model.layers[0].attn_q;
    const row_size = try llama_cpu.tensorRowByteSize(.q6_k, cols);
    const matrix = try llama_cpu.tensorBytes(&model, tensor);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.0625) + 0.05;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.runMatVecQ6KToBuffer(backend, matrix[0 .. rows * row_size], input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = undefined;
    var dequantized_row: [cols]f32 = undefined;
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q6_k, row_bytes, cols);
        expected[row] = dot(&dequantized_row, &input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.02);
    }
}

test "metal q6k matvec writes to dst offset" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 512;
    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q6_k);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q6k-matvec-offset.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q6k-matvec-offset.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const row_size = try llama_cpu.tensorRowByteSize(.q6_k, cols);
    const matrix = try llama_cpu.tensorBytes(&model, model.layers[0].attn_q);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.0625) + 0.05;
    }

    var expected = [_]f32{ -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0 };
    var actual = expected;
    var dequantized_row: [cols]f32 = undefined;
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q6_k, row_bytes, cols);
        expected[1 + row] = dot(&dequantized_row, &input);
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);
    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, actual.len);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.writeBufferF32(output_buffer, &actual);
    try metal_backend.runMatVecQ6KToDstBuffer(backend, matrix[0 .. rows * row_size], input_buffer, output_buffer, @sizeOf(f32), rows, cols);
    try metal_backend.readBufferF32(output_buffer, &actual);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.02);
    }
}

test "metal q6k fused add matches cpu dequantized reference for dominant llama shape" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q6_k);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q6k-add.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q6k-add.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const rows = 3;
    const cols = model.feed_forward_length;
    const tensor = model.layers[0].ffn_down;
    const row_size = try llama_cpu.tensorRowByteSize(.q6_k, cols);
    const matrix = try llama_cpu.tensorBytes(&model, tensor);

    const input = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(input);
    for (input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 37)) - 18)) * 0.02) + 0.01;
    }

    var base: [rows]f32 = .{ -0.5, 0.75, 1.25 };
    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, input);
    try metal_backend.writeBufferF32(output_buffer, &base);
    try metal_backend.runMatVecQ6KAddToBuffer(backend, matrix[0 .. rows * row_size], input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = base;
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(dequantized_row, .q6_k, row_bytes, cols);
        expected[row] += dot(dequantized_row, input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.05);
    }
}

test "metal MoonQuant q6k matvec matches cpu dequantized reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 512;
    var raw_matrix: [rows * (cols / 256) * 210]u8 = undefined;
    fillQ6KMatrix(&raw_matrix, rows, cols);
    var packed_matrix = try moon_quant.packQ6KTensor(std.testing.allocator, &raw_matrix, rows, cols);
    defer packed_matrix.deinit(std.testing.allocator);

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
    try metal_backend.runMatVecMoonQuantQ6KToBuffer(backend, packed_matrix.bytes, input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = undefined;
    var dequantized_row: [cols]f32 = undefined;
    const row_size = try llama_cpu.tensorRowByteSize(.q6_k, cols);
    for (0..rows) |row| {
        const row_bytes = raw_matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q6_k, row_bytes, cols);
        expected[row] = dot(&dequantized_row, &input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.01);
    }
}

test "metal MoonQuant q6k fused add matches cpu reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 512;
    var raw_matrix: [rows * (cols / 256) * 210]u8 = undefined;
    fillQ6KMatrix(&raw_matrix, rows, cols);
    var packed_matrix = try moon_quant.packQ6KTensor(std.testing.allocator, &raw_matrix, rows, cols);
    defer packed_matrix.deinit(std.testing.allocator);

    var input: [cols]f32 = undefined;
    var base: [rows]f32 = .{ 0.5, -1.25, 2.0 };
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
    try metal_backend.writeBufferF32(output_buffer, &base);
    try metal_backend.runMatVecMoonQuantQ6KAddToBuffer(backend, packed_matrix.bytes, input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = base;
    var dequantized_row: [cols]f32 = undefined;
    const row_size = try llama_cpu.tensorRowByteSize(.q6_k, cols);
    for (0..rows) |row| {
        const row_bytes = raw_matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q6_k, row_bytes, cols);
        expected[row] += dot(&dequantized_row, &input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.01);
    }
}

test "metal q6k fused argmax matches cpu dequantized reference for output projection" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q6_k);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q6k-argmax.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q6k-argmax.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const tensor = model.output;
    const rows = try tensor.rowCount();
    const cols = 512;
    const row_size = try llama_cpu.tensorRowByteSize(.q6_k, cols);
    const matrix = try llama_cpu.tensorBytes(&model, tensor);

    const input = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(input);
    for (input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 41)) - 20)) * 0.015625) + 0.02;
    }

    var expected_best_token: u32 = 0;
    var expected_best_logit = -std.math.inf(f32);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(dequantized_row, .q6_k, row_bytes, cols);
        const logit = dot(dequantized_row, input);
        if (logit > expected_best_logit) {
            expected_best_logit = logit;
            expected_best_token = @intCast(row);
        }
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const packed_buffer = try metal_backend.createByteScratchBuffer(backend, 3 * @sizeOf(u32));
    defer metal_backend.destroyBuffer(packed_buffer);

    try metal_backend.writeBufferF32(input_buffer, input);
    try metal_backend.writeBufferU32(packed_buffer, &.{ 0, 0, std.math.maxInt(u32) });
    try metal_backend.runMatVecQ6KArgmaxToBuffer(backend, matrix, input_buffer, packed_buffer, rows, cols);

    var argmax_state: [3]u32 = .{ 0, 0, 0 };
    try metal_backend.readBufferU32(packed_buffer, &argmax_state);
    const actual_token = argmax_state[2];
    try std.testing.expectEqual(expected_best_token, actual_token);
}

test "metal q4k fused argmax matches cpu dequantized reference for output projection" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q4_k);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q4k-argmax.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q4k-argmax.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const tensor = model.output;
    const rows = try tensor.rowCount();
    const cols = 512;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);
    const matrix = try llama_cpu.tensorBytes(&model, tensor);

    const input = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(input);
    for (input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 41)) - 20)) * 0.015625) + 0.02;
    }

    var expected_best_token: u32 = 0;
    var expected_best_logit = -std.math.inf(f32);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, row_bytes, cols);
        const logit = dot(dequantized_row, input);
        if (logit > expected_best_logit) {
            expected_best_logit = logit;
            expected_best_token = @intCast(row);
        }
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const packed_buffer = try metal_backend.createByteScratchBuffer(backend, 3 * @sizeOf(u32));
    defer metal_backend.destroyBuffer(packed_buffer);

    try metal_backend.writeBufferF32(input_buffer, input);
    try metal_backend.writeBufferU32(packed_buffer, &.{ 0, 0, std.math.maxInt(u32) });
    try metal_backend.runMatVecQ4KArgmaxToBuffer(backend, matrix, input_buffer, packed_buffer, rows, cols);

    var argmax_state: [3]u32 = .{ 0, 0, 0 };
    try metal_backend.readBufferU32(packed_buffer, &argmax_state);
    const actual_token = argmax_state[2];
    try std.testing.expectEqual(expected_best_token, actual_token);
}

test "metal q8_0 matvec matches cpu dequantized reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 256;
    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q8_0);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q80-matvec.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q80-matvec.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const tensor = model.layers[0].attn_q;
    const row_size = try llama_cpu.tensorRowByteSize(.q8_0, cols);
    const matrix = try llama_cpu.tensorBytes(&model, tensor);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 13)) - 6)) * 0.125) + 0.05;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);
    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.runMatVecQ8_0ToBuffer(backend, matrix[0 .. rows * row_size], input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = undefined;
    var dequantized_row: [cols]f32 = undefined;
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q8_0, row_bytes, cols);
        expected[row] = dot(&dequantized_row, &input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.02);
    }
}

test "metal q8_0 fused add matches cpu dequantized reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 256;
    const fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q8_0);
    defer std.testing.allocator.free(fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "q80-add.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "q80-add.gguf");
    defer std.testing.allocator.free(path);

    var model = try llama_cpu.loadModel(std.testing.allocator, path);
    defer model.deinit(std.testing.allocator);

    const tensor = model.layers[0].attn_q;
    const row_size = try llama_cpu.tensorRowByteSize(.q8_0, cols);
    const matrix = try llama_cpu.tensorBytes(&model, tensor);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 13)) - 6)) * 0.125) + 0.05;
    }
    var base: [rows]f32 = .{ 0.5, -0.25, 1.0 };

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);
    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.writeBufferF32(output_buffer, &base);
    try metal_backend.runMatVecQ8_0AddToBuffer(backend, matrix[0 .. rows * row_size], input_buffer, output_buffer, rows, cols);

    var actual: [rows]f32 = undefined;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected: [rows]f32 = base;
    var dequantized_row: [cols]f32 = undefined;
    for (0..rows) |row| {
        const row_bytes = matrix[row * row_size ..][0..row_size];
        try llama_cpu.dequantizeRow(&dequantized_row, .q8_0, row_bytes, cols);
        expected[row] += dot(&dequantized_row, &input);
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.02);
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

test "metal dense fused add matches cpu reference for dominant llama shape" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const rows = 3;
    const cols = 2048;
    const matrix = try std.testing.allocator.alloc(f32, rows * cols);
    defer std.testing.allocator.free(matrix);
    const input = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(input);
    var expected: [rows]f32 = .{ 0.5, -1.0, 2.5 };
    var actual: [rows]f32 = expected;

    for (input, 0..) |*value, col| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast(col % 29)) - 14)) * 0.02;
    }
    for (0..cols) |col| {
        for (0..rows) |row| {
            matrix[row + col * rows] = (@as(f32, @floatFromInt((row * 11 + col * 5) % 37)) - 18.0) * 0.015;
        }
    }
    for (0..rows) |row| {
        for (0..cols) |col| expected[row] += matrix[row + col * rows] * input[col];
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);
    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, input);
    try metal_backend.writeBufferF32(output_buffer, &actual);
    try metal_backend.runMatVecAddToBuffer(backend, matrix, input_buffer, output_buffer, rows, cols);
    try metal_backend.readBufferF32(output_buffer, &actual);

    for (0..rows) |row| {
        try std.testing.expectApproxEqAbs(expected[row], actual[row], 0.005);
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
    var k_cache: [context_length * kv_dim]f16 = undefined;
    var v_cache: [context_length * kv_dim]f16 = undefined;
    var k_cache_f32: [context_length * kv_dim]f32 = undefined;
    var v_cache_f32: [context_length * kv_dim]f32 = undefined;
    var expected: [head_count * head_dim]f32 = undefined;
    var actual: [head_count * head_dim]f32 = undefined;

    for (&q, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.07) + 0.02;
    }
    for (&k_cache_f32, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 17)) - 8)) * 0.06) - 0.01;
        k_cache[index] = @floatCast(value.*);
    }
    for (&v_cache_f32, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 13)) - 6)) * 0.08) + 0.03;
        v_cache[index] = @floatCast(value.*);
    }

    attentionReference(&expected, &q, &k_cache_f32, &v_cache_f32, head_count, head_count_kv, head_dim, kv_dim, position, layer_base, scale);

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const q_buffer = try metal_backend.createScratchBuffer(backend, q.len);
    defer metal_backend.destroyBuffer(q_buffer);

    const k_buffer = try metal_backend.createByteScratchBuffer(backend, k_cache.len * @sizeOf(f16));
    defer metal_backend.destroyBuffer(k_buffer);
    const v_buffer = try metal_backend.createByteScratchBuffer(backend, v_cache.len * @sizeOf(f16));
    defer metal_backend.destroyBuffer(v_buffer);

    const output_buffer = try metal_backend.createScratchBuffer(backend, actual.len);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(q_buffer, &q);
    try metal_backend.writeBufferF16(k_buffer, &k_cache);
    try metal_backend.writeBufferF16(v_buffer, &v_cache);
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
        try std.testing.expectApproxEqAbs(expected[index], actual[index], 0.005);
    }
}

test "metal rope-to-dst writes rotated kv slice directly" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 6;
    const position = 3;
    const freq_base: f32 = 10000;
    const dst_prefix = 5;

    var src: [head_count * head_dim]f32 = undefined;
    for (&src, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index)) - 7)) * 0.125) + 0.05;
    }

    var expected = [_]f32{-9.0} ** (dst_prefix + src.len + 3);
    var actual = expected;
    applyRoPEReference(expected[dst_prefix .. dst_prefix + src.len], &src, head_count, head_dim, rope_dim, position, freq_base, 0);

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const src_buffer = try metal_backend.createScratchBuffer(backend, src.len);
    defer metal_backend.destroyBuffer(src_buffer);
    const dst_buffer = try metal_backend.createScratchBuffer(backend, actual.len);
    defer metal_backend.destroyBuffer(dst_buffer);

    try metal_backend.writeBufferF32(src_buffer, &src);
    try metal_backend.writeBufferF32(dst_buffer, &actual);
    try metal_backend.applyRoPEToDst(
        backend,
        src_buffer,
        dst_buffer,
        dst_prefix * @sizeOf(f32),
        head_count,
        head_dim,
        rope_dim,
        position,
        freq_base,
        0,
    );
    try metal_backend.readBufferF32(dst_buffer, &actual);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0005);
    }
}

test "metal rope-to-half-dst writes rotated kv slice directly" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 6;
    const position = 3;
    const freq_base: f32 = 10000;
    const dst_prefix = 5;

    var src: [head_count * head_dim]f32 = undefined;
    for (&src, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index)) - 7)) * 0.125) + 0.05;
    }

    var expected_f32 = [_]f32{-9.0} ** (dst_prefix + src.len + 3);
    applyRoPEReference(expected_f32[dst_prefix .. dst_prefix + src.len], &src, head_count, head_dim, rope_dim, position, freq_base, 0);
    var expected = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + src.len + 3);
    for (expected_f32, 0..) |value, index| expected[index] = @floatCast(value);
    var actual = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + src.len + 3);

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const src_buffer = try metal_backend.createScratchBuffer(backend, src.len);
    defer metal_backend.destroyBuffer(src_buffer);
    const dst_buffer = try metal_backend.createByteScratchBuffer(backend, actual.len * @sizeOf(f16));
    defer metal_backend.destroyBuffer(dst_buffer);

    try metal_backend.writeBufferF32(src_buffer, &src);
    try metal_backend.writeBufferF16(dst_buffer, &actual);
    try metal_backend.applyRoPEToHalfDst(
        backend,
        src_buffer,
        dst_buffer,
        dst_prefix,
        head_count,
        head_dim,
        rope_dim,
        position,
        freq_base,
        0,
    );
    try metal_backend.readBufferF16(dst_buffer, &actual);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.001);
    }
}

test "metal q4k k-half matches separate matvec rope reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 8;
    const rows = head_count * head_dim;
    const cols = 512;
    const position = 3;
    const freq_base: f32 = 10000;
    const dst_prefix = 5;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);

    var matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&matrix, rows, cols);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.09375) - 0.15;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const k_cache_buffer = try metal_backend.createByteScratchBuffer(backend, (dst_prefix + rows + 3) * @sizeOf(f16));
    defer metal_backend.destroyBuffer(k_cache_buffer);

    var actual = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.writeBufferF16(k_cache_buffer, &actual);

    try metal_backend.runMatVecQ4KKHalf(
        backend,
        &matrix,
        input_buffer,
        k_cache_buffer,
        dst_prefix,
        head_count,
        head_dim,
        rope_dim,
        cols,
        position,
        freq_base,
        0,
    );
    try metal_backend.readBufferF16(k_cache_buffer, &actual);

    const expected_f32 = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected_f32);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);
    for (0..rows) |row| {
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, matrix[row * row_size ..][0..row_size], cols);
        expected_f32[row] = dot(dequantized_row, &input);
    }
    applyRoPEReference(expected_f32, expected_f32, head_count, head_dim, rope_dim, position, freq_base, 0);

    var expected = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    for (0..rows) |index| expected[dst_prefix + index] = @floatCast(expected_f32[index]);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.03);
    }
}

test "metal q4k q-rope matches separate matvec rope reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 8;
    const rows = head_count * head_dim;
    const cols = 512;
    const position = 3;
    const freq_base: f32 = 10000;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);

    var matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&matrix, rows, cols);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.09375) - 0.15;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.runMatVecQ4KQRope(
        backend,
        &matrix,
        input_buffer,
        output_buffer,
        head_count,
        head_dim,
        rope_dim,
        cols,
        position,
        freq_base,
        0,
    );

    var actual = [_]f32{0} ** rows;
    try metal_backend.readBufferF32(output_buffer, &actual);

    var expected = [_]f32{0} ** rows;
    var dequantized_row: [cols]f32 = undefined;
    for (0..rows) |row| {
        try llama_cpu.dequantizeRow(&dequantized_row, .q4_k, matrix[row * row_size ..][0..row_size], cols);
        expected[row] = dot(&dequantized_row, &input);
    }
    applyRoPEReference(&expected, &expected, head_count, head_dim, rope_dim, position, freq_base, 0);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.03);
    }
}

test "metal MoonQuant q4k k-half matches separate matvec rope reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 8;
    const rows = head_count * head_dim;
    const cols = 512;
    const position = 3;
    const freq_base: f32 = 10000;
    const dst_prefix = 5;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);

    var raw_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&raw_matrix, rows, cols);
    var packed_matrix = try moon_quant.packQ4KTensor(std.testing.allocator, &raw_matrix, rows, cols);
    defer packed_matrix.deinit(std.testing.allocator);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.09375) - 0.15;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const k_cache_buffer = try metal_backend.createByteScratchBuffer(backend, (dst_prefix + rows + 3) * @sizeOf(f16));
    defer metal_backend.destroyBuffer(k_cache_buffer);

    var actual = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.writeBufferF16(k_cache_buffer, &actual);

    try metal_backend.runMatVecMoonQuantQ4KKHalf(
        backend,
        packed_matrix.bytes,
        input_buffer,
        k_cache_buffer,
        dst_prefix,
        head_count,
        head_dim,
        rope_dim,
        cols,
        position,
        freq_base,
        0,
    );
    try metal_backend.readBufferF16(k_cache_buffer, &actual);

    const expected_f32 = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected_f32);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);
    for (0..rows) |row| {
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, raw_matrix[row * row_size ..][0..row_size], cols);
        expected_f32[row] = dot(dequantized_row, &input);
    }
    applyRoPEReference(expected_f32, expected_f32, head_count, head_dim, rope_dim, position, freq_base, 0);

    var expected = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    for (0..rows) |index| expected[dst_prefix + index] = @floatCast(expected_f32[index]);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.03);
    }
}

test "metal MoonQuant q4k q-rope matches separate matvec rope reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 8;
    const rows = head_count * head_dim;
    const cols = 512;
    const position = 3;
    const freq_base: f32 = 10000;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);

    var raw_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&raw_matrix, rows, cols);
    var packed_matrix = try moon_quant.packQ4KTensor(std.testing.allocator, &raw_matrix, rows, cols);
    defer packed_matrix.deinit(std.testing.allocator);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.09375) - 0.15;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const output_buffer = try metal_backend.createScratchBuffer(backend, rows);
    defer metal_backend.destroyBuffer(output_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.runMatVecMoonQuantQ4KQRope(
        backend,
        packed_matrix.bytes,
        input_buffer,
        output_buffer,
        head_count,
        head_dim,
        rope_dim,
        cols,
        position,
        freq_base,
        0,
    );

    var actual = [_]f32{0} ** rows;
    try metal_backend.readBufferF32(output_buffer, &actual);

    const expected = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);
    for (0..rows) |row| {
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, raw_matrix[row * row_size ..][0..row_size], cols);
        expected[row] = dot(dequantized_row, &input);
    }
    applyRoPEReference(expected, expected, head_count, head_dim, rope_dim, position, freq_base, 0);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.03);
    }
}

test "metal q4k dual kv-half matches separate matvec rope and store reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 8;
    const rows = head_count * head_dim;
    const cols = 512;
    const position = 3;
    const freq_base: f32 = 10000;
    const dst_prefix = 5;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);

    var k_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    var v_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&k_matrix, rows, cols);
    fillQ4KMatrix(&v_matrix, rows, cols);
    for (&v_matrix, 0..) |*byte, index| {
        byte.* ^=
            @as(u8, @intCast((index * 7 + 11) & 0x0F)) |
            (@as(u8, @intCast((index * 5 + 3) & 0x0F)) << 4);
    }

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.09375) - 0.15;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const k_cache_buffer = try metal_backend.createByteScratchBuffer(backend, (dst_prefix + rows + 3) * @sizeOf(f16));
    defer metal_backend.destroyBuffer(k_cache_buffer);
    const v_cache_buffer = try metal_backend.createByteScratchBuffer(backend, (dst_prefix + rows + 3) * @sizeOf(f16));
    defer metal_backend.destroyBuffer(v_cache_buffer);

    var actual_k = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    var actual_v = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.writeBufferF16(k_cache_buffer, &actual_k);
    try metal_backend.writeBufferF16(v_cache_buffer, &actual_v);

    try metal_backend.runMatVecQ4KDualKvHalf(
        backend,
        &k_matrix,
        &v_matrix,
        input_buffer,
        k_cache_buffer,
        v_cache_buffer,
        dst_prefix,
        head_count,
        head_dim,
        rope_dim,
        cols,
        position,
        freq_base,
        0,
    );
    try metal_backend.readBufferF16(k_cache_buffer, &actual_k);
    try metal_backend.readBufferF16(v_cache_buffer, &actual_v);

    const expected_k_f32 = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected_k_f32);
    const expected_v_f32 = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected_v_f32);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);

    for (0..rows) |row| {
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, k_matrix[row * row_size ..][0..row_size], cols);
        expected_k_f32[row] = dot(dequantized_row, &input);
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, v_matrix[row * row_size ..][0..row_size], cols);
        expected_v_f32[row] = dot(dequantized_row, &input);
    }
    applyRoPEReference(expected_k_f32, expected_k_f32, head_count, head_dim, rope_dim, position, freq_base, 0);

    var expected_k = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    var expected_v = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    for (0..rows) |index| {
        expected_k[dst_prefix + index] = @floatCast(expected_k_f32[index]);
        expected_v[dst_prefix + index] = @floatCast(expected_v_f32[index]);
    }

    for (expected_k, actual_k) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.03);
    }
    for (expected_v, actual_v) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.03);
    }
}

test "metal MoonQuant q4k dual kv-half matches separate matvec rope and store reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 8;
    const rows = head_count * head_dim;
    const cols = 512;
    const position = 3;
    const freq_base: f32 = 10000;
    const dst_prefix = 5;
    const row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);

    var raw_k_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    var raw_v_matrix: [rows * (cols / 256) * 144]u8 = undefined;
    fillQ4KMatrix(&raw_k_matrix, rows, cols);
    fillQ4KMatrix(&raw_v_matrix, rows, cols);
    for (&raw_v_matrix, 0..) |*byte, index| {
        byte.* ^=
            @as(u8, @intCast((index * 7 + 11) & 0x0F)) |
            (@as(u8, @intCast((index * 5 + 3) & 0x0F)) << 4);
    }

    var packed_k = try moon_quant.packQ4KTensor(std.testing.allocator, &raw_k_matrix, rows, cols);
    defer packed_k.deinit(std.testing.allocator);
    var packed_v = try moon_quant.packQ4KTensor(std.testing.allocator, &raw_v_matrix, rows, cols);
    defer packed_v.deinit(std.testing.allocator);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.09375) - 0.15;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const k_cache_buffer = try metal_backend.createByteScratchBuffer(backend, (dst_prefix + rows + 3) * @sizeOf(f16));
    defer metal_backend.destroyBuffer(k_cache_buffer);
    const v_cache_buffer = try metal_backend.createByteScratchBuffer(backend, (dst_prefix + rows + 3) * @sizeOf(f16));
    defer metal_backend.destroyBuffer(v_cache_buffer);

    var actual_k = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    var actual_v = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.writeBufferF16(k_cache_buffer, &actual_k);
    try metal_backend.writeBufferF16(v_cache_buffer, &actual_v);

    try metal_backend.runMatVecMoonQuantQ4KDualKvHalf(
        backend,
        packed_k.bytes,
        packed_v.bytes,
        input_buffer,
        k_cache_buffer,
        v_cache_buffer,
        dst_prefix,
        head_count,
        head_dim,
        rope_dim,
        cols,
        position,
        freq_base,
        0,
    );
    try metal_backend.readBufferF16(k_cache_buffer, &actual_k);
    try metal_backend.readBufferF16(v_cache_buffer, &actual_v);

    const expected_k_f32 = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected_k_f32);
    const expected_v_f32 = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected_v_f32);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);

    for (0..rows) |row| {
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, raw_k_matrix[row * row_size ..][0..row_size], cols);
        expected_k_f32[row] = dot(dequantized_row, &input);
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, raw_v_matrix[row * row_size ..][0..row_size], cols);
        expected_v_f32[row] = dot(dequantized_row, &input);
    }
    applyRoPEReference(expected_k_f32, expected_k_f32, head_count, head_dim, rope_dim, position, freq_base, 0);

    var expected_k = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    var expected_v = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    for (0..rows) |index| {
        expected_k[dst_prefix + index] = @floatCast(expected_k_f32[index]);
        expected_v[dst_prefix + index] = @floatCast(expected_v_f32[index]);
    }

    for (expected_k, actual_k) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.03);
    }
    for (expected_v, actual_v) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.03);
    }
}

test "metal q4k/q6k dual kv-half matches separate matvec rope and store reference" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 8;
    const rows = head_count * head_dim;
    const cols = 512;
    const position = 3;
    const freq_base: f32 = 10000;
    const dst_prefix = 5;

    const q4_fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q4_k);
    defer std.testing.allocator.free(q4_fixture);
    const q6_fixture = try llama_fixture.makeLlamaBenchmarkFixture(std.testing.allocator, .q6_k);
    defer std.testing.allocator.free(q6_fixture);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try llama_fixture.writeFixtureFile(tmp.dir, "mixed-k-q4.gguf", q4_fixture);
    try llama_fixture.writeFixtureFile(tmp.dir, "mixed-v-q6.gguf", q6_fixture);
    const q4_path = try tmp.dir.realpathAlloc(std.testing.allocator, "mixed-k-q4.gguf");
    defer std.testing.allocator.free(q4_path);
    const q6_path = try tmp.dir.realpathAlloc(std.testing.allocator, "mixed-v-q6.gguf");
    defer std.testing.allocator.free(q6_path);

    var q4_model = try llama_cpu.loadModel(std.testing.allocator, q4_path);
    defer q4_model.deinit(std.testing.allocator);
    var q6_model = try llama_cpu.loadModel(std.testing.allocator, q6_path);
    defer q6_model.deinit(std.testing.allocator);

    const k_tensor = q4_model.layers[0].attn_k;
    const v_tensor = q6_model.layers[0].attn_v;
    const k_row_size = try llama_cpu.tensorRowByteSize(.q4_k, cols);
    const v_row_size = try llama_cpu.tensorRowByteSize(.q6_k, cols);
    const k_matrix = try llama_cpu.tensorBytes(&q4_model, k_tensor);
    const v_matrix = try llama_cpu.tensorBytes(&q6_model, v_tensor);

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index % 19)) - 9)) * 0.09375) - 0.15;
    }

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input_buffer = try metal_backend.createScratchBuffer(backend, cols);
    defer metal_backend.destroyBuffer(input_buffer);
    const k_cache_buffer = try metal_backend.createByteScratchBuffer(backend, (dst_prefix + rows + 3) * @sizeOf(f16));
    defer metal_backend.destroyBuffer(k_cache_buffer);
    const v_cache_buffer = try metal_backend.createByteScratchBuffer(backend, (dst_prefix + rows + 3) * @sizeOf(f16));
    defer metal_backend.destroyBuffer(v_cache_buffer);

    var actual_k = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    var actual_v = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.writeBufferF16(k_cache_buffer, &actual_k);
    try metal_backend.writeBufferF16(v_cache_buffer, &actual_v);

    try metal_backend.runMatVecQ4KQ6KDualKvHalf(
        backend,
        k_matrix[0 .. rows * k_row_size],
        v_matrix[0 .. rows * v_row_size],
        input_buffer,
        k_cache_buffer,
        v_cache_buffer,
        dst_prefix,
        head_count,
        head_dim,
        rope_dim,
        cols,
        position,
        freq_base,
        0,
    );
    try metal_backend.readBufferF16(k_cache_buffer, &actual_k);
    try metal_backend.readBufferF16(v_cache_buffer, &actual_v);

    const expected_k_f32 = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected_k_f32);
    const expected_v_f32 = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected_v_f32);
    const dequantized_row = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(dequantized_row);

    for (0..rows) |row| {
        try llama_cpu.dequantizeRow(dequantized_row, .q4_k, k_matrix[row * k_row_size ..][0..k_row_size], cols);
        expected_k_f32[row] = dot(dequantized_row, &input);
        try llama_cpu.dequantizeRow(dequantized_row, .q6_k, v_matrix[row * v_row_size ..][0..v_row_size], cols);
        expected_v_f32[row] = dot(dequantized_row, &input);
    }
    applyRoPEReference(expected_k_f32, expected_k_f32, head_count, head_dim, rope_dim, position, freq_base, 0);

    var expected_k = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    var expected_v = [_]f16{@as(f16, @floatCast(-9.0))} ** (dst_prefix + rows + 3);
    for (0..rows) |index| {
        expected_k[dst_prefix + index] = @floatCast(expected_k_f32[index]);
        expected_v[dst_prefix + index] = @floatCast(expected_v_f32[index]);
    }

    for (expected_k, actual_k) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.03);
    }
    for (expected_v, actual_v) |want, got| {
        try std.testing.expectApproxEqAbs(@as(f32, want), @as(f32, got), 0.05);
    }
}

test "metal rope at offset rotates kv slice in place" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const head_count = 2;
    const head_dim = 8;
    const rope_dim = 6;
    const position = 3;
    const freq_base: f32 = 10000;
    const dst_prefix = 5;

    var expected = [_]f32{-9.0} ** (dst_prefix + head_count * head_dim + 3);
    var actual = expected;
    for (actual[dst_prefix .. dst_prefix + head_count * head_dim], 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt(@as(i32, @intCast(index)) - 7)) * 0.125) + 0.05;
    }
    @memcpy(expected[dst_prefix .. dst_prefix + head_count * head_dim], actual[dst_prefix .. dst_prefix + head_count * head_dim]);
    applyRoPEReference(
        expected[dst_prefix .. dst_prefix + head_count * head_dim],
        expected[dst_prefix .. dst_prefix + head_count * head_dim],
        head_count,
        head_dim,
        rope_dim,
        position,
        freq_base,
        0,
    );

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const buffer = try metal_backend.createScratchBuffer(backend, actual.len);
    defer metal_backend.destroyBuffer(buffer);

    try metal_backend.writeBufferF32(buffer, &actual);
    try metal_backend.applyRoPEAtOffset(
        backend,
        buffer,
        dst_prefix * @sizeOf(f32),
        head_count,
        head_dim,
        rope_dim,
        position,
        freq_base,
        0,
    );
    try metal_backend.readBufferF32(buffer, &actual);

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0005);
    }
}

test "metal argmax returns best token index" {
    if (!metal_backend.buildEnabled()) return error.SkipZigTest;
    const supported = try metal_backend.canInitialize(std.testing.allocator);
    if (!supported) return error.SkipZigTest;

    const backend = try metal_backend.create(std.testing.allocator);
    defer backend.deinit(std.testing.allocator);

    const input = [_]f32{ -1.0, 4.0, 2.5, 4.5, 3.0 };
    const input_buffer = try metal_backend.createScratchBuffer(backend, input.len);
    defer metal_backend.destroyBuffer(input_buffer);
    const token_buffer = try metal_backend.createScratchBuffer(backend, 1);
    defer metal_backend.destroyBuffer(token_buffer);

    try metal_backend.writeBufferF32(input_buffer, &input);
    try metal_backend.argmax(backend, input_buffer, token_buffer, input.len);

    var token: [1]u32 = .{0};
    try metal_backend.readBufferU32(token_buffer, &token);
    try std.testing.expectEqual(@as(u32, 3), token[0]);
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

fn fillQ6KMatrix(buffer: []u8, rows: usize, cols: usize) void {
    const row_size = cols / 256 * 210;
    for (0..rows) |row| {
        for (0..cols / 256) |block| {
            const base = row * row_size + block * 210;
            @memset(buffer[base .. base + 210], 0);
            @memset(buffer[base + 128 .. base + 192], 0xAA);
            @memset(buffer[base + 192 .. base + 208], 1);
            writeHalf(buffer[base + 208 ..][0..2], 1);
            for (0..256) |index| {
                const nibble: u8 = @intCast((row + block + index) & 1);
                setQ6KNibble(buffer[base .. base + 210], index, nibble);
            }
        }
    }
}

fn setQ6KNibble(block: []u8, index: usize, nibble: u8) void {
    if (index < 32) {
        block[index] = (block[index] & 0xF0) | nibble;
    } else if (index < 64) {
        const offset = 32 + (index - 32);
        block[offset] = (block[offset] & 0xF0) | nibble;
    } else if (index < 96) {
        const offset = index - 64;
        block[offset] = (block[offset] & 0x0F) | (nibble << 4);
    } else if (index < 128) {
        const offset = 32 + (index - 96);
        block[offset] = (block[offset] & 0x0F) | (nibble << 4);
    } else if (index < 160) {
        const offset = 64 + (index - 128);
        block[offset] = (block[offset] & 0xF0) | nibble;
    } else if (index < 192) {
        const offset = 96 + (index - 160);
        block[offset] = (block[offset] & 0xF0) | nibble;
    } else if (index < 224) {
        const offset = 64 + (index - 192);
        block[offset] = (block[offset] & 0x0F) | (nibble << 4);
    } else {
        const offset = 96 + (index - 224);
        block[offset] = (block[offset] & 0x0F) | (nibble << 4);
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

fn applyRoPEReference(values: []f32, src: []const f32, head_count: usize, head_dim: usize, rope_dim: usize, position: usize, freq_base: f32, rope_style: u32) void {
    @memcpy(values, src);
    const n_rot = @min(rope_dim, head_dim);
    const pos_f32 = @as(f32, @floatFromInt(position));
    for (0..head_count) |head_index| {
        const head = values[head_index * head_dim ..][0..head_dim];
        if (rope_style == 0) {
            var pair: usize = 0;
            while (pair + 1 < n_rot) : (pair += 2) {
                const exponent = @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(n_rot));
                const theta = pos_f32 / std.math.pow(f32, freq_base, exponent);
                const cos_theta = @cos(theta);
                const sin_theta = @sin(theta);
                const x0 = src[head_index * head_dim + pair];
                const x1 = src[head_index * head_dim + pair + 1];
                head[pair] = x0 * cos_theta - x1 * sin_theta;
                head[pair + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        } else {
            const half_rot = n_rot / 2;
            for (0..half_rot) |i| {
                const exponent = @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(n_rot));
                const theta = pos_f32 / std.math.pow(f32, freq_base, exponent);
                const cos_theta = @cos(theta);
                const sin_theta = @sin(theta);
                const x0 = src[head_index * head_dim + i];
                const x1 = src[head_index * head_dim + i + half_rot];
                head[i] = x0 * cos_theta - x1 * sin_theta;
                head[i + half_rot] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}
