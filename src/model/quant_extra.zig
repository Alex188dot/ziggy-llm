const std = @import("std");

const qk_k: usize = 256;
const q3_k_block_size: usize = 110;
const iq3_xxs_block_size: usize = 98;
const iq4_xs_block_size: usize = 136;

const kvalues_iq4nl = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

const iq3xxs_grid = [256]u32{
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

pub fn dequantizeRowQ3K(out: []f32, row: []const u8, row_len: usize) !void {
    if (row_len % qk_k != 0 or row.len != (row_len / qk_k) * q3_k_block_size) return error.InvalidTensorMetadata;

    var out_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += q3_k_block_size) {
        const block = row[block_index .. block_index + q3_k_block_size];
        const hmask = block[0..32];
        const qs = block[32..96];
        const scales_raw = block[96..108];
        const d_all = readF16AsF32(block[108..110]);
        const scales = expandQ3KScales(scales_raw);

        var q_offset: usize = 0;
        var mask: u8 = 1;
        var scale_index: usize = 0;
        for (0..2) |_| {
            var shift: u3 = 0;
            for (0..4) |_| {
                const dl_0 = d_all * @as(f32, @floatFromInt(scales[scale_index]));
                scale_index += 1;
                for (0..16) |lane| {
                    out[out_offset] = dl_0 * @as(f32, @floatFromInt(decodeQ3Value(qs[q_offset + lane], shift, hmask[lane], mask)));
                    out_offset += 1;
                }

                const dl_1 = d_all * @as(f32, @floatFromInt(scales[scale_index]));
                scale_index += 1;
                for (0..16) |lane| {
                    out[out_offset] = dl_1 * @as(f32, @floatFromInt(decodeQ3Value(qs[q_offset + 16 + lane], shift, hmask[16 + lane], mask)));
                    out_offset += 1;
                }

                q_offset += 32;
                shift += 2;
                mask <<= 1;
            }
        }
    }
}

pub fn dotQ3KRow(row: []const u8, row_len: usize, input: []const f32) !f32 {
    if (row_len % qk_k != 0 or row.len != (row_len / qk_k) * q3_k_block_size) return error.InvalidTensorMetadata;

    var sum: f32 = 0;
    var input_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += q3_k_block_size) {
        const block = row[block_index .. block_index + q3_k_block_size];
        const hmask = block[0..32];
        const qs = block[32..96];
        const scales = expandQ3KScales(block[96..108]);
        const d_all = readF16AsF32(block[108..110]);

        var q_offset: usize = 0;
        var mask: u8 = 1;
        var scale_index: usize = 0;
        for (0..2) |_| {
            var shift: u3 = 0;
            for (0..4) |_| {
                const scale_0 = d_all * @as(f32, @floatFromInt(scales[scale_index]));
                scale_index += 1;
                for (0..16) |lane| {
                    sum = @mulAdd(f32, scale_0 * @as(f32, @floatFromInt(decodeQ3Value(qs[q_offset + lane], shift, hmask[lane], mask))), input[input_offset], sum);
                    input_offset += 1;
                }

                const scale_1 = d_all * @as(f32, @floatFromInt(scales[scale_index]));
                scale_index += 1;
                for (0..16) |lane| {
                    sum = @mulAdd(f32, scale_1 * @as(f32, @floatFromInt(decodeQ3Value(qs[q_offset + 16 + lane], shift, hmask[16 + lane], mask))), input[input_offset], sum);
                    input_offset += 1;
                }

                q_offset += 32;
                shift += 2;
                mask <<= 1;
            }
        }
    }
    return sum;
}

pub fn dequantizeRowIQ3XXS(out: []f32, row: []const u8, row_len: usize) !void {
    if (row_len % qk_k != 0 or row.len != (row_len / qk_k) * iq3_xxs_block_size) return error.InvalidTensorMetadata;

    var out_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += iq3_xxs_block_size) {
        const block = row[block_index .. block_index + iq3_xxs_block_size];
        const d = readF16AsF32(block[0..2]);
        var qs: []const u8 = block[2 .. 2 + qk_k / 4];
        const scales_and_signs = block[2 + qk_k / 4 ..][0 .. qk_k / 8];

        for (0..qk_k / 32) |ib32| {
            const aux32 = std.mem.readInt(u32, scales_and_signs[ib32 * 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32 >> 28))) * 0.5;
            for (0..4) |l| {
                const signs = iq2Signs(@truncate((aux32 >> @as(u5, @intCast(7 * l))) & 0x7f));
                const grid1 = iq3xxs_grid[qs[2 * l]];
                const grid2 = iq3xxs_grid[qs[2 * l + 1]];
                for (0..4) |j| {
                    out[out_offset + j + 0] = db * @as(f32, @floatFromInt(gridByte(grid1, j))) * iqSign(signs, j + 0);
                    out[out_offset + j + 4] = db * @as(f32, @floatFromInt(gridByte(grid2, j))) * iqSign(signs, j + 4);
                }
                out_offset += 8;
            }
            qs = qs[8..];
        }
    }
}

pub fn dotIQ3XXSRow(row: []const u8, row_len: usize, input: []const f32) !f32 {
    if (row_len % qk_k != 0 or row.len != (row_len / qk_k) * iq3_xxs_block_size) return error.InvalidTensorMetadata;

    var sum: f32 = 0;
    var input_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += iq3_xxs_block_size) {
        const block = row[block_index .. block_index + iq3_xxs_block_size];
        const d = readF16AsF32(block[0..2]);
        var qs: []const u8 = block[2 .. 2 + qk_k / 4];
        const scales_and_signs = block[2 + qk_k / 4 ..][0 .. qk_k / 8];

        for (0..qk_k / 32) |ib32| {
            const aux32 = std.mem.readInt(u32, scales_and_signs[ib32 * 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32 >> 28))) * 0.5;
            for (0..4) |l| {
                const signs = iq2Signs(@truncate((aux32 >> @as(u5, @intCast(7 * l))) & 0x7f));
                const grid1 = iq3xxs_grid[qs[2 * l]];
                const grid2 = iq3xxs_grid[qs[2 * l + 1]];
                for (0..4) |j| {
                    sum = @mulAdd(f32, db * @as(f32, @floatFromInt(gridByte(grid1, j))) * iqSign(signs, j + 0), input[input_offset], sum);
                    input_offset += 1;
                }
                for (0..4) |j| {
                    sum = @mulAdd(f32, db * @as(f32, @floatFromInt(gridByte(grid2, j))) * iqSign(signs, j + 4), input[input_offset], sum);
                    input_offset += 1;
                }
            }
            qs = qs[8..];
        }
    }
    return sum;
}

pub fn dequantizeRowIQ4XS(out: []f32, row: []const u8, row_len: usize) !void {
    if (row_len % qk_k != 0 or row.len != (row_len / qk_k) * iq4_xs_block_size) return error.InvalidTensorMetadata;

    var out_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += iq4_xs_block_size) {
        const block = row[block_index .. block_index + iq4_xs_block_size];
        const d = readF16AsF32(block[0..2]);
        const scales_h = std.mem.readInt(u16, block[2..4], .little);
        const scales_l = block[4..8];
        var qs: []const u8 = block[8..136];

        for (0..qk_k / 32) |ib| {
            const low = (scales_l[ib / 2] >> @as(u3, @intCast(4 * (ib % 2)))) & 0x0f;
            const high = (scales_h >> @as(u4, @intCast(2 * ib))) & 0x03;
            const dl = d * @as(f32, @floatFromInt(@as(i32, low | (high << 4)) - 32));
            for (0..16) |j| {
                out[out_offset + j + 0] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] & 0x0f]));
                out[out_offset + j + 16] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] >> 4]));
            }
            out_offset += 32;
            qs = qs[16..];
        }
    }
}

pub fn dotIQ4XSRow(row: []const u8, row_len: usize, input: []const f32) !f32 {
    if (row_len % qk_k != 0 or row.len != (row_len / qk_k) * iq4_xs_block_size) return error.InvalidTensorMetadata;

    var sum: f32 = 0;
    var input_offset: usize = 0;
    var block_index: usize = 0;
    while (block_index < row.len) : (block_index += iq4_xs_block_size) {
        const block = row[block_index .. block_index + iq4_xs_block_size];
        const d = readF16AsF32(block[0..2]);
        const scales_h = std.mem.readInt(u16, block[2..4], .little);
        const scales_l = block[4..8];
        var qs: []const u8 = block[8..136];

        for (0..qk_k / 32) |ib| {
            const low = (scales_l[ib / 2] >> @as(u3, @intCast(4 * (ib % 2)))) & 0x0f;
            const high = (scales_h >> @as(u4, @intCast(2 * ib))) & 0x03;
            const dl = d * @as(f32, @floatFromInt(@as(i32, low | (high << 4)) - 32));
            for (0..16) |j| {
                sum = @mulAdd(f32, dl * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] & 0x0f])), input[input_offset], sum);
                input_offset += 1;
            }
            for (0..16) |j| {
                sum = @mulAdd(f32, dl * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] >> 4])), input[input_offset], sum);
                input_offset += 1;
            }
            qs = qs[16..];
        }
    }
    return sum;
}

fn expandQ3KScales(scales_raw: []const u8) [16]i32 {
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;

    var aux = [_]u32{
        std.mem.readInt(u32, scales_raw[0..4], .little),
        std.mem.readInt(u32, scales_raw[4..8], .little),
        std.mem.readInt(u32, scales_raw[8..12], .little),
        0,
    };
    const tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    var out: [16]i32 = undefined;
    var byte_index: usize = 0;
    for (aux) |word| {
        for (0..4) |lane| {
            const byte = @as(u8, @truncate(word >> @as(u5, @intCast(8 * lane))));
            out[byte_index] = @as(i32, byte) - 32;
            byte_index += 1;
        }
    }
    return out;
}

fn decodeQ3Value(q: u8, shift: u3, high_mask: u8, active_mask: u8) i32 {
    const low = (q >> shift) & 0x03;
    return @as(i32, low) - (if ((high_mask & active_mask) != 0) @as(i32, 0) else @as(i32, 4));
}

fn iq2Signs(code: u7) u8 {
    const raw = @as(u8, code);
    return raw | (if ((@popCount(raw) & 1) == 1) @as(u8, 0x80) else @as(u8, 0));
}

fn iqSign(signs: u8, bit_index: usize) f32 {
    return if ((signs & (@as(u8, 1) << @as(u3, @intCast(bit_index)))) != 0) -1.0 else 1.0;
}

fn gridByte(grid: u32, index: usize) u8 {
    return @truncate(grid >> @as(u5, @intCast(index * 8)));
}

fn readF16AsF32(bytes: []const u8) f32 {
    const raw = std.mem.readInt(u16, bytes[0..2], .little);
    return @as(f32, @floatCast(@as(f16, @bitCast(raw))));
}
