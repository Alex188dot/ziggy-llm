
#include <metal_stdlib>
using namespace metal;

constant uint ZIGGY_MAX_ROW_SIMDGROUPS = 8;
constant uint ZIGGY_MAX_NORM_SIMDGROUPS = 8;

kernel void matvec_f32(
    device const float *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    threadgroup float partial_sums[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum = 0.0f;

    uint col = lane * 4;
    while (col + 3 < cols) {
        const float4 input_values = float4(
            input[col + 0],
            input[col + 1],
            input[col + 2],
            input[col + 3]
        );
        const float4 matrix_values = float4(
            matrix[row + (col + 0) * rows],
            matrix[row + (col + 1) * rows],
            matrix[row + (col + 2) * rows],
            matrix[row + (col + 3) * rows]
        );
        local_sum += dot(matrix_values, input_values);
        col += threads_per_group * 4;
    }

    for (; col < cols; col += threads_per_group) {
        local_sum += matrix[row + col * rows] * input[col];
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

inline float read_half_le(device const uchar *bytes, uint offset) {
    const ushort raw = ushort(bytes[offset]) | (ushort(bytes[offset + 1]) << 8);
    return float(as_type<half>(raw));
}

inline uchar get_scale_k4(device const uchar *scales, uint index) {
    return index < 4
        ? (scales[index] & 63)
        : ((scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4));
}

inline uchar get_min_k4(device const uchar *scales, uint index) {
    return index < 4
        ? (scales[index + 4] & 63)
        : ((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
}

constant uint ZIGGY_Q4K_VALUES_PER_BLOCK = 256;
constant uint ZIGGY_Q4K_BYTES_PER_BLOCK = 144;
constant uint ZIGGY_Q4K_GROUPS_PER_BLOCK = 4;
constant uint ZIGGY_Q4K_PACKED_BYTES_PER_GROUP = 32;
constant uint ZIGGY_Q4K_VALUES_PER_GROUP = 64;
constant uint ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP = 8;
constant uint ZIGGY_MAX_Q4K_SIMDGROUPS = 8;
constant uint ZIGGY_Q5K_VALUES_PER_BLOCK = 256;
constant uint ZIGGY_Q5K_BYTES_PER_BLOCK = 176;
constant uint ZIGGY_Q5K_GROUPS_PER_BLOCK = 4;
constant uint ZIGGY_Q5K_PACKED_BYTES_PER_GROUP = 32;
constant uint ZIGGY_Q5K_VALUES_PER_GROUP = 64;
constant uint ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP = 8;
constant uint ZIGGY_MAX_Q5K_SIMDGROUPS = 8;
constant uint ZIGGY_Q6K_VALUES_PER_BLOCK = 256;
constant uint ZIGGY_Q6K_BYTES_PER_BLOCK = 210;
constant uint ZIGGY_Q6K_CHUNKS_PER_BLOCK = 64;
constant uint ZIGGY_Q6K_VALUES_PER_VECTOR_CHUNK = 4;
constant uint ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF = 8;
constant uint ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK = 16;
constant uint ZIGGY_Q8_0_VALUES_PER_BLOCK = 32;
constant uint ZIGGY_Q8_0_BYTES_PER_BLOCK = 34;
constant uint ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK = 160;
constant uint ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0 = 2048;
constant uint ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1 = 5632;
constant uint ZIGGY_MAX_HEAD_DIM = 256;
constant uint ZIGGY_IQ3_XXS_VALUES_PER_BLOCK = 256;
constant uint ZIGGY_IQ3_XXS_BYTES_PER_BLOCK = 98;
constant uint ZIGGY_IQ4_XS_VALUES_PER_BLOCK = 256;
constant uint ZIGGY_IQ4_XS_BYTES_PER_BLOCK = 136;

inline float4 ziggy_load_float4(device const float *values, uint offset) {
    return float4(
        values[offset + 0],
        values[offset + 1],
        values[offset + 2],
        values[offset + 3]
    );
}

inline uchar4 ziggy_load_uchar4(device const uchar *values) {
    return uchar4(values[0], values[1], values[2], values[3]);
}

inline float4 ziggy_i8x4_to_float4(uchar4 values) {
    return float4(
        float(as_type<char>(values[0])),
        float(as_type<char>(values[1])),
        float(as_type<char>(values[2])),
        float(as_type<char>(values[3]))
    );
}

inline float4 ziggy_q6k_decode_low_nibbles(uchar4 ql_values, uchar4 qh_values, uint qh_shift) {
    const uint4 low = uint4(ql_values) & uint4(0x0Fu);
    const uint4 high = ((uint4(qh_values) >> qh_shift) & uint4(0x03u)) << 4;
    return float4(low | high) - 32.0f;
}

inline float4 ziggy_q6k_decode_high_nibbles(uchar4 ql_values, uchar4 qh_values, uint qh_shift) {
    const uint4 low = uint4(ql_values) >> 4;
    const uint4 high = ((uint4(qh_values) >> qh_shift) & uint4(0x03u)) << 4;
    return float4(low | high) - 32.0f;
}

inline float ziggy_q6k_chunk_dot(
    device const uchar *block,
    device const float *input,
    uint block_half,
    uint l,
    uint input_offset
) {
    const device uchar *ql = block + block_half * 64;
    const device uchar *qh = block + 128 + block_half * 32;
    const device uchar *scales = block + 192 + block_half * 8;
    const float d = read_half_le(block, 208);
    const uint scale_index = l / 16;
    const float s0 = d * float(as_type<char>(scales[scale_index + 0]));
    const float s2 = d * float(as_type<char>(scales[scale_index + 2]));
    const float s4 = d * float(as_type<char>(scales[scale_index + 4]));
    const float s6 = d * float(as_type<char>(scales[scale_index + 6]));
    const uchar4 qh_values = ziggy_load_uchar4(qh + l);
    const uchar4 ql_low_values = ziggy_load_uchar4(ql + l);
    const uchar4 ql_high_values = ziggy_load_uchar4(ql + l + 32);

    const float4 q0 = ziggy_q6k_decode_low_nibbles(ql_low_values, qh_values, 0) * s0;
    const float4 q1 = ziggy_q6k_decode_low_nibbles(ql_high_values, qh_values, 2) * s2;
    const float4 q2 = ziggy_q6k_decode_high_nibbles(ql_low_values, qh_values, 4) * s4;
    const float4 q3 = ziggy_q6k_decode_high_nibbles(ql_high_values, qh_values, 6) * s6;

    return dot(q0, ziggy_load_float4(input, input_offset + 0))
        + dot(q1, ziggy_load_float4(input, input_offset + 32))
        + dot(q2, ziggy_load_float4(input, input_offset + 64))
        + dot(q3, ziggy_load_float4(input, input_offset + 96));
}

inline float ziggy_q8_0_block_dot(
    device const uchar *block,
    device const float *input,
    uint input_offset
) {
    const float d = read_half_le(block, 0);
    const device uchar *q = block + 2;
    float block_sum = 0.0f;

    for (uint value_index = 0; value_index < ZIGGY_Q8_0_VALUES_PER_BLOCK; value_index += 4) {
        block_sum += dot(
            ziggy_i8x4_to_float4(ziggy_load_uchar4(q + value_index)),
            ziggy_load_float4(input, input_offset + value_index)
        );
    }

    return d * block_sum;
}

constant char ZIGGY_IQ4_NL_VALUES[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
};

constant uint ZIGGY_IQ3_XXS_GRID[256] = {
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

inline uchar ziggy_iq2_signs(uchar code) {
    return code | ((((popcount(code) & 1) == 1) ? uchar(0x80) : uchar(0x00)));
}

inline float ziggy_iq_sign(uchar signs, uint bit_index) {
    return ((signs & (uchar(1) << bit_index)) != 0) ? -1.0f : 1.0f;
}

inline uchar ziggy_grid_byte(uint grid, uint index) {
    return uchar((grid >> (index * 8)) & 0xff);
}

kernel void store_kv_half(
    device const float *src [[buffer(0)]],
    device half *dst [[buffer(1)]],
    constant uint &dst_offset [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    dst[dst_offset + index] = half(src[index]);
}

#define ZIGGY_DENSE_MATVEC_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const float *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    threadgroup float partial_sums[ZIGGY_MAX_ROW_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    uint col = lane * 4; \
    while (col + 3 < effective_cols) { \
        const float4 input_values = float4( \
            input[col + 0], \
            input[col + 1], \
            input[col + 2], \
            input[col + 3] \
        ); \
        const float4 matrix_values = float4( \
            matrix[row + (col + 0) * rows], \
            matrix[row + (col + 1) * rows], \
            matrix[row + (col + 2) * rows], \
            matrix[row + (col + 3) * rows] \
        ); \
        local_sum += dot(matrix_values, input_values); \
        col += threads_per_group * 4; \
    } \
    for (; col < effective_cols; col += threads_per_group) { \
        local_sum += matrix[row + col * rows] * input[col]; \
    } \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        output[row] += sum; \
    } \
}

ZIGGY_DENSE_MATVEC_ADD_KERNEL(matvec_add_f32, 0)
ZIGGY_DENSE_MATVEC_ADD_KERNEL(matvec_add_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_DENSE_MATVEC_ADD_KERNEL(matvec_add_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_DENSE_MATVEC_ADD_KERNEL

kernel void matvec_q4k_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / ZIGGY_Q4K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP;
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) {
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP;
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP;
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK;
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK;
        const uint q_offset = chunk_in_group * 4;
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset;

        const device uchar *block = row_bytes + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK;
        const float d = read_half_le(block, 0);
        const float dmin = read_half_le(block, 2);
        const device uchar *scales = block + 4;
        const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset;
        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]);
        const uchar4 low_q = packed & uchar4(0x0F);
        const uchar4 high_q = packed >> 4;
        const float4 input_low = float4(
            input[input_offset + 0],
            input[input_offset + 1],
            input[input_offset + 2],
            input[input_offset + 3]
        );
        const float4 input_high = float4(
            input[input_offset + 32 + 0],
            input[input_offset + 32 + 1],
            input[input_offset + 32 + 2],
            input[input_offset + 32 + 3]
        );

        const uint scale_index = group * 2;
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0));
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0));
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1));
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1));
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low);
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high);
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

#define ZIGGY_Q4K_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes = matrix + row * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device uchar *block = row_bytes + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
        const float d = read_half_le(block, 0); \
        const float dmin = read_half_le(block, 2); \
        const device uchar *scales = block + 4; \
        const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
        const uchar4 low_q = packed & uchar4(0x0F); \
        const uchar4 high_q = packed >> 4; \
        const float4 input_low = float4( \
            input[input_offset + 0], \
            input[input_offset + 1], \
            input[input_offset + 2], \
            input[input_offset + 3] \
        ); \
        const float4 input_high = float4( \
            input[input_offset + 32 + 0], \
            input[input_offset + 32 + 1], \
            input[input_offset + 32 + 2], \
            input[input_offset + 32 + 3] \
        ); \
        const uint scale_index = group * 2; \
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low); \
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high); \
    } \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        output[row] += sum; \
    } \
}

ZIGGY_Q4K_ADD_KERNEL(matvec_q4k_add_f32, 0)
ZIGGY_Q4K_ADD_KERNEL(matvec_q4k_add_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_ADD_KERNEL(matvec_q4k_add_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_Q4K_ADD_KERNEL

kernel void matvec_q5k_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / ZIGGY_Q5K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q5K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q5K_GROUPS_PER_BLOCK * ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP;
    threadgroup float partial_sums[ZIGGY_MAX_Q5K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) {
        const uint block_group_index = chunk_index / ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP;
        const uint chunk_in_group = chunk_index % ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP;
        const uint block_index = block_group_index / ZIGGY_Q5K_GROUPS_PER_BLOCK;
        const uint group = block_group_index % ZIGGY_Q5K_GROUPS_PER_BLOCK;
        const uint q_offset = chunk_in_group * 4;
        const uint input_offset = block_index * ZIGGY_Q5K_VALUES_PER_BLOCK + group * ZIGGY_Q5K_VALUES_PER_GROUP + q_offset;

        const device uchar *block = row_bytes + block_index * ZIGGY_Q5K_BYTES_PER_BLOCK;
        const float d = read_half_le(block, 0);
        const float dmin = read_half_le(block, 2);
        const device uchar *scales = block + 4;
        const device uchar *qh = block + 16 + q_offset;
        const device uchar *q = block + 48 + group * ZIGGY_Q5K_PACKED_BYTES_PER_GROUP + q_offset;

        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]);
        const uchar4 qh_packed = uchar4(qh[0], qh[1], qh[2], qh[3]);
        const uchar4 low_q_raw = packed & uchar4(0x0F);
        const uchar4 high_q_raw = packed >> 4;

        const uint low_mask_u = 1u << (group * 2);
        const uint high_mask_u = 2u << (group * 2);
        const uint4 qh_bits = uint4(qh_packed);
        const uint4 low_bit_set = select(uint4(0), uint4(16), (qh_bits & uint4(low_mask_u)) != uint4(0));
        const uint4 high_bit_set = select(uint4(0), uint4(16), (qh_bits & uint4(high_mask_u)) != uint4(0));
        const uchar4 low_q = low_q_raw | uchar4(low_bit_set);
        const uchar4 high_q = high_q_raw | uchar4(high_bit_set);

        const float4 input_low = float4(
            input[input_offset + 0],
            input[input_offset + 1],
            input[input_offset + 2],
            input[input_offset + 3]
        );
        const float4 input_high = float4(
            input[input_offset + 32 + 0],
            input[input_offset + 32 + 1],
            input[input_offset + 32 + 2],
            input[input_offset + 32 + 3]
        );

        const uint scale_index = group * 2;
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0));
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0));
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1));
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1));
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low);
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high);
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

#define ZIGGY_Q5K_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q5K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q5K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes = matrix + row * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q5K_GROUPS_PER_BLOCK * ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_sums[ZIGGY_MAX_Q5K_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q5K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q5K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q5K_VALUES_PER_BLOCK + group * ZIGGY_Q5K_VALUES_PER_GROUP + q_offset; \
        const device uchar *block = row_bytes + block_index * ZIGGY_Q5K_BYTES_PER_BLOCK; \
        const float d = read_half_le(block, 0); \
        const float dmin = read_half_le(block, 2); \
        const device uchar *scales = block + 4; \
        const device uchar *qh = block + 16 + q_offset; \
        const device uchar *q = block + 48 + group * ZIGGY_Q5K_PACKED_BYTES_PER_GROUP + q_offset; \
        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
        const uchar4 qh_packed = uchar4(qh[0], qh[1], qh[2], qh[3]); \
        const uchar4 low_q_raw = packed & uchar4(0x0F); \
        const uchar4 high_q_raw = packed >> 4; \
        const uint low_mask_u = 1u << (group * 2); \
        const uint high_mask_u = 2u << (group * 2); \
        const uint4 qh_bits = uint4(qh_packed); \
        const uint4 low_bit_set = select(uint4(0), uint4(16), (qh_bits & uint4(low_mask_u)) != uint4(0)); \
        const uint4 high_bit_set = select(uint4(0), uint4(16), (qh_bits & uint4(high_mask_u)) != uint4(0)); \
        const uchar4 low_q = low_q_raw | uchar4(low_bit_set); \
        const uchar4 high_q = high_q_raw | uchar4(high_bit_set); \
        const float4 input_low = float4( \
            input[input_offset + 0], \
            input[input_offset + 1], \
            input[input_offset + 2], \
            input[input_offset + 3] \
        ); \
        const float4 input_high = float4( \
            input[input_offset + 32 + 0], \
            input[input_offset + 32 + 1], \
            input[input_offset + 32 + 2], \
            input[input_offset + 32 + 3] \
        ); \
        const uint scale_index = group * 2; \
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low); \
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high); \
    } \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        output[row] += sum; \
    } \
}

ZIGGY_Q5K_ADD_KERNEL(matvec_q5k_add_f32, 0)

#undef ZIGGY_Q5K_ADD_KERNEL

kernel void matvec_q6k_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / ZIGGY_Q6K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q6K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    const uint chunks_per_row = blocks_per_row * ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint chunk_index = lane; chunk_index < chunks_per_row; chunk_index += threads_per_group) {
        const uint block_index = chunk_index / ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;
        const uint block_chunk = chunk_index % ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;
        const uint block_half = block_chunk / ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF;
        const uint l = (block_chunk % ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF) * ZIGGY_Q6K_VALUES_PER_VECTOR_CHUNK;
        const device uchar *block = row_bytes + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l;
        local_sum += ziggy_q6k_chunk_dot(block, input, block_half, l, input_offset);
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

inline uint ziggy_ordered_float_bits(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
}

kernel void matvec_q6k_argmax_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device atomic_uint *output_state [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / ZIGGY_Q6K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q6K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    const uint chunks_per_row = blocks_per_row * ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint chunk_index = lane; chunk_index < chunks_per_row; chunk_index += threads_per_group) {
        const uint block_index = chunk_index / ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;
        const uint block_chunk = chunk_index % ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;
        const uint block_half = block_chunk / ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF;
        const uint l = (block_chunk % ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF) * ZIGGY_Q6K_VALUES_PER_VECTOR_CHUNK;
        const device uchar *block = row_bytes + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l;
        local_sum += ziggy_q6k_chunk_dot(block, input, block_half, l, input_offset);
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }

        device atomic_uint *best_value = output_state;
        device atomic_uint *best_token = output_state + 1;
        uint observed = atomic_load_explicit(best_value, memory_order_relaxed);
        const uint ordered = ziggy_ordered_float_bits(sum);
        while (ordered > observed) {
            if (atomic_compare_exchange_weak_explicit(
                    best_value,
                    &observed,
                    ordered,
                    memory_order_relaxed,
                    memory_order_relaxed)) {
                atomic_store_explicit(best_token, row, memory_order_relaxed);
                break;
            }
        }
    }
}

kernel void matvec_q8_0_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / ZIGGY_Q8_0_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q8_0_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint block_index = lane; block_index < blocks_per_row; block_index += threads_per_group) {
        const device uchar *block = row_bytes + block_index * ZIGGY_Q8_0_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q8_0_VALUES_PER_BLOCK;
        local_sum += ziggy_q8_0_block_dot(block, input, input_offset);
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

kernel void matvec_q8_0_add_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / ZIGGY_Q8_0_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q8_0_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint block_index = lane; block_index < blocks_per_row; block_index += threads_per_group) {
        const device uchar *block = row_bytes + block_index * ZIGGY_Q8_0_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q8_0_VALUES_PER_BLOCK;
        local_sum += ziggy_q8_0_block_dot(block, input, input_offset);
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] += sum;
    }
}

constant uint ZIGGY_Q3K_VALUES_PER_BLOCK = 256;
constant uint ZIGGY_Q3K_BYTES_PER_BLOCK = 110;

inline float ziggy_dot_q3k_block(
    device const uchar *block,
    device const float *input
) {
    const float d_all = read_half_le(block, 108);
    const device uchar *hm = block;
    const device uchar *q = block + 32;
    const device uchar *scales_raw = block + 96;

    uint aux[4];
    aux[0] = uint(scales_raw[0]) | (uint(scales_raw[1]) << 8) | (uint(scales_raw[2]) << 16) | (uint(scales_raw[3]) << 24);
    aux[1] = uint(scales_raw[4]) | (uint(scales_raw[5]) << 8) | (uint(scales_raw[6]) << 16) | (uint(scales_raw[7]) << 24);
    aux[2] = uint(scales_raw[8]) | (uint(scales_raw[9]) << 8) | (uint(scales_raw[10]) << 16) | (uint(scales_raw[11]) << 24);
    aux[3] = 0;

    const uint kmask1 = 0x03030303;
    const uint kmask2 = 0x0f0f0f0f;
    uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    float sum = 0.0f;
    int is = 0;
    uint8_t m = 1;
    uint input_offset = 0;

    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            const float dl0 = d_all * float(as_type<char>(uchar(aux[is / 4] >> (8 * (is % 4)))) - 32);
            is++;
            for (int l = 0; l < 16; ++l) {
                const int8_t quant = int8_t(((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                sum += dl0 * float(quant) * input[input_offset + l];
            }

            const float dl1 = d_all * float(as_type<char>(uchar(aux[is / 4] >> (8 * (is % 4)))) - 32);
            is++;
            for (int l = 0; l < 16; ++l) {
                const int8_t quant = int8_t(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                sum += dl1 * float(quant) * input[input_offset + 16 + l];
            }

            shift += 2;
            m <<= 1;
            input_offset += 32;
        }
        q += 32;
    }

    return sum;
}

kernel void matvec_q3k_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / ZIGGY_Q3K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q3K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint block_index = lane; block_index < blocks_per_row; block_index += threads_per_group) {
        const device uchar *block = row_bytes + block_index * ZIGGY_Q3K_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q3K_VALUES_PER_BLOCK;
        local_sum += ziggy_dot_q3k_block(block, input + input_offset);
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

kernel void matvec_q3k_add_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / ZIGGY_Q3K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q3K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint block_index = lane; block_index < blocks_per_row; block_index += threads_per_group) {
        const device uchar *block = row_bytes + block_index * ZIGGY_Q3K_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q3K_VALUES_PER_BLOCK;
        local_sum += ziggy_dot_q3k_block(block, input + input_offset);
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] += sum;
    }
}


inline float ziggy_dense_row_partial(
    device const float *matrix,
    device const float *input,
    uint row,
    uint rows,
    uint cols,
    uint lane,
    uint threads_per_group
) {
    float local_sum = 0.0f;
    uint col = lane * 4;
    while (col + 3 < cols) {
        const float4 input_values = float4(
            input[col + 0],
            input[col + 1],
            input[col + 2],
            input[col + 3]
        );
        const float4 matrix_values = float4(
            matrix[row + (col + 0) * rows],
            matrix[row + (col + 1) * rows],
            matrix[row + (col + 2) * rows],
            matrix[row + (col + 3) * rows]
        );
        local_sum += dot(matrix_values, input_values);
        col += threads_per_group * 4;
    }
    for (; col < cols; col += threads_per_group) {
        local_sum += matrix[row + col * rows] * input[col];
    }
    return local_sum;
}

inline float ziggy_q4k_row_partial(
    device const uchar *matrix,
    device const float *input,
    uint row,
    uint rows,
    uint cols,
    uint lane,
    uint threads_per_group
) {
    (void)rows;
    const uint blocks_per_row = cols / ZIGGY_Q4K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;

    float local_sum = 0.0f;
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP;
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) {
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP;
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP;
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK;
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK;
        const uint q_offset = chunk_in_group * 4;
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset;

        const device uchar *block = row_bytes + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK;
        const float d = read_half_le(block, 0);
        const float dmin = read_half_le(block, 2);
        const device uchar *scales = block + 4;
        const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset;
        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]);
        const uchar4 low_q = packed & uchar4(0x0F);
        const uchar4 high_q = packed >> 4;
        const float4 input_low = float4(
            input[input_offset + 0],
            input[input_offset + 1],
            input[input_offset + 2],
            input[input_offset + 3]
        );
        const float4 input_high = float4(
            input[input_offset + 32 + 0],
            input[input_offset + 32 + 1],
            input[input_offset + 32 + 2],
            input[input_offset + 32 + 3]
        );

        const uint scale_index = group * 2;
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0));
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0));
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1));
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1));
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low);
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high);
    }
    return local_sum;
}

inline float ziggy_q5k_row_partial(
    device const uchar *matrix,
    device const float *input,
    uint row,
    uint rows,
    uint cols,
    uint lane,
    uint threads_per_group
) {
    (void)rows;
    const uint blocks_per_row = cols / ZIGGY_Q5K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q5K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q5K_GROUPS_PER_BLOCK * ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP;

    float local_sum = 0.0f;
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) {
        const uint block_group_index = chunk_index / ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP;
        const uint chunk_in_group = chunk_index % ZIGGY_Q5K_PACKED_CHUNKS_PER_GROUP;
        const uint block_index = block_group_index / ZIGGY_Q5K_GROUPS_PER_BLOCK;
        const uint group = block_group_index % ZIGGY_Q5K_GROUPS_PER_BLOCK;
        const uint q_offset = chunk_in_group * 4;
        const uint input_offset = block_index * ZIGGY_Q5K_VALUES_PER_BLOCK + group * ZIGGY_Q5K_VALUES_PER_GROUP + q_offset;

        const device uchar *block = row_bytes + block_index * ZIGGY_Q5K_BYTES_PER_BLOCK;
        const float d = read_half_le(block, 0);
        const float dmin = read_half_le(block, 2);
        const device uchar *scales = block + 4;
        const device uchar *qh = block + 16 + q_offset;
        const device uchar *q = block + 48 + group * ZIGGY_Q5K_PACKED_BYTES_PER_GROUP + q_offset;

        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]);
        const uchar4 qh_packed = uchar4(qh[0], qh[1], qh[2], qh[3]);
        const uchar4 low_q_raw = packed & uchar4(0x0F);
        const uchar4 high_q_raw = packed >> 4;

        const uint low_mask_u = 1u << (group * 2);
        const uint high_mask_u = 2u << (group * 2);
        const uint4 qh_bits = uint4(qh_packed);
        const uint4 low_bit_set = select(uint4(0), uint4(16), (qh_bits & uint4(low_mask_u)) != uint4(0));
        const uint4 high_bit_set = select(uint4(0), uint4(16), (qh_bits & uint4(high_mask_u)) != uint4(0));
        const uchar4 low_q = low_q_raw | uchar4(low_bit_set);
        const uchar4 high_q = high_q_raw | uchar4(high_bit_set);

        const float4 input_low = float4(
            input[input_offset + 0],
            input[input_offset + 1],
            input[input_offset + 2],
            input[input_offset + 3]
        );
        const float4 input_high = float4(
            input[input_offset + 32 + 0],
            input[input_offset + 32 + 1],
            input[input_offset + 32 + 2],
            input[input_offset + 32 + 3]
        );

        const uint scale_index = group * 2;
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0));
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0));
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1));
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1));
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low);
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high);
    }
    return local_sum;
}

inline float ziggy_q6k_row_partial(
    device const uchar *matrix,
    device const float *input,
    uint row,
    uint rows,
    uint cols,
    uint lane,
    uint threads_per_group
) {
    (void)rows;
    const uint blocks_per_row = cols / ZIGGY_Q6K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q6K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;
    const uint chunks_per_row = blocks_per_row * ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;

    float local_sum = 0.0f;
    for (uint chunk_index = lane; chunk_index < chunks_per_row; chunk_index += threads_per_group) {
        const uint block_index = chunk_index / ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;
        const uint block_chunk = chunk_index % ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK;
        const uint block_half = block_chunk / ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF;
        const uint l = (block_chunk % ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF) * ZIGGY_Q6K_VALUES_PER_VECTOR_CHUNK;
        const device uchar *block = row_bytes + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l;
        local_sum += ziggy_q6k_chunk_dot(block, input, block_half, l, input_offset);
    }
    return local_sum;
}

inline float ziggy_q8_0_row_partial(
    device const uchar *matrix,
    device const float *input,
    uint row,
    uint rows,
    uint cols,
    uint lane,
    uint threads_per_group
) {
    (void)rows;
    const uint blocks_per_row = cols / ZIGGY_Q8_0_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q8_0_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;

    float local_sum = 0.0f;
    for (uint block_index = lane; block_index < blocks_per_row; block_index += threads_per_group) {
        const device uchar *block = row_bytes + block_index * ZIGGY_Q8_0_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q8_0_VALUES_PER_BLOCK;
        local_sum += ziggy_q8_0_block_dot(block, input, input_offset);
    }
    return local_sum;
}

inline float ziggy_q3k_row_partial(
    device const uchar *matrix,
    device const float *input,
    uint row,
    uint rows,
    uint cols,
    uint lane,
    uint threads_per_group
) {
    (void)rows;
    const uint blocks_per_row = cols / ZIGGY_Q3K_VALUES_PER_BLOCK;
    const uint row_stride = blocks_per_row * ZIGGY_Q3K_BYTES_PER_BLOCK;
    const device uchar *row_bytes = matrix + row * row_stride;

    float local_sum = 0.0f;
    for (uint block_index = lane; block_index < blocks_per_row; block_index += threads_per_group) {
        const device uchar *block = row_bytes + block_index * ZIGGY_Q3K_BYTES_PER_BLOCK;
        const uint input_offset = block_index * ZIGGY_Q3K_VALUES_PER_BLOCK;
        local_sum += ziggy_dot_q3k_block(block, input + input_offset);
    }
    return local_sum;
}

inline void ziggy_store_rotated_half_pair(
    device half *dst,
    uint index0,
    uint index1,
    float x0,
    float x1,
    uint pair,
    uint pair_count,
    uint position,
    float freq_base
) {
    const float exponent = float(pair * 2) / float(pair_count * 2);
    const float theta = float(position) / pow(freq_base, exponent);
    const float cos_theta = cos(theta);
    const float sin_theta = sin(theta);
    dst[index0] = half(x0 * cos_theta - x1 * sin_theta);
    dst[index1] = half(x0 * sin_theta + x1 * cos_theta);
}

#define ZIGGY_BIAS_KERNEL(NAME, MATRIX_T, PARTIAL_FN, MAX_SIMDGROUPS) \
kernel void NAME( \
    device const MATRIX_T *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    device const float *bias [[buffer(3)]], \
    constant uint &rows [[buffer(4)]], \
    constant uint &cols [[buffer(5)]], \
    constant uint &use_bias [[buffer(6)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    threadgroup float partial_sums[MAX_SIMDGROUPS]; \
    const float local_sum = PARTIAL_FN(matrix, input, row, rows, cols, lane, threads_per_group); \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        if (use_bias != 0) sum += bias[row]; \
        output[row] = sum; \
    } \
}

#define ZIGGY_DUAL_KERNEL(NAME, MATRIX_T, PARTIAL_FN, MAX_SIMDGROUPS) \
kernel void NAME( \
    device const MATRIX_T *matrix_a [[buffer(0)]], \
    device const MATRIX_T *matrix_b [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device float *output_a [[buffer(3)]], \
    device float *output_b [[buffer(4)]], \
    constant uint &rows [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    threadgroup float partial_sums_a[MAX_SIMDGROUPS]; \
    threadgroup float partial_sums_b[MAX_SIMDGROUPS]; \
    const float local_sum_a = PARTIAL_FN(matrix_a, input, row, rows, cols, lane, threads_per_group); \
    const float local_sum_b = PARTIAL_FN(matrix_b, input, row, rows, cols, lane, threads_per_group); \
    const float simd_sum_a = simd_sum(local_sum_a); \
    const float simd_sum_b = simd_sum(local_sum_b); \
    if (simd_lane == 0) { \
        partial_sums_a[simd_group] = simd_sum_a; \
        partial_sums_b[simd_group] = simd_sum_b; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum_a = 0.0f; \
        float sum_b = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            sum_a += partial_sums_a[index]; \
            sum_b += partial_sums_b[index]; \
        } \
        output_a[row] = sum_a; \
        output_b[row] = sum_b; \
    } \
}

#define ZIGGY_ARGMAX_KERNEL(NAME, MATRIX_T, PARTIAL_FN, MAX_SIMDGROUPS) \
kernel void NAME( \
    device const MATRIX_T *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device atomic_uint *output_state [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    threadgroup float partial_sums[MAX_SIMDGROUPS]; \
    const float local_sum = PARTIAL_FN(matrix, input, row, rows, cols, lane, threads_per_group); \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        device atomic_uint *best_value = output_state; \
        device atomic_uint *best_token = output_state + 1; \
        uint observed = atomic_load_explicit(best_value, memory_order_relaxed); \
        const uint ordered = ziggy_ordered_float_bits(sum); \
        while (ordered > observed) { \
            if (atomic_compare_exchange_weak_explicit( \
                    best_value, \
                    &observed, \
                    ordered, \
                    memory_order_relaxed, \
                    memory_order_relaxed)) { \
                atomic_store_explicit(best_token, row, memory_order_relaxed); \
                break; \
            } \
        } \
    } \
}

#define ZIGGY_KV_STORE_KERNEL(NAME, MATRIX_T, PARTIAL_FN, MAX_SIMDGROUPS) \
kernel void NAME( \
    device const MATRIX_T *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device half *dst [[buffer(2)]], \
    device const float *bias [[buffer(3)]], \
    constant uint &rows [[buffer(4)]], \
    constant uint &cols [[buffer(5)]], \
    constant uint &dst_offset [[buffer(6)]], \
    constant uint &apply_rope [[buffer(7)]], \
    constant uint &head_count [[buffer(8)]], \
    constant uint &head_dim [[buffer(9)]], \
    constant uint &rope_dim [[buffer(10)]], \
    constant uint &position [[buffer(11)]], \
    constant float &freq_base [[buffer(12)]], \
    constant uint &rope_style [[buffer(13)]], \
    constant uint &use_bias [[buffer(14)]], \
    uint item [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    const uint pair_count = min(rope_dim, head_dim) / 2; \
    if (apply_rope == 0 || pair_count == 0 || head_count == 0 || head_dim == 0) { \
        if (item >= rows) return; \
        threadgroup float partial_sums[MAX_SIMDGROUPS]; \
        const float local_sum = PARTIAL_FN(matrix, input, item, rows, cols, lane, threads_per_group); \
        const float simd_sum_value = simd_sum(local_sum); \
        if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (lane == 0) { \
            float sum = 0.0f; \
            const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
            for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
            if (use_bias != 0) sum += bias[item]; \
            dst[dst_offset + item] = half(sum); \
        } \
        return; \
    } \
    const uint n_rot = pair_count * 2; \
    const uint items_per_head = pair_count + (head_dim - n_rot); \
    const uint total_items = head_count * items_per_head; \
    if (item >= total_items) return; \
    const uint head = item / items_per_head; \
    const uint local = item % items_per_head; \
    const uint head_base = head * head_dim; \
    if (local < pair_count) { \
        const uint row0 = head_base + (rope_style == 0 ? local * 2 : local); \
        const uint row1 = head_base + (rope_style == 0 ? local * 2 + 1 : local + pair_count); \
        threadgroup float partial_sums_a[MAX_SIMDGROUPS]; \
        threadgroup float partial_sums_b[MAX_SIMDGROUPS]; \
        const float local_sum_a = PARTIAL_FN(matrix, input, row0, rows, cols, lane, threads_per_group); \
        const float local_sum_b = PARTIAL_FN(matrix, input, row1, rows, cols, lane, threads_per_group); \
        const float simd_sum_a = simd_sum(local_sum_a); \
        const float simd_sum_b = simd_sum(local_sum_b); \
        if (simd_lane == 0) { \
            partial_sums_a[simd_group] = simd_sum_a; \
            partial_sums_b[simd_group] = simd_sum_b; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (lane == 0) { \
            float sum_a = 0.0f; \
            float sum_b = 0.0f; \
            const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
            for (uint index = 0; index < simd_group_count; index += 1) { \
                sum_a += partial_sums_a[index]; \
                sum_b += partial_sums_b[index]; \
            } \
            if (use_bias != 0) { \
                sum_a += bias[row0]; \
                sum_b += bias[row1]; \
            } \
            ziggy_store_rotated_half_pair(dst, dst_offset + row0, dst_offset + row1, sum_a, sum_b, local, pair_count, position, freq_base); \
        } \
        return; \
    } \
    const uint tail_row = head_base + n_rot + (local - pair_count); \
    if (tail_row >= rows) return; \
    threadgroup float partial_sums[MAX_SIMDGROUPS]; \
    const float local_sum = PARTIAL_FN(matrix, input, tail_row, rows, cols, lane, threads_per_group); \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        if (use_bias != 0) sum += bias[tail_row]; \
        dst[dst_offset + tail_row] = half(sum); \
    } \
}

ZIGGY_BIAS_KERNEL(matvec_bias_f32, float, ziggy_dense_row_partial, ZIGGY_MAX_ROW_SIMDGROUPS)
ZIGGY_BIAS_KERNEL(matvec_q4k_bias_f32, uchar, ziggy_q4k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_BIAS_KERNEL(matvec_q5k_bias_f32, uchar, ziggy_q5k_row_partial, ZIGGY_MAX_Q5K_SIMDGROUPS)
ZIGGY_BIAS_KERNEL(matvec_q6k_bias_f32, uchar, ziggy_q6k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_BIAS_KERNEL(matvec_q8_0_bias_f32, uchar, ziggy_q8_0_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_BIAS_KERNEL(matvec_q3k_bias_f32, uchar, ziggy_q3k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)

ZIGGY_DUAL_KERNEL(dual_matvec_f32, float, ziggy_dense_row_partial, ZIGGY_MAX_ROW_SIMDGROUPS)
ZIGGY_DUAL_KERNEL(dual_matvec_q4k_f32, uchar, ziggy_q4k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_DUAL_KERNEL(dual_matvec_q5k_f32, uchar, ziggy_q5k_row_partial, ZIGGY_MAX_Q5K_SIMDGROUPS)
ZIGGY_DUAL_KERNEL(dual_matvec_q6k_f32, uchar, ziggy_q6k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_DUAL_KERNEL(dual_matvec_q8_0_f32, uchar, ziggy_q8_0_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_DUAL_KERNEL(dual_matvec_q3k_f32, uchar, ziggy_q3k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)

ZIGGY_ARGMAX_KERNEL(matvec_q4k_argmax_f32, uchar, ziggy_q4k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_ARGMAX_KERNEL(matvec_q8_0_argmax_f32, uchar, ziggy_q8_0_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_ARGMAX_KERNEL(matvec_q3k_argmax_f32, uchar, ziggy_q3k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)

ZIGGY_KV_STORE_KERNEL(matvec_store_kv_half_f32, float, ziggy_dense_row_partial, ZIGGY_MAX_ROW_SIMDGROUPS)
ZIGGY_KV_STORE_KERNEL(matvec_q4k_store_kv_half_f32, uchar, ziggy_q4k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_KV_STORE_KERNEL(matvec_q5k_store_kv_half_f32, uchar, ziggy_q5k_row_partial, ZIGGY_MAX_Q5K_SIMDGROUPS)
ZIGGY_KV_STORE_KERNEL(matvec_q6k_store_kv_half_f32, uchar, ziggy_q6k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_KV_STORE_KERNEL(matvec_q8_0_store_kv_half_f32, uchar, ziggy_q8_0_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)
ZIGGY_KV_STORE_KERNEL(matvec_q3k_store_kv_half_f32, uchar, ziggy_q3k_row_partial, ZIGGY_MAX_Q4K_SIMDGROUPS)

#undef ZIGGY_KV_STORE_KERNEL
#undef ZIGGY_ARGMAX_KERNEL
#undef ZIGGY_DUAL_KERNEL
#undef ZIGGY_BIAS_KERNEL

#define ZIGGY_Q6K_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q6K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q6K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes = matrix + row * row_stride; \
    const uint chunks_per_row = blocks_per_row * ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK; \
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    for (uint chunk_index = lane; chunk_index < chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_index = chunk_index / ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK; \
        const uint block_chunk = chunk_index % ZIGGY_Q6K_VECTOR_CHUNKS_PER_BLOCK; \
        const uint block_half = block_chunk / ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF; \
        const uint l = (block_chunk % ZIGGY_Q6K_VECTOR_CHUNKS_PER_HALF) * ZIGGY_Q6K_VALUES_PER_VECTOR_CHUNK; \
        const device uchar *block = row_bytes + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; \
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l; \
        local_sum += ziggy_q6k_chunk_dot(block, input, block_half, l, input_offset); \
    } \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        output[row] += sum; \
    } \
}

ZIGGY_Q6K_ADD_KERNEL(matvec_q6k_add_f32, 0)
ZIGGY_Q6K_ADD_KERNEL(matvec_q6k_add_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q6K_ADD_KERNEL(matvec_q6k_add_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_Q6K_ADD_KERNEL

#define ZIGGY_MOONQ_Q4K_KERNEL(NAME, ADD_TO_DST, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &cols [[buffer(4)]], \
    uint row_group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    uint r0 = row_group * 4 + 0; \
    uint r1 = row_group * 4 + 1; \
    uint r2 = row_group * 4 + 2; \
    uint r3 = row_group * 4 + 3; \
    if (r0 >= rows) return; \
    const bool valid1 = r1 < rows; \
    const bool valid2 = r2 < rows; \
    const bool valid3 = r3 < rows; \
    \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + r0 * row_stride; \
    const device uchar *row_bytes1 = valid1 ? (matrix + r1 * row_stride) : row_bytes0; \
    const device uchar *row_bytes2 = valid2 ? (matrix + r2 * row_stride) : row_bytes0; \
    const device uchar *row_bytes3 = valid3 ? (matrix + r3 * row_stride) : row_bytes0; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    \
    threadgroup float partial_sums0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums2[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums3[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    \
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        \
        const device float4* in_ptr_low = (const device float4*)(input + input_offset); \
        const device float4* in_ptr_high = (const device float4*)(input + input_offset + 32); \
        const float4 input_low = *in_ptr_low; \
        const float4 input_high = *in_ptr_high; \
        const uint scale_index = group * 2; \
        \
        { \
            const device uchar *block = row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint*)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint*)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = row_bytes2 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint*)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum2 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = row_bytes3 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint*)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum3 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
    } \
    \
    const float s0 = simd_sum(sum0); \
    const float s1 = simd_sum(sum1); \
    const float s2 = simd_sum(sum2); \
    const float s3 = simd_sum(sum3); \
    if (simd_lane == 0) { \
        partial_sums0[simd_group] = s0; \
        partial_sums1[simd_group] = s1; \
        partial_sums2[simd_group] = s2; \
        partial_sums3[simd_group] = s3; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (lane == 0) { \
        float fs0 = 0.0f, fs1 = 0.0f, fs2 = 0.0f, fs3 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            fs0 += partial_sums0[index]; \
            fs1 += partial_sums1[index]; \
            fs2 += partial_sums2[index]; \
            fs3 += partial_sums3[index]; \
        } \
        if (ADD_TO_DST) { \
                     output[r0] += fs0; \
            if (valid1) output[r1] += fs1; \
            if (valid2) output[r2] += fs2; \
            if (valid3) output[r3] += fs3; \
        } else { \
                     output[r0] = fs0; \
            if (valid1) output[r1] = fs1; \
            if (valid2) output[r2] = fs2; \
            if (valid3) output[r3] = fs3; \
        } \
    } \
}

ZIGGY_MOONQ_Q4K_KERNEL(matvec_moonq_q4k_f32, false, 0)
ZIGGY_MOONQ_Q4K_KERNEL(matvec_moonq_q4k_2048_f32, false, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_KERNEL(matvec_moonq_q4k_5632_f32, false, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)
ZIGGY_MOONQ_Q4K_KERNEL(matvec_moonq_q4k_add_f32, true, 0)
ZIGGY_MOONQ_Q4K_KERNEL(matvec_moonq_q4k_add_2048_f32, true, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_KERNEL(matvec_moonq_q4k_add_5632_f32, true, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q4K_KERNEL

kernel void apply_rope_f32(
    device float *vector [[buffer(0)]],
    constant uint &vector_base [[buffer(1)]],
    constant uint &head_count [[buffer(2)]],
    constant uint &head_dim [[buffer(3)]],
    constant uint &pair_count [[buffer(4)]],
    constant uint &position [[buffer(5)]],
    constant float &freq_base [[buffer(6)]],
    constant uint &rope_style [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    if (pair_count == 0) return;
    const uint total_pairs = head_count * pair_count;
    if (index >= total_pairs) return;

    const uint head = index / pair_count;
    const uint pair = index % pair_count;
    const uint base = vector_base + head * head_dim;

    const float exponent = float(pair * 2) / float(pair_count * 2);
    const float theta = float(position) / pow(freq_base, exponent);
    const float cos_theta = cos(theta);
    const float sin_theta = sin(theta);

    if (rope_style == 0) {
        const uint idx = base + pair * 2;
        const float x0 = vector[idx];
        const float x1 = vector[idx + 1];
        vector[idx] = x0 * cos_theta - x1 * sin_theta;
        vector[idx + 1] = x0 * sin_theta + x1 * cos_theta;
    } else {
        const uint idx0 = base + pair;
        const uint idx1 = base + pair + pair_count;
        const float x0 = vector[idx0];
        const float x1 = vector[idx1];
        vector[idx0] = x0 * cos_theta - x1 * sin_theta;
        vector[idx1] = x0 * sin_theta + x1 * cos_theta;
    }
}

kernel void apply_rope_to_dst_f32(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &dst_base [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &pair_count [[buffer(5)]],
    constant uint &position [[buffer(6)]],
    constant float &freq_base [[buffer(7)]],
    constant uint &rope_style [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {
    const uint total_values = head_count * head_dim;
    if (index >= total_values) return;

    const uint head = index / head_dim;
    const uint dim = index % head_dim;
    const uint dst_index = dst_base + index;
    if (pair_count == 0 || dim >= pair_count * 2) {
        dst[dst_index] = src[index];
        return;
    }

    if (rope_style == 0) {
        const uint pair = dim / 2;
        const uint base = head * head_dim + pair * 2;
        const float exponent = float(pair * 2) / float(pair_count * 2);
        const float theta = float(position) / pow(freq_base, exponent);
        const float cos_theta = cos(theta);
        const float sin_theta = sin(theta);
        const float x0 = src[base];
        const float x1 = src[base + 1];

        if ((dim & 1) == 0) {
            dst[dst_index] = x0 * cos_theta - x1 * sin_theta;
        } else {
            dst[dst_index] = x0 * sin_theta + x1 * cos_theta;
        }
    } else {
        const uint pair = dim % pair_count;
        const bool is_first_half = dim < pair_count;
        const uint base = head * head_dim;
        const float exponent = float(pair * 2) / float(pair_count * 2);
        const float theta = float(position) / pow(freq_base, exponent);
        const float cos_theta = cos(theta);
        const float sin_theta = sin(theta);
        const float x0 = src[base + pair];
        const float x1 = src[base + pair + pair_count];

        if (is_first_half) {
            dst[dst_index] = x0 * cos_theta - x1 * sin_theta;
        } else {
            dst[dst_index] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

kernel void attention_fused_f32(
    device const float *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &head_count [[buffer(4)]],
    constant uint &head_count_kv [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &kv_dim [[buffer(7)]],
    constant uint &context_length [[buffer(8)]],
    constant uint &position [[buffer(9)]],
    constant uint &layer_base [[buffer(10)]],
    constant uint &window_start [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    constant float &softcap [[buffer(13)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (head >= head_count) return;
    const uint kv_group = head_count / head_count_kv;
    const uint kv_head = head / kv_group;
    const uint kv_offset = kv_head * head_dim;
    device const float *q_head = q + head * head_dim;

    const uint token_count = position - window_start + 1;

    float local_m = -INFINITY;
    float local_l = 0.0f;
    float local_out[ZIGGY_MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    for (uint token_index = lane; token_index < token_count; token_index += threads_per_group) {
        const uint token = window_start + token_index;
        device const half *k_head = k_cache + layer_base + token * kv_dim + kv_offset;
        device const half *v_head = v_cache + layer_base + token * kv_dim + kv_offset;

        float s = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            s += q_head[d] * float(k_head[d]);
        }
        s *= scale;
        if (softcap > 0.0f) {
            s = tanh(s / softcap) * softcap;
        }

        float m_new = max(local_m, s);
        float exp_diff = exp(local_m - m_new);
        float p = exp(s - m_new);

        local_l = local_l * exp_diff + p;
        for (uint d = 0; d < head_dim; d++) {
            local_out[d] = local_out[d] * exp_diff + p * float(v_head[d]);
        }
        local_m = m_new;
    }

    threadgroup float tg_m[ZIGGY_MAX_ROW_SIMDGROUPS];
    float simd_m = simd_max(local_m);
    if (simd_lane == 0) tg_m[simd_group] = simd_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_m = -INFINITY;
    const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
    if (lane == 0) {
        for (uint i = 0; i < simd_group_count; i++) {
            global_m = max(global_m, tg_m[i]);
        }
        tg_m[0] = global_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_m = tg_m[0];

    float scaled_l = local_l * exp(local_m - global_m);
    float simd_l = simd_sum(scaled_l);
    threadgroup float tg_l[ZIGGY_MAX_ROW_SIMDGROUPS];
    if (simd_lane == 0) tg_l[simd_group] = simd_l;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_l = 0.0f;
    if (lane == 0) {
        for (uint i = 0; i < simd_group_count; i++) {
            global_l += tg_l[i];
        }
        tg_l[0] = global_l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_l = tg_l[0];

    threadgroup float tg_out[ZIGGY_MAX_ROW_SIMDGROUPS * ZIGGY_MAX_HEAD_DIM];
    float thread_scale = exp(local_m - global_m) / global_l;

    for (uint d = 0; d < head_dim; d++) {
        float scaled_out = local_out[d] * thread_scale;
        float simd_out = simd_sum(scaled_out);
        if (simd_lane == 0) {
            tg_out[simd_group * ZIGGY_MAX_HEAD_DIM + d] = simd_out;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        for (uint d = simd_lane; d < head_dim; d += threads_per_simdgroup) {
            float sum = 0.0f;
            for (uint i = 0; i < simd_group_count; i++) {
                sum += tg_out[i * ZIGGY_MAX_HEAD_DIM + d];
            }
            output[head * head_dim + d] = sum;
        }
    }
}

kernel void silu_mul_f32(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    const float value = gate[index];
    gate[index] = (value / (1.0f + exp(-value))) * up[index];
}

kernel void gelu_mul_f32(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    const float value = gate[index];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float cubic = value * value * value;
    const float gelu = 0.5f * value * (1.0f + tanh(sqrt_2_over_pi * (value + 0.044715f * cubic)));
    gate[index] = gelu * up[index];
}

kernel void add_in_place_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    dst[index] += src[index];
}

kernel void rms_norm_f32(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    threadgroup float partial_sums[ZIGGY_MAX_NORM_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint i = lane; i < count; i += threads_per_group) {
        const float value = input[i];
        local_sum += value * value;
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        partial_sums[0] = 1.0f / sqrt(sum / float(count) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = partial_sums[0];
    for (uint index = lane; index < count; index += threads_per_group) {
        output[index] = input[index] * scale * (weights[index] + weight_offset);
    }
}

kernel void rms_norm_per_head_f32(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    constant float &weight_offset [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (head >= head_count) return;

    device const float *h_in = input + head * head_dim;
    device const float *h_w = weights;
    device float *h_out = output + head * head_dim;

    threadgroup float partial_sums[ZIGGY_MAX_NORM_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint i = lane; i < head_dim; i += threads_per_group) {
        const float val = h_in[i];
        local_sum += val * val;
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        partial_sums[0] = 1.0f / sqrt(sum / float(head_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = partial_sums[0];
    for (uint i = lane; i < head_dim; i += threads_per_group) {
        h_out[i] = h_in[i] * scale * (h_w[i] + weight_offset);
    }
}

kernel void argmax_f32(
    device const float *input [[buffer(0)]],
    device uint *output_token [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &thread_count [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (count == 0 || thread_count == 0 || thread_count > 256) return;
    if (tid >= thread_count) return;

    float best_value = -INFINITY;
    uint best_index = 0;
    for (uint i = tid; i < count; i += thread_count) {
        const float value = input[i];
        if (value > best_value || (value == best_value && i < best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    threadgroup float scratch_values[256];
    threadgroup uint scratch_indices[256];
    scratch_values[tid] = best_value;
    scratch_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = thread_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            const float rhs_value = scratch_values[tid + stride];
            const uint rhs_index = scratch_indices[tid + stride];
            const float lhs_value = scratch_values[tid];
            const uint lhs_index = scratch_indices[tid];
            if (rhs_value > lhs_value || (rhs_value == lhs_value && rhs_index < lhs_index)) {
                scratch_values[tid] = rhs_value;
                scratch_indices[tid] = rhs_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid != 0) return;
    output_token[0] = scratch_indices[0];
}

constant uint ZIGGY_SHORTLIST_MAX_K = 64;
constant uint ZIGGY_SHORTLIST_THREAD_COUNT = 32;

struct ZiggyShortlistEntry {
    uint token_id;
    float score;
};

kernel void topk_f32(
    device const float *input [[buffer(0)]],
    device ZiggyShortlistEntry *output_entries [[buffer(1)]],
    constant uint &count [[buffer(3)]],
    constant uint &top_k [[buffer(4)]],
    constant uint &thread_count [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (count == 0 || top_k == 0 || top_k > ZIGGY_SHORTLIST_MAX_K || thread_count == 0 || thread_count > ZIGGY_SHORTLIST_THREAD_COUNT) return;
    if (tid >= thread_count) return;

    float best_values[ZIGGY_SHORTLIST_MAX_K];
    uint best_indices[ZIGGY_SHORTLIST_MAX_K];
    for (uint i = 0; i < top_k; i += 1) {
        best_values[i] = -INFINITY;
        best_indices[i] = 0;
    }

    for (uint i = tid; i < count; i += thread_count) {
        const float value = input[i];
        uint insert_at = top_k;
        for (uint slot = 0; slot < top_k; slot += 1) {
            const bool better_value = value > best_values[slot];
            const bool equal_value = value == best_values[slot];
            const bool better_tie = equal_value && i < best_indices[slot];
            if (better_value || better_tie) {
                insert_at = slot;
                break;
            }
        }
        if (insert_at == top_k) continue;

        for (uint shift = top_k - 1; shift > insert_at; shift -= 1) {
            best_values[shift] = best_values[shift - 1];
            best_indices[shift] = best_indices[shift - 1];
        }
        best_values[insert_at] = value;
        best_indices[insert_at] = i;
    }

    threadgroup float scratch_values[ZIGGY_SHORTLIST_THREAD_COUNT * ZIGGY_SHORTLIST_MAX_K];
    threadgroup uint scratch_indices[ZIGGY_SHORTLIST_THREAD_COUNT * ZIGGY_SHORTLIST_MAX_K];

    const uint scratch_base = tid * ZIGGY_SHORTLIST_MAX_K;
    for (uint i = 0; i < top_k; i += 1) {
        scratch_values[scratch_base + i] = best_values[i];
        scratch_indices[scratch_base + i] = best_indices[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) return;

    for (uint worker = 1; worker < thread_count; worker += 1) {
        const uint worker_base = worker * ZIGGY_SHORTLIST_MAX_K;
        for (uint candidate_index = 0; candidate_index < top_k; candidate_index += 1) {
            const float value = scratch_values[worker_base + candidate_index];
            const uint token = scratch_indices[worker_base + candidate_index];
            uint insert_at = top_k;
            for (uint slot = 0; slot < top_k; slot += 1) {
                const bool better_value = value > best_values[slot];
                const bool equal_value = value == best_values[slot];
                const bool better_tie = equal_value && token < best_indices[slot];
                if (better_value || better_tie) {
                    insert_at = slot;
                    break;
                }
            }
            if (insert_at == top_k) continue;

            for (uint shift = top_k - 1; shift > insert_at; shift -= 1) {
                best_values[shift] = best_values[shift - 1];
                best_indices[shift] = best_indices[shift - 1];
            }
            best_values[insert_at] = value;
            best_indices[insert_at] = token;
        }
    }

    for (uint i = 0; i < top_k; i += 1) {
        output_entries[i].token_id = best_indices[i];
        output_entries[i].score = best_values[i];
    }
}

kernel void normalize_topk_f32(
    device ZiggyShortlistEntry *entries [[buffer(0)]],
    constant uint &top_k [[buffer(1)]],
    constant bool &apply_softmax [[buffer(2)]],
    constant bool &normalize_weights [[buffer(3)]],
    constant float &scale [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (top_k == 0 || top_k > ZIGGY_SHORTLIST_MAX_K) return;
    if (tid != 0) return;

    float weights[ZIGGY_SHORTLIST_MAX_K];
    float total = 0.0f;
    float max_score = -INFINITY;

    if (apply_softmax) {
        for (uint i = 0; i < top_k; i += 1) {
            max_score = max(max_score, entries[i].score);
        }
        for (uint i = 0; i < top_k; i += 1) {
            const float weight = exp(entries[i].score - max_score);
            weights[i] = weight;
            total += weight;
        }
    } else {
        for (uint i = 0; i < top_k; i += 1) {
            const float value = isfinite(entries[i].score) && entries[i].score > 0.0f ? entries[i].score : 0.0f;
            weights[i] = value;
            total += value;
        }
    }

    if (!(total > 0.0f) || !isfinite(total)) {
        for (uint i = 0; i < top_k; i += 1) {
            entries[i].score = 0.0f;
        }
        return;
    }

    const float inv_total = normalize_weights ? (1.0f / total) : 1.0f;
    for (uint i = 0; i < top_k; i += 1) {
        entries[i].score = weights[i] * inv_total * scale;
    }
}

kernel void sample_topk_f32(
    device const float *input [[buffer(0)]],
    device uint *output_token [[buffer(1)]],
    constant uint &count [[buffer(3)]],
    constant uint &top_k [[buffer(4)]],
    constant uint &thread_count [[buffer(5)]],
    constant float &temperature [[buffer(6)]],
    constant float &random_uniform [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (count == 0 || top_k == 0 || top_k > ZIGGY_SHORTLIST_MAX_K || thread_count == 0 || thread_count > ZIGGY_SHORTLIST_THREAD_COUNT) return;
    if (tid >= thread_count) return;

    float best_values[ZIGGY_SHORTLIST_MAX_K];
    uint best_indices[ZIGGY_SHORTLIST_MAX_K];
    for (uint i = 0; i < top_k; i += 1) {
        best_values[i] = -INFINITY;
        best_indices[i] = 0;
    }

    for (uint i = tid; i < count; i += thread_count) {
        const float value = input[i];
        uint insert_at = top_k;
        for (uint slot = 0; slot < top_k; slot += 1) {
            const bool better_value = value > best_values[slot];
            const bool equal_value = value == best_values[slot];
            const bool better_tie = equal_value && i < best_indices[slot];
            if (better_value || better_tie) {
                insert_at = slot;
                break;
            }
        }
        if (insert_at == top_k) continue;

        for (uint shift = top_k - 1; shift > insert_at; shift -= 1) {
            best_values[shift] = best_values[shift - 1];
            best_indices[shift] = best_indices[shift - 1];
        }
        best_values[insert_at] = value;
        best_indices[insert_at] = i;
    }

    threadgroup float scratch_values[ZIGGY_SHORTLIST_THREAD_COUNT * ZIGGY_SHORTLIST_MAX_K];
    threadgroup uint scratch_indices[ZIGGY_SHORTLIST_THREAD_COUNT * ZIGGY_SHORTLIST_MAX_K];
    const uint scratch_base = tid * ZIGGY_SHORTLIST_MAX_K;
    for (uint i = 0; i < top_k; i += 1) {
        scratch_values[scratch_base + i] = best_values[i];
        scratch_indices[scratch_base + i] = best_indices[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;

    for (uint worker = 1; worker < thread_count; worker += 1) {
        const uint worker_base = worker * ZIGGY_SHORTLIST_MAX_K;
        for (uint candidate_index = 0; candidate_index < top_k; candidate_index += 1) {
            const float value = scratch_values[worker_base + candidate_index];
            const uint token = scratch_indices[worker_base + candidate_index];
            uint insert_at = top_k;
            for (uint slot = 0; slot < top_k; slot += 1) {
                const bool better_value = value > best_values[slot];
                const bool equal_value = value == best_values[slot];
                const bool better_tie = equal_value && token < best_indices[slot];
                if (better_value || better_tie) {
                    insert_at = slot;
                    break;
                }
            }
            if (insert_at == top_k) continue;
            for (uint shift = top_k - 1; shift > insert_at; shift -= 1) {
                best_values[shift] = best_values[shift - 1];
                best_indices[shift] = best_indices[shift - 1];
            }
            best_values[insert_at] = value;
            best_indices[insert_at] = token;
        }
    }

    float max_logit = best_values[0];
    for (uint i = 1; i < top_k; i += 1) {
        max_logit = max(max_logit, best_values[i]);
    }

    float weights[ZIGGY_SHORTLIST_MAX_K];
    float total_weight = 0.0f;
    for (uint i = 0; i < top_k; i += 1) {
        const float shifted = (best_values[i] - max_logit) / temperature;
        const float weight = exp(shifted);
        weights[i] = weight;
        total_weight += weight;
    }

    if (!(total_weight > 0.0f) || !isfinite(total_weight)) {
        output_token[0] = best_indices[0];
        return;
    }

    const float clamped_uniform = clamp(random_uniform, 0.0f, 0.99999994f);
    const float target = clamped_uniform * total_weight;
    float cumulative = 0.0f;
    for (uint i = 0; i < top_k; i += 1) {
        cumulative += weights[i];
        if (target <= cumulative) {
            output_token[0] = best_indices[i];
            return;
        }
    }
    output_token[0] = best_indices[top_k - 1];
}

kernel void weighted_sum_topk_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    device const ZiggyShortlistEntry *entries [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &slot_idx [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    dst[index] += src[index] * entries[slot_idx].score;
}

kernel void sigmoid_scale_add_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    device const float *scalar [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    const float gate = 1.0f / (1.0f + exp(-scalar[0]));
    dst[index] += src[index] * gate;
}

inline float ziggy_dot_iq3_xxs_block(
    device const uchar *block,
    device const float *input
) {
    const float d = read_half_le(block, 0);
    device const uchar *qs = block + 2;
    device const uchar *scales_and_signs = block + 66;
    float sum = 0.0f;

    for (uint ib32 = 0; ib32 < 8; ib32 += 1) {
        const uint aux32 =
            uint(scales_and_signs[ib32 * 4 + 0]) |
            (uint(scales_and_signs[ib32 * 4 + 1]) << 8) |
            (uint(scales_and_signs[ib32 * 4 + 2]) << 16) |
            (uint(scales_and_signs[ib32 * 4 + 3]) << 24);
        const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
        for (uint l = 0; l < 4; l += 1) {
            const uchar signs = ziggy_iq2_signs(uchar((aux32 >> (7 * l)) & 0x7f));
            const uint grid1 = ZIGGY_IQ3_XXS_GRID[qs[2 * l + 0]];
            const uint grid2 = ZIGGY_IQ3_XXS_GRID[qs[2 * l + 1]];
            for (uint j = 0; j < 4; j += 1) {
                const float value = db * float(ziggy_grid_byte(grid1, j)) * ziggy_iq_sign(signs, j + 0);
                sum += value * input[j + 0];
            }
            for (uint j = 0; j < 4; j += 1) {
                const float value = db * float(ziggy_grid_byte(grid2, j)) * ziggy_iq_sign(signs, j + 4);
                sum += value * input[j + 4];
            }
            input += 8;
        }
        qs += 8;
    }
    return sum;
}

inline float ziggy_dot_iq4_xs_block(
    device const uchar *block,
    device const float *input
) {
    const float d = read_half_le(block, 0);
    const ushort scales_h = ushort(block[2]) | (ushort(block[3]) << 8);
    device const uchar *scales_l = block + 4;
    device const uchar *qs = block + 8;
    float sum = 0.0f;

    for (uint ib = 0; ib < 8; ib += 1) {
        const uchar low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
        const ushort high = (scales_h >> (2 * ib)) & 0x03;
        const float dl = d * float(int(low | uchar(high << 4)) - 32);
        for (uint j = 0; j < 16; j += 1) {
            sum += dl * float(ZIGGY_IQ4_NL_VALUES[qs[j] & 0x0f]) * input[j];
        }
        for (uint j = 0; j < 16; j += 1) {
            sum += dl * float(ZIGGY_IQ4_NL_VALUES[qs[j] >> 4]) * input[16 + j];
        }
        qs += 16;
        input += 32;
    }
    return sum;
}

constant uint ZIGGY_BATCH_ARGMAX_MAX_DRAFTS = 8;

kernel void batch_argmax_f32(
    device const float *input [[buffer(0)]],
    device uint *output_tokens [[buffer(1)]],
    constant uint &vocab_size [[buffer(2)]],
    constant uint &batch_count [[buffer(3)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_count) return;
    if (vocab_size == 0 || thread_count == 0 || thread_count > 256) return;
    if (tid >= thread_count) return;

    device const float *batch_input = input + batch_idx * vocab_size;

    float best_value = -INFINITY;
    uint best_index = 0;
    for (uint i = tid; i < vocab_size; i += thread_count) {
        const float value = batch_input[i];
        if (value > best_value || (value == best_value && i < best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    threadgroup float scratch_values[256];
    threadgroup uint scratch_indices[256];
    scratch_values[tid] = best_value;
    scratch_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = thread_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            const float rhs_value = scratch_values[tid + stride];
            const uint rhs_index = scratch_indices[tid + stride];
            const float lhs_value = scratch_values[tid];
            const uint lhs_index = scratch_indices[tid];
            if (rhs_value > lhs_value || (rhs_value == lhs_value && rhs_index < lhs_index)) {
                scratch_values[tid] = rhs_value;
                scratch_indices[tid] = rhs_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid != 0) return;
    output_tokens[batch_idx] = scratch_indices[0];
}

kernel void batch_matvec_add_f32(
    device const float *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant uint &batch_idx [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;

    device const float *batch_input = input + batch_idx * cols;
    device float *batch_output = output + batch_idx * rows;

    threadgroup float partial_sums[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum = 0.0f;

    uint col = lane * 4;
    while (col + 3 < cols) {
        const float4 input_values = float4(
            batch_input[col + 0],
            batch_input[col + 1],
            batch_input[col + 2],
            batch_input[col + 3]
        );
        const float4 matrix_values = float4(
            matrix[row + (col + 0) * rows],
            matrix[row + (col + 1) * rows],
            matrix[row + (col + 2) * rows],
            matrix[row + (col + 3) * rows]
        );
        local_sum += dot(matrix_values, input_values);
        col += threads_per_group * 4;
    }

    for (; col < cols; col += threads_per_group) {
        local_sum += matrix[row + col * rows] * batch_input[col];
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        batch_output[row] = sum;
    }
}

#define ZIGGY_BATCH_Q4K_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &cols [[buffer(4)]], \
    constant uint &batch_idx [[buffer(5)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    device const float *batch_input = input + batch_idx * effective_cols; \
    device float *batch_output = output + batch_idx * rows; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes = matrix + row * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device uchar *block = row_bytes + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
        const float d = read_half_le(block, 0); \
        const float dmin = read_half_le(block, 2); \
        const device uchar *scales = block + 4; \
        const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
        const uchar4 low_q = packed & uchar4(0x0F); \
        const uchar4 high_q = packed >> 4; \
        const float4 input_low = float4( \
            batch_input[input_offset + 0], \
            batch_input[input_offset + 1], \
            batch_input[input_offset + 2], \
            batch_input[input_offset + 3] \
        ); \
        const float4 input_high = float4( \
            batch_input[input_offset + 32 + 0], \
            batch_input[input_offset + 32 + 1], \
            batch_input[input_offset + 32 + 2], \
            batch_input[input_offset + 32 + 3] \
        ); \
        const uint scale_index = group * 2; \
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low); \
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high); \
    } \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        batch_output[row] = sum; \
    } \
}

ZIGGY_BATCH_Q4K_ADD_KERNEL(batch_matvec_q4k_add_f32, 0)
ZIGGY_BATCH_Q4K_ADD_KERNEL(batch_matvec_q4k_add_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_BATCH_Q4K_ADD_KERNEL(batch_matvec_q4k_add_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_BATCH_Q4K_ADD_KERNEL

#define ZIGGY_BATCH_Q4K_MATVEC_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &cols [[buffer(4)]], \
    constant uint &batch_idx [[buffer(5)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    device const float *batch_input = input + batch_idx * effective_cols; \
    device float *batch_output = output + batch_idx * rows; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes = matrix + row * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device uchar *block = row_bytes + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
        const float d = read_half_le(block, 0); \
        const float dmin = read_half_le(block, 2); \
        const device uchar *scales = block + 4; \
        const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
        const uchar4 low_q = packed & uchar4(0x0F); \
        const uchar4 high_q = packed >> 4; \
        const float4 input_low = float4( \
            batch_input[input_offset + 0], \
            batch_input[input_offset + 1], \
            batch_input[input_offset + 2], \
            batch_input[input_offset + 3] \
        ); \
        const float4 input_high = float4( \
            batch_input[input_offset + 32 + 0], \
            batch_input[input_offset + 32 + 1], \
            batch_input[input_offset + 32 + 2], \
            batch_input[input_offset + 32 + 3] \
        ); \
        const uint scale_index = group * 2; \
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low); \
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high); \
    } \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        batch_output[row] = sum; \
    } \
}

ZIGGY_BATCH_Q4K_MATVEC_KERNEL(batch_matvec_q4k_f32, 0)

#undef ZIGGY_BATCH_Q4K_MATVEC_KERNEL

kernel void batch_silu_mul_f32(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &batch_idx [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    device float *batch_gate = gate + batch_idx * count;
    device const float *batch_up = up + batch_idx * count;
    const float value = batch_gate[index];
    batch_gate[index] = (value / (1.0f + exp(-value))) * batch_up[index];
}

kernel void batch_add_in_place_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &batch_idx [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    device float *batch_dst = dst + batch_idx * count;
    device const float *batch_src = src + batch_idx * count;
    batch_dst[index] += batch_src[index];
}

kernel void batch_rms_norm_f32(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    constant uint &batch_idx [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    device const float *batch_input = input + batch_idx * count;
    device float *batch_output = output + batch_idx * count;
    threadgroup float partial_sums[ZIGGY_MAX_NORM_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint i = lane; i < count; i += threads_per_group) {
        const float value = batch_input[i];
        local_sum += value * value;
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        partial_sums[0] = 1.0f / sqrt(sum / float(count) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = partial_sums[0];
    for (uint index = lane; index < count; index += threads_per_group) {
        batch_output[index] = batch_input[index] * scale * (weights[index] + weight_offset);
    }
}

#define ZIGGY_Q4K_SILU_DOWN_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *gate [[buffer(1)]], \
    device const float *up [[buffer(2)]], \
    device float *output [[buffer(3)]], \
    constant uint &rows [[buffer(4)]], \
    constant uint &cols [[buffer(5)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row >= rows) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes = matrix + row * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device uchar *block = row_bytes + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
        const float d = read_half_le(block, 0); \
        const float dmin = read_half_le(block, 2); \
        const device uchar *scales = block + 4; \
        const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
        const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
        const uchar4 low_q = packed & uchar4(0x0F); \
        const uchar4 high_q = packed >> 4; \
        const device float4* gate_ptr_low = (const device float4*)(gate + input_offset); \
        const device float4* gate_ptr_high = (const device float4*)(gate + input_offset + 32); \
        const device float4* up_ptr_low = (const device float4*)(up + input_offset); \
        const device float4* up_ptr_high = (const device float4*)(up + input_offset + 32); \
        const float4 g_low = *gate_ptr_low; \
        const float4 g_high = *gate_ptr_high; \
        const float4 u_low = *up_ptr_low; \
        const float4 u_high = *up_ptr_high; \
        const float4 input_low = (g_low / (1.0f + exp(-g_low))) * u_low; \
        const float4 input_high = (g_high / (1.0f + exp(-g_high))) * u_high; \
        const uint scale_index = group * 2; \
        const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
        const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
        const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
        const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
        local_sum += dot(float4(low_q) * d1 - float4(m1), input_low); \
        local_sum += dot(float4(high_q) * d2 - float4(m2), input_high); \
    } \
    const float simd_sum_value = simd_sum(local_sum); \
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) sum += partial_sums[index]; \
        output[row] += sum; \
    } \
}

ZIGGY_Q4K_SILU_DOWN_ADD_KERNEL(matvec_q4k_silu_down_add_f32, 0)
ZIGGY_Q4K_SILU_DOWN_ADD_KERNEL(matvec_q4k_silu_down_add_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_SILU_DOWN_ADD_KERNEL(matvec_q4k_silu_down_add_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)


#define ZIGGY_MOONQ_Q4K_SILU_DOWN_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *gate [[buffer(1)]], \
    device const float *up [[buffer(2)]], \
    device float *output [[buffer(3)]], \
    constant uint &rows [[buffer(4)]], \
    constant uint &cols [[buffer(5)]], \
    uint row_group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    uint r0 = row_group * 4 + 0; \
    uint r1 = row_group * 4 + 1; \
    uint r2 = row_group * 4 + 2; \
    uint r3 = row_group * 4 + 3; \
    if (r0 >= rows) return; \
    const bool valid1 = r1 < rows; \
    const bool valid2 = r2 < rows; \
    const bool valid3 = r3 < rows; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + r0 * row_stride; \
    const device uchar *row_bytes1 = valid1 ? (matrix + r1 * row_stride) : row_bytes0; \
    const device uchar *row_bytes2 = valid2 ? (matrix + r2 * row_stride) : row_bytes0; \
    const device uchar *row_bytes3 = valid3 ? (matrix + r3 * row_stride) : row_bytes0; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_sums0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums2[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums3[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device float4* gate_ptr_low = (const device float4*)(gate + input_offset); \
        const device float4* gate_ptr_high = (const device float4*)(gate + input_offset + 32); \
        const device float4* up_ptr_low = (const device float4*)(up + input_offset); \
        const device float4* up_ptr_high = (const device float4*)(up + input_offset + 32); \
        const float4 g_low = *gate_ptr_low; \
        const float4 g_high = *gate_ptr_high; \
        const float4 u_low = *up_ptr_low; \
        const float4 u_high = *up_ptr_high; \
        const float4 input_low = (g_low / (1.0f + exp(-g_low))) * u_low; \
        const float4 input_high = (g_high / (1.0f + exp(-g_high))) * u_high; \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint*)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint*)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = row_bytes2 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint*)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum2 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = row_bytes3 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint*)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum3 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
    } \
    const float s0 = simd_sum(sum0); \
    const float s1 = simd_sum(sum1); \
    const float s2 = simd_sum(sum2); \
    const float s3 = simd_sum(sum3); \
    if (simd_lane == 0) { \
        partial_sums0[simd_group] = s0; \
        partial_sums1[simd_group] = s1; \
        partial_sums2[simd_group] = s2; \
        partial_sums3[simd_group] = s3; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float fs0 = 0.0f, fs1 = 0.0f, fs2 = 0.0f, fs3 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            fs0 += partial_sums0[index]; \
            fs1 += partial_sums1[index]; \
            fs2 += partial_sums2[index]; \
            fs3 += partial_sums3[index]; \
        } \
        output[r0] += fs0; \
        if (valid1) output[r1] += fs1; \
        if (valid2) output[r2] += fs2; \
        if (valid3) output[r3] += fs3; \
    } \
}

ZIGGY_MOONQ_Q4K_SILU_DOWN_ADD_KERNEL(matvec_moonq_q4k_silu_down_add_f32, 0)
ZIGGY_MOONQ_Q4K_SILU_DOWN_ADD_KERNEL(matvec_moonq_q4k_silu_down_add_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_SILU_DOWN_ADD_KERNEL(matvec_moonq_q4k_silu_down_add_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q4K_SILU_DOWN_ADD_KERNEL

kernel void indexed_matvec_iq3_xxs_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    device const ZiggyShortlistEntry *entries [[buffer(5)]],
    constant uint &slot_idx [[buffer(6)]],
    constant uint &rows_per_expert [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows || cols == 0) return;

    const uint block_count = cols / ZIGGY_IQ3_XXS_VALUES_PER_BLOCK;
    const uint row_bytes = block_count * ZIGGY_IQ3_XXS_BYTES_PER_BLOCK;
    const uint expert_index = entries[slot_idx].token_id;
    const uint matrix_row = expert_index * rows_per_expert + row;
    const device uchar *row_ptr = matrix + matrix_row * row_bytes;

    threadgroup float partial_sums[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum = 0.0f;

    for (uint block_index = lane; block_index < block_count; block_index += threads_per_group) {
        local_sum += ziggy_dot_iq3_xxs_block(
            row_ptr + block_index * ZIGGY_IQ3_XXS_BYTES_PER_BLOCK,
            input + block_index * ZIGGY_IQ3_XXS_VALUES_PER_BLOCK
        );
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

kernel void dual_indexed_matvec_iq3_xxs_f32(
    device const uchar *matrix_a [[buffer(0)]],
    device const uchar *matrix_b [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output_a [[buffer(3)]],
    device float *output_b [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    device const ZiggyShortlistEntry *entries [[buffer(7)]],
    constant uint &slot_idx [[buffer(8)]],
    constant uint &rows_per_expert [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows || cols == 0) return;

    const uint block_count = cols / ZIGGY_IQ3_XXS_VALUES_PER_BLOCK;
    const uint row_bytes = block_count * ZIGGY_IQ3_XXS_BYTES_PER_BLOCK;
    const uint expert_index = entries[slot_idx].token_id;
    const uint matrix_row = expert_index * rows_per_expert + row;
    const device uchar *row_ptr_a = matrix_a + matrix_row * row_bytes;
    const device uchar *row_ptr_b = matrix_b + matrix_row * row_bytes;

    threadgroup float partial_sums_a[ZIGGY_MAX_ROW_SIMDGROUPS];
    threadgroup float partial_sums_b[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum_a = 0.0f;
    float local_sum_b = 0.0f;

    for (uint block_index = lane; block_index < block_count; block_index += threads_per_group) {
        const device float *block_input = input + block_index * ZIGGY_IQ3_XXS_VALUES_PER_BLOCK;
        local_sum_a += ziggy_dot_iq3_xxs_block(
            row_ptr_a + block_index * ZIGGY_IQ3_XXS_BYTES_PER_BLOCK,
            block_input
        );
        local_sum_b += ziggy_dot_iq3_xxs_block(
            row_ptr_b + block_index * ZIGGY_IQ3_XXS_BYTES_PER_BLOCK,
            block_input
        );
    }

    const float simd_sum_a = simd_sum(local_sum_a);
    const float simd_sum_b = simd_sum(local_sum_b);
    if (simd_lane == 0) {
        partial_sums_a[simd_group] = simd_sum_a;
        partial_sums_b[simd_group] = simd_sum_b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum_a += partial_sums_a[index];
            sum_b += partial_sums_b[index];
        }
        output_a[row] = sum_a;
        output_b[row] = sum_b;
    }
}

kernel void indexed_matvec_iq4_xs_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    device const ZiggyShortlistEntry *entries [[buffer(5)]],
    constant uint &slot_idx [[buffer(6)]],
    constant uint &rows_per_expert [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows || cols == 0) return;

    const uint block_count = cols / ZIGGY_IQ4_XS_VALUES_PER_BLOCK;
    const uint row_bytes = block_count * ZIGGY_IQ4_XS_BYTES_PER_BLOCK;
    const uint expert_index = entries[slot_idx].token_id;
    const uint matrix_row = expert_index * rows_per_expert + row;
    const device uchar *row_ptr = matrix + matrix_row * row_bytes;

    threadgroup float partial_sums[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum = 0.0f;

    for (uint block_index = lane; block_index < block_count; block_index += threads_per_group) {
        local_sum += ziggy_dot_iq4_xs_block(
            row_ptr + block_index * ZIGGY_IQ4_XS_BYTES_PER_BLOCK,
            input + block_index * ZIGGY_IQ4_XS_VALUES_PER_BLOCK
        );
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

kernel void indexed_matvec_iq4_xs_add_weighted_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    device const ZiggyShortlistEntry *entries [[buffer(5)]],
    constant uint &slot_idx [[buffer(6)]],
    constant uint &rows_per_expert [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows || cols == 0) return;

    const uint block_count = cols / ZIGGY_IQ4_XS_VALUES_PER_BLOCK;
    const uint row_bytes = block_count * ZIGGY_IQ4_XS_BYTES_PER_BLOCK;
    const uint expert_index = entries[slot_idx].token_id;
    const uint matrix_row = expert_index * rows_per_expert + row;
    const device uchar *row_ptr = matrix + matrix_row * row_bytes;

    threadgroup float partial_sums[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum = 0.0f;

    for (uint block_index = lane; block_index < block_count; block_index += threads_per_group) {
        local_sum += ziggy_dot_iq4_xs_block(
            row_ptr + block_index * ZIGGY_IQ4_XS_BYTES_PER_BLOCK,
            input + block_index * ZIGGY_IQ4_XS_VALUES_PER_BLOCK
        );
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] += sum * entries[slot_idx].score;
    }
}

kernel void indexed_matvec_q3k_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    device const ZiggyShortlistEntry *entries [[buffer(5)]],
    constant uint &slot_idx [[buffer(6)]],
    constant uint &rows_per_expert [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows || cols == 0) return;

    const uint block_count = cols / ZIGGY_Q3K_VALUES_PER_BLOCK;
    const uint row_bytes = block_count * ZIGGY_Q3K_BYTES_PER_BLOCK;
    const uint expert_index = entries[slot_idx].token_id;
    const uint matrix_row = expert_index * rows_per_expert + row;
    const device uchar *row_ptr = matrix + matrix_row * row_bytes;

    threadgroup float partial_sums[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum = 0.0f;

    for (uint block_index = lane; block_index < block_count; block_index += threads_per_group) {
        local_sum += ziggy_dot_q3k_block(
            row_ptr + block_index * ZIGGY_Q3K_BYTES_PER_BLOCK,
            input + block_index * ZIGGY_Q3K_VALUES_PER_BLOCK
        );
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] = sum;
    }
}

kernel void dual_indexed_matvec_q3k_f32(
    device const uchar *matrix_a [[buffer(0)]],
    device const uchar *matrix_b [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output_a [[buffer(3)]],
    device float *output_b [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    device const ZiggyShortlistEntry *entries [[buffer(7)]],
    constant uint &slot_idx [[buffer(8)]],
    constant uint &rows_per_expert [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows || cols == 0) return;

    const uint block_count = cols / ZIGGY_Q3K_VALUES_PER_BLOCK;
    const uint row_bytes = block_count * ZIGGY_Q3K_BYTES_PER_BLOCK;
    const uint expert_index = entries[slot_idx].token_id;
    const uint matrix_row = expert_index * rows_per_expert + row;
    const device uchar *row_ptr_a = matrix_a + matrix_row * row_bytes;
    const device uchar *row_ptr_b = matrix_b + matrix_row * row_bytes;

    threadgroup float partial_sums_a[ZIGGY_MAX_ROW_SIMDGROUPS];
    threadgroup float partial_sums_b[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum_a = 0.0f;
    float local_sum_b = 0.0f;

    for (uint block_index = lane; block_index < block_count; block_index += threads_per_group) {
        const device float *block_input = input + block_index * ZIGGY_Q3K_VALUES_PER_BLOCK;
        local_sum_a += ziggy_dot_q3k_block(
            row_ptr_a + block_index * ZIGGY_Q3K_BYTES_PER_BLOCK,
            block_input
        );
        local_sum_b += ziggy_dot_q3k_block(
            row_ptr_b + block_index * ZIGGY_Q3K_BYTES_PER_BLOCK,
            block_input
        );
    }

    const float simd_sum_a = simd_sum(local_sum_a);
    const float simd_sum_b = simd_sum(local_sum_b);
    if (simd_lane == 0) {
        partial_sums_a[simd_group] = simd_sum_a;
        partial_sums_b[simd_group] = simd_sum_b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum_a += partial_sums_a[index];
            sum_b += partial_sums_b[index];
        }
        output_a[row] = sum_a;
        output_b[row] = sum_b;
    }
}

kernel void indexed_matvec_q3k_add_weighted_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    device const ZiggyShortlistEntry *entries [[buffer(5)]],
    constant uint &slot_idx [[buffer(6)]],
    constant uint &rows_per_expert [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows || cols == 0) return;

    const uint block_count = cols / ZIGGY_Q3K_VALUES_PER_BLOCK;
    const uint row_bytes = block_count * ZIGGY_Q3K_BYTES_PER_BLOCK;
    const uint expert_index = entries[slot_idx].token_id;
    const float weight = entries[slot_idx].score;
    const uint matrix_row = expert_index * rows_per_expert + row;
    const device uchar *row_ptr = matrix + matrix_row * row_bytes;

    threadgroup float partial_sums[ZIGGY_MAX_ROW_SIMDGROUPS];
    float local_sum = 0.0f;

    for (uint block_index = lane; block_index < block_count; block_index += threads_per_group) {
        local_sum += ziggy_dot_q3k_block(
            row_ptr + block_index * ZIGGY_Q3K_BYTES_PER_BLOCK,
            input + block_index * ZIGGY_Q3K_VALUES_PER_BLOCK
        );
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (simd_lane == 0) partial_sums[simd_group] = simd_sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float sum = 0.0f;
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            sum += partial_sums[index];
        }
        output[row] += sum * weight;
    }
}

inline float ziggy_softplus(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

inline float ziggy_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float ziggy_silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void linear_conv1d_f32(
    device const float *qkv [[buffer(0)]],
    device float *conv_state [[buffer(1)]],
    device const float *conv_weights [[buffer(2)]],
    device float *conv_out [[buffer(3)]],
    constant uint &layer_index [[buffer(4)]],
    constant uint &block_count [[buffer(5)]],
    constant uint &kernel_dim [[buffer(6)]],
    constant uint &qkv_dim [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= qkv_dim) return;

    const uint channel = tid;
    const uint conv_state_per_layer = (kernel_dim - 1) * qkv_dim;
    const uint state_base = layer_index * conv_state_per_layer;

    float acc = 0.0f;
    for (uint k = 0; k < kernel_dim - 1; k++) {
        uint state_idx = state_base + k * qkv_dim + channel;
        uint weight_idx = k * qkv_dim + channel;
        acc += conv_state[state_idx] * conv_weights[weight_idx];
    }
    uint final_weight_idx = (kernel_dim - 1) * qkv_dim + channel;
    acc += qkv[channel] * conv_weights[final_weight_idx];
    conv_out[channel] = ziggy_silu(acc);

    if (kernel_dim > 1) {
        for (uint k = 0; k < kernel_dim - 2; k++) {
            uint dst = state_base + k * qkv_dim + channel;
            uint src = state_base + (k + 1) * qkv_dim + channel;
            conv_state[dst] = conv_state[src];
        }
        uint last = state_base + (kernel_dim - 2) * qkv_dim + channel;
        conv_state[last] = qkv[channel];
    }
}

kernel void linear_recurrent_norm_f32(
    device const float *conv_out [[buffer(0)]],
    device float *recurrent_state [[buffer(1)]],
    device const float *z [[buffer(2)]],
    device const float *a [[buffer(3)]],
    device const float *b [[buffer(4)]],
    device const float *dt_bias [[buffer(5)]],
    device const float *A_log [[buffer(6)]],
    device const float *norm_weights [[buffer(7)]],
    device float *out [[buffer(8)]],
    constant uint &layer_index [[buffer(9)]],
    constant uint &num_key_heads [[buffer(10)]],
    constant uint &num_value_heads [[buffer(11)]],
    constant uint &key_head_dim [[buffer(12)]],
    constant uint &value_head_dim [[buffer(13)]],
    constant uint &qkv_dim [[buffer(14)]],
    constant float &rms_norm_eps [[buffer(15)]],
    constant float &scale [[buffer(16)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    const uint head = tid / value_head_dim;
    const uint col = lid;

    if (head >= num_value_heads) return;
    if (col >= value_head_dim) return;

    const uint q_dim = num_key_heads * key_head_dim;
    const uint v_dim = num_value_heads * value_head_dim;

    const float a_val = a[head];
    const float dt_bias_val = dt_bias[head];
    const float a_log_val = A_log[head];
    const float gate = a_log_val * ziggy_softplus(a_val + dt_bias_val);
    const float beta = ziggy_sigmoid(b[head]);
    const float decay = exp(gate);

    const uint qk_head = head % num_key_heads;
    const uint q_base = qk_head * key_head_dim;
    const uint k_base = qk_head * key_head_dim;
    const uint v_base = head * value_head_dim;

    threadgroup float tg_q_scale;
    threadgroup float tg_k_scale;
    threadgroup float tg_norm_scale;
    threadgroup float tg_out[256];

    if (col == 0) {
        float q_sum_sq = 0.0f;
        float k_sum_sq = 0.0f;
        for (uint i = 0; i < key_head_dim; i++) {
            q_sum_sq += conv_out[q_base + i] * conv_out[q_base + i];
            k_sum_sq += conv_out[q_dim + k_base + i] * conv_out[q_dim + k_base + i];
        }
        float q_norm = sqrt(q_sum_sq);
        float k_norm = sqrt(k_sum_sq);
        tg_q_scale = 1.0f / max(q_norm, rms_norm_eps);
        tg_k_scale = 1.0f / max(k_norm, rms_norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_scale = tg_q_scale;
    float k_scale = tg_k_scale;

    const uint state_per_head = key_head_dim * value_head_dim;
    const uint state_per_layer = num_value_heads * state_per_head;
    const uint state_base = layer_index * state_per_layer + head * state_per_head;

    for (uint row = 0; row < key_head_dim; row++) {
        recurrent_state[state_base + row * value_head_dim + col] *= decay;
    }

    float sk = 0.0f;
    for (uint row = 0; row < key_head_dim; row++) {
        sk += recurrent_state[state_base + row * value_head_dim + col] * conv_out[q_dim + k_base + row] * k_scale;
    }

    float delta = (conv_out[q_dim + q_dim + v_base + col] - sk) * beta;
    for (uint row = 0; row < key_head_dim; row++) {
        recurrent_state[state_base + row * value_head_dim + col] += conv_out[q_dim + k_base + row] * k_scale * delta;
    }

    float output = 0.0f;
    for (uint row = 0; row < key_head_dim; row++) {
        output += recurrent_state[state_base + row * value_head_dim + col] * conv_out[q_base + row] * q_scale * scale;
    }

    tg_out[col] = output;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col == 0) {
        float sum_sq = 0.0f;
        for (uint i = 0; i < value_head_dim; i++) {
            sum_sq += tg_out[i] * tg_out[i];
        }
        tg_norm_scale = 1.0f / sqrt(sum_sq / float(value_head_dim) + rms_norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float z_val = z[v_base + col];
    out[v_base + col] = tg_out[col] * tg_norm_scale * norm_weights[col] * ziggy_silu(z_val);
}

kernel void split_packed_q_f32(
    device const float *packed [[buffer(0)]],
    device float *q [[buffer(1)]],
    device float *q_gate [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= head_count * head_dim) return;
    const uint head = idx / head_dim;
    const uint col = idx % head_dim;
    const uint packed_base = head * head_dim * 2;
    q[idx] = packed[packed_base + col];
    q_gate[idx] = packed[packed_base + head_dim + col];
}

kernel void sigmoid_mul_gate_f32(
    device float *output [[buffer(0)]],
    device const float *q_gate [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    output[idx] *= ziggy_sigmoid(q_gate[idx]);
}
