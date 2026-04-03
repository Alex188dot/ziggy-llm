#include <metal_stdlib>
using namespace metal;

constant uint ZIGGY_MAX_ROW_SIMDGROUPS = 8;

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
constant uint ZIGGY_Q6K_VALUES_PER_BLOCK = 256;
constant uint ZIGGY_Q6K_BYTES_PER_BLOCK = 210;
constant uint ZIGGY_Q6K_CHUNKS_PER_BLOCK = 64;
constant uint ZIGGY_Q8_0_VALUES_PER_BLOCK = 32;
constant uint ZIGGY_Q8_0_BYTES_PER_BLOCK = 34;
constant uint ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK = 160;
constant uint ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0 = 2048;
constant uint ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1 = 5632;

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
    const uint chunks_per_row = blocks_per_row * ZIGGY_Q6K_CHUNKS_PER_BLOCK;
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint chunk_index = lane; chunk_index < chunks_per_row; chunk_index += threads_per_group) {
        const uint block_index = chunk_index / ZIGGY_Q6K_CHUNKS_PER_BLOCK;
        const uint block_chunk = chunk_index % ZIGGY_Q6K_CHUNKS_PER_BLOCK;
        const uint block_half = block_chunk / 32;
        const uint l = block_chunk % 32;
        const device uchar *block = row_bytes + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK;
        const device uchar *ql = block + block_half * 64;
        const device uchar *qh = block + 128 + block_half * 32;
        const device uchar *scales = block + 192 + block_half * 8;
        const float d = read_half_le(block, 208);
        const uint scale_index = l / 16;
        const float s0 = d * float(as_type<char>(scales[scale_index + 0]));
        const float s2 = d * float(as_type<char>(scales[scale_index + 2]));
        const float s4 = d * float(as_type<char>(scales[scale_index + 4]));
        const float s6 = d * float(as_type<char>(scales[scale_index + 6]));
        const uchar qh_value = qh[l];
        const uchar ql_low = ql[l];
        const uchar ql_high = ql[l + 32];
        const float q1 = float(int(ql_low & 0x0F) | (int((qh_value >> 0) & 0x03) << 4)) - 32.0f;
        const float q2 = float(int(ql_high & 0x0F) | (int((qh_value >> 2) & 0x03) << 4)) - 32.0f;
        const float q3 = float(int(ql_low >> 4) | (int((qh_value >> 4) & 0x03) << 4)) - 32.0f;
        const float q4 = float(int(ql_high >> 4) | (int((qh_value >> 6) & 0x03) << 4)) - 32.0f;
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l;

        local_sum += s0 * q1 * input[input_offset + 0];
        local_sum += s2 * q2 * input[input_offset + 32];
        local_sum += s4 * q3 * input[input_offset + 64];
        local_sum += s6 * q4 * input[input_offset + 96];
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
        const float d = read_half_le(block, 0);
        const uint input_offset = block_index * ZIGGY_Q8_0_VALUES_PER_BLOCK;
        for (uint value_index = 0; value_index < ZIGGY_Q8_0_VALUES_PER_BLOCK; value_index += 1) {
            const float q = float(as_type<char>(block[2 + value_index]));
            local_sum += d * q * input[input_offset + value_index];
        }
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
        const float d = read_half_le(block, 0);
        const uint input_offset = block_index * ZIGGY_Q8_0_VALUES_PER_BLOCK;
        for (uint value_index = 0; value_index < ZIGGY_Q8_0_VALUES_PER_BLOCK; value_index += 1) {
            const float q = float(as_type<char>(block[2 + value_index]));
            local_sum += d * q * input[input_offset + value_index];
        }
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
    const uint chunks_per_row = blocks_per_row * ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    for (uint chunk_index = lane; chunk_index < chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_index = chunk_index / ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_chunk = chunk_index % ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_half = block_chunk / 32; \
        const uint l = block_chunk % 32; \
        const device uchar *block = row_bytes + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; \
        const device uchar *ql = block + block_half * 64; \
        const device uchar *qh = block + 128 + block_half * 32; \
        const device uchar *scales = block + 192 + block_half * 8; \
        const float d = read_half_le(block, 208); \
        const uint scale_index = l / 16; \
        const float s0 = d * float(as_type<char>(scales[scale_index + 0])); \
        const float s2 = d * float(as_type<char>(scales[scale_index + 2])); \
        const float s4 = d * float(as_type<char>(scales[scale_index + 4])); \
        const float s6 = d * float(as_type<char>(scales[scale_index + 6])); \
        const uchar qh_value = qh[l]; \
        const uchar ql_low = ql[l]; \
        const uchar ql_high = ql[l + 32]; \
        const float q1 = float(int(ql_low & 0x0F) | (int((qh_value >> 0) & 0x03) << 4)) - 32.0f; \
        const float q2 = float(int(ql_high & 0x0F) | (int((qh_value >> 2) & 0x03) << 4)) - 32.0f; \
        const float q3 = float(int(ql_low >> 4) | (int((qh_value >> 4) & 0x03) << 4)) - 32.0f; \
        const float q4 = float(int(ql_high >> 4) | (int((qh_value >> 6) & 0x03) << 4)) - 32.0f; \
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l; \
        local_sum += s0 * q1 * input[input_offset + 0]; \
        local_sum += s2 * q2 * input[input_offset + 32]; \
        local_sum += s4 * q3 * input[input_offset + 64]; \
        local_sum += s6 * q4 * input[input_offset + 96]; \
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
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
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
        const device uchar *block = row_bytes + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
        const float d = read_half_le(block, 0); \
        const float dmin = read_half_le(block, 2); \
        const device uchar *block_scales = block + 4; \
        const device uchar *block_mins = block + 12; \
        const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
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
        const float d1 = d * float(block_scales[scale_index + 0]); \
        const float m1 = dmin * float(block_mins[scale_index + 0]); \
        const float d2 = d * float(block_scales[scale_index + 1]); \
        const float m2 = dmin * float(block_mins[scale_index + 1]); \
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
        if (ADD_TO_DST) output[row] += sum; else output[row] = sum; \
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
    constant uint &head_count [[buffer(1)]],
    constant uint &head_dim [[buffer(2)]],
    constant uint &pair_count [[buffer(3)]],
    constant uint &position [[buffer(4)]],
    constant float &freq_base [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (pair_count == 0) return;
    const uint total_pairs = head_count * pair_count;
    if (index >= total_pairs) return;

    const uint head = index / pair_count;
    const uint pair = index % pair_count;
    const uint base = head * head_dim + pair * 2;
    const float exponent = float(pair * 2) / float(pair_count * 2);
    const float theta = float(position) / pow(freq_base, exponent);
    const float cos_theta = cos(theta);
    const float sin_theta = sin(theta);
    const float x0 = vector[base];
    const float x1 = vector[base + 1];
    vector[base] = x0 * cos_theta - x1 * sin_theta;
    vector[base + 1] = x0 * sin_theta + x1 * cos_theta;
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
}

constant uint ZIGGY_MAX_ATTENTION_CONTEXT = 4096;

kernel void attention_fused_f32(
    device const float *q [[buffer(0)]],
    device const float *k_cache [[buffer(1)]],
    device const float *v_cache [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &head_count [[buffer(4)]],
    constant uint &head_count_kv [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &kv_dim [[buffer(7)]],
    constant uint &context_length [[buffer(8)]],
    constant uint &position [[buffer(9)]],
    constant uint &layer_base [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (head >= head_count || context_length > ZIGGY_MAX_ATTENTION_CONTEXT) return;
    const uint kv_group = head_count / head_count_kv;
    const uint kv_head = head / kv_group;
    const uint kv_offset = kv_head * head_dim;
    device const float *q_head = q + head * head_dim;
    threadgroup float scores[ZIGGY_MAX_ATTENTION_CONTEXT];
    threadgroup float partial_values[ZIGGY_MAX_ROW_SIMDGROUPS];
    const uint token_count = position + 1;

    float local_max = -INFINITY;
    for (uint token = lane; token < token_count; token += threads_per_group) {
        device const float *k_head = k_cache + layer_base + token * kv_dim + kv_offset;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += 1) {
            dot += q_head[d] * k_head[d];
        }
        const float value = dot * scale;
        scores[token] = value;
        local_max = max(local_max, value);
    }

    float simd_max_value = simd_max(local_max);
    if (simd_lane == 0) partial_values[simd_group] = simd_max_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_value = -INFINITY;
    if (lane == 0) {
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            max_value = max(max_value, partial_values[index]);
        }
        partial_values[0] = max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_value = partial_values[0];

    float local_denom = 0.0f;
    for (uint token = lane; token < token_count; token += threads_per_group) {
        const float shifted = exp(scores[token] - max_value);
        scores[token] = shifted;
        local_denom += shifted;
    }

    const float simd_denom_value = simd_sum(local_denom);
    if (simd_lane == 0) partial_values[simd_group] = simd_denom_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float denom = 0.0f;
    if (lane == 0) {
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
        for (uint index = 0; index < simd_group_count; index += 1) {
            denom += partial_values[index];
        }
        partial_values[0] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denom = partial_values[0];

    for (uint token = lane; token < token_count; token += threads_per_group) {
        scores[token] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < head_dim; dim += threads_per_group) {
        float sum = 0.0f;
        for (uint token = 0; token < token_count; token += 1) {
            sum += scores[token] * v_cache[layer_base + token * kv_dim + kv_offset + dim];
        }
        output[head * head_dim + dim] = sum;
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
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;

    float sum = 0.0f;
    for (uint i = 0; i < count; i += 1) {
        const float value = input[i];
        sum += value * value;
    }
    const float scale = 1.0f / sqrt(sum / float(count) + eps);
    output[index] = input[index] * scale * weights[index];
}

kernel void argmax_f32(
    device const float *input [[buffer(0)]],
    device uint *output_token [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0 || count == 0) return;
    uint best_index = 0;
    float best_value = input[0];
    for (uint i = 1; i < count; i += 1) {
        const float value = input[i];
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    output_token[0] = best_index;
}
