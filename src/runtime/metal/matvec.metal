
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
constant uint ZIGGY_Q6K_VALUES_PER_BLOCK = 256;
constant uint ZIGGY_Q6K_BYTES_PER_BLOCK = 210;
constant uint ZIGGY_Q6K_CHUNKS_PER_BLOCK = 64;
constant uint ZIGGY_Q8_0_VALUES_PER_BLOCK = 32;
constant uint ZIGGY_Q8_0_BYTES_PER_BLOCK = 34;
constant uint ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK = 160;
constant uint ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0 = 2048;
constant uint ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1 = 5632;
constant uint ZIGGY_MAX_HEAD_DIM = 256;

inline float ziggy_normed_value(
    device const float *input,
    device const float *weights,
    float scale,
    uint index
) {
    return input[index] * weights[index] * scale;
}

inline float4 ziggy_normed_value4(
    device const float *input,
    device const float *weights,
    float scale,
    uint index
) {
    return float4(
        ziggy_normed_value(input, weights, scale, index + 0),
        ziggy_normed_value(input, weights, scale, index + 1),
        ziggy_normed_value(input, weights, scale, index + 2),
        ziggy_normed_value(input, weights, scale, index + 3)
    );
}

kernel void rms_norm_scale_f32(
    device const float *input [[buffer(0)]],
    device float *output_scale [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    threadgroup float partial_sums[ZIGGY_MAX_NORM_SIMDGROUPS];

    float local_sum = 0.0f;
    for (uint index = lane; index < count; index += threads_per_group) {
        const float value = input[index];
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
        output_scale[0] = 1.0f / sqrt(sum / float(count) + eps);
    }
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

kernel void pack_kv_half_f32(
    device const float *k_src [[buffer(0)]],
    device const float *v_src [[buffer(1)]],
    device half *k_dst [[buffer(2)]],
    device half *v_dst [[buffer(3)]],
    constant uint &dst_base [[buffer(4)]],
    constant uint &head_count [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &pair_count [[buffer(7)]],
    constant uint &position [[buffer(8)]],
    constant float &freq_base [[buffer(9)]],
    constant uint &rope_style [[buffer(10)]],
    uint index [[thread_position_in_grid]]
) {
    const uint total_values = head_count * head_dim;
    if (index >= total_values) return;

    const uint head = index / head_dim;
    const uint dim = index % head_dim;
    const uint dst_index = dst_base + index;
    v_dst[dst_index] = half(v_src[index]);

    if (pair_count == 0 || dim >= pair_count * 2) {
        k_dst[dst_index] = half(k_src[index]);
        return;
    }

    float result = 0.0f;
    if (rope_style == 0) {
        const uint pair = dim / 2;
        const uint base = head * head_dim + pair * 2;
        const float exponent = float(pair * 2) / float(pair_count * 2);
        const float theta = float(position) / pow(freq_base, exponent);
        const float cos_theta = cos(theta);
        const float sin_theta = sin(theta);
        const float x0 = k_src[base];
        const float x1 = k_src[base + 1];

        if ((dim & 1) == 0) {
            result = x0 * cos_theta - x1 * sin_theta;
        } else {
            result = x0 * sin_theta + x1 * cos_theta;
        }
    } else {
        const uint pair = dim % pair_count;
        const bool is_first_half = dim < pair_count;
        const uint base = head * head_dim;
        const float exponent = float(pair * 2) / float(pair_count * 2);
        const float theta = float(position) / pow(freq_base, exponent);
        const float cos_theta = cos(theta);
        const float sin_theta = sin(theta);
        const float x0 = k_src[base + pair];
        const float x1 = k_src[base + pair + pair_count];

        if (is_first_half) {
            result = x0 * cos_theta - x1 * sin_theta;
        } else {
            result = x0 * sin_theta + x1 * cos_theta;
        }
    }
    k_dst[dst_index] = half(result);
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

#define ZIGGY_Q4K_DUAL_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix0 [[buffer(0)]], \
    device const uchar *matrix1 [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device float *output0 [[buffer(3)]], \
    device float *output1 [[buffer(4)]], \
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
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix0 + row * row_stride; \
    const device uchar *row_bytes1 = matrix1 + row * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_sums0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_sum0 = 0.0f; \
    float local_sum1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
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
        { \
            const device uchar *block = row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local_sum0 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local_sum0 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local_sum1 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local_sum1 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
    } \
    const float simd_sum_value0 = simd_sum(local_sum0); \
    const float simd_sum_value1 = simd_sum(local_sum1); \
    if (simd_lane == 0) { \
        partial_sums0[simd_group] = simd_sum_value0; \
        partial_sums1[simd_group] = simd_sum_value1; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float sum0 = 0.0f; \
        float sum1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            sum0 += partial_sums0[index]; \
            sum1 += partial_sums1[index]; \
        } \
        output0[row] = sum0; \
        output1[row] = sum1; \
    } \
}

ZIGGY_Q4K_DUAL_KERNEL(matvec_q4k_dual_f32, 0)
ZIGGY_Q4K_DUAL_KERNEL(matvec_q4k_dual_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_DUAL_KERNEL(matvec_q4k_dual_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_Q4K_DUAL_KERNEL

#define ZIGGY_Q4K_K_HALF_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device half *k_cache [[buffer(2)]], \
    constant uint &dst_base [[buffer(3)]], \
    constant uint &head_count [[buffer(4)]], \
    constant uint &head_dim [[buffer(5)]], \
    constant uint &pair_count [[buffer(6)]], \
    constant uint &cols [[buffer(7)]], \
    constant uint &position [[buffer(8)]], \
    constant float &freq_base [[buffer(9)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + row0 * row_stride; \
    const device uchar *row_bytes1 = matrix + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local0 = 0.0f; \
    float local1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = float4(input[input_offset + 0], input[input_offset + 1], input[input_offset + 2], input[input_offset + 3]); \
        const float4 input_high = float4(input[input_offset + 32 + 0], input[input_offset + 32 + 1], input[input_offset + 32 + 2], input[input_offset + 32 + 3]); \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local0 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local0 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local1 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local1 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
    } \
    const float sum0 = simd_sum(local0); \
    const float sum1 = simd_sum(local1); \
    if (simd_lane == 0) { \
        partial0[simd_group] = sum0; \
        partial1[simd_group] = sum1; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            k0 += partial0[index]; \
            k1 += partial1[index]; \
        } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
    } \
}

ZIGGY_Q4K_K_HALF_KERNEL(matvec_q4k_k_half_f32, 0)
ZIGGY_Q4K_K_HALF_KERNEL(matvec_q4k_k_half_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_K_HALF_KERNEL(matvec_q4k_k_half_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_Q4K_K_HALF_RMS_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device const float *norm_weights [[buffer(2)]], \
    device const float *norm_scale [[buffer(3)]], \
    device half *k_cache [[buffer(4)]], \
    constant uint &dst_base [[buffer(5)]], \
    constant uint &head_count [[buffer(6)]], \
    constant uint &head_dim [[buffer(7)]], \
    constant uint &pair_count [[buffer(8)]], \
    constant uint &cols [[buffer(9)]], \
    constant uint &position [[buffer(10)]], \
    constant float &freq_base [[buffer(11)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const float input_scale = norm_scale[0]; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + row0 * row_stride; \
    const device uchar *row_bytes1 = matrix + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local0 = 0.0f; \
    float local1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset); \
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32); \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local0 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local0 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local1 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local1 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
    } \
    const float sum0 = simd_sum(local0); \
    const float sum1 = simd_sum(local1); \
    if (simd_lane == 0) { partial0[simd_group] = sum0; partial1[simd_group] = sum1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float final0 = 0.0f, final1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            final0 += partial0[index]; \
            final1 += partial1[index]; \
        } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(final0 * cos_theta - final1 * sin_theta); \
        k_cache[dst_base + row1] = half(final0 * sin_theta + final1 * cos_theta); \
    } \
}

ZIGGY_Q4K_K_HALF_RMS_KERNEL(matvec_q4k_k_half_rms_f32, 0)
ZIGGY_Q4K_K_HALF_RMS_KERNEL(matvec_q4k_k_half_rms_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_K_HALF_RMS_KERNEL(matvec_q4k_k_half_rms_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_Q4K_K_HALF_RMS_KERNEL

#undef ZIGGY_Q4K_K_HALF_KERNEL

#define ZIGGY_Q4K_Q_ROPE_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &head_count [[buffer(3)]], \
    constant uint &head_dim [[buffer(4)]], \
    constant uint &pair_count [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
    constant uint &position [[buffer(7)]], \
    constant float &freq_base [[buffer(8)]], \
    constant uint &rope_style [[buffer(9)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = rope_style == 0 ? (head * head_dim + pair * 2) : (head * head_dim + pair); \
    const uint row1 = rope_style == 0 ? (row0 + 1) : (row0 + pair_count); \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + row0 * row_stride; \
    const device uchar *row_bytes1 = matrix + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float sum0 = 0.0f; \
    float sum1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = float4(input[input_offset + 0], input[input_offset + 1], input[input_offset + 2], input[input_offset + 3]); \
        const float4 input_high = float4(input[input_offset + 32 + 0], input[input_offset + 32 + 1], input[input_offset + 32 + 2], input[input_offset + 32 + 3]); \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            sum0 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            sum0 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            sum1 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            sum1 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
    } \
    const float s0 = simd_sum(sum0); \
    const float s1 = simd_sum(sum1); \
    if (simd_lane == 0) { partial0[simd_group] = s0; partial1[simd_group] = s1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float fs0 = 0.0f, fs1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { fs0 += partial0[index]; fs1 += partial1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        output[row0] = fs0 * cos_theta - fs1 * sin_theta; \
        output[row1] = fs0 * sin_theta + fs1 * cos_theta; \
    } \
}

ZIGGY_Q4K_Q_ROPE_KERNEL(matvec_q4k_q_rope_f32, 0)
ZIGGY_Q4K_Q_ROPE_KERNEL(matvec_q4k_q_rope_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_Q_ROPE_KERNEL(matvec_q4k_q_rope_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_Q4K_Q_ROPE_RMS_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device const float *norm_weights [[buffer(2)]], \
    device const float *norm_scale [[buffer(3)]], \
    device float *output [[buffer(4)]], \
    constant uint &head_count [[buffer(5)]], \
    constant uint &head_dim [[buffer(6)]], \
    constant uint &pair_count [[buffer(7)]], \
    constant uint &cols [[buffer(8)]], \
    constant uint &position [[buffer(9)]], \
    constant float &freq_base [[buffer(10)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const float input_scale = norm_scale[0]; \
    const uint row0 = row_pair * 2; \
    const uint row1 = row0 + 1; \
    const uint pair = row_pair % pair_count; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + row0 * row_stride; \
    const device uchar *row_bytes1 = matrix + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float sum0 = 0.0f; \
    float sum1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset); \
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32); \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            sum0 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            sum0 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            sum1 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            sum1 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
    } \
    const float s0 = simd_sum(sum0); \
    const float s1 = simd_sum(sum1); \
    if (simd_lane == 0) { partial0[simd_group] = s0; partial1[simd_group] = s1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float fs0 = 0.0f, fs1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { fs0 += partial0[index]; fs1 += partial1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        output[row0] = fs0 * cos_theta - fs1 * sin_theta; \
        output[row1] = fs0 * sin_theta + fs1 * cos_theta; \
    } \
}

ZIGGY_Q4K_Q_ROPE_RMS_KERNEL(matvec_q4k_q_rope_rms_f32, 0)
ZIGGY_Q4K_Q_ROPE_RMS_KERNEL(matvec_q4k_q_rope_rms_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_Q_ROPE_RMS_KERNEL(matvec_q4k_q_rope_rms_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_Q4K_Q_ROPE_RMS_KERNEL

#undef ZIGGY_Q4K_Q_ROPE_KERNEL

#define ZIGGY_Q4K_DUAL_KV_HALF_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix_k [[buffer(0)]], \
    device const uchar *matrix_v [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device half *k_cache [[buffer(3)]], \
    device half *v_cache [[buffer(4)]], \
    constant uint &dst_base [[buffer(5)]], \
    constant uint &head_count [[buffer(6)]], \
    constant uint &head_dim [[buffer(7)]], \
    constant uint &pair_count [[buffer(8)]], \
    constant uint &cols [[buffer(9)]], \
    constant uint &position [[buffer(10)]], \
    constant float &freq_base [[buffer(11)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *k_row_bytes0 = matrix_k + row0 * row_stride; \
    const device uchar *k_row_bytes1 = matrix_k + row1 * row_stride; \
    const device uchar *v_row_bytes0 = matrix_v + row0 * row_stride; \
    const device uchar *v_row_bytes1 = matrix_v + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_k0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_k1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_k0 = 0.0f; \
    float local_k1 = 0.0f; \
    float local_v0 = 0.0f; \
    float local_v1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = float4(input[input_offset + 0], input[input_offset + 1], input[input_offset + 2], input[input_offset + 3]); \
        const float4 input_high = float4(input[input_offset + 32 + 0], input[input_offset + 32 + 1], input[input_offset + 32 + 2], input[input_offset + 32 + 3]); \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = k_row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local_k0 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local_k0 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = k_row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local_k1 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local_k1 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = v_row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local_v0 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local_v0 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = v_row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local_v1 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local_v1 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
    } \
    const float sum_k0 = simd_sum(local_k0); \
    const float sum_k1 = simd_sum(local_k1); \
    const float sum_v0 = simd_sum(local_v0); \
    const float sum_v1 = simd_sum(local_v1); \
    if (simd_lane == 0) { \
        partial_k0[simd_group] = sum_k0; \
        partial_k1[simd_group] = sum_k1; \
        partial_v0[simd_group] = sum_v0; \
        partial_v1[simd_group] = sum_v1; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            k0 += partial_k0[index]; \
            k1 += partial_k1[index]; \
            v0 += partial_v0[index]; \
            v1 += partial_v1[index]; \
        } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
        v_cache[dst_base + row0] = half(v0); \
        v_cache[dst_base + row1] = half(v1); \
    } \
}

ZIGGY_Q4K_DUAL_KV_HALF_KERNEL(matvec_q4k_dual_kv_half_f32, 0)
ZIGGY_Q4K_DUAL_KV_HALF_KERNEL(matvec_q4k_dual_kv_half_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_DUAL_KV_HALF_KERNEL(matvec_q4k_dual_kv_half_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_Q4K_DUAL_KV_HALF_RMS_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix_k [[buffer(0)]], \
    device const uchar *matrix_v [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device const float *norm_weights [[buffer(3)]], \
    device const float *norm_scale [[buffer(4)]], \
    device half *k_cache [[buffer(5)]], \
    device half *v_cache [[buffer(6)]], \
    constant uint &dst_base [[buffer(7)]], \
    constant uint &head_count [[buffer(8)]], \
    constant uint &head_dim [[buffer(9)]], \
    constant uint &pair_count [[buffer(10)]], \
    constant uint &cols [[buffer(11)]], \
    constant uint &position [[buffer(12)]], \
    constant float &freq_base [[buffer(13)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const float input_scale = norm_scale[0]; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const device uchar *k_row_bytes0 = matrix_k + row0 * row_stride; \
    const device uchar *k_row_bytes1 = matrix_k + row1 * row_stride; \
    const device uchar *v_row_bytes0 = matrix_v + row0 * row_stride; \
    const device uchar *v_row_bytes1 = matrix_v + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_k0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_k1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_k0 = 0.0f; \
    float local_k1 = 0.0f; \
    float local_v0 = 0.0f; \
    float local_v1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset); \
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32); \
        const uint scale_index = group * 2; \
        { const device uchar *block = k_row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const device uchar *scales = block + 4; const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); const uchar4 low_q = packed & uchar4(0x0F); const uchar4 high_q = packed >> 4; const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); local_k0 += dot(float4(low_q) * d1 - float4(m1), input_low); local_k0 += dot(float4(high_q) * d2 - float4(m2), input_high); } \
        { const device uchar *block = k_row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const device uchar *scales = block + 4; const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); const uchar4 low_q = packed & uchar4(0x0F); const uchar4 high_q = packed >> 4; const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); local_k1 += dot(float4(low_q) * d1 - float4(m1), input_low); local_k1 += dot(float4(high_q) * d2 - float4(m2), input_high); } \
        { const device uchar *block = v_row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const device uchar *scales = block + 4; const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); const uchar4 low_q = packed & uchar4(0x0F); const uchar4 high_q = packed >> 4; const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); local_v0 += dot(float4(low_q) * d1 - float4(m1), input_low); local_v0 += dot(float4(high_q) * d2 - float4(m2), input_high); } \
        { const device uchar *block = v_row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const device uchar *scales = block + 4; const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); const uchar4 low_q = packed & uchar4(0x0F); const uchar4 high_q = packed >> 4; const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); local_v1 += dot(float4(low_q) * d1 - float4(m1), input_low); local_v1 += dot(float4(high_q) * d2 - float4(m2), input_high); } \
    } \
    const float sum_k0 = simd_sum(local_k0); const float sum_k1 = simd_sum(local_k1); const float sum_v0 = simd_sum(local_v0); const float sum_v1 = simd_sum(local_v1); \
    if (simd_lane == 0) { partial_k0[simd_group] = sum_k0; partial_k1[simd_group] = sum_k1; partial_v0[simd_group] = sum_v0; partial_v1[simd_group] = sum_v1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { k0 += partial_k0[index]; k1 += partial_k1[index]; v0 += partial_v0[index]; v1 += partial_v1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
        v_cache[dst_base + row0] = half(v0); \
        v_cache[dst_base + row1] = half(v1); \
    } \
}

ZIGGY_Q4K_DUAL_KV_HALF_RMS_KERNEL(matvec_q4k_dual_kv_half_rms_f32, 0)
ZIGGY_Q4K_DUAL_KV_HALF_RMS_KERNEL(matvec_q4k_dual_kv_half_rms_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_DUAL_KV_HALF_RMS_KERNEL(matvec_q4k_dual_kv_half_rms_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_Q4K_DUAL_KV_HALF_RMS_KERNEL

#undef ZIGGY_Q4K_DUAL_KV_HALF_KERNEL

#define ZIGGY_Q4K_Q6K_DUAL_KV_HALF_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix_k [[buffer(0)]], \
    device const uchar *matrix_v [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device half *k_cache [[buffer(3)]], \
    device half *v_cache [[buffer(4)]], \
    constant uint &dst_base [[buffer(5)]], \
    constant uint &head_count [[buffer(6)]], \
    constant uint &head_dim [[buffer(7)]], \
    constant uint &pair_count [[buffer(8)]], \
    constant uint &cols [[buffer(9)]], \
    constant uint &position [[buffer(10)]], \
    constant float &freq_base [[buffer(11)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint q4_blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint q4_row_stride = q4_blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const uint q6_blocks_per_row = effective_cols / ZIGGY_Q6K_VALUES_PER_BLOCK; \
    const uint q6_row_stride = q6_blocks_per_row * ZIGGY_Q6K_BYTES_PER_BLOCK; \
    const device uchar *k_row_bytes0 = matrix_k + row0 * q4_row_stride; \
    const device uchar *k_row_bytes1 = matrix_k + row1 * q4_row_stride; \
    const device uchar *v_row_bytes0 = matrix_v + row0 * q6_row_stride; \
    const device uchar *v_row_bytes1 = matrix_v + row1 * q6_row_stride; \
    const uint q4_chunks_per_row = q4_blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    const uint q6_chunks_per_row = q6_blocks_per_row * ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
    threadgroup float partial_k0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_k1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_k0 = 0.0f; \
    float local_k1 = 0.0f; \
    float local_v0 = 0.0f; \
    float local_v1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < q4_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = float4(input[input_offset + 0], input[input_offset + 1], input[input_offset + 2], input[input_offset + 3]); \
        const float4 input_high = float4(input[input_offset + 32 + 0], input[input_offset + 32 + 1], input[input_offset + 32 + 2], input[input_offset + 32 + 3]); \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = k_row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local_k0 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local_k0 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
        { \
            const device uchar *block = k_row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const device uchar *scales = block + 4; \
            const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); \
            const uchar4 low_q = packed & uchar4(0x0F); \
            const uchar4 high_q = packed >> 4; \
            const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); \
            const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); \
            const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); \
            const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); \
            local_k1 += dot(float4(low_q) * d1 - float4(m1), input_low); \
            local_k1 += dot(float4(high_q) * d2 - float4(m2), input_high); \
        } \
    } \
    for (uint chunk_index = lane; chunk_index < q6_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_index = chunk_index / ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_chunk = chunk_index % ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_half = block_chunk / 32; \
        const uint l = block_chunk % 32; \
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l; \
        { \
            const device uchar *block = v_row_bytes0 + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; \
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
            local_v0 += s0 * q1 * input[input_offset + 0]; \
            local_v0 += s2 * q2 * input[input_offset + 32]; \
            local_v0 += s4 * q3 * input[input_offset + 64]; \
            local_v0 += s6 * q4 * input[input_offset + 96]; \
        } \
        { \
            const device uchar *block = v_row_bytes1 + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; \
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
            local_v1 += s0 * q1 * input[input_offset + 0]; \
            local_v1 += s2 * q2 * input[input_offset + 32]; \
            local_v1 += s4 * q3 * input[input_offset + 64]; \
            local_v1 += s6 * q4 * input[input_offset + 96]; \
        } \
    } \
    const float sum_k0 = simd_sum(local_k0); \
    const float sum_k1 = simd_sum(local_k1); \
    const float sum_v0 = simd_sum(local_v0); \
    const float sum_v1 = simd_sum(local_v1); \
    if (simd_lane == 0) { \
        partial_k0[simd_group] = sum_k0; \
        partial_k1[simd_group] = sum_k1; \
        partial_v0[simd_group] = sum_v0; \
        partial_v1[simd_group] = sum_v1; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            k0 += partial_k0[index]; \
            k1 += partial_k1[index]; \
            v0 += partial_v0[index]; \
            v1 += partial_v1[index]; \
        } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
        v_cache[dst_base + row0] = half(v0); \
        v_cache[dst_base + row1] = half(v1); \
    } \
}

ZIGGY_Q4K_Q6K_DUAL_KV_HALF_KERNEL(matvec_q4k_q6k_dual_kv_half_f32, 0)
ZIGGY_Q4K_Q6K_DUAL_KV_HALF_KERNEL(matvec_q4k_q6k_dual_kv_half_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_Q6K_DUAL_KV_HALF_KERNEL(matvec_q4k_q6k_dual_kv_half_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix_k [[buffer(0)]], \
    device const uchar *matrix_v [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device const float *norm_weights [[buffer(3)]], \
    device const float *norm_scale [[buffer(4)]], \
    device half *k_cache [[buffer(5)]], \
    device half *v_cache [[buffer(6)]], \
    constant uint &dst_base [[buffer(7)]], \
    constant uint &head_count [[buffer(8)]], \
    constant uint &head_dim [[buffer(9)]], \
    constant uint &pair_count [[buffer(10)]], \
    constant uint &cols [[buffer(11)]], \
    constant uint &position [[buffer(12)]], \
    constant float &freq_base [[buffer(13)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const float input_scale = norm_scale[0]; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride_k = blocks_per_row * ZIGGY_Q4K_BYTES_PER_BLOCK; \
    const uint row_stride_v = (effective_cols / ZIGGY_Q6K_VALUES_PER_BLOCK) * ZIGGY_Q6K_BYTES_PER_BLOCK; \
    const device uchar *k_row_bytes0 = matrix_k + row0 * row_stride_k; \
    const device uchar *k_row_bytes1 = matrix_k + row1 * row_stride_k; \
    const device uchar *v_row_bytes0 = matrix_v + row0 * row_stride_v; \
    const device uchar *v_row_bytes1 = matrix_v + row1 * row_stride_v; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_k0[ZIGGY_MAX_Q4K_SIMDGROUPS]; threadgroup float partial_k1[ZIGGY_MAX_Q4K_SIMDGROUPS]; threadgroup float partial_v0[ZIGGY_MAX_Q4K_SIMDGROUPS]; threadgroup float partial_v1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_k0 = 0.0f, local_k1 = 0.0f, local_v0 = 0.0f, local_v1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset); \
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32); \
        const uint scale_index = group * 2; \
        { const device uchar *block = k_row_bytes0 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const device uchar *scales = block + 4; const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); const uchar4 low_q = packed & uchar4(0x0F); const uchar4 high_q = packed >> 4; const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); local_k0 += dot(float4(low_q) * d1 - float4(m1), input_low); local_k0 += dot(float4(high_q) * d2 - float4(m2), input_high); } \
        { const device uchar *block = k_row_bytes1 + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const device uchar *scales = block + 4; const device uchar *q = block + 16 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uchar4 packed = uchar4(q[0], q[1], q[2], q[3]); const uchar4 low_q = packed & uchar4(0x0F); const uchar4 high_q = packed >> 4; const float d1 = d * float(get_scale_k4(scales, scale_index + 0)); const float m1 = dmin * float(get_min_k4(scales, scale_index + 0)); const float d2 = d * float(get_scale_k4(scales, scale_index + 1)); const float m2 = dmin * float(get_min_k4(scales, scale_index + 1)); local_k1 += dot(float4(low_q) * d1 - float4(m1), input_low); local_k1 += dot(float4(high_q) * d2 - float4(m2), input_high); } \
        for (uint v_block_chunk = group * 16 + q_offset; v_block_chunk < group * 16 + q_offset + 4; v_block_chunk += 1) { \
            const uint block_half = v_block_chunk / 32; const uint l = v_block_chunk % 32; const device uchar *block0 = v_row_bytes0 + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; const device uchar *block1 = v_row_bytes1 + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; const device uchar *ql0 = block0 + block_half * 64; const device uchar *qh0 = block0 + 128 + block_half * 32; const device uchar *scales0 = block0 + 192 + block_half * 8; const device uchar *ql1 = block1 + block_half * 64; const device uchar *qh1 = block1 + 128 + block_half * 32; const device uchar *scales1 = block1 + 192 + block_half * 8; const float d0 = read_half_le(block0, 208); const float d1v = read_half_le(block1, 208); const uint s_index = l / 16; const float sv0 = d0 * float(as_type<char>(scales0[s_index + 0])); const float sv2 = d0 * float(as_type<char>(scales0[s_index + 2])); const float sv4 = d0 * float(as_type<char>(scales0[s_index + 4])); const float sv6 = d0 * float(as_type<char>(scales0[s_index + 6])); const float tv0 = d1v * float(as_type<char>(scales1[s_index + 0])); const float tv2 = d1v * float(as_type<char>(scales1[s_index + 2])); const float tv4 = d1v * float(as_type<char>(scales1[s_index + 4])); const float tv6 = d1v * float(as_type<char>(scales1[s_index + 6])); const uchar qh0v = qh0[l]; const uchar ql0_low = ql0[l]; const uchar ql0_high = ql0[l + 32]; const uchar qh1v = qh1[l]; const uchar ql1_low = ql1[l]; const uchar ql1_high = ql1[l + 32]; const float q01 = float(int(ql0_low & 0x0F) | (int((qh0v >> 0) & 0x03) << 4)) - 32.0f; const float q02 = float(int(ql0_high & 0x0F) | (int((qh0v >> 2) & 0x03) << 4)) - 32.0f; const float q03 = float(int(ql0_low >> 4) | (int((qh0v >> 4) & 0x03) << 4)) - 32.0f; const float q04 = float(int(ql0_high >> 4) | (int((qh0v >> 6) & 0x03) << 4)) - 32.0f; const float r01 = float(int(ql1_low & 0x0F) | (int((qh1v >> 0) & 0x03) << 4)) - 32.0f; const float r02 = float(int(ql1_high & 0x0F) | (int((qh1v >> 2) & 0x03) << 4)) - 32.0f; const float r03 = float(int(ql1_low >> 4) | (int((qh1v >> 4) & 0x03) << 4)) - 32.0f; const float r04 = float(int(ql1_high >> 4) | (int((qh1v >> 6) & 0x03) << 4)) - 32.0f; const uint v_input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l; local_v0 += sv0 * q01 * ziggy_normed_value(input, norm_weights, input_scale, v_input_offset + 0); local_v0 += sv2 * q02 * ziggy_normed_value(input, norm_weights, input_scale, v_input_offset + 32); local_v0 += sv4 * q03 * ziggy_normed_value(input, norm_weights, input_scale, v_input_offset + 64); local_v0 += sv6 * q04 * ziggy_normed_value(input, norm_weights, input_scale, v_input_offset + 96); local_v1 += tv0 * r01 * ziggy_normed_value(input, norm_weights, input_scale, v_input_offset + 0); local_v1 += tv2 * r02 * ziggy_normed_value(input, norm_weights, input_scale, v_input_offset + 32); local_v1 += tv4 * r03 * ziggy_normed_value(input, norm_weights, input_scale, v_input_offset + 64); local_v1 += tv6 * r04 * ziggy_normed_value(input, norm_weights, input_scale, v_input_offset + 96); \
        } \
    } \
    const float sum_k0 = simd_sum(local_k0), sum_k1 = simd_sum(local_k1), sum_v0 = simd_sum(local_v0), sum_v1 = simd_sum(local_v1); \
    if (simd_lane == 0) { partial_k0[simd_group] = sum_k0; partial_k1[simd_group] = sum_k1; partial_v0[simd_group] = sum_v0; partial_v1[simd_group] = sum_v1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { k0 += partial_k0[index]; k1 += partial_k1[index]; v0 += partial_v0[index]; v1 += partial_v1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
        v_cache[dst_base + row0] = half(v0); \
        v_cache[dst_base + row1] = half(v1); \
    } \
}

ZIGGY_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL(matvec_q4k_q6k_dual_kv_half_rms_f32, 0)
ZIGGY_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL(matvec_q4k_q6k_dual_kv_half_rms_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL(matvec_q4k_q6k_dual_kv_half_rms_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL

#undef ZIGGY_Q4K_Q6K_DUAL_KV_HALF_KERNEL

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

        device atomic_uint *state_lock = output_state;
        device atomic_uint *best_value = output_state + 1;
        device atomic_uint *best_token = output_state + 2;
        const uint bits = as_type<uint>(sum);
        const uint ordered = (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
        while (true) {
            const uint current_best = atomic_load_explicit(best_value, memory_order_relaxed);
            const uint current_token = atomic_load_explicit(best_token, memory_order_relaxed);
            if (ordered < current_best || (ordered == current_best && row >= current_token)) break;
            if (atomic_exchange_explicit(state_lock, 1u, memory_order_relaxed) == 0u) {
                const uint locked_best = atomic_load_explicit(best_value, memory_order_relaxed);
                const uint locked_token = atomic_load_explicit(best_token, memory_order_relaxed);
                if (ordered > locked_best || (ordered == locked_best && row < locked_token)) {
                    atomic_store_explicit(best_value, ordered, memory_order_relaxed);
                    atomic_store_explicit(best_token, row, memory_order_relaxed);
                }
                atomic_store_explicit(state_lock, 0u, memory_order_relaxed);
                break;
            }
        }
    }
}

kernel void matvec_q6k_argmax_rms_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const float *norm_weights [[buffer(2)]],
    device const float *norm_scale [[buffer(3)]],
    device atomic_uint *output_state [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;
    const float input_scale = norm_scale[0];
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
        local_sum += s0 * q1 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 0);
        local_sum += s2 * q2 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 32);
        local_sum += s4 * q3 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 64);
        local_sum += s6 * q4 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 96);
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
        device atomic_uint *state_lock = output_state;
        device atomic_uint *best_value = output_state + 1;
        device atomic_uint *best_token = output_state + 2;
        const uint bits = as_type<uint>(sum);
        const uint ordered = (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
        while (true) {
            const uint current_best = atomic_load_explicit(best_value, memory_order_relaxed);
            const uint current_token = atomic_load_explicit(best_token, memory_order_relaxed);
            if (ordered < current_best || (ordered == current_best && row >= current_token)) break;
            if (atomic_exchange_explicit(state_lock, 1u, memory_order_relaxed) == 0u) {
                const uint locked_best = atomic_load_explicit(best_value, memory_order_relaxed);
                const uint locked_token = atomic_load_explicit(best_token, memory_order_relaxed);
                if (ordered > locked_best || (ordered == locked_best && row < locked_token)) {
                    atomic_store_explicit(best_value, ordered, memory_order_relaxed);
                    atomic_store_explicit(best_token, row, memory_order_relaxed);
                }
                atomic_store_explicit(state_lock, 0u, memory_order_relaxed);
                break;
            }
        }
    }
}

kernel void matvec_q4k_argmax_f32(
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

        device atomic_uint *state_lock = output_state;
        device atomic_uint *best_value = output_state + 1;
        device atomic_uint *best_token = output_state + 2;
        const uint bits = as_type<uint>(sum);
        const uint ordered = (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
        while (true) {
            const uint current_best = atomic_load_explicit(best_value, memory_order_relaxed);
            const uint current_token = atomic_load_explicit(best_token, memory_order_relaxed);
            if (ordered < current_best || (ordered == current_best && row >= current_token)) break;
            if (atomic_exchange_explicit(state_lock, 1u, memory_order_relaxed) == 0u) {
                const uint locked_best = atomic_load_explicit(best_value, memory_order_relaxed);
                const uint locked_token = atomic_load_explicit(best_token, memory_order_relaxed);
                if (ordered > locked_best || (ordered == locked_best && row < locked_token)) {
                    atomic_store_explicit(best_value, ordered, memory_order_relaxed);
                    atomic_store_explicit(best_token, row, memory_order_relaxed);
                }
                atomic_store_explicit(state_lock, 0u, memory_order_relaxed);
                break;
            }
        }
    }
}

kernel void matvec_q4k_argmax_rms_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const float *norm_weights [[buffer(2)]],
    device const float *norm_scale [[buffer(3)]],
    device atomic_uint *output_state [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (row >= rows) return;
    const float input_scale = norm_scale[0];
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
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset);
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32);
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
        device atomic_uint *state_lock = output_state;
        device atomic_uint *best_value = output_state + 1;
        device atomic_uint *best_token = output_state + 2;
        const uint bits = as_type<uint>(sum);
        const uint ordered = (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
        while (true) {
            const uint current_best = atomic_load_explicit(best_value, memory_order_relaxed);
            const uint current_token = atomic_load_explicit(best_token, memory_order_relaxed);
            if (ordered < current_best || (ordered == current_best && row >= current_token)) break;
            if (atomic_exchange_explicit(state_lock, 1u, memory_order_relaxed) == 0u) {
                const uint locked_best = atomic_load_explicit(best_value, memory_order_relaxed);
                const uint locked_token = atomic_load_explicit(best_token, memory_order_relaxed);
                if (ordered > locked_best || (ordered == locked_best && row < locked_token)) {
                    atomic_store_explicit(best_value, ordered, memory_order_relaxed);
                    atomic_store_explicit(best_token, row, memory_order_relaxed);
                }
                atomic_store_explicit(state_lock, 0u, memory_order_relaxed);
                break;
            }
        }
    }
}

#define ZIGGY_MOONQ_Q6K_KERNEL(NAME, ADD_TO_DST, STATIC_COLS) \
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
    const uint row_stride = blocks_per_row * 224; \
    const device uchar *row_bytes = matrix + row * row_stride; \
    const uint chunks_per_row = blocks_per_row * ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
    threadgroup float partial_sums[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_sum = 0.0f; \
    for (uint chunk_index = lane; chunk_index < chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_index = chunk_index / ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_chunk = chunk_index % ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_half = block_chunk / 32; \
        const uint l = block_chunk % 32; \
        const device uchar *block = row_bytes + block_index * 224; \
        const device uchar *scales = block + 2 + block_half * 8; \
        const device uchar *ql = block + 18 + block_half * 64; \
        const device uchar *qh = block + 146 + block_half * 32; \
        const float d = read_half_le(block, 0); \
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
        if (ADD_TO_DST) output[row] += sum; else output[row] = sum; \
    } \
}

ZIGGY_MOONQ_Q6K_KERNEL(matvec_moonq_q6k_f32, false, 0)
ZIGGY_MOONQ_Q6K_KERNEL(matvec_moonq_q6k_2048_f32, false, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q6K_KERNEL(matvec_moonq_q6k_5632_f32, false, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)
ZIGGY_MOONQ_Q6K_KERNEL(matvec_moonq_q6k_add_f32, true, 0)
ZIGGY_MOONQ_Q6K_KERNEL(matvec_moonq_q6k_add_2048_f32, true, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q6K_KERNEL(matvec_moonq_q6k_add_5632_f32, true, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q6K_KERNEL

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

#define ZIGGY_MOONQ_Q4K_DUAL_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix0 [[buffer(0)]], \
    device const uchar *matrix1 [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device float *output0 [[buffer(3)]], \
    device float *output1 [[buffer(4)]], \
    constant uint &rows [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
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
    const device uchar *row_bytes00 = matrix0 + r0 * row_stride; \
    const device uchar *row_bytes01 = valid1 ? (matrix0 + r1 * row_stride) : row_bytes00; \
    const device uchar *row_bytes02 = valid2 ? (matrix0 + r2 * row_stride) : row_bytes00; \
    const device uchar *row_bytes03 = valid3 ? (matrix0 + r3 * row_stride) : row_bytes00; \
    const device uchar *row_bytes10 = matrix1 + r0 * row_stride; \
    const device uchar *row_bytes11 = valid1 ? (matrix1 + r1 * row_stride) : row_bytes10; \
    const device uchar *row_bytes12 = valid2 ? (matrix1 + r2 * row_stride) : row_bytes10; \
    const device uchar *row_bytes13 = valid3 ? (matrix1 + r3 * row_stride) : row_bytes10; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_sums00[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums01[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums02[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums03[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums10[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums11[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums12[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_sums13[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float sum00 = 0.0f, sum01 = 0.0f, sum02 = 0.0f, sum03 = 0.0f; \
    float sum10 = 0.0f, sum11 = 0.0f, sum12 = 0.0f, sum13 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device float4 *in_ptr_low = (const device float4 *)(input + input_offset); \
        const device float4 *in_ptr_high = (const device float4 *)(input + input_offset + 32); \
        const float4 input_low = *in_ptr_low; \
        const float4 input_high = *in_ptr_high; \
        const uint scale_index = group * 2; \
        { const device uchar *block = row_bytes00 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum00 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes01 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum01 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes02 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum02 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes03 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum03 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes10 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum10 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes11 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum11 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes12 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum12 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes13 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum13 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
    } \
    const float s00 = simd_sum(sum00), s01 = simd_sum(sum01), s02 = simd_sum(sum02), s03 = simd_sum(sum03); \
    const float s10 = simd_sum(sum10), s11 = simd_sum(sum11), s12 = simd_sum(sum12), s13 = simd_sum(sum13); \
    if (simd_lane == 0) { partial_sums00[simd_group] = s00; partial_sums01[simd_group] = s01; partial_sums02[simd_group] = s02; partial_sums03[simd_group] = s03; partial_sums10[simd_group] = s10; partial_sums11[simd_group] = s11; partial_sums12[simd_group] = s12; partial_sums13[simd_group] = s13; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float fs00 = 0.0f, fs01 = 0.0f, fs02 = 0.0f, fs03 = 0.0f; \
        float fs10 = 0.0f, fs11 = 0.0f, fs12 = 0.0f, fs13 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { fs00 += partial_sums00[index]; fs01 += partial_sums01[index]; fs02 += partial_sums02[index]; fs03 += partial_sums03[index]; fs10 += partial_sums10[index]; fs11 += partial_sums11[index]; fs12 += partial_sums12[index]; fs13 += partial_sums13[index]; } \
        output0[r0] = fs00; output1[r0] = fs10; \
        if (valid1) { output0[r1] = fs01; output1[r1] = fs11; } \
        if (valid2) { output0[r2] = fs02; output1[r2] = fs12; } \
        if (valid3) { output0[r3] = fs03; output1[r3] = fs13; } \
    } \
}

ZIGGY_MOONQ_Q4K_DUAL_KERNEL(matvec_moonq_q4k_dual_f32, 0)
ZIGGY_MOONQ_Q4K_DUAL_KERNEL(matvec_moonq_q4k_dual_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_DUAL_KERNEL(matvec_moonq_q4k_dual_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q4K_DUAL_KERNEL

#define ZIGGY_MOONQ_Q4K_K_HALF_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device half *k_cache [[buffer(2)]], \
    constant uint &dst_base [[buffer(3)]], \
    constant uint &head_count [[buffer(4)]], \
    constant uint &head_dim [[buffer(5)]], \
    constant uint &pair_count [[buffer(6)]], \
    constant uint &cols [[buffer(7)]], \
    constant uint &position [[buffer(8)]], \
    constant float &freq_base [[buffer(9)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + row0 * row_stride; \
    const device uchar *row_bytes1 = matrix + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float sum0 = 0.0f; \
    float sum1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device float4 *in_ptr_low = (const device float4 *)(input + input_offset); \
        const device float4 *in_ptr_high = (const device float4 *)(input + input_offset + 32); \
        const float4 input_low = *in_ptr_low; \
        const float4 input_high = *in_ptr_high; \
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
            const uint packed = *(const device uint *)q; \
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
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
    } \
    const float s0 = simd_sum(sum0); \
    const float s1 = simd_sum(sum1); \
    if (simd_lane == 0) { \
        partial0[simd_group] = s0; \
        partial1[simd_group] = s1; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float fs0 = 0.0f, fs1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            fs0 += partial0[index]; \
            fs1 += partial1[index]; \
        } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(fs0 * cos_theta - fs1 * sin_theta); \
        k_cache[dst_base + row1] = half(fs0 * sin_theta + fs1 * cos_theta); \
    } \
}

ZIGGY_MOONQ_Q4K_K_HALF_KERNEL(matvec_moonq_q4k_k_half_f32, 0)
ZIGGY_MOONQ_Q4K_K_HALF_KERNEL(matvec_moonq_q4k_k_half_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_K_HALF_KERNEL(matvec_moonq_q4k_k_half_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_MOONQ_Q4K_K_HALF_RMS_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device const float *norm_weights [[buffer(2)]], \
    device const float *norm_scale [[buffer(3)]], \
    device half *k_cache [[buffer(4)]], \
    constant uint &dst_base [[buffer(5)]], \
    constant uint &head_count [[buffer(6)]], \
    constant uint &head_dim [[buffer(7)]], \
    constant uint &pair_count [[buffer(8)]], \
    constant uint &cols [[buffer(9)]], \
    constant uint &position [[buffer(10)]], \
    constant float &freq_base [[buffer(11)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const float input_scale = norm_scale[0]; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + row0 * row_stride; \
    const device uchar *row_bytes1 = matrix + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float sum0 = 0.0f; \
    float sum1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset); \
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32); \
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
            const uint packed = *(const device uint *)q; \
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
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
    } \
    const float s0 = simd_sum(sum0); \
    const float s1 = simd_sum(sum1); \
    if (simd_lane == 0) { partial0[simd_group] = s0; partial1[simd_group] = s1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float fs0 = 0.0f, fs1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { fs0 += partial0[index]; fs1 += partial1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(fs0 * cos_theta - fs1 * sin_theta); \
        k_cache[dst_base + row1] = half(fs0 * sin_theta + fs1 * cos_theta); \
    } \
}

ZIGGY_MOONQ_Q4K_K_HALF_RMS_KERNEL(matvec_moonq_q4k_k_half_rms_f32, 0)
ZIGGY_MOONQ_Q4K_K_HALF_RMS_KERNEL(matvec_moonq_q4k_k_half_rms_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_K_HALF_RMS_KERNEL(matvec_moonq_q4k_k_half_rms_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q4K_K_HALF_RMS_KERNEL

#undef ZIGGY_MOONQ_Q4K_K_HALF_KERNEL

#define ZIGGY_MOONQ_Q4K_Q_ROPE_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &head_count [[buffer(3)]], \
    constant uint &head_dim [[buffer(4)]], \
    constant uint &pair_count [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
    constant uint &position [[buffer(7)]], \
    constant float &freq_base [[buffer(8)]], \
    constant uint &rope_style [[buffer(9)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = rope_style == 0 ? (head * head_dim + pair * 2) : (head * head_dim + pair); \
    const uint row1 = rope_style == 0 ? (row0 + 1) : (row0 + pair_count); \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + row0 * row_stride; \
    const device uchar *row_bytes1 = matrix + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float sum0 = 0.0f; \
    float sum1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device float4 *in_ptr_low = (const device float4 *)(input + input_offset); \
        const device float4 *in_ptr_high = (const device float4 *)(input + input_offset + 32); \
        const float4 input_low = *in_ptr_low; \
        const float4 input_high = *in_ptr_high; \
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
            const uint packed = *(const device uint *)q; \
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
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
    } \
    const float s0 = simd_sum(sum0); \
    const float s1 = simd_sum(sum1); \
    if (simd_lane == 0) { partial0[simd_group] = s0; partial1[simd_group] = s1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float fs0 = 0.0f, fs1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { fs0 += partial0[index]; fs1 += partial1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        output[row0] = fs0 * cos_theta - fs1 * sin_theta; \
        output[row1] = fs0 * sin_theta + fs1 * cos_theta; \
    } \
}

ZIGGY_MOONQ_Q4K_Q_ROPE_KERNEL(matvec_moonq_q4k_q_rope_f32, 0)
ZIGGY_MOONQ_Q4K_Q_ROPE_KERNEL(matvec_moonq_q4k_q_rope_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_Q_ROPE_KERNEL(matvec_moonq_q4k_q_rope_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_MOONQ_Q4K_Q_ROPE_RMS_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device const float *norm_weights [[buffer(2)]], \
    device const float *norm_scale [[buffer(3)]], \
    device float *output [[buffer(4)]], \
    constant uint &head_count [[buffer(5)]], \
    constant uint &head_dim [[buffer(6)]], \
    constant uint &pair_count [[buffer(7)]], \
    constant uint &cols [[buffer(8)]], \
    constant uint &position [[buffer(9)]], \
    constant float &freq_base [[buffer(10)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const float input_scale = norm_scale[0]; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = row_pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const device uchar *row_bytes0 = matrix + row0 * row_stride; \
    const device uchar *row_bytes1 = matrix + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float sum0 = 0.0f; \
    float sum1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset); \
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32); \
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
            const uint packed = *(const device uint *)q; \
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
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            sum1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
    } \
    const float s0 = simd_sum(sum0); \
    const float s1 = simd_sum(sum1); \
    if (simd_lane == 0) { partial0[simd_group] = s0; partial1[simd_group] = s1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float fs0 = 0.0f, fs1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { fs0 += partial0[index]; fs1 += partial1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        output[row0] = fs0 * cos_theta - fs1 * sin_theta; \
        output[row1] = fs0 * sin_theta + fs1 * cos_theta; \
    } \
}

ZIGGY_MOONQ_Q4K_Q_ROPE_RMS_KERNEL(matvec_moonq_q4k_q_rope_rms_f32, 0)
ZIGGY_MOONQ_Q4K_Q_ROPE_RMS_KERNEL(matvec_moonq_q4k_q_rope_rms_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_Q_ROPE_RMS_KERNEL(matvec_moonq_q4k_q_rope_rms_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q4K_Q_ROPE_RMS_KERNEL

#undef ZIGGY_MOONQ_Q4K_Q_ROPE_KERNEL

#define ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix_k [[buffer(0)]], \
    device const uchar *matrix_v [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device half *k_cache [[buffer(3)]], \
    device half *v_cache [[buffer(4)]], \
    constant uint &dst_base [[buffer(5)]], \
    constant uint &head_count [[buffer(6)]], \
    constant uint &head_dim [[buffer(7)]], \
    constant uint &pair_count [[buffer(8)]], \
    constant uint &cols [[buffer(9)]], \
    constant uint &position [[buffer(10)]], \
    constant float &freq_base [[buffer(11)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const device uchar *k_row_bytes0 = matrix_k + row0 * row_stride; \
    const device uchar *k_row_bytes1 = matrix_k + row1 * row_stride; \
    const device uchar *v_row_bytes0 = matrix_v + row0 * row_stride; \
    const device uchar *v_row_bytes1 = matrix_v + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_k0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_k1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_k0 = 0.0f, local_k1 = 0.0f, local_v0 = 0.0f, local_v1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device float4 *in_ptr_low = (const device float4 *)(input + input_offset); \
        const device float4 *in_ptr_high = (const device float4 *)(input + input_offset + 32); \
        const float4 input_low = *in_ptr_low; \
        const float4 input_high = *in_ptr_high; \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = k_row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            local_k0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = k_row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            local_k1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = v_row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            local_v0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = v_row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            local_v1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
    } \
    const float sum_k0 = simd_sum(local_k0); \
    const float sum_k1 = simd_sum(local_k1); \
    const float sum_v0 = simd_sum(local_v0); \
    const float sum_v1 = simd_sum(local_v1); \
    if (simd_lane == 0) { \
        partial_k0[simd_group] = sum_k0; \
        partial_k1[simd_group] = sum_k1; \
        partial_v0[simd_group] = sum_v0; \
        partial_v1[simd_group] = sum_v1; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            k0 += partial_k0[index]; \
            k1 += partial_k1[index]; \
            v0 += partial_v0[index]; \
            v1 += partial_v1[index]; \
        } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
        v_cache[dst_base + row0] = half(v0); \
        v_cache[dst_base + row1] = half(v1); \
    } \
}

ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_KERNEL(matvec_moonq_q4k_dual_kv_half_f32, 0)
ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_KERNEL(matvec_moonq_q4k_dual_kv_half_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_KERNEL(matvec_moonq_q4k_dual_kv_half_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_RMS_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix_k [[buffer(0)]], \
    device const uchar *matrix_v [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device const float *norm_weights [[buffer(3)]], \
    device const float *norm_scale [[buffer(4)]], \
    device half *k_cache [[buffer(5)]], \
    device half *v_cache [[buffer(6)]], \
    constant uint &dst_base [[buffer(7)]], \
    constant uint &head_count [[buffer(8)]], \
    constant uint &head_dim [[buffer(9)]], \
    constant uint &pair_count [[buffer(10)]], \
    constant uint &cols [[buffer(11)]], \
    constant uint &position [[buffer(12)]], \
    constant float &freq_base [[buffer(13)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const float input_scale = norm_scale[0]; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint row_stride = blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const device uchar *k_row_bytes0 = matrix_k + row0 * row_stride; \
    const device uchar *k_row_bytes1 = matrix_k + row1 * row_stride; \
    const device uchar *v_row_bytes0 = matrix_v + row0 * row_stride; \
    const device uchar *v_row_bytes1 = matrix_v + row1 * row_stride; \
    const uint packed_chunks_per_row = blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    threadgroup float partial_k0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_k1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_k0 = 0.0f, local_k1 = 0.0f, local_v0 = 0.0f, local_v1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < packed_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset); \
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32); \
        const uint scale_index = group * 2; \
        { const device uchar *block = k_row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint *)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); local_k0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = k_row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint *)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); local_k1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = v_row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint *)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); local_v0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = v_row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint *)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); local_v1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
    } \
    const float sum_k0 = simd_sum(local_k0); \
    const float sum_k1 = simd_sum(local_k1); \
    const float sum_v0 = simd_sum(local_v0); \
    const float sum_v1 = simd_sum(local_v1); \
    if (simd_lane == 0) { partial_k0[simd_group] = sum_k0; partial_k1[simd_group] = sum_k1; partial_v0[simd_group] = sum_v0; partial_v1[simd_group] = sum_v1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { k0 += partial_k0[index]; k1 += partial_k1[index]; v0 += partial_v0[index]; v1 += partial_v1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
        v_cache[dst_base + row0] = half(v0); \
        v_cache[dst_base + row1] = half(v1); \
    } \
}

ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_RMS_KERNEL(matvec_moonq_q4k_dual_kv_half_rms_f32, 0)
ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_RMS_KERNEL(matvec_moonq_q4k_dual_kv_half_rms_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_RMS_KERNEL(matvec_moonq_q4k_dual_kv_half_rms_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_RMS_KERNEL

#undef ZIGGY_MOONQ_Q4K_DUAL_KV_HALF_KERNEL

#define ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix_k [[buffer(0)]], \
    device const uchar *matrix_v [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device half *k_cache [[buffer(3)]], \
    device half *v_cache [[buffer(4)]], \
    constant uint &dst_base [[buffer(5)]], \
    constant uint &head_count [[buffer(6)]], \
    constant uint &head_dim [[buffer(7)]], \
    constant uint &pair_count [[buffer(8)]], \
    constant uint &cols [[buffer(9)]], \
    constant uint &position [[buffer(10)]], \
    constant float &freq_base [[buffer(11)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint q4_blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint q4_row_stride = q4_blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const uint q6_blocks_per_row = effective_cols / ZIGGY_Q6K_VALUES_PER_BLOCK; \
    const uint q6_row_stride = q6_blocks_per_row * ZIGGY_Q6K_BYTES_PER_BLOCK; \
    const device uchar *k_row_bytes0 = matrix_k + row0 * q4_row_stride; \
    const device uchar *k_row_bytes1 = matrix_k + row1 * q4_row_stride; \
    const device uchar *v_row_bytes0 = matrix_v + row0 * q6_row_stride; \
    const device uchar *v_row_bytes1 = matrix_v + row1 * q6_row_stride; \
    const uint q4_chunks_per_row = q4_blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    const uint q6_chunks_per_row = q6_blocks_per_row * ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
    threadgroup float partial_k0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_k1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_k0 = 0.0f, local_k1 = 0.0f, local_v0 = 0.0f, local_v1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < q4_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const device float4 *in_ptr_low = (const device float4 *)(input + input_offset); \
        const device float4 *in_ptr_high = (const device float4 *)(input + input_offset + 32); \
        const float4 input_low = *in_ptr_low; \
        const float4 input_high = *in_ptr_high; \
        const uint scale_index = group * 2; \
        { \
            const device uchar *block = k_row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            local_k0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
        { \
            const device uchar *block = k_row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
            const float d = read_half_le(block, 0); \
            const float dmin = read_half_le(block, 2); \
            const float d1 = d * float(block[4 + scale_index + 0]); \
            const float m1 = dmin * float(block[12 + scale_index + 0]); \
            const float d2 = d * float(block[4 + scale_index + 1]); \
            const float m2 = dmin * float(block[12 + scale_index + 1]); \
            const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; \
            const uint packed = *(const device uint *)q; \
            const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); \
            const float4 low_q = float4(q_vec & 0x0F); \
            const float4 high_q = float4(q_vec >> 4); \
            local_k1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
        } \
    } \
    for (uint chunk_index = lane; chunk_index < q6_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_index = chunk_index / ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_chunk = chunk_index % ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_half = block_chunk / 32; \
        const uint l = block_chunk % 32; \
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l; \
        { \
            const device uchar *block = v_row_bytes0 + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; \
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
            local_v0 += s0 * q1 * input[input_offset + 0]; \
            local_v0 += s2 * q2 * input[input_offset + 32]; \
            local_v0 += s4 * q3 * input[input_offset + 64]; \
            local_v0 += s6 * q4 * input[input_offset + 96]; \
        } \
        { \
            const device uchar *block = v_row_bytes1 + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; \
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
            local_v1 += s0 * q1 * input[input_offset + 0]; \
            local_v1 += s2 * q2 * input[input_offset + 32]; \
            local_v1 += s4 * q3 * input[input_offset + 64]; \
            local_v1 += s6 * q4 * input[input_offset + 96]; \
        } \
    } \
    const float sum_k0 = simd_sum(local_k0); \
    const float sum_k1 = simd_sum(local_k1); \
    const float sum_v0 = simd_sum(local_v0); \
    const float sum_v1 = simd_sum(local_v1); \
    if (simd_lane == 0) { \
        partial_k0[simd_group] = sum_k0; \
        partial_k1[simd_group] = sum_k1; \
        partial_v0[simd_group] = sum_v0; \
        partial_v1[simd_group] = sum_v1; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { \
            k0 += partial_k0[index]; \
            k1 += partial_k1[index]; \
            v0 += partial_v0[index]; \
            v1 += partial_v1[index]; \
        } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
        v_cache[dst_base + row0] = half(v0); \
        v_cache[dst_base + row1] = half(v1); \
    } \
}

ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_KERNEL(matvec_moonq_q4k_q6k_dual_kv_half_f32, 0)
ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_KERNEL(matvec_moonq_q4k_q6k_dual_kv_half_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_KERNEL(matvec_moonq_q4k_q6k_dual_kv_half_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix_k [[buffer(0)]], \
    device const uchar *matrix_v [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device const float *norm_weights [[buffer(3)]], \
    device const float *norm_scale [[buffer(4)]], \
    device half *k_cache [[buffer(5)]], \
    device half *v_cache [[buffer(6)]], \
    constant uint &dst_base [[buffer(7)]], \
    constant uint &head_count [[buffer(8)]], \
    constant uint &head_dim [[buffer(9)]], \
    constant uint &pair_count [[buffer(10)]], \
    constant uint &cols [[buffer(11)]], \
    constant uint &position [[buffer(12)]], \
    constant float &freq_base [[buffer(13)]], \
    uint row_pair [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint threads_per_group [[threads_per_threadgroup]], \
    uint threads_per_simdgroup [[threads_per_simdgroup]] \
) { \
    if (row_pair >= head_count * pair_count) return; \
    constexpr uint kStaticCols = STATIC_COLS; \
    const uint effective_cols = kStaticCols == 0 ? cols : kStaticCols; \
    if (kStaticCols != 0 && cols != kStaticCols) return; \
    const float input_scale = norm_scale[0]; \
    const uint head = row_pair / pair_count; \
    const uint pair = row_pair % pair_count; \
    const uint row0 = head * head_dim + pair * 2; \
    const uint row1 = row0 + 1; \
    const uint q4_blocks_per_row = effective_cols / ZIGGY_Q4K_VALUES_PER_BLOCK; \
    const uint q4_row_stride = q4_blocks_per_row * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; \
    const uint q6_blocks_per_row = effective_cols / ZIGGY_Q6K_VALUES_PER_BLOCK; \
    const uint q6_row_stride = q6_blocks_per_row * ZIGGY_Q6K_BYTES_PER_BLOCK; \
    const device uchar *k_row_bytes0 = matrix_k + row0 * q4_row_stride; \
    const device uchar *k_row_bytes1 = matrix_k + row1 * q4_row_stride; \
    const device uchar *v_row_bytes0 = matrix_v + row0 * q6_row_stride; \
    const device uchar *v_row_bytes1 = matrix_v + row1 * q6_row_stride; \
    const uint q4_chunks_per_row = q4_blocks_per_row * ZIGGY_Q4K_GROUPS_PER_BLOCK * ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
    const uint q6_chunks_per_row = q6_blocks_per_row * ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
    threadgroup float partial_k0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_k1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v0[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    threadgroup float partial_v1[ZIGGY_MAX_Q4K_SIMDGROUPS]; \
    float local_k0 = 0.0f, local_k1 = 0.0f, local_v0 = 0.0f, local_v1 = 0.0f; \
    for (uint chunk_index = lane; chunk_index < q4_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_group_index = chunk_index / ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint group = block_group_index % ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        const uint q_offset = chunk_in_group * 4; \
        const uint input_offset = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK + group * ZIGGY_Q4K_VALUES_PER_GROUP + q_offset; \
        const float4 input_low = ziggy_normed_value4(input, norm_weights, input_scale, input_offset); \
        const float4 input_high = ziggy_normed_value4(input, norm_weights, input_scale, input_offset + 32); \
        const uint scale_index = group * 2; \
        { const device uchar *block = k_row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint *)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); local_k0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = k_row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint *)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); local_k1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
    } \
    for (uint chunk_index = lane; chunk_index < q6_chunks_per_row; chunk_index += threads_per_group) { \
        const uint block_index = chunk_index / ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_chunk = chunk_index % ZIGGY_Q6K_CHUNKS_PER_BLOCK; \
        const uint block_half = block_chunk / 32; \
        const uint l = block_chunk % 32; \
        const uint input_offset = block_index * ZIGGY_Q6K_VALUES_PER_BLOCK + block_half * 128 + l; \
        { \
            const device uchar *block = v_row_bytes0 + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; \
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
            local_v0 += s0 * q1 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 0); \
            local_v0 += s2 * q2 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 32); \
            local_v0 += s4 * q3 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 64); \
            local_v0 += s6 * q4 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 96); \
        } \
        { \
            const device uchar *block = v_row_bytes1 + block_index * ZIGGY_Q6K_BYTES_PER_BLOCK; \
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
            local_v1 += s0 * q1 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 0); \
            local_v1 += s2 * q2 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 32); \
            local_v1 += s4 * q3 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 64); \
            local_v1 += s6 * q4 * ziggy_normed_value(input, norm_weights, input_scale, input_offset + 96); \
        } \
    } \
    const float sum_k0 = simd_sum(local_k0); \
    const float sum_k1 = simd_sum(local_k1); \
    const float sum_v0 = simd_sum(local_v0); \
    const float sum_v1 = simd_sum(local_v1); \
    if (simd_lane == 0) { partial_k0[simd_group] = sum_k0; partial_k1[simd_group] = sum_k1; partial_v0[simd_group] = sum_v0; partial_v1[simd_group] = sum_v1; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { \
        float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f; \
        const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; \
        for (uint index = 0; index < simd_group_count; index += 1) { k0 += partial_k0[index]; k1 += partial_k1[index]; v0 += partial_v0[index]; v1 += partial_v1[index]; } \
        const float exponent = float(pair * 2) / float(pair_count * 2); \
        const float theta = float(position) / pow(freq_base, exponent); \
        const float cos_theta = cos(theta); \
        const float sin_theta = sin(theta); \
        k_cache[dst_base + row0] = half(k0 * cos_theta - k1 * sin_theta); \
        k_cache[dst_base + row1] = half(k0 * sin_theta + k1 * cos_theta); \
        v_cache[dst_base + row0] = half(v0); \
        v_cache[dst_base + row1] = half(v1); \
    } \
}

ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL(matvec_moonq_q4k_q6k_dual_kv_half_rms_f32, 0)
ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL(matvec_moonq_q4k_q6k_dual_kv_half_rms_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL(matvec_moonq_q4k_q6k_dual_kv_half_rms_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_RMS_KERNEL

#undef ZIGGY_MOONQ_Q4K_Q6K_DUAL_KV_HALF_KERNEL

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

kernel void apply_rope_to_half_dst_f32(
    device const float *src [[buffer(0)]],
    device half *dst [[buffer(1)]],
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
        dst[dst_index] = half(src[index]);
        return;
    }

    float result = 0.0f;
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
            result = x0 * cos_theta - x1 * sin_theta;
        } else {
            result = x0 * sin_theta + x1 * cos_theta;
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
            result = x0 * cos_theta - x1 * sin_theta;
        } else {
            result = x0 * sin_theta + x1 * cos_theta;
        }
    }
    dst[dst_index] = half(result);
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
    constant float &scale [[buffer(11)]],
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

    const uint token_count = position + 1;

    float local_m = -INFINITY;
    float local_l = 0.0f;
    float local_out[ZIGGY_MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    for (uint token = lane; token < token_count; token += threads_per_group) {
        device const half *k_head = k_cache + layer_base + token * kv_dim + kv_offset;
        device const half *v_head = v_cache + layer_base + token * kv_dim + kv_offset;

        float s = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            s += q_head[d] * float(k_head[d]);
        }
        s *= scale;

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

kernel void attention_fused_hd64_f32(
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
    constant float &scale [[buffer(11)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    if (head >= head_count || head_dim > 64) return;
    const uint kv_group = head_count / head_count_kv;
    const uint kv_head = head / kv_group;
    const uint kv_offset = kv_head * head_dim;
    device const float *q_head = q + head * head_dim;
    const uint token_count = min(position + 1, context_length);

    float local_m = -INFINITY;
    float local_l = 0.0f;
    float local_out[64];
    for (uint d = 0; d < head_dim; d++) local_out[d] = 0.0f;

    for (uint token = lane; token < token_count; token += threads_per_group) {
        device const half *k_head = k_cache + layer_base + token * kv_dim + kv_offset;
        device const half *v_head = v_cache + layer_base + token * kv_dim + kv_offset;
        float s = 0.0f;
        for (uint d = 0; d < head_dim; d++) s += q_head[d] * float(k_head[d]);
        s *= scale;
        const float m_new = max(local_m, s);
        const float exp_diff = exp(local_m - m_new);
        const float p = exp(s - m_new);
        local_l = local_l * exp_diff + p;
        for (uint d = 0; d < head_dim; d++) local_out[d] = local_out[d] * exp_diff + p * float(v_head[d]);
        local_m = m_new;
    }

    threadgroup float tg_m[ZIGGY_MAX_ROW_SIMDGROUPS];
    const float simd_m = simd_max(local_m);
    if (simd_lane == 0) tg_m[simd_group] = simd_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_m = -INFINITY;
    const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup;
    if (lane == 0) {
        for (uint i = 0; i < simd_group_count; i++) global_m = max(global_m, tg_m[i]);
        tg_m[0] = global_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_m = tg_m[0];

    const float scaled_l = local_l * exp(local_m - global_m);
    const float simd_l = simd_sum(scaled_l);
    threadgroup float tg_l[ZIGGY_MAX_ROW_SIMDGROUPS];
    if (simd_lane == 0) tg_l[simd_group] = simd_l;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_l = 0.0f;
    if (lane == 0) {
        for (uint i = 0; i < simd_group_count; i++) global_l += tg_l[i];
        tg_l[0] = global_l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_l = tg_l[0];

    threadgroup float tg_out[ZIGGY_MAX_ROW_SIMDGROUPS * 64];
    const float thread_scale = exp(local_m - global_m) / global_l;
    for (uint d = 0; d < head_dim; d++) {
        const float scaled_out = local_out[d] * thread_scale;
        const float simd_out = simd_sum(scaled_out);
        if (simd_lane == 0) tg_out[simd_group * 64 + d] = simd_out;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        for (uint d = simd_lane; d < head_dim; d += threads_per_simdgroup) {
            float sum = 0.0f;
            for (uint i = 0; i < simd_group_count; i++) sum += tg_out[i * 64 + d];
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
        output[index] = input[index] * scale * weights[index];
    }
}

kernel void rms_norm_per_head_f32(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
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
        h_out[i] = h_in[i] * scale * h_w[i];
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
    constant uint &batch_idx [[buffer(5)]],
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
        batch_output[index] = batch_input[index] * scale * weights[index];
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

kernel void build_ffn_gate_block_mask_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device atomic_uint *mask [[buffer(2)]],
    device atomic_uint *stats [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant float &threshold [[buffer(5)]],
    uint block_index [[thread_position_in_grid]]
) {
    const uint block_start = block_index * ZIGGY_Q4K_VALUES_PER_BLOCK;
    if (block_start >= cols) return;

    bool active = false;
    const uint block_end = min(cols, block_start + ZIGGY_Q4K_VALUES_PER_BLOCK);
    for (uint index = block_start; index < block_end; index += 1) {
        const float g = gate[index];
        const float value = (g / (1.0f + exp(-g))) * up[index];
        if (fabs(value) > threshold) {
            active = true;
            break;
        }
    }

    atomic_store_explicit(mask + block_index, active ? 1u : 0u, memory_order_relaxed);
    if (block_index == 0) {
        atomic_store_explicit(stats + 0, 0u, memory_order_relaxed);
        atomic_store_explicit(stats + 1, (cols + ZIGGY_Q4K_VALUES_PER_BLOCK - 1) / ZIGGY_Q4K_VALUES_PER_BLOCK, memory_order_relaxed);
    }
    if (active) atomic_fetch_add_explicit(stats + 0, 1u, memory_order_relaxed);
}


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

#define ZIGGY_Q4K_GATED_SILU_DOWN_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *gate [[buffer(1)]], \
    device const float *up [[buffer(2)]], \
    device const uint *mask [[buffer(3)]], \
    device float *output [[buffer(4)]], \
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
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        if (mask[block_index] == 0) continue; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
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
        const device uchar *block = row_bytes + block_index * ZIGGY_Q4K_BYTES_PER_BLOCK; \
        const uint scale_index = group * 2; \
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
        local_sum += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); \
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

ZIGGY_Q4K_GATED_SILU_DOWN_ADD_KERNEL(matvec_q4k_gated_silu_down_add_f32, 0)
ZIGGY_Q4K_GATED_SILU_DOWN_ADD_KERNEL(matvec_q4k_gated_silu_down_add_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_Q4K_GATED_SILU_DOWN_ADD_KERNEL(matvec_q4k_gated_silu_down_add_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#define ZIGGY_MOONQ_Q4K_GATED_SILU_DOWN_ADD_KERNEL(NAME, STATIC_COLS) \
kernel void NAME( \
    device const uchar *matrix [[buffer(0)]], \
    device const float *gate [[buffer(1)]], \
    device const float *up [[buffer(2)]], \
    device const uint *mask [[buffer(3)]], \
    device float *output [[buffer(4)]], \
    constant uint &rows [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
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
        const uint block_index = block_group_index / ZIGGY_Q4K_GROUPS_PER_BLOCK; \
        if (mask[block_index] == 0) continue; \
        const uint chunk_in_group = chunk_index % ZIGGY_Q4K_PACKED_CHUNKS_PER_GROUP; \
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
        { const device uchar *block = row_bytes0 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum0 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes1 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum1 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes2 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum2 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
        { const device uchar *block = row_bytes3 + block_index * ZIGGY_MOONQ_Q4K_BYTES_PER_BLOCK; const float d = read_half_le(block, 0); const float dmin = read_half_le(block, 2); const float d1 = d * float(block[4 + scale_index + 0]); const float m1 = dmin * float(block[12 + scale_index + 0]); const float d2 = d * float(block[4 + scale_index + 1]); const float m2 = dmin * float(block[12 + scale_index + 1]); const device uchar *q = block + 20 + group * ZIGGY_Q4K_PACKED_BYTES_PER_GROUP + q_offset; const uint packed = *(const device uint*)q; const uint4 q_vec = uint4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF); const float4 low_q = float4(q_vec & 0x0F); const float4 high_q = float4(q_vec >> 4); sum3 += dot(low_q * d1 - m1, input_low) + dot(high_q * d2 - m2, input_high); } \
    } \
    const float simd_sum0 = simd_sum(sum0); const float simd_sum1 = simd_sum(sum1); const float simd_sum2 = simd_sum(sum2); const float simd_sum3 = simd_sum(sum3); \
    if (simd_lane == 0) { partial_sums0[simd_group] = simd_sum0; partial_sums1[simd_group] = simd_sum1; partial_sums2[simd_group] = simd_sum2; partial_sums3[simd_group] = simd_sum3; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f; const uint simd_group_count = (threads_per_group + threads_per_simdgroup - 1) / threads_per_simdgroup; for (uint index = 0; index < simd_group_count; index += 1) { total0 += partial_sums0[index]; total1 += partial_sums1[index]; total2 += partial_sums2[index]; total3 += partial_sums3[index]; } output[r0] += total0; if (valid1) output[r1] += total1; if (valid2) output[r2] += total2; if (valid3) output[r3] += total3; } \
}

ZIGGY_MOONQ_Q4K_GATED_SILU_DOWN_ADD_KERNEL(matvec_moonq_q4k_gated_silu_down_add_f32, 0)
ZIGGY_MOONQ_Q4K_GATED_SILU_DOWN_ADD_KERNEL(matvec_moonq_q4k_gated_silu_down_add_2048_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_0)
ZIGGY_MOONQ_Q4K_GATED_SILU_DOWN_ADD_KERNEL(matvec_moonq_q4k_gated_silu_down_add_5632_f32, ZIGGY_MOONQ_Q4K_SPECIAL_COLS_1)

#undef ZIGGY_MOONQ_Q4K_SILU_DOWN_ADD_KERNEL
