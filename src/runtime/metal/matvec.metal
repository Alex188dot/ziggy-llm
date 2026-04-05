
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
    const device uchar *row_bytes1 = matrix + r1 * row_stride; \
    const device uchar *row_bytes2 = matrix + r2 * row_stride; \
    const device uchar *row_bytes3 = matrix + r3 * row_stride; \
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
        if (valid1) { \
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
        if (valid2) { \
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
        if (valid3) { \
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
    uint index [[thread_position_in_grid]]
) {
    if (pair_count == 0) return;
    const uint total_pairs = head_count * pair_count;
    if (index >= total_pairs) return;

    const uint head = index / pair_count;
    const uint pair = index % pair_count;
    const uint base = vector_base + head * head_dim + pair * 2;
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
    const device uchar *row_bytes1 = matrix + r1 * row_stride; \
    const device uchar *row_bytes2 = matrix + r2 * row_stride; \
    const device uchar *row_bytes3 = matrix + r3 * row_stride; \
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
        if (valid1) { \
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
        if (valid2) { \
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
        if (valid3) { \
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
