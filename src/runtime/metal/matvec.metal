#include <metal_stdlib>
using namespace metal;

kernel void matvec_f32(
    device const float *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;

    float sum = 0.0f;
    for (uint col = 0; col < cols; col += 1) {
        sum += matrix[row + col * rows] * input[col];
    }
    output[row] = sum;
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

kernel void matvec_q4k_f32(
    device const uchar *matrix [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;

    const uint blocks_per_row = cols / 256;
    const uint row_stride = blocks_per_row * 144;
    const device uchar *row_bytes = matrix + row * row_stride;

    float sum = 0.0f;
    uint input_offset = 0;
    for (uint block_index = 0; block_index < blocks_per_row; block_index += 1) {
        const device uchar *block = row_bytes + block_index * 144;
        const float d = read_half_le(block, 0);
        const float dmin = read_half_le(block, 2);
        const device uchar *scales = block + 4;
        const device uchar *q = block + 16;

        for (uint group = 0; group < 4; group += 1) {
            const uchar scale0 = get_scale_k4(scales, group * 2 + 0);
            const uchar scale1 = get_scale_k4(scales, group * 2 + 1);
            const uchar min0 = get_min_k4(scales, group * 2 + 0);
            const uchar min1 = get_min_k4(scales, group * 2 + 1);
            const float d1 = d * float(scale0);
            const float m1 = dmin * float(min0);
            const float d2 = d * float(scale1);
            const float m2 = dmin * float(min1);

            for (uint l = 0; l < 32; l += 1) {
                const uchar packed = q[l];
                sum += (d1 * float(packed & 0x0F) - m1) * input[input_offset + l];
                sum += (d2 * float(packed >> 4) - m2) * input[input_offset + 32 + l];
            }

            input_offset += 64;
            q += 32;
        }
    }

    output[row] = sum;
}

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
    uint dim [[thread_position_in_threadgroup]]
) {
    if (head >= head_count || context_length > ZIGGY_MAX_ATTENTION_CONTEXT) return;
    const uint kv_group = head_count / head_count_kv;
    const uint kv_head = head / kv_group;
    const uint kv_offset = kv_head * head_dim;
    device const float *q_head = q + head * head_dim;
    threadgroup float scores[ZIGGY_MAX_ATTENTION_CONTEXT];

    if (dim == 0) {
        float max_value = -INFINITY;
        for (uint token = 0; token <= position; token += 1) {
            device const float *k_head = k_cache + layer_base + token * kv_dim + kv_offset;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d += 1) {
                dot += q_head[d] * k_head[d];
            }
            const float value = dot * scale;
            scores[token] = value;
            if (value > max_value) max_value = value;
        }

        float denom = 0.0f;
        for (uint token = 0; token <= position; token += 1) {
            const float shifted = exp(scores[token] - max_value);
            scores[token] = shifted;
            denom += shifted;
        }
        for (uint token = 0; token <= position; token += 1) {
            scores[token] /= denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (dim >= head_dim) return;

    float sum = 0.0f;
    for (uint token = 0; token <= position; token += 1) {
        sum += scores[token] * v_cache[layer_base + token * kv_dim + kv_offset + dim];
    }
    output[head * head_dim + dim] = sum;
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
