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

kernel void attention_scores_f32(
    device const float *q [[buffer(0)]],
    device const float *k_cache [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_count_kv [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant uint &kv_dim [[buffer(6)]],
    constant uint &context_length [[buffer(7)]],
    constant uint &position [[buffer(8)]],
    constant uint &layer_base [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    uint head [[thread_position_in_grid]]
) {
    if (head >= head_count) return;

    const uint kv_group = head_count / head_count_kv;
    const uint kv_head = head / kv_group;
    const uint kv_offset = kv_head * head_dim;
    device float *head_scores = scores + head * context_length;
    device const float *q_head = q + head * head_dim;

    float max_value = -INFINITY;
    for (uint token = 0; token <= position; token += 1) {
        device const float *k_head = k_cache + layer_base + token * kv_dim + kv_offset;
        float sum = 0.0f;
        for (uint dim = 0; dim < head_dim; dim += 1) {
            sum += q_head[dim] * k_head[dim];
        }
        const float value = sum * scale;
        head_scores[token] = value;
        if (value > max_value) max_value = value;
    }

    float denom = 0.0f;
    for (uint token = 0; token <= position; token += 1) {
        const float shifted = exp(head_scores[token] - max_value);
        head_scores[token] = shifted;
        denom += shifted;
    }

    for (uint token = 0; token <= position; token += 1) {
        head_scores[token] /= denom;
    }
}

kernel void attention_values_f32(
    device const float *scores [[buffer(0)]],
    device const float *v_cache [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_count_kv [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant uint &kv_dim [[buffer(6)]],
    constant uint &context_length [[buffer(7)]],
    constant uint &position [[buffer(8)]],
    constant uint &layer_base [[buffer(9)]],
    uint index [[thread_position_in_grid]]
) {
    const uint total = head_count * head_dim;
    if (index >= total) return;

    const uint head = index / head_dim;
    const uint dim = index % head_dim;
    const uint kv_group = head_count / head_count_kv;
    const uint kv_head = head / kv_group;
    const uint kv_offset = kv_head * head_dim + dim;
    device const float *head_scores = scores + head * context_length;

    float sum = 0.0f;
    for (uint token = 0; token <= position; token += 1) {
        sum += head_scores[token] * v_cache[layer_base + token * kv_dim + kv_offset];
    }
    output[index] = sum;
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
