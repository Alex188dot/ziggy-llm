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
