#ifndef ZIGGY_METAL_BRIDGE_H
#define ZIGGY_METAL_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef void ZiggyMetalContext;
typedef void ZiggyMetalBuffer;

typedef struct {
    char name[128];
    uint64_t recommended_max_working_set_size;
    bool has_unified_memory;
    bool low_power;
} ZiggyMetalDeviceInfo;

enum {
    ZIGGY_METAL_OK = 0,
    ZIGGY_METAL_UNAVAILABLE = 1,
    ZIGGY_METAL_INITIALIZATION_FAILED = 2,
    ZIGGY_METAL_COMPILATION_FAILED = 3,
    ZIGGY_METAL_BUFFER_FAILED = 4,
    ZIGGY_METAL_EXECUTION_FAILED = 5,
};

int ziggy_metal_create_context(
    const char *shader_source,
    size_t shader_source_len,
    ZiggyMetalContext **out_ctx,
    ZiggyMetalDeviceInfo *out_info,
    char *error_message,
    size_t error_message_len
);
void ziggy_metal_destroy_context(ZiggyMetalContext *ctx);

int ziggy_metal_create_buffer(
    ZiggyMetalContext *ctx,
    const void *bytes,
    size_t length,
    ZiggyMetalBuffer **out_buffer,
    char *error_message,
    size_t error_message_len
);
int ziggy_metal_create_empty_buffer(
    ZiggyMetalContext *ctx,
    size_t length,
    ZiggyMetalBuffer **out_buffer,
    char *error_message,
    size_t error_message_len
);
void ziggy_metal_destroy_buffer(ZiggyMetalBuffer *buffer);
size_t ziggy_metal_buffer_length(const ZiggyMetalBuffer *buffer);

int ziggy_metal_write_buffer(
    ZiggyMetalBuffer *buffer,
    const void *bytes,
    size_t length,
    size_t offset,
    char *error_message,
    size_t error_message_len
);
int ziggy_metal_read_buffer(
    const ZiggyMetalBuffer *buffer,
    void *out_bytes,
    size_t length,
    size_t offset,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_run_matvec_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_run_matvec_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_run_matvec_q4k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_run_matvec_q4k_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_run_matvec_q6k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_run_matvec_q6k_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_run_matvec_moonq_q4k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_run_matvec_moonq_q4k_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_copy_buffer_region(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *src,
    size_t src_offset,
    ZiggyMetalBuffer *dst,
    size_t dst_offset,
    size_t length,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_apply_rope_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *vector,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t position,
    float freq_base,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_attention_fused_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *q,
    const ZiggyMetalBuffer *k_cache,
    const ZiggyMetalBuffer *v_cache,
    ZiggyMetalBuffer *output,
    uint32_t head_count,
    uint32_t head_count_kv,
    uint32_t head_dim,
    uint32_t kv_dim,
    uint32_t context_length,
    uint32_t position,
    uint32_t layer_base,
    float scale,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_silu_mul_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    uint32_t count,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_begin_sequence(
    ZiggyMetalContext *ctx,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_commit_sequence(
    ZiggyMetalContext *ctx,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_add_in_place_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *dst,
    const ZiggyMetalBuffer *src,
    uint32_t count,
    char *error_message,
    size_t error_message_len
);

int ziggy_metal_rms_norm_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *input,
    const ZiggyMetalBuffer *weights,
    ZiggyMetalBuffer *output,
    uint32_t count,
    float eps,
    char *error_message,
    size_t error_message_len
);

#endif
