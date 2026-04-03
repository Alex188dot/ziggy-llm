#import "bridge.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <stdio.h>
#import <string.h>

@interface ZiggyMetalState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4K2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4K5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ropePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFusedPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> siluMulPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> addInPlacePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsNormPipeline;
@property(nonatomic, strong) id<MTLCommandBuffer> pendingCommandBuffer;
@end

@implementation ZiggyMetalState
@end

@interface ZiggyMetalBufferState : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) size_t length;
@end

@implementation ZiggyMetalBufferState
@end

static void ziggy_write_error(char *buffer, size_t buffer_len, NSString *message) {
    if (buffer == NULL || buffer_len == 0) return;
    const char *utf8 = message.UTF8String;
    if (utf8 == NULL) utf8 = "unknown metal error";
    snprintf(buffer, buffer_len, "%s", utf8);
    buffer[buffer_len - 1] = '\0';
}

static ZiggyMetalState *ziggy_state(ZiggyMetalContext *ctx) {
    return (__bridge ZiggyMetalState *)ctx;
}

static ZiggyMetalBufferState *ziggy_buffer(ZiggyMetalBuffer *buffer) {
    return (__bridge ZiggyMetalBufferState *)buffer;
}

static const ZiggyMetalBufferState *ziggy_const_buffer(const ZiggyMetalBuffer *buffer) {
    return (__bridge ZiggyMetalBufferState *)buffer;
}

static id<MTLComputePipelineState> ziggy_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString *name,
    NSError **error
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) return nil;
    return [device newComputePipelineStateWithFunction:function error:error];
}

static void ziggy_dispatch_standard(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger grid_width
) {
    NSUInteger thread_width = pipeline.threadExecutionWidth;
    if (thread_width == 0) thread_width = 1;
    NSUInteger threads_per_group = thread_width < grid_width ? thread_width : grid_width;
    if (threads_per_group == 0) threads_per_group = 1;
    MTLSize grid_size = MTLSizeMake(grid_width, 1, 1);
    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
}

static void ziggy_dispatch_q4k_rows(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger row_count
) {
    NSUInteger thread_width = pipeline.threadExecutionWidth;
    if (thread_width == 0) thread_width = 1;
    NSUInteger max_total_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threads_per_group = thread_width * 4;
    if (threads_per_group > max_total_threads) {
        threads_per_group = max_total_threads - (max_total_threads % thread_width);
    }
    if (threads_per_group == 0) threads_per_group = thread_width;
    if (threads_per_group > 256) threads_per_group = 256;

    MTLSize grid_size = MTLSizeMake(row_count, 1, 1);
    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
}

static id<MTLComputePipelineState> ziggy_select_moonq_q4k_pipeline(
    ZiggyMetalState *state,
    uint32_t cols,
    bool add_to_dst
) {
    if (add_to_dst) {
        if (cols == 2048) return state.matvecMoonQ4KAdd2048Pipeline;
        if (cols == 5632) return state.matvecMoonQ4KAdd5632Pipeline;
        return state.matvecMoonQ4KAddPipeline;
    }

    if (cols == 2048) return state.matvecMoonQ4K2048Pipeline;
    if (cols == 5632) return state.matvecMoonQ4K5632Pipeline;
    return state.matvecMoonQ4KPipeline;
}

static id<MTLComputePipelineState> ziggy_select_dense_add_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecAdd2048Pipeline;
    if (cols == 5632) return state.matvecAdd5632Pipeline;
    return state.matvecAddPipeline;
}

static id<MTLComputePipelineState> ziggy_select_q4k_add_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecQ4KAdd2048Pipeline;
    if (cols == 5632) return state.matvecQ4KAdd5632Pipeline;
    return state.matvecQ4KAddPipeline;
}

static id<MTLComputePipelineState> ziggy_select_q6k_add_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecQ6KAdd2048Pipeline;
    if (cols == 5632) return state.matvecQ6KAdd5632Pipeline;
    return state.matvecQ6KAddPipeline;
}

static void ziggy_dispatch_rowwise(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger row_count
) {
    NSUInteger thread_width = pipeline.threadExecutionWidth;
    if (thread_width == 0) thread_width = 1;
    NSUInteger max_total_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threads_per_group = thread_width * 4;
    if (threads_per_group > max_total_threads) {
        threads_per_group = max_total_threads - (max_total_threads % thread_width);
    }
    if (threads_per_group == 0) threads_per_group = thread_width;
    if (threads_per_group > 256) threads_per_group = 256;

    MTLSize grid_size = MTLSizeMake(row_count, 1, 1);
    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
}

static int ziggy_commit_pending(
    ZiggyMetalState *state,
    char *error_message,
    size_t error_message_len
) {
    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    if (command_buffer == nil) return ZIGGY_METAL_OK;

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    state.pendingCommandBuffer = nil;

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal command buffer failed");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    return ZIGGY_METAL_OK;
}

static int ziggy_run_compute(
    ZiggyMetalState *state,
    id<MTLComputePipelineState> pipeline,
    NSUInteger grid_width,
    void (^encode)(id<MTLComputeCommandEncoder> encoder),
    char *error_message,
    size_t error_message_len
) {
    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    const bool has_pending = command_buffer != nil;
    if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    [encoder setComputePipelineState:pipeline];
    encode(encoder);
    ziggy_dispatch_standard(encoder, pipeline, grid_width);
    [encoder endEncoding];

    if (has_pending) return ZIGGY_METAL_OK;

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal command buffer failed");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    return ZIGGY_METAL_OK;
}

int ziggy_metal_create_context(
    const char *shader_source,
    size_t shader_source_len,
    ZiggyMetalContext **out_ctx,
    ZiggyMetalDeviceInfo *out_info,
    char *error_message,
    size_t error_message_len
) {
    if (out_ctx == NULL || shader_source == NULL) {
        ziggy_write_error(error_message, error_message_len, @"missing Metal context output or shader source");
        return ZIGGY_METAL_INITIALIZATION_FAILED;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            ziggy_write_error(error_message, error_message_len, @"no Metal device available");
            return ZIGGY_METAL_UNAVAILABLE;
        }

        NSString *source = [[NSString alloc] initWithBytes:shader_source length:shader_source_len encoding:NSUTF8StringEncoding];
        if (source == nil) {
            ziggy_write_error(error_message, error_message_len, @"shader source was not valid UTF-8");
            return ZIGGY_METAL_COMPILATION_FAILED;
        }

        NSError *compile_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&compile_error];
        if (library == nil) {
            ziggy_write_error(error_message, error_message_len, compile_error.localizedDescription ?: @"failed to compile Metal shaders");
            return ZIGGY_METAL_COMPILATION_FAILED;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal command queue");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        NSError *pipeline_error = nil;
        id<MTLComputePipelineState> matvec_pipeline = ziggy_pipeline(device, library, @"matvec_f32", &pipeline_error);
        if (matvec_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_add_pipeline = ziggy_pipeline(device, library, @"matvec_add_f32", &pipeline_error);
        if (matvec_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal fused-add matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_add_2048_f32", &pipeline_error);
        if (matvec_add_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal fused-add matvec 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_add_5632_f32", &pipeline_error);
        if (matvec_add_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal fused-add matvec 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> matvec_q4k_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_f32", &pipeline_error);
        if (matvec_q4k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_add_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_add_f32", &pipeline_error);
        if (matvec_q4k_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_add_2048_f32", &pipeline_error);
        if (matvec_q4k_add_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k add 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_add_5632_f32", &pipeline_error);
        if (matvec_q4k_add_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k add 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> matvec_q6k_pipeline = ziggy_pipeline(device, library, @"matvec_q6k_f32", &pipeline_error);
        if (matvec_q6k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q6k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q6k_add_pipeline = ziggy_pipeline(device, library, @"matvec_q6k_add_f32", &pipeline_error);
        if (matvec_q6k_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q6k add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q6k_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q6k_add_2048_f32", &pipeline_error);
        if (matvec_q6k_add_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q6k add 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q6k_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q6k_add_5632_f32", &pipeline_error);
        if (matvec_q6k_add_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q6k add 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> matvec_moon_q4k_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_f32", &pipeline_error);
        if (matvec_moon_q4k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_2048_f32", &pipeline_error);
        if (matvec_moon_q4k_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_5632_f32", &pipeline_error);
        if (matvec_moon_q4k_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_add_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_add_f32", &pipeline_error);
        if (matvec_moon_q4k_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_add_2048_f32", &pipeline_error);
        if (matvec_moon_q4k_add_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k add 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_add_5632_f32", &pipeline_error);
        if (matvec_moon_q4k_add_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k add 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> rope_pipeline = ziggy_pipeline(device, library, @"apply_rope_f32", &pipeline_error);
        if (rope_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal rope pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> attention_fused_pipeline = ziggy_pipeline(device, library, @"attention_fused_f32", &pipeline_error);
        if (attention_fused_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal fused-attention pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> silu_mul_pipeline = ziggy_pipeline(device, library, @"silu_mul_f32", &pipeline_error);
        if (silu_mul_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal silu pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> add_in_place_pipeline = ziggy_pipeline(device, library, @"add_in_place_f32", &pipeline_error);
        if (add_in_place_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> rms_norm_pipeline = ziggy_pipeline(device, library, @"rms_norm_f32", &pipeline_error);
        if (rms_norm_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal RMSNorm pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        ZiggyMetalState *state = [ZiggyMetalState new];
        state.device = device;
        state.queue = queue;
        state.matvecPipeline = matvec_pipeline;
        state.matvecAddPipeline = matvec_add_pipeline;
        state.matvecAdd2048Pipeline = matvec_add_2048_pipeline;
        state.matvecAdd5632Pipeline = matvec_add_5632_pipeline;
        state.matvecQ4KPipeline = matvec_q4k_pipeline;
        state.matvecQ4KAddPipeline = matvec_q4k_add_pipeline;
        state.matvecQ4KAdd2048Pipeline = matvec_q4k_add_2048_pipeline;
        state.matvecQ4KAdd5632Pipeline = matvec_q4k_add_5632_pipeline;
        state.matvecQ6KPipeline = matvec_q6k_pipeline;
        state.matvecQ6KAddPipeline = matvec_q6k_add_pipeline;
        state.matvecQ6KAdd2048Pipeline = matvec_q6k_add_2048_pipeline;
        state.matvecQ6KAdd5632Pipeline = matvec_q6k_add_5632_pipeline;
        state.matvecMoonQ4KPipeline = matvec_moon_q4k_pipeline;
        state.matvecMoonQ4K2048Pipeline = matvec_moon_q4k_2048_pipeline;
        state.matvecMoonQ4K5632Pipeline = matvec_moon_q4k_5632_pipeline;
        state.matvecMoonQ4KAddPipeline = matvec_moon_q4k_add_pipeline;
        state.matvecMoonQ4KAdd2048Pipeline = matvec_moon_q4k_add_2048_pipeline;
        state.matvecMoonQ4KAdd5632Pipeline = matvec_moon_q4k_add_5632_pipeline;
        state.ropePipeline = rope_pipeline;
        state.attentionFusedPipeline = attention_fused_pipeline;
        state.siluMulPipeline = silu_mul_pipeline;
        state.addInPlacePipeline = add_in_place_pipeline;
        state.rmsNormPipeline = rms_norm_pipeline;
        *out_ctx = (__bridge_retained void *)state;

        if (out_info != NULL) {
            memset(out_info, 0, sizeof(*out_info));
            snprintf(out_info->name, sizeof(out_info->name), "%s", device.name.UTF8String ?: "Metal device");
            if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
                out_info->recommended_max_working_set_size = device.recommendedMaxWorkingSetSize;
            }
            out_info->has_unified_memory = device.hasUnifiedMemory;
            out_info->low_power = device.lowPower;
        }
        return ZIGGY_METAL_OK;
    }
}

void ziggy_metal_destroy_context(ZiggyMetalContext *ctx) {
    if (ctx == NULL) return;
    CFBridgingRelease(ctx);
}

int ziggy_metal_create_buffer(
    ZiggyMetalContext *ctx,
    const void *bytes,
    size_t length,
    ZiggyMetalBuffer **out_buffer,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || out_buffer == NULL || bytes == NULL || length == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal buffer creation request");
        return ZIGGY_METAL_BUFFER_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        id<MTLBuffer> buffer = [state.device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal buffer");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        ZiggyMetalBufferState *wrapper = [ZiggyMetalBufferState new];
        wrapper.buffer = buffer;
        wrapper.length = length;
        *out_buffer = (__bridge_retained void *)wrapper;
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_create_empty_buffer(
    ZiggyMetalContext *ctx,
    size_t length,
    ZiggyMetalBuffer **out_buffer,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || out_buffer == NULL || length == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid empty Metal buffer creation request");
        return ZIGGY_METAL_BUFFER_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        id<MTLBuffer> buffer = [state.device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate empty Metal buffer");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        ZiggyMetalBufferState *wrapper = [ZiggyMetalBufferState new];
        wrapper.buffer = buffer;
        wrapper.length = length;
        *out_buffer = (__bridge_retained void *)wrapper;
        return ZIGGY_METAL_OK;
    }
}

void ziggy_metal_destroy_buffer(ZiggyMetalBuffer *buffer) {
    if (buffer == NULL) return;
    CFBridgingRelease(buffer);
}

size_t ziggy_metal_buffer_length(const ZiggyMetalBuffer *buffer) {
    if (buffer == NULL) return 0;
    return ziggy_const_buffer(buffer).length;
}

int ziggy_metal_write_buffer(
    ZiggyMetalBuffer *buffer,
    const void *bytes,
    size_t length,
    size_t offset,
    char *error_message,
    size_t error_message_len
) {
    if (buffer == NULL || bytes == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal buffer write request");
        return ZIGGY_METAL_BUFFER_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalBufferState *wrapper = ziggy_buffer(buffer);
        if (offset + length > wrapper.length) {
            ziggy_write_error(error_message, error_message_len, @"Metal buffer write exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
        memcpy((uint8_t *)wrapper.buffer.contents + offset, bytes, length);
        [wrapper.buffer didModifyRange:NSMakeRange(offset, length)];
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_read_buffer(
    const ZiggyMetalBuffer *buffer,
    void *out_bytes,
    size_t length,
    size_t offset,
    char *error_message,
    size_t error_message_len
) {
    if (buffer == NULL || out_bytes == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal buffer read request");
        return ZIGGY_METAL_BUFFER_FAILED;
    }

    @autoreleasepool {
        const ZiggyMetalBufferState *wrapper = ziggy_const_buffer(buffer);
        if (offset + length > wrapper.length) {
            ziggy_write_error(error_message, error_message_len, @"Metal buffer read exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
        memcpy(out_bytes, (const uint8_t *)wrapper.buffer.contents + offset, length);
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.matvecPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_rowwise(encoder, state.matvecPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal matvec command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal fused-add matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputePipelineState> pipeline = ziggy_select_dense_add_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_rowwise(encoder, pipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal fused-add matvec command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q4k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q4k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.matvecQ4KPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, state.matvecQ4KPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q4k_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q4k add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputePipelineState> pipeline = ziggy_select_q4k_add_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q4k add command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q6k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q6k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.matvecQ6KPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, state.matvecQ6KPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q6k command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q6k_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q6k add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputePipelineState> pipeline = ziggy_select_q6k_add_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q6k add command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_moonq_q4k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal MoonQuant q4k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_pipeline(state, cols, false);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal MoonQuant command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_moonq_q4k_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal MoonQuant q4k add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_pipeline(state, cols, true);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal MoonQuant add command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_copy_buffer_region(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *src,
    size_t src_offset,
    ZiggyMetalBuffer *dst,
    size_t dst_offset,
    size_t length,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || src == NULL || dst == NULL || length == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal buffer copy request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *src_buffer = ziggy_const_buffer(src);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);
        if (src_offset + length > src_buffer.length || dst_offset + length > dst_buffer.length) {
            ziggy_write_error(error_message, error_message_len, @"Metal buffer copy exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        if (state.pendingCommandBuffer != nil) {
            id<MTLBlitCommandEncoder> encoder = [state.pendingCommandBuffer blitCommandEncoder];
            if (encoder == nil) {
                ziggy_write_error(error_message, error_message_len, @"failed to create Metal blit encoder");
                return ZIGGY_METAL_EXECUTION_FAILED;
            }

            [encoder copyFromBuffer:src_buffer.buffer sourceOffset:src_offset toBuffer:dst_buffer.buffer destinationOffset:dst_offset size:length];
            [encoder endEncoding];
            return ZIGGY_METAL_OK;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal blit encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder copyFromBuffer:src_buffer.buffer sourceOffset:src_offset toBuffer:dst_buffer.buffer destinationOffset:dst_offset size:length];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal buffer copy failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

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
) {
    if (ctx == NULL || vector == NULL || head_count == 0 || head_dim == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal rope request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (pair_count == 0) return ZIGGY_METAL_OK;

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *buffer = ziggy_buffer(vector);
        return ziggy_run_compute(
            state,
            state.ropePipeline,
            head_count * pair_count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:buffer.buffer offset:0 atIndex:0];
                [encoder setBytes:&head_count length:sizeof(head_count) atIndex:1];
                [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:2];
                [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:3];
                [encoder setBytes:&position length:sizeof(position) atIndex:4];
                [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:5];
            },
            error_message,
            error_message_len
        );
    }
}

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
) {
    if (ctx == NULL || q == NULL || k_cache == NULL || v_cache == NULL || output == NULL || head_count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal fused-attention request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *q_buffer = ziggy_const_buffer(q);
        const ZiggyMetalBufferState *k_buffer = ziggy_const_buffer(k_cache);
        const ZiggyMetalBufferState *v_buffer = ziggy_const_buffer(v_cache);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.attentionFusedPipeline];
        [encoder setBuffer:q_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:k_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:v_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:3];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:4];
        [encoder setBytes:&head_count_kv length:sizeof(head_count_kv) atIndex:5];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        [encoder setBytes:&kv_dim length:sizeof(kv_dim) atIndex:7];
        [encoder setBytes:&context_length length:sizeof(context_length) atIndex:8];
        [encoder setBytes:&position length:sizeof(position) atIndex:9];
        [encoder setBytes:&layer_base length:sizeof(layer_base) atIndex:10];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:11];

        NSUInteger thread_width = state.attentionFusedPipeline.threadExecutionWidth;
        if (thread_width == 0) thread_width = 1;
        NSUInteger max_total_threads = state.attentionFusedPipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger work_items = position + 1;
        if (head_dim > work_items) work_items = head_dim;
        NSUInteger threads_per_group = thread_width;
        while (threads_per_group < work_items && threads_per_group + thread_width <= max_total_threads && threads_per_group < 256) {
            threads_per_group += thread_width;
        }
        if (threads_per_group > max_total_threads) threads_per_group = max_total_threads;
        MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize grid_size = MTLSizeMake(head_count, 1, 1);
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal fused-attention command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_silu_mul_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    uint32_t count,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || gate == NULL || up == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal silu request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *gate_buffer = ziggy_buffer(gate);
        const ZiggyMetalBufferState *up_buffer = ziggy_const_buffer(up);
        return ziggy_run_compute(
            state,
            state.siluMulPipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:gate_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:up_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_begin_sequence(
    ZiggyMetalContext *ctx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal sequence request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        if (state.pendingCommandBuffer != nil) {
            ziggy_write_error(error_message, error_message_len, @"Metal sequence already active");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        state.pendingCommandBuffer = [state.queue commandBuffer];
        if (state.pendingCommandBuffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_commit_sequence(
    ZiggyMetalContext *ctx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal sequence commit request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_commit_pending(state, error_message, error_message_len);
    }
}

int ziggy_metal_add_in_place_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *dst,
    const ZiggyMetalBuffer *src,
    uint32_t count,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || dst == NULL || src == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);
        const ZiggyMetalBufferState *src_buffer = ziggy_const_buffer(src);
        return ziggy_run_compute(
            state,
            state.addInPlacePipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:dst_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:src_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_rms_norm_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *input,
    const ZiggyMetalBuffer *weights,
    ZiggyMetalBuffer *output,
    uint32_t count,
    float eps,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || input == NULL || weights == NULL || output == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal RMSNorm request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        const ZiggyMetalBufferState *weights_buffer = ziggy_const_buffer(weights);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        return ziggy_run_compute(
            state,
            state.rmsNormPipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:weights_buffer.buffer offset:0 atIndex:1];
                [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
                [encoder setBytes:&eps length:sizeof(eps) atIndex:4];
            },
            error_message,
            error_message_len
        );
    }
}
