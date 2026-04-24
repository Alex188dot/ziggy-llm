
#import "bridge.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <stdio.h>
#import <string.h>
#import <time.h>

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
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KArgmaxPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ80Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ80AddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ3KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ3KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexedMatvecQ3KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> dualIndexedMatvecQ3KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexedMatvecQ3KAddWeightedPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4K2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4K5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ropePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ropeToDstPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> storeKvHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFusedPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> siluMulPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> geluMulPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> addInPlacePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsNormPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsNormPerHeadPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> argmaxPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> topKPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> normalizeTopKPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sampleTopKPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> weightedSumTopKPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sigmoidScaleAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchArgmaxPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecQ4KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecQ4KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecQ4KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecQ4KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ5KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ5KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ5KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ5KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KSiluDownAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KSiluDownAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KSiluDownAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KSiluDownAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KSiluDownAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KSiluDownAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchSiluMulPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchAddInPlacePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchRmsNormPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexedMatvecIQ3XXSPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> dualIndexedMatvecIQ3XXSPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexedMatvecIQ4XSPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexedMatvecIQ4XSAddWeightedPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> linearConv1dPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> linearRecurrentNormPipeline;
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

static uint64_t ziggy_now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

static void ziggy_fill_commit_stats(
    id<MTLCommandBuffer> command_buffer,
    uint64_t cpu_wait_ns,
    ZiggyMetalCommitStats *out_stats
) {
    if (out_stats == NULL) return;
    out_stats->cpu_wait_ns = cpu_wait_ns;
    out_stats->gpu_elapsed_ns = 0;
    out_stats->gpu_timestamps_valid = false;

    if (![command_buffer respondsToSelector:@selector(GPUStartTime)] ||
        ![command_buffer respondsToSelector:@selector(GPUEndTime)]) {
        return;
    }

    const CFTimeInterval gpu_start = command_buffer.GPUStartTime;
    const CFTimeInterval gpu_end = command_buffer.GPUEndTime;
    if (!(gpu_end > gpu_start) || gpu_start <= 0.0) return;

    out_stats->gpu_elapsed_ns = (uint64_t)((gpu_end - gpu_start) * 1000000000.0);
    out_stats->gpu_timestamps_valid = out_stats->gpu_elapsed_ns > 0;
}

static id<MTLCommandBuffer> ziggy_new_command_buffer(id<MTLCommandQueue> queue, char *error_message, size_t error_message_len) {
    if (@available(macOS 11.0, *)) {
        MTLCommandBufferDescriptor *desc = [[MTLCommandBufferDescriptor alloc] init];
        desc.retainedReferences = NO;
        desc.errorOptions = MTLCommandBufferErrorOptionNone;
        id<MTLCommandBuffer> cb = [queue commandBufferWithDescriptor:desc];
        if (cb == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
        }
        return cb;
    } else {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (cb == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
        }
        return cb;
    }
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

static void ziggy_dispatch_moonq_q4k_rows(
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

    NSUInteger grid_width = (row_count + 3) / 4; 
    MTLSize grid_size = MTLSizeMake(grid_width, 1, 1);
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

static id<MTLComputePipelineState> ziggy_select_q5k_add_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    (void)cols;
    return state.matvecQ5KAddPipeline;
}

static NSUInteger ziggy_rowwise_thread_count(id<MTLComputePipelineState> pipeline) {
    NSUInteger thread_width = pipeline.threadExecutionWidth;
    if (thread_width == 0) thread_width = 1;
    NSUInteger max_total_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threads_per_group = thread_width * 4;
    if (threads_per_group > max_total_threads) {
        threads_per_group = max_total_threads - (max_total_threads % thread_width);
    }
    if (threads_per_group == 0) threads_per_group = thread_width;
    if (threads_per_group > 256) threads_per_group = 256;
    return threads_per_group;
}

static void ziggy_dispatch_rowwise(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger row_count
) {
    NSUInteger threads_per_group = ziggy_rowwise_thread_count(pipeline);

    MTLSize grid_size = MTLSizeMake(row_count, 1, 1);
    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
}

static void ziggy_dispatch_indexed_rowwise(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger row_count,
    NSUInteger cols
) {
    // Indexed matvec kernels dequantize one block per thread iteration.
    // Use exactly one thread per block so no lanes are idle.
    const NSUInteger values_per_block = 256;
    NSUInteger block_count = cols / values_per_block;
    if (block_count == 0) block_count = 1;

    NSUInteger threads_per_group = block_count;
    if (threads_per_group > 128) threads_per_group = 128;
    if (threads_per_group < 1) threads_per_group = 1;

    MTLSize grid_size = MTLSizeMake(row_count, 1, 1);
    MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
}

static int ziggy_commit_pending(
    ZiggyMetalState *state,
    ZiggyMetalCommitStats *out_stats,
    char *error_message,
    size_t error_message_len
) {
    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    if (command_buffer == nil) {
        if (out_stats != NULL) memset(out_stats, 0, sizeof(*out_stats));
        return ZIGGY_METAL_OK;
    }

    const uint64_t wait_start_ns = ziggy_now_ns();
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    const uint64_t cpu_wait_ns = ziggy_now_ns() - wait_start_ns;
    state.pendingCommandBuffer = nil;
    ziggy_fill_commit_stats(command_buffer, cpu_wait_ns, out_stats);

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
    if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
    if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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

static int ziggy_run_single_threadgroup(
    ZiggyMetalState *state,
    id<MTLComputePipelineState> pipeline,
    NSUInteger thread_count,
    void (^encode)(id<MTLComputeCommandEncoder> encoder),
    char *error_message,
    size_t error_message_len
) {
    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    const bool has_pending = command_buffer != nil;
    if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
    if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    [encoder setComputePipelineState:pipeline];
    encode(encoder);
    MTLSize grid_size = MTLSizeMake(1, 1, 1);
    MTLSize group_size = MTLSizeMake(thread_count, 1, 1);
    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
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

static int ziggy_run_rowwise_matvec(
    ZiggyMetalState *state,
    id<MTLComputePipelineState> pipeline,
    const ZiggyMetalBufferState *matrix_buffer,
    const ZiggyMetalBufferState *input_buffer,
    ZiggyMetalBufferState *output_buffer,
    size_t output_offset_bytes,
    uint32_t rows,
    uint32_t cols,
    int dispatch_type, // 0 = std, 1 = q4, 2 = moonq_q4
    NSString *invalid_message,
    NSString *command_error_message,
    char *error_message,
    size_t error_message_len
) {
    if (matrix_buffer == NULL || input_buffer == NULL || output_buffer == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, invalid_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    if (output_offset_bytes + ((size_t)rows * sizeof(float)) > output_buffer.length) {
        ziggy_write_error(error_message, error_message_len, @"Metal matvec output exceeded allocation");
        return ZIGGY_METAL_BUFFER_FAILED;
    }

    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    const bool has_pending = command_buffer != nil;
    if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
    if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
    [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer.buffer offset:output_offset_bytes atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];

    if (dispatch_type == 2) {
        ziggy_dispatch_moonq_q4k_rows(encoder, pipeline, rows);
    } else if (dispatch_type == 1) {
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
    } else {
        ziggy_dispatch_rowwise(encoder, pipeline, rows);
    }
    [encoder endEncoding];

    if (has_pending) return ZIGGY_METAL_OK;

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: command_error_message);
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

        MTLCompileOptions *compile_options = [[MTLCompileOptions alloc] init];
        compile_options.fastMathEnabled = YES;
        
        NSError *compile_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:compile_options error:&compile_error];
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

        id<MTLComputePipelineState> matvec_q5k_pipeline = ziggy_pipeline(device, library, @"matvec_q5k_f32", &pipeline_error);
        if (matvec_q5k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q5k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q5k_add_pipeline = ziggy_pipeline(device, library, @"matvec_q5k_add_f32", &pipeline_error);
        if (matvec_q5k_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q5k add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> matvec_q6k_pipeline = ziggy_pipeline(device, library, @"matvec_q6k_f32", &pipeline_error);
        if (matvec_q6k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q6k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q6k_argmax_pipeline = ziggy_pipeline(device, library, @"matvec_q6k_argmax_f32", &pipeline_error);
        if (matvec_q6k_argmax_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q6k argmax pipeline");
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
        id<MTLComputePipelineState> matvec_q80_pipeline = ziggy_pipeline(device, library, @"matvec_q8_0_f32", &pipeline_error);
        if (matvec_q80_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q8_0 matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q80_add_pipeline = ziggy_pipeline(device, library, @"matvec_q8_0_add_f32", &pipeline_error);
        if (matvec_q80_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q8_0 add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q3k_pipeline = ziggy_pipeline(device, library, @"matvec_q3k_f32", &pipeline_error);
        if (matvec_q3k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q3k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q3k_add_pipeline = ziggy_pipeline(device, library, @"matvec_q3k_add_f32", &pipeline_error);
        if (matvec_q3k_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q3k add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> indexed_matvec_q3k_pipeline = ziggy_pipeline(device, library, @"indexed_matvec_q3k_f32", &pipeline_error);
        if (indexed_matvec_q3k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal indexed q3k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> dual_indexed_matvec_q3k_pipeline = ziggy_pipeline(device, library, @"dual_indexed_matvec_q3k_f32", &pipeline_error);
        if (dual_indexed_matvec_q3k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal dual indexed q3k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> indexed_matvec_q3k_add_weighted_pipeline = ziggy_pipeline(device, library, @"indexed_matvec_q3k_add_weighted_f32", &pipeline_error);
        if (indexed_matvec_q3k_add_weighted_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal indexed q3k add weighted pipeline");
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

        id<MTLComputePipelineState> matvec_q4k_silu_down_add_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_silu_down_add_f32", &pipeline_error);
        if (matvec_q4k_silu_down_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k silu down add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_silu_down_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_silu_down_add_2048_f32", &pipeline_error);
        id<MTLComputePipelineState> matvec_q4k_silu_down_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_silu_down_add_5632_f32", &pipeline_error);

        id<MTLComputePipelineState> matvec_moon_q4k_silu_down_add_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_silu_down_add_f32", &pipeline_error);
        if (matvec_moon_q4k_silu_down_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal moonq q4k silu down add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_silu_down_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_silu_down_add_2048_f32", &pipeline_error);
        id<MTLComputePipelineState> matvec_moon_q4k_silu_down_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_silu_down_add_5632_f32", &pipeline_error);

        id<MTLComputePipelineState> rope_pipeline = ziggy_pipeline(device, library, @"apply_rope_f32", &pipeline_error);
        if (rope_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal rope pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> rope_to_dst_pipeline = ziggy_pipeline(device, library, @"apply_rope_to_dst_f32", &pipeline_error);
        if (rope_to_dst_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal rope-to-dst pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> store_kv_half_pipeline = ziggy_pipeline(device, library, @"store_kv_half", &pipeline_error);
        if (store_kv_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal store_kv_half pipeline");
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
        id<MTLComputePipelineState> gelu_mul_pipeline = ziggy_pipeline(device, library, @"gelu_mul_f32", &pipeline_error);
        if (gelu_mul_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal gelu pipeline");
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
        id<MTLComputePipelineState> rms_norm_per_head_pipeline = ziggy_pipeline(device, library, @"rms_norm_per_head_f32", &pipeline_error);
        if (rms_norm_per_head_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal RMSNorm Per Head pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> argmax_pipeline = ziggy_pipeline(device, library, @"argmax_f32", &pipeline_error);
        if (argmax_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal argmax pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> topk_pipeline = ziggy_pipeline(device, library, @"topk_f32", &pipeline_error);
        if (topk_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal top-k pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> normalize_topk_pipeline = ziggy_pipeline(device, library, @"normalize_topk_f32", &pipeline_error);
        if (normalize_topk_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal normalize top-k pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> sample_topk_pipeline = ziggy_pipeline(device, library, @"sample_topk_f32", &pipeline_error);
        if (sample_topk_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal sample top-k pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> weighted_sum_topk_pipeline = ziggy_pipeline(device, library, @"weighted_sum_topk_f32", &pipeline_error);
        if (weighted_sum_topk_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal weighted sum top-k pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> sigmoid_scale_add_pipeline = ziggy_pipeline(device, library, @"sigmoid_scale_add_f32", &pipeline_error);
        if (sigmoid_scale_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal sigmoid scale add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> batch_argmax_pipeline = ziggy_pipeline(device, library, @"batch_argmax_f32", &pipeline_error);
        if (batch_argmax_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch argmax pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> batch_matvec_add_pipeline = ziggy_pipeline(device, library, @"batch_matvec_add_f32", &pipeline_error);
        if (batch_matvec_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch matvec add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> batch_matvec_q4k_add_pipeline = ziggy_pipeline(device, library, @"batch_matvec_q4k_add_f32", &pipeline_error);
        if (batch_matvec_q4k_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch q4k add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> batch_matvec_q4k_add_2048_pipeline = ziggy_pipeline(device, library, @"batch_matvec_q4k_add_2048_f32", &pipeline_error);
        if (batch_matvec_q4k_add_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch q4k add 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> batch_matvec_q4k_add_5632_pipeline = ziggy_pipeline(device, library, @"batch_matvec_q4k_add_5632_f32", &pipeline_error);
        if (batch_matvec_q4k_add_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch q4k add 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> batch_matvec_q4k_pipeline = ziggy_pipeline(device, library, @"batch_matvec_q4k_f32", &pipeline_error);
        if (batch_matvec_q4k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch q4k pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> batch_silu_mul_pipeline = ziggy_pipeline(device, library, @"batch_silu_mul_f32", &pipeline_error);
        if (batch_silu_mul_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch silu pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> batch_add_in_place_pipeline = ziggy_pipeline(device, library, @"batch_add_in_place_f32", &pipeline_error);
        if (batch_add_in_place_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> batch_rms_norm_pipeline = ziggy_pipeline(device, library, @"batch_rms_norm_f32", &pipeline_error);
        if (batch_rms_norm_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal batch RMSNorm pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> indexed_matvec_iq3_xxs_pipeline = ziggy_pipeline(device, library, @"indexed_matvec_iq3_xxs_f32", &pipeline_error);
        if (indexed_matvec_iq3_xxs_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal indexed iq3_xxs pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> dual_indexed_matvec_iq3_xxs_pipeline = ziggy_pipeline(device, library, @"dual_indexed_matvec_iq3_xxs_f32", &pipeline_error);
        if (dual_indexed_matvec_iq3_xxs_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal dual indexed iq3_xxs pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> indexed_matvec_iq4_xs_pipeline = ziggy_pipeline(device, library, @"indexed_matvec_iq4_xs_f32", &pipeline_error);
        if (indexed_matvec_iq4_xs_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal indexed iq4_xs pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> indexed_matvec_iq4_xs_add_weighted_pipeline = ziggy_pipeline(device, library, @"indexed_matvec_iq4_xs_add_weighted_f32", &pipeline_error);
        if (indexed_matvec_iq4_xs_add_weighted_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal indexed iq4_xs add-weighted pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        id<MTLComputePipelineState> linear_conv1d_pipeline = ziggy_pipeline(device, library, @"linear_conv1d_f32", &pipeline_error);
        if (linear_conv1d_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal linear conv1d pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> linear_recurrent_norm_pipeline = ziggy_pipeline(device, library, @"linear_recurrent_norm_f32", &pipeline_error);
        if (linear_recurrent_norm_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal linear recurrent norm pipeline");
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
        state.matvecQ5KPipeline = matvec_q5k_pipeline;
        state.matvecQ5KAddPipeline = matvec_q5k_add_pipeline;
        state.matvecQ6KPipeline = matvec_q6k_pipeline;
        state.matvecQ6KArgmaxPipeline = matvec_q6k_argmax_pipeline;
        state.matvecQ6KAddPipeline = matvec_q6k_add_pipeline;
        state.matvecQ6KAdd2048Pipeline = matvec_q6k_add_2048_pipeline;
        state.matvecQ6KAdd5632Pipeline = matvec_q6k_add_5632_pipeline;
        state.matvecQ80Pipeline = matvec_q80_pipeline;
        state.matvecQ80AddPipeline = matvec_q80_add_pipeline;
        state.matvecQ3KPipeline = matvec_q3k_pipeline;
        state.matvecQ3KAddPipeline = matvec_q3k_add_pipeline;
        state.indexedMatvecQ3KPipeline = indexed_matvec_q3k_pipeline;
        state.dualIndexedMatvecQ3KPipeline = dual_indexed_matvec_q3k_pipeline;
        state.indexedMatvecQ3KAddWeightedPipeline = indexed_matvec_q3k_add_weighted_pipeline;
        state.matvecMoonQ4KPipeline = matvec_moon_q4k_pipeline;
        state.matvecMoonQ4K2048Pipeline = matvec_moon_q4k_2048_pipeline;
        state.matvecMoonQ4K5632Pipeline = matvec_moon_q4k_5632_pipeline;
        state.matvecMoonQ4KAddPipeline = matvec_moon_q4k_add_pipeline;
        state.matvecMoonQ4KAdd2048Pipeline = matvec_moon_q4k_add_2048_pipeline;
        state.matvecMoonQ4KAdd5632Pipeline = matvec_moon_q4k_add_5632_pipeline;
        state.matvecQ4KSiluDownAddPipeline = matvec_q4k_silu_down_add_pipeline;
        state.matvecQ4KSiluDownAdd2048Pipeline = matvec_q4k_silu_down_add_2048_pipeline;
        state.matvecQ4KSiluDownAdd5632Pipeline = matvec_q4k_silu_down_add_5632_pipeline;
        state.matvecMoonQ4KSiluDownAddPipeline = matvec_moon_q4k_silu_down_add_pipeline;
        state.matvecMoonQ4KSiluDownAdd2048Pipeline = matvec_moon_q4k_silu_down_add_2048_pipeline;
        state.matvecMoonQ4KSiluDownAdd5632Pipeline = matvec_moon_q4k_silu_down_add_5632_pipeline;
        state.ropePipeline = rope_pipeline;
        state.ropeToDstPipeline = rope_to_dst_pipeline;
        state.storeKvHalfPipeline = store_kv_half_pipeline;
        state.attentionFusedPipeline = attention_fused_pipeline;
        state.siluMulPipeline = silu_mul_pipeline;
        state.geluMulPipeline = gelu_mul_pipeline;
        state.addInPlacePipeline = add_in_place_pipeline;
        state.rmsNormPipeline = rms_norm_pipeline;
        state.rmsNormPerHeadPipeline = rms_norm_per_head_pipeline;
        state.argmaxPipeline = argmax_pipeline;
        state.topKPipeline = topk_pipeline;
        state.normalizeTopKPipeline = normalize_topk_pipeline;
        state.sampleTopKPipeline = sample_topk_pipeline;
        state.weightedSumTopKPipeline = weighted_sum_topk_pipeline;
        state.sigmoidScaleAddPipeline = sigmoid_scale_add_pipeline;
        state.batchArgmaxPipeline = batch_argmax_pipeline;
        state.batchMatvecAddPipeline = batch_matvec_add_pipeline;
        state.batchMatvecQ4KAddPipeline = batch_matvec_q4k_add_pipeline;
        state.batchMatvecQ4KAdd2048Pipeline = batch_matvec_q4k_add_2048_pipeline;
        state.batchMatvecQ4KAdd5632Pipeline = batch_matvec_q4k_add_5632_pipeline;
        state.batchMatvecQ4KPipeline = batch_matvec_q4k_pipeline;
        state.batchSiluMulPipeline = batch_silu_mul_pipeline;
        state.batchAddInPlacePipeline = batch_add_in_place_pipeline;
        state.batchRmsNormPipeline = batch_rms_norm_pipeline;
        state.indexedMatvecIQ3XXSPipeline = indexed_matvec_iq3_xxs_pipeline;
        state.dualIndexedMatvecIQ3XXSPipeline = dual_indexed_matvec_iq3_xxs_pipeline;
        state.indexedMatvecIQ4XSPipeline = indexed_matvec_iq4_xs_pipeline;
        state.indexedMatvecIQ4XSAddWeightedPipeline = indexed_matvec_iq4_xs_add_weighted_pipeline;
        state.linearConv1dPipeline = linear_conv1d_pipeline;
        state.linearRecurrentNormPipeline = linear_recurrent_norm_pipeline;
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
    return ziggy_metal_run_matvec_f32_to_dst(
        ctx,
        matrix,
        input,
        output,
        0,
        rows,
        cols,
        error_message,
        error_message_len
    );
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
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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

int ziggy_metal_run_matvec_f32_to_dst(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    size_t output_offset_bytes,
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
        return ziggy_run_rowwise_matvec(
            state,
            state.matvecPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            output_offset_bytes,
            rows,
            cols,
            0,
            @"invalid Metal matvec request",
            @"Metal matvec command failed",
            error_message,
            error_message_len
        );
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
    return ziggy_metal_run_matvec_q4k_f32_to_dst(
        ctx,
        matrix,
        input,
        output,
        0,
        rows,
        cols,
        error_message,
        error_message_len
    );
}

int ziggy_metal_run_matvec_q4k_f32_to_dst(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    size_t output_offset_bytes,
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
        return ziggy_run_rowwise_matvec(
            state,
            state.matvecQ4KPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            output_offset_bytes,
            rows,
            cols,
            1,
            @"invalid Metal q4k matvec request",
            @"Metal command buffer failed",
            error_message,
            error_message_len
        );
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
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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

int ziggy_metal_run_matvec_q5k_f32(
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
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q5k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_rowwise_matvec(
            state,
            state.matvecQ5KPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            0,
            rows,
            cols,
            1,
            @"invalid Metal q5k matvec request",
            @"Metal command buffer failed",
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_run_matvec_q5k_f32_to_dst(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    size_t output_offset_bytes,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q5k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_rowwise_matvec(
            state,
            state.matvecQ5KPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            output_offset_bytes,
            rows,
            cols,
            1,
            @"invalid Metal q5k matvec request",
            @"Metal command buffer failed",
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_run_matvec_q5k_add_f32(
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
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q5k add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputePipelineState> pipeline = ziggy_select_q5k_add_pipeline(state, cols);
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
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q5k add command buffer failed");
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
    return ziggy_metal_run_matvec_q6k_f32_to_dst(
        ctx,
        matrix,
        input,
        output,
        0,
        rows,
        cols,
        error_message,
        error_message_len
    );
}

int ziggy_metal_run_matvec_q6k_f32_to_dst(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    size_t output_offset_bytes,
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
        return ziggy_run_rowwise_matvec(
            state,
            state.matvecQ6KPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            output_offset_bytes,
            rows,
            cols,
            1,
            @"invalid Metal q6k matvec request",
            @"Metal q6k command buffer failed",
            error_message,
            error_message_len
        );
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
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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

int ziggy_metal_run_matvec_q6k_argmax_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output_packed,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output_packed == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q6k argmax request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output_packed);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        if (input_buffer.length < ((size_t)cols * sizeof(float)) ||
            output_buffer.length < sizeof(uint64_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal q6k argmax exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        [encoder setComputePipelineState:state.matvecQ6KArgmaxPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, state.matvecQ6KArgmaxPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q6k argmax command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q8_0_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    return ziggy_metal_run_matvec_q8_0_f32_to_dst(ctx, matrix, input, output, 0, rows, cols, error_message, error_message_len);
}

int ziggy_metal_run_matvec_q8_0_f32_to_dst(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    size_t output_offset_bytes,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q8_0 matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_rowwise_matvec(
            state,
            state.matvecQ80Pipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            output_offset_bytes,
            rows,
            cols,
            1,
            @"invalid Metal q8_0 matvec request",
            @"Metal q8_0 command buffer failed",
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_run_matvec_q8_0_add_f32(
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
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q8_0 add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.matvecQ80AddPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, state.matvecQ80AddPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q8_0 add command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q3k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    return ziggy_metal_run_matvec_q3k_f32_to_dst(
        ctx,
        matrix,
        input,
        output,
        0,
        rows,
        cols,
        error_message,
        error_message_len
    );
}

int ziggy_metal_run_matvec_q3k_f32_to_dst(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    size_t output_offset_bytes,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q3k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_rowwise_matvec(
            state,
            state.matvecQ3KPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            output_offset_bytes,
            rows,
            cols,
            1,
            @"invalid Metal q3k matvec request",
            @"Metal q3k command buffer failed",
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_run_matvec_q3k_add_f32(
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
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q3k add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.matvecQ3KAddPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, state.matvecQ3KAddPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q3k add command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_indexed_matvec_q3k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    const ZiggyMetalBuffer *entries,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || entries == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal indexed q3k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        const ZiggyMetalBufferState *entries_buffer = ziggy_const_buffer(entries);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.indexedMatvecQ3KPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setBuffer:entries_buffer.buffer offset:0 atIndex:5];
        [encoder setBytes:&slot_idx length:sizeof(slot_idx) atIndex:6];
        [encoder setBytes:&rows_per_expert length:sizeof(rows_per_expert) atIndex:7];
        ziggy_dispatch_q4k_rows(encoder, state.indexedMatvecQ3KPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal indexed q3k command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_dual_indexed_matvec_q3k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix_a,
    const ZiggyMetalBuffer *matrix_b,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output_a,
    ZiggyMetalBuffer *output_b,
    uint32_t rows,
    uint32_t cols,
    const ZiggyMetalBuffer *entries,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix_a == NULL || matrix_b == NULL || input == NULL || output_a == NULL || output_b == NULL || entries == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal dual indexed q3k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_a_buffer = ziggy_const_buffer(matrix_a);
        const ZiggyMetalBufferState *matrix_b_buffer = ziggy_const_buffer(matrix_b);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_a_buffer = ziggy_buffer(output_a);
        ZiggyMetalBufferState *output_b_buffer = ziggy_buffer(output_b);
        const ZiggyMetalBufferState *entries_buffer = ziggy_const_buffer(entries);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.dualIndexedMatvecQ3KPipeline];
        [encoder setBuffer:matrix_a_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:matrix_b_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:output_a_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:output_b_buffer.buffer offset:0 atIndex:4];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:6];
        [encoder setBuffer:entries_buffer.buffer offset:0 atIndex:7];
        [encoder setBytes:&slot_idx length:sizeof(slot_idx) atIndex:8];
        [encoder setBytes:&rows_per_expert length:sizeof(rows_per_expert) atIndex:9];
        ziggy_dispatch_q4k_rows(encoder, state.dualIndexedMatvecQ3KPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal dual indexed q3k command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_indexed_matvec_q3k_add_weighted_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    const ZiggyMetalBuffer *entries,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || entries == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal indexed q3k add weighted request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        const ZiggyMetalBufferState *entries_buffer = ziggy_const_buffer(entries);
        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.indexedMatvecQ3KAddWeightedPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setBuffer:entries_buffer.buffer offset:0 atIndex:5];
        [encoder setBytes:&slot_idx length:sizeof(slot_idx) atIndex:6];
        [encoder setBytes:&rows_per_expert length:sizeof(rows_per_expert) atIndex:7];
        ziggy_dispatch_q4k_rows(encoder, state.indexedMatvecQ3KAddWeightedPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal indexed q3k add weighted command buffer failed");
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
    return ziggy_metal_run_matvec_moonq_q4k_f32_to_dst(
        ctx,
        matrix,
        input,
        output,
        0,
        rows,
        cols,
        error_message,
        error_message_len
    );
}

int ziggy_metal_run_matvec_moonq_q4k_f32_to_dst(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    size_t output_offset_bytes,
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
        id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_pipeline(state, cols, false);
        return ziggy_run_rowwise_matvec(
            state,
            pipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            output_offset_bytes,
            rows,
            cols,
            2,
            @"invalid Metal MoonQuant q4k matvec request",
            @"Metal MoonQuant command buffer failed",
            error_message,
            error_message_len
        );
    }
}

static int ziggy_run_fused_silu_down_matvec(
    ZiggyMetalState *state,
    id<MTLComputePipelineState> pipeline,
    const ZiggyMetalBufferState *matrix_buffer,
    const ZiggyMetalBufferState *gate_buffer,
    const ZiggyMetalBufferState *up_buffer,
    ZiggyMetalBufferState *output_buffer,
    uint32_t rows,
    uint32_t cols,
    int dispatch_type,
    NSString *invalid_message,
    NSString *command_error_message,
    char *error_message,
    size_t error_message_len
) {
    if (matrix_buffer == NULL || gate_buffer == NULL || up_buffer == NULL || output_buffer == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, invalid_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    const bool has_pending = command_buffer != nil;
    if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
    if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
    [encoder setBuffer:gate_buffer.buffer offset:0 atIndex:1];
    [encoder setBuffer:up_buffer.buffer offset:0 atIndex:2];
    [encoder setBuffer:output_buffer.buffer offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:5];

    if (dispatch_type == 2) {
        ziggy_dispatch_moonq_q4k_rows(encoder, pipeline, rows);
    } else {
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
    }
    [encoder endEncoding];

    if (has_pending) return ZIGGY_METAL_OK;

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: command_error_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    return ZIGGY_METAL_OK;
}

int ziggy_metal_run_matvec_q4k_silu_down_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        id<MTLComputePipelineState> pipeline = cols == 2048 ? state.matvecQ4KSiluDownAdd2048Pipeline : (cols == 5632 ? state.matvecQ4KSiluDownAdd5632Pipeline : state.matvecQ4KSiluDownAddPipeline);
        return ziggy_run_fused_silu_down_matvec(state, pipeline, ziggy_const_buffer(matrix), ziggy_const_buffer(gate), ziggy_const_buffer(up), ziggy_buffer(output), rows, cols, 1, @"invalid Metal q4k silu down add request", @"Metal command buffer failed", error_message, error_message_len);
    }
}

int ziggy_metal_run_matvec_moonq_q4k_silu_down_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        id<MTLComputePipelineState> pipeline = cols == 2048 ? state.matvecMoonQ4KSiluDownAdd2048Pipeline : (cols == 5632 ? state.matvecMoonQ4KSiluDownAdd5632Pipeline : state.matvecMoonQ4KSiluDownAddPipeline);
        return ziggy_run_fused_silu_down_matvec(state, pipeline, ziggy_const_buffer(matrix), ziggy_const_buffer(gate), ziggy_const_buffer(up), ziggy_buffer(output), rows, cols, 2, @"invalid Metal MoonQuant silu down add request", @"Metal command buffer failed", error_message, error_message_len);
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
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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
        ziggy_dispatch_moonq_q4k_rows(encoder, pipeline, rows);
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

        id<MTLCommandBuffer> command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    return ziggy_metal_apply_rope_at_offset_f32(
        ctx,
        vector,
        0,
        head_count,
        head_dim,
        rope_dim,
        position,
        freq_base,
        rope_style,
        error_message,
        error_message_len
    );
}

int ziggy_metal_apply_rope_at_offset_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *vector,
    size_t vector_offset_bytes,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
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
        const size_t element_count = (size_t)head_count * (size_t)head_dim;
        const size_t byte_count = element_count * sizeof(float);
        if (byte_count == 0 || vector_offset_bytes + byte_count > buffer.length) {
            ziggy_write_error(error_message, error_message_len, @"Metal rope exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
        const uint32_t vector_base = (uint32_t)(vector_offset_bytes / sizeof(float));
        return ziggy_run_compute(
            state,
            state.ropePipeline,
            head_count * pair_count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:buffer.buffer offset:0 atIndex:0];
                [encoder setBytes:&vector_base length:sizeof(vector_base) atIndex:1];
                [encoder setBytes:&head_count length:sizeof(head_count) atIndex:2];
                [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
                [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:4];
                [encoder setBytes:&position length:sizeof(position) atIndex:5];
                [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:6];
                [encoder setBytes:&rope_style length:sizeof(rope_style) atIndex:7];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_apply_rope_to_dst_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *src,
    ZiggyMetalBuffer *dst,
    size_t dst_offset_bytes,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || src == NULL || dst == NULL || head_count == 0 || head_dim == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal rope-to-dst request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *src_buffer = ziggy_const_buffer(src);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);
        const size_t element_count = (size_t)head_count * (size_t)head_dim;
        const size_t byte_count = element_count * sizeof(float);
        if (byte_count == 0 || src_buffer.length < byte_count || dst_offset_bytes + byte_count > dst_buffer.length) {
            ziggy_write_error(error_message, error_message_len, @"Metal rope-to-dst exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        const uint32_t dst_base = (uint32_t)(dst_offset_bytes / sizeof(float));
        return ziggy_run_compute(
            state,
            state.ropeToDstPipeline,
            element_count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:src_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:dst_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&dst_base length:sizeof(dst_base) atIndex:2];
                [encoder setBytes:&head_count length:sizeof(head_count) atIndex:3];
                [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
                [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:5];
                [encoder setBytes:&position length:sizeof(position) atIndex:6];
                [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:7];
                [encoder setBytes:&rope_style length:sizeof(rope_style) atIndex:8];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_store_kv_half(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *src,
    ZiggyMetalBuffer *dst,
    size_t dst_offset_elements,
    uint32_t count,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || src == NULL || dst == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal store_kv_half request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *src_buffer = ziggy_const_buffer(src);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);

        if (src_buffer.length < count * sizeof(float) ||
            dst_buffer.length < (dst_offset_elements + count) * 2) {
            ziggy_write_error(error_message, error_message_len, @"Metal store_kv_half exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        uint32_t offset = (uint32_t)dst_offset_elements;
        return ziggy_run_compute(
            state,
            state.storeKvHalfPipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:src_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:dst_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&offset length:sizeof(offset) atIndex:2];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
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
    uint32_t window_start,
    float scale,
    float softcap,
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
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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
        [encoder setBytes:&window_start length:sizeof(window_start) atIndex:11];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:12];
        [encoder setBytes:&softcap length:sizeof(softcap) atIndex:13];

        NSUInteger thread_width = state.attentionFusedPipeline.threadExecutionWidth;
        if (thread_width == 0) thread_width = 1;
        NSUInteger max_total_threads = state.attentionFusedPipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger work_items = position - window_start + 1;
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

int ziggy_metal_gelu_mul_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    uint32_t count,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || gate == NULL || up == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal gelu request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *gate_buffer = ziggy_buffer(gate);
        const ZiggyMetalBufferState *up_buffer = ziggy_const_buffer(up);
        return ziggy_run_compute(
            state,
            state.geluMulPipeline,
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
        state.pendingCommandBuffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (state.pendingCommandBuffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;
        
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
        return ziggy_commit_pending(state, NULL, error_message, error_message_len);
    }
}

int ziggy_metal_commit_sequence_timed(
    ZiggyMetalContext *ctx,
    ZiggyMetalCommitStats *out_stats,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal sequence commit request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_commit_pending(state, out_stats, error_message, error_message_len);
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

int ziggy_metal_add_bias_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *dst,
    const void *bias_weights,
    uint32_t count,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || dst == NULL || bias_weights == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal add bias request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);

        id<MTLBuffer> bias_buffer = [state.device newBufferWithBytes:bias_weights length:count * sizeof(float) options:MTLResourceStorageModeShared];
        if (bias_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal bias buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        ZiggyMetalBufferState *bias_buffer_state = [[ZiggyMetalBufferState alloc] init];
        bias_buffer_state.buffer = bias_buffer;
        bias_buffer_state.length = count * sizeof(float);

        return ziggy_run_compute(
            state,
            state.addInPlacePipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:dst_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:bias_buffer_state.buffer offset:0 atIndex:1];
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
    float weight_offset,
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
        const NSUInteger thread_count = ziggy_rowwise_thread_count(state.rmsNormPipeline);
        if (thread_count == 0 || thread_count > 256) {
            ziggy_write_error(error_message, error_message_len, @"Metal RMSNorm threadgroup sizing failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        if (input_buffer.length < ((size_t)count * sizeof(float)) ||
            weights_buffer.length < ((size_t)count * sizeof(float)) ||
            output_buffer.length < ((size_t)count * sizeof(float))) {
            ziggy_write_error(error_message, error_message_len, @"Metal RMSNorm exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
        return ziggy_run_single_threadgroup(
            state,
            state.rmsNormPipeline,
            thread_count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:weights_buffer.buffer offset:0 atIndex:1];
                [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
                [encoder setBytes:&eps length:sizeof(eps) atIndex:4];
                [encoder setBytes:&weight_offset length:sizeof(weight_offset) atIndex:5];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_rms_norm_per_head_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *input,
    const ZiggyMetalBuffer *weights,
    ZiggyMetalBuffer *output,
    uint32_t head_count,
    uint32_t head_dim,
    float eps,
    float weight_offset,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || input == NULL || weights == NULL || output == NULL || head_count == 0 || head_dim == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal RMSNorm Per Head request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        const ZiggyMetalBufferState *weights_buffer = ziggy_const_buffer(weights);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        
        const NSUInteger thread_count = ziggy_rowwise_thread_count(state.rmsNormPerHeadPipeline);
        if (thread_count == 0 || thread_count > 256) {
            ziggy_write_error(error_message, error_message_len, @"Metal RMSNorm Per Head threadgroup sizing failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        const size_t req_in_bytes = (size_t)head_count * head_dim * sizeof(float);
        const size_t req_w_bytes = (size_t)head_dim * sizeof(float);
        if (input_buffer.length < req_in_bytes || weights_buffer.length < req_w_bytes || output_buffer.length < req_in_bytes) {
            ziggy_write_error(error_message, error_message_len, @"Metal RMSNorm Per Head exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.rmsNormPerHeadPipeline];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:weights_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:3];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:5];
        [encoder setBytes:&weight_offset length:sizeof(weight_offset) atIndex:6];
        
        ziggy_dispatch_rowwise(encoder, state.rmsNormPerHeadPipeline, head_count);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal RMSNorm Per Head command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_linear_conv1d_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *qkv,
    ZiggyMetalBuffer *conv_state,
    const ZiggyMetalBuffer *conv_weights,
    ZiggyMetalBuffer *conv_out,
    uint32_t layer_index,
    uint32_t block_count,
    uint32_t kernel_dim,
    uint32_t qkv_dim,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || qkv == NULL || conv_state == NULL || conv_weights == NULL || conv_out == NULL || kernel_dim == 0 || qkv_dim == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal linear conv1d request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *qkv_buffer = ziggy_const_buffer(qkv);
        ZiggyMetalBufferState *conv_state_buffer = ziggy_buffer(conv_state);
        const ZiggyMetalBufferState *conv_weights_buffer = ziggy_const_buffer(conv_weights);
        ZiggyMetalBufferState *conv_out_buffer = ziggy_buffer(conv_out);

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.linearConv1dPipeline];
        [encoder setBuffer:qkv_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:conv_state_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:conv_weights_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:conv_out_buffer.buffer offset:0 atIndex:3];
        [encoder setBytes:&layer_index length:sizeof(layer_index) atIndex:4];
        [encoder setBytes:&block_count length:sizeof(block_count) atIndex:5];
        [encoder setBytes:&kernel_dim length:sizeof(kernel_dim) atIndex:6];
        [encoder setBytes:&qkv_dim length:sizeof(qkv_dim) atIndex:7];

        const NSUInteger thread_count = qkv_dim;
        const NSUInteger threads_per_group = MIN(thread_count, 256);
        const NSUInteger group_count = (thread_count + threads_per_group - 1) / threads_per_group;
        MTLSize grid_size = MTLSizeMake(group_count, 1, 1);
        MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal linear conv1d command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_linear_recurrent_norm_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *conv_out,
    ZiggyMetalBuffer *recurrent_state,
    const ZiggyMetalBuffer *z,
    const ZiggyMetalBuffer *a,
    const ZiggyMetalBuffer *b,
    const ZiggyMetalBuffer *dt_bias,
    const ZiggyMetalBuffer *A_log,
    const ZiggyMetalBuffer *norm_weights,
    ZiggyMetalBuffer *out,
    uint32_t layer_index,
    uint32_t num_key_heads,
    uint32_t num_value_heads,
    uint32_t key_head_dim,
    uint32_t value_head_dim,
    uint32_t qkv_dim,
    float rms_norm_eps,
    float scale,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || conv_out == NULL || recurrent_state == NULL || z == NULL || a == NULL || b == NULL || dt_bias == NULL || A_log == NULL || norm_weights == NULL || out == NULL || num_value_heads == 0 || value_head_dim == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal linear recurrent norm request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *conv_out_buffer = ziggy_const_buffer(conv_out);
        ZiggyMetalBufferState *recurrent_state_buffer = ziggy_buffer(recurrent_state);
        const ZiggyMetalBufferState *z_buffer = ziggy_const_buffer(z);
        const ZiggyMetalBufferState *a_buffer = ziggy_const_buffer(a);
        const ZiggyMetalBufferState *b_buffer = ziggy_const_buffer(b);
        const ZiggyMetalBufferState *dt_bias_buffer = ziggy_const_buffer(dt_bias);
        const ZiggyMetalBufferState *A_log_buffer = ziggy_const_buffer(A_log);
        const ZiggyMetalBufferState *norm_weights_buffer = ziggy_const_buffer(norm_weights);
        ZiggyMetalBufferState *out_buffer = ziggy_buffer(out);

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.linearRecurrentNormPipeline];
        [encoder setBuffer:conv_out_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:recurrent_state_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:z_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:a_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:b_buffer.buffer offset:0 atIndex:4];
        [encoder setBuffer:dt_bias_buffer.buffer offset:0 atIndex:5];
        [encoder setBuffer:A_log_buffer.buffer offset:0 atIndex:6];
        [encoder setBuffer:norm_weights_buffer.buffer offset:0 atIndex:7];
        [encoder setBuffer:out_buffer.buffer offset:0 atIndex:8];
        [encoder setBytes:&layer_index length:sizeof(layer_index) atIndex:9];
        [encoder setBytes:&num_key_heads length:sizeof(num_key_heads) atIndex:10];
        [encoder setBytes:&num_value_heads length:sizeof(num_value_heads) atIndex:11];
        [encoder setBytes:&key_head_dim length:sizeof(key_head_dim) atIndex:12];
        [encoder setBytes:&value_head_dim length:sizeof(value_head_dim) atIndex:13];
        [encoder setBytes:&qkv_dim length:sizeof(qkv_dim) atIndex:14];
        [encoder setBytes:&rms_norm_eps length:sizeof(rms_norm_eps) atIndex:15];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:16];

        const NSUInteger thread_count = num_value_heads * value_head_dim;
        const NSUInteger threads_per_group = MIN(value_head_dim, 256);
        const NSUInteger group_count = num_value_heads;
        MTLSize grid_size = MTLSizeMake(group_count, 1, 1);
        MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal linear recurrent norm command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_argmax_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output_token,
    uint32_t count,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || input == NULL || output_token == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal argmax request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output_token);
        const NSUInteger thread_width = state.argmaxPipeline.threadExecutionWidth;
        const NSUInteger max_total_threads = state.argmaxPipeline.maxTotalThreadsPerThreadgroup;
        uint32_t thread_count = 256;
        if (thread_width > 0 && thread_count < thread_width) thread_count = (uint32_t)thread_width;
        if (max_total_threads > 0 && thread_count > max_total_threads) thread_count = (uint32_t)max_total_threads;
        if (thread_count == 0 || thread_count > 256) {
            ziggy_write_error(error_message, error_message_len, @"Metal argmax threadgroup sizing failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        if (input_buffer.length < ((size_t)count * sizeof(float)) || output_buffer.length < sizeof(uint32_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal argmax exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
        return ziggy_run_single_threadgroup(
            state,
            state.argmaxPipeline,
            thread_count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:output_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
                [encoder setBytes:&thread_count length:sizeof(thread_count) atIndex:3];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_topk_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output_entries,
    uint32_t count,
    uint32_t top_k,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || input == NULL || output_entries == NULL || count == 0 || top_k == 0 || top_k > 64) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal top-k request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *entry_buffer = ziggy_buffer(output_entries);
        const NSUInteger thread_width = state.topKPipeline.threadExecutionWidth;
        const NSUInteger max_total_threads = state.topKPipeline.maxTotalThreadsPerThreadgroup;
        uint32_t thread_count = 32;
        if (thread_width > 0 && thread_count < thread_width) thread_count = (uint32_t)thread_width;
        if (max_total_threads > 0 && thread_count > max_total_threads) thread_count = (uint32_t)max_total_threads;
        if (thread_count == 0 || thread_count > 32) {
            ziggy_write_error(error_message, error_message_len, @"Metal top-k threadgroup sizing failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        if (input_buffer.length < ((size_t)count * sizeof(float)) ||
            entry_buffer.length < ((size_t)top_k * (sizeof(uint32_t) + sizeof(float)))) {
            ziggy_write_error(error_message, error_message_len, @"Metal top-k exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        return ziggy_run_single_threadgroup(
            state,
            state.topKPipeline,
            thread_count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:entry_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
                [encoder setBytes:&top_k length:sizeof(top_k) atIndex:4];
                [encoder setBytes:&thread_count length:sizeof(thread_count) atIndex:5];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_normalize_topk_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *entries,
    uint32_t top_k,
    bool apply_softmax,
    bool normalize_weights,
    float scale,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || entries == NULL || top_k == 0 || top_k > 64) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal normalize top-k request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *entry_buffer = ziggy_buffer(entries);
        if (entry_buffer.length < ((size_t)top_k * (sizeof(uint32_t) + sizeof(float)))) {
            ziggy_write_error(error_message, error_message_len, @"Metal normalize top-k exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
        return ziggy_run_single_threadgroup(
            state,
            state.normalizeTopKPipeline,
            1,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:entry_buffer.buffer offset:0 atIndex:0];
                [encoder setBytes:&top_k length:sizeof(top_k) atIndex:1];
                [encoder setBytes:&apply_softmax length:sizeof(apply_softmax) atIndex:2];
                [encoder setBytes:&normalize_weights length:sizeof(normalize_weights) atIndex:3];
                [encoder setBytes:&scale length:sizeof(scale) atIndex:4];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_sample_topk_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output_token,
    uint32_t count,
    uint32_t top_k,
    float temperature,
    float random_uniform,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || input == NULL || output_token == NULL || count == 0 || top_k == 0 || top_k > 64 || !(temperature > 0.0f)) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal sample top-k request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *token_buffer = ziggy_buffer(output_token);
        const NSUInteger thread_width = state.sampleTopKPipeline.threadExecutionWidth;
        const NSUInteger max_total_threads = state.sampleTopKPipeline.maxTotalThreadsPerThreadgroup;
        uint32_t thread_count = 32;
        if (thread_width > 0 && thread_count < thread_width) thread_count = (uint32_t)thread_width;
        if (max_total_threads > 0 && thread_count > max_total_threads) thread_count = (uint32_t)max_total_threads;
        if (thread_count == 0 || thread_count > 32) {
            ziggy_write_error(error_message, error_message_len, @"Metal sample top-k threadgroup sizing failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        if (input_buffer.length < ((size_t)count * sizeof(float)) || token_buffer.length < sizeof(uint32_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal sample top-k exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        return ziggy_run_single_threadgroup(
            state,
            state.sampleTopKPipeline,
            thread_count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:token_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
                [encoder setBytes:&top_k length:sizeof(top_k) atIndex:4];
                [encoder setBytes:&thread_count length:sizeof(thread_count) atIndex:5];
                [encoder setBytes:&temperature length:sizeof(temperature) atIndex:6];
                [encoder setBytes:&random_uniform length:sizeof(random_uniform) atIndex:7];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_weighted_sum_topk_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *dst,
    const ZiggyMetalBuffer *src,
    const ZiggyMetalBuffer *entries,
    uint32_t count,
    uint32_t slot_idx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || dst == NULL || src == NULL || entries == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal weighted sum top-k request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);
        const ZiggyMetalBufferState *src_buffer = ziggy_const_buffer(src);
        const ZiggyMetalBufferState *entry_buffer = ziggy_const_buffer(entries);
        if (dst_buffer.length < ((size_t)count * sizeof(float)) ||
            src_buffer.length < ((size_t)count * sizeof(float)) ||
            entry_buffer.length < ((size_t)(slot_idx + 1) * (sizeof(uint32_t) + sizeof(float)))) {
            ziggy_write_error(error_message, error_message_len, @"Metal weighted sum top-k exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
        return ziggy_run_compute(
            state,
            state.weightedSumTopKPipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:dst_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:src_buffer.buffer offset:0 atIndex:1];
                [encoder setBuffer:entry_buffer.buffer offset:0 atIndex:2];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
                [encoder setBytes:&slot_idx length:sizeof(slot_idx) atIndex:4];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_sigmoid_scale_add_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *dst,
    const ZiggyMetalBuffer *src,
    const ZiggyMetalBuffer *scalar,
    uint32_t count,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || dst == NULL || src == NULL || scalar == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal sigmoid scale add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);
        const ZiggyMetalBufferState *src_buffer = ziggy_const_buffer(src);
        const ZiggyMetalBufferState *scalar_buffer = ziggy_const_buffer(scalar);
        if (dst_buffer.length < ((size_t)count * sizeof(float)) ||
            src_buffer.length < ((size_t)count * sizeof(float)) ||
            scalar_buffer.length < sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal sigmoid scale add exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
        return ziggy_run_compute(
            state,
            state.sigmoidScaleAddPipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:dst_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:src_buffer.buffer offset:0 atIndex:1];
                [encoder setBuffer:scalar_buffer.buffer offset:0 atIndex:2];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_batch_argmax_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output_tokens,
    uint32_t vocab_size,
    uint32_t batch_count,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || input == NULL || output_tokens == NULL || vocab_size == 0 || batch_count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal batch argmax request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output_tokens);
        const NSUInteger thread_width = state.batchArgmaxPipeline.threadExecutionWidth;
        const NSUInteger max_total_threads = state.batchArgmaxPipeline.maxTotalThreadsPerThreadgroup;
        uint32_t thread_count = 256;
        if (thread_width > 0 && thread_count < thread_width) thread_count = (uint32_t)thread_width;
        if (max_total_threads > 0 && thread_count > max_total_threads) thread_count = (uint32_t)max_total_threads;
        if (thread_count == 0 || thread_count > 256) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch argmax threadgroup sizing failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        if (input_buffer.length < ((size_t)batch_count * vocab_size * sizeof(float)) ||
            output_buffer.length < ((size_t)batch_count * sizeof(uint32_t))) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch argmax exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.batchArgmaxPipeline];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:1];
        [encoder setBytes:&vocab_size length:sizeof(vocab_size) atIndex:2];
        [encoder setBytes:&batch_count length:sizeof(batch_count) atIndex:3];
        [encoder setBytes:&thread_count length:sizeof(thread_count) atIndex:4];

        MTLSize grid_size = MTLSizeMake(batch_count, 1, 1);
        MTLSize group_size = MTLSizeMake(thread_count, 1, 1);
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal batch argmax command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

static id<MTLComputePipelineState> ziggy_select_batch_q4k_add_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.batchMatvecQ4KAdd2048Pipeline;
    if (cols == 5632) return state.batchMatvecQ4KAdd5632Pipeline;
    return state.batchMatvecQ4KAddPipeline;
}

int ziggy_metal_batch_matvec_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t batch_idx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal batch matvec add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        if (output_buffer.length < ((size_t)batch_idx + 1) * rows * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch matvec add exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.batchMatvecAddPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:5];
        ziggy_dispatch_rowwise(encoder, state.batchMatvecAddPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal batch matvec add command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_batch_matvec_q4k_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t batch_idx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal batch q4k add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        if (output_buffer.length < ((size_t)batch_idx + 1) * rows * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch q4k add exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputePipelineState> pipeline = ziggy_select_batch_q4k_add_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:5];
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal batch q4k add command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_batch_silu_mul_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    uint32_t count,
    uint32_t batch_idx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || gate == NULL || up == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal batch silu request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *gate_buffer = ziggy_buffer(gate);
        const ZiggyMetalBufferState *up_buffer = ziggy_const_buffer(up);
        if (gate_buffer.length < ((size_t)batch_idx + 1) * count * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch silu exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        return ziggy_run_compute(
            state,
            state.batchSiluMulPipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:gate_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:up_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
                [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:3];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_batch_add_in_place_f32(
    ZiggyMetalContext *ctx,
    ZiggyMetalBuffer *dst,
    const ZiggyMetalBuffer *src,
    uint32_t count,
    uint32_t batch_idx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || dst == NULL || src == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal batch add request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);
        const ZiggyMetalBufferState *src_buffer = ziggy_const_buffer(src);
        if (dst_buffer.length < ((size_t)batch_idx + 1) * count * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch add exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        return ziggy_run_compute(
            state,
            state.batchAddInPlacePipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:dst_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:src_buffer.buffer offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
                [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:3];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_batch_rms_norm_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *input,
    const ZiggyMetalBuffer *weights,
    ZiggyMetalBuffer *output,
    uint32_t count,
    float eps,
    float weight_offset,
    uint32_t batch_idx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || input == NULL || weights == NULL || output == NULL || count == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal batch RMSNorm request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        const ZiggyMetalBufferState *weights_buffer = ziggy_const_buffer(weights);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        const NSUInteger thread_count = ziggy_rowwise_thread_count(state.batchRmsNormPipeline);
        if (thread_count == 0 || thread_count > 256) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch RMSNorm threadgroup sizing failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        const size_t required_len = ((size_t)batch_idx + 1) * count * sizeof(float);
        if (input_buffer.length < required_len ||
            weights_buffer.length < ((size_t)count * sizeof(float)) ||
            output_buffer.length < required_len) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch RMSNorm exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        return ziggy_run_single_threadgroup(
            state,
            state.batchRmsNormPipeline,
            thread_count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:weights_buffer.buffer offset:0 atIndex:1];
                [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
                [encoder setBytes:&eps length:sizeof(eps) atIndex:4];
                [encoder setBytes:&weight_offset length:sizeof(weight_offset) atIndex:5];
                [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:6];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_batch_matvec_q4k_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t batch_idx,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal batch q4k matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        if (output_buffer.length < ((size_t)batch_idx + 1) * rows * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal batch q4k matvec exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        [encoder setComputePipelineState:state.batchMatvecQ4KPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:5];
        ziggy_dispatch_q4k_rows(encoder, state.batchMatvecQ4KPipeline, rows);
        [encoder endEncoding];

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal batch q4k matvec command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

static int ziggy_run_indexed_rowwise_matvec(
    ZiggyMetalState *state,
    id<MTLComputePipelineState> pipeline,
    const ZiggyMetalBufferState *matrix_buffer,
    const ZiggyMetalBufferState *input_buffer,
    ZiggyMetalBufferState *output_buffer,
    const ZiggyMetalBufferState *entry_buffer,
    uint32_t rows,
    uint32_t cols,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    NSString *invalid_message,
    NSString *command_error_message,
    char *error_message,
    size_t error_message_len
) {
    if (matrix_buffer == NULL || input_buffer == NULL || output_buffer == NULL || entry_buffer == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, invalid_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    if (output_buffer.length < ((size_t)rows * sizeof(float))) {
        ziggy_write_error(error_message, error_message_len, @"Metal indexed matvec output exceeded allocation");
        return ZIGGY_METAL_BUFFER_FAILED;
    }

    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    const bool has_pending = command_buffer != nil;
    if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
    if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
    [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [encoder setBuffer:entry_buffer.buffer offset:0 atIndex:5];
    [encoder setBytes:&slot_idx length:sizeof(slot_idx) atIndex:6];
    [encoder setBytes:&rows_per_expert length:sizeof(rows_per_expert) atIndex:7];
    ziggy_dispatch_indexed_rowwise(encoder, pipeline, rows, cols);
    [encoder endEncoding];

    if (has_pending) return ZIGGY_METAL_OK;

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: command_error_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    return ZIGGY_METAL_OK;
}

static int ziggy_run_dual_indexed_rowwise_matvec(
    ZiggyMetalState *state,
    id<MTLComputePipelineState> pipeline,
    const ZiggyMetalBufferState *matrix_a_buffer,
    const ZiggyMetalBufferState *matrix_b_buffer,
    const ZiggyMetalBufferState *input_buffer,
    ZiggyMetalBufferState *output_a_buffer,
    ZiggyMetalBufferState *output_b_buffer,
    const ZiggyMetalBufferState *entry_buffer,
    uint32_t rows,
    uint32_t cols,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    NSString *validation_error_message,
    NSString *command_error_message,
    char *error_message,
    size_t error_message_len
) {
    if (rows == 0 || cols == 0 || rows_per_expert == 0) {
        ziggy_write_error(error_message, error_message_len, validation_error_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    if (input_buffer.length < ((size_t)cols * sizeof(float)) ||
        output_a_buffer.length < ((size_t)rows * sizeof(float)) ||
        output_b_buffer.length < ((size_t)rows * sizeof(float)))
    {
        ziggy_write_error(error_message, error_message_len, @"Metal dual indexed matvec buffers exceeded allocation");
        return ZIGGY_METAL_BUFFER_FAILED;
    }

    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    const bool has_pending = command_buffer != nil;
    if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
    if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matrix_a_buffer.buffer offset:0 atIndex:0];
    [encoder setBuffer:matrix_b_buffer.buffer offset:0 atIndex:1];
    [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
    [encoder setBuffer:output_a_buffer.buffer offset:0 atIndex:3];
    [encoder setBuffer:output_b_buffer.buffer offset:0 atIndex:4];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:6];
    [encoder setBuffer:entry_buffer.buffer offset:0 atIndex:7];
    [encoder setBytes:&slot_idx length:sizeof(slot_idx) atIndex:8];
    [encoder setBytes:&rows_per_expert length:sizeof(rows_per_expert) atIndex:9];
    ziggy_dispatch_indexed_rowwise(encoder, pipeline, rows, cols);
    [encoder endEncoding];

    if (has_pending) return ZIGGY_METAL_OK;

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: command_error_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    return ZIGGY_METAL_OK;
}

int ziggy_metal_indexed_matvec_iq3_xxs_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    const ZiggyMetalBuffer *entries,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || entries == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal indexed iq3_xxs request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_indexed_rowwise_matvec(
            state,
            state.indexedMatvecIQ3XXSPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            ziggy_const_buffer(entries),
            rows,
            cols,
            slot_idx,
            rows_per_expert,
            @"invalid Metal indexed iq3_xxs request",
            @"Metal indexed iq3_xxs command failed",
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_dual_indexed_matvec_iq3_xxs_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix_a,
    const ZiggyMetalBuffer *matrix_b,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output_a,
    ZiggyMetalBuffer *output_b,
    uint32_t rows,
    uint32_t cols,
    const ZiggyMetalBuffer *entries,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix_a == NULL || matrix_b == NULL || input == NULL || output_a == NULL || output_b == NULL || entries == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal dual indexed iq3_xxs request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_dual_indexed_rowwise_matvec(
            state,
            state.dualIndexedMatvecIQ3XXSPipeline,
            ziggy_const_buffer(matrix_a),
            ziggy_const_buffer(matrix_b),
            ziggy_const_buffer(input),
            ziggy_buffer(output_a),
            ziggy_buffer(output_b),
            ziggy_const_buffer(entries),
            rows,
            cols,
            slot_idx,
            rows_per_expert,
            @"invalid Metal dual indexed iq3_xxs request",
            @"Metal dual indexed iq3_xxs command failed",
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_indexed_matvec_iq4_xs_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    const ZiggyMetalBuffer *entries,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || entries == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal indexed iq4_xs request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_indexed_rowwise_matvec(
            state,
            state.indexedMatvecIQ4XSPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            ziggy_const_buffer(entries),
            rows,
            cols,
            slot_idx,
            rows_per_expert,
            @"invalid Metal indexed iq4_xs request",
            @"Metal indexed iq4_xs command failed",
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_indexed_matvec_iq4_xs_add_weighted_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    const ZiggyMetalBuffer *entries,
    uint32_t slot_idx,
    uint32_t rows_per_expert,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || entries == NULL) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal indexed iq4_xs add-weighted request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_indexed_rowwise_matvec(
            state,
            state.indexedMatvecIQ4XSAddWeightedPipeline,
            ziggy_const_buffer(matrix),
            ziggy_const_buffer(input),
            ziggy_buffer(output),
            ziggy_const_buffer(entries),
            rows,
            cols,
            slot_idx,
            rows_per_expert,
            @"invalid Metal indexed iq4_xs add-weighted request",
            @"Metal indexed iq4_xs add-weighted command failed",
            error_message,
            error_message_len
        );
    }
}
