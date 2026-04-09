
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
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KDualPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KDual2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KDual5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KKHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KKHalf2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KKHalf5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KQRopePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KQRope2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KQRope5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KDualKvHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KDualKvHalf2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KDualKvHalf5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KQ6KDualKvHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KQ6KDualKvHalf2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KQ6KDualKvHalf5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KArgmaxPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ6KArgmaxPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ80Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ80AddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4K2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4K5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KKHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KKHalf2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KKHalf5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KQRopePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KQRope2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KQRope5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KDualPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KDual2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KDual5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KDualKvHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KDualKvHalf2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KDualKvHalf5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KQ6KDualKvHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KQ6KDualKvHalf2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KQ6KDualKvHalf5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ropePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ropeToDstPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ropeToHalfDstPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> storeKvHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> packKvHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFusedPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> siluMulPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> addInPlacePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsNormPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsNormPerHeadPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> argmaxPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> topKPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sampleTopKPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchArgmaxPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecQ4KAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecQ4KAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecQ4KAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchMatvecQ4KPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KSiluDownAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KSiluDownAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KSiluDownAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> buildFfnGateBlockMaskPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KGatedSiluDownAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KGatedSiluDownAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecQ4KGatedSiluDownAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KSiluDownAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KSiluDownAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KSiluDownAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KGatedSiluDownAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KGatedSiluDownAdd2048Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matvecMoonQ4KGatedSiluDownAdd5632Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchSiluMulPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchAddInPlacePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> batchRmsNormPipeline;
@property(nonatomic, strong) id<MTLCommandBuffer> pendingCommandBuffer;
@property(nonatomic, strong) id<MTLComputeCommandEncoder> pendingComputeEncoder;
@property(nonatomic) uint32_t pendingEncoderCount;
@property(nonatomic) uint32_t pendingDispatchCount;
@end

@implementation ZiggyMetalState
@end

@interface ZiggyMetalBufferState : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) size_t length;
@property(nonatomic) bool hostVisible;
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
    uint32_t encoder_count,
    uint32_t dispatch_count,
    ZiggyMetalCommitStats *out_stats
) {
    if (out_stats == NULL) return;
    out_stats->cpu_wait_ns = cpu_wait_ns;
    out_stats->gpu_elapsed_ns = 0;
    out_stats->non_gpu_wait_ns = 0;
    out_stats->gpu_timestamps_valid = false;
    out_stats->command_buffer_count = command_buffer != nil ? 1 : 0;
    out_stats->encoder_count = encoder_count;
    out_stats->dispatch_count = dispatch_count;

    if (![command_buffer respondsToSelector:@selector(GPUStartTime)] ||
        ![command_buffer respondsToSelector:@selector(GPUEndTime)]) {
        return;
    }

    const CFTimeInterval gpu_start = command_buffer.GPUStartTime;
    const CFTimeInterval gpu_end = command_buffer.GPUEndTime;
    if (!(gpu_end > gpu_start) || gpu_start <= 0.0) return;

    out_stats->gpu_elapsed_ns = (uint64_t)((gpu_end - gpu_start) * 1000000000.0);
    out_stats->gpu_timestamps_valid = out_stats->gpu_elapsed_ns > 0;
    if (out_stats->gpu_timestamps_valid && cpu_wait_ns > out_stats->gpu_elapsed_ns) {
        out_stats->non_gpu_wait_ns = cpu_wait_ns - out_stats->gpu_elapsed_ns;
    }
}

static void ziggy_record_encoder(ZiggyMetalState *state) {
    state.pendingEncoderCount += 1;
}

static void ziggy_record_dispatch(ZiggyMetalState *state) {
    state.pendingDispatchCount += 1;
}

static id<MTLComputeCommandEncoder> ziggy_new_compute_encoder(
    ZiggyMetalState *state,
    id<MTLCommandBuffer> command_buffer,
    char *error_message,
    size_t error_message_len
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
        return nil;
    }
    ziggy_record_encoder(state);
    return encoder;
}

static id<MTLBlitCommandEncoder> ziggy_new_blit_encoder(
    ZiggyMetalState *state,
    id<MTLCommandBuffer> command_buffer,
    char *error_message,
    size_t error_message_len
) {
    id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
    if (encoder == nil) {
        ziggy_write_error(error_message, error_message_len, @"failed to create Metal blit encoder");
        return nil;
    }
    ziggy_record_encoder(state);
    return encoder;
}

static void ziggy_end_pending_compute_encoder(ZiggyMetalState *state) {
    if (state.pendingComputeEncoder != nil) {
        [state.pendingComputeEncoder endEncoding];
        state.pendingComputeEncoder = nil;
    }
}

static id<MTLComputeCommandEncoder> ziggy_acquire_compute_encoder(
    ZiggyMetalState *state,
    id<MTLCommandBuffer> command_buffer,
    bool has_pending,
    char *error_message,
    size_t error_message_len
) {
    if (has_pending) {
        if (state.pendingComputeEncoder == nil) {
            state.pendingComputeEncoder = ziggy_new_compute_encoder(state, command_buffer, error_message, error_message_len);
        }
        return state.pendingComputeEncoder;
    }
    return ziggy_new_compute_encoder(state, command_buffer, error_message, error_message_len);
}

static void ziggy_finish_compute_encoder(
    ZiggyMetalState *state,
    id<MTLComputeCommandEncoder> encoder,
    bool has_pending
) {
    if (has_pending) return;
    [encoder endEncoding];
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

static id<MTLComputePipelineState> ziggy_select_moonq_q4k_dual_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecMoonQ4KDual2048Pipeline;
    if (cols == 5632) return state.matvecMoonQ4KDual5632Pipeline;
    return state.matvecMoonQ4KDualPipeline;
}

static id<MTLComputePipelineState> ziggy_select_moonq_q4k_k_half_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecMoonQ4KKHalf2048Pipeline;
    if (cols == 5632) return state.matvecMoonQ4KKHalf5632Pipeline;
    return state.matvecMoonQ4KKHalfPipeline;
}

static id<MTLComputePipelineState> ziggy_select_moonq_q4k_q_rope_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecMoonQ4KQRope2048Pipeline;
    if (cols == 5632) return state.matvecMoonQ4KQRope5632Pipeline;
    return state.matvecMoonQ4KQRopePipeline;
}

static id<MTLComputePipelineState> ziggy_select_moonq_q4k_dual_kv_half_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecMoonQ4KDualKvHalf2048Pipeline;
    if (cols == 5632) return state.matvecMoonQ4KDualKvHalf5632Pipeline;
    return state.matvecMoonQ4KDualKvHalfPipeline;
}

static id<MTLComputePipelineState> ziggy_select_moonq_q4k_q6k_dual_kv_half_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecMoonQ4KQ6KDualKvHalf2048Pipeline;
    if (cols == 5632) return state.matvecMoonQ4KQ6KDualKvHalf5632Pipeline;
    return state.matvecMoonQ4KQ6KDualKvHalfPipeline;
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

static id<MTLComputePipelineState> ziggy_select_q4k_dual_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecQ4KDual2048Pipeline;
    if (cols == 5632) return state.matvecQ4KDual5632Pipeline;
    return state.matvecQ4KDualPipeline;
}

static id<MTLComputePipelineState> ziggy_select_q4k_k_half_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecQ4KKHalf2048Pipeline;
    if (cols == 5632) return state.matvecQ4KKHalf5632Pipeline;
    return state.matvecQ4KKHalfPipeline;
}

static id<MTLComputePipelineState> ziggy_select_q4k_q_rope_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecQ4KQRope2048Pipeline;
    if (cols == 5632) return state.matvecQ4KQRope5632Pipeline;
    return state.matvecQ4KQRopePipeline;
}

static id<MTLComputePipelineState> ziggy_select_q4k_dual_kv_half_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecQ4KDualKvHalf2048Pipeline;
    if (cols == 5632) return state.matvecQ4KDualKvHalf5632Pipeline;
    return state.matvecQ4KDualKvHalfPipeline;
}

static id<MTLComputePipelineState> ziggy_select_q4k_q6k_dual_kv_half_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecQ4KQ6KDualKvHalf2048Pipeline;
    if (cols == 5632) return state.matvecQ4KQ6KDualKvHalf5632Pipeline;
    return state.matvecQ4KQ6KDualKvHalfPipeline;
}

static id<MTLComputePipelineState> ziggy_select_q6k_add_pipeline(
    ZiggyMetalState *state,
    uint32_t cols
) {
    if (cols == 2048) return state.matvecQ6KAdd2048Pipeline;
    if (cols == 5632) return state.matvecQ6KAdd5632Pipeline;
    return state.matvecQ6KAddPipeline;
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

    const uint32_t encoder_count = state.pendingEncoderCount;
    const uint32_t dispatch_count = state.pendingDispatchCount;
    ziggy_end_pending_compute_encoder(state);
    const uint64_t wait_start_ns = ziggy_now_ns();
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    const uint64_t cpu_wait_ns = ziggy_now_ns() - wait_start_ns;
    state.pendingCommandBuffer = nil;
    state.pendingEncoderCount = 0;
    state.pendingDispatchCount = 0;
    ziggy_fill_commit_stats(command_buffer, cpu_wait_ns, encoder_count, dispatch_count, out_stats);

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

    id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
    if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    [encoder setComputePipelineState:pipeline];
    encode(encoder);
    ziggy_dispatch_standard(encoder, pipeline, grid_width);
    ziggy_record_dispatch(state);
    ziggy_finish_compute_encoder(state, encoder, has_pending);

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

    id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
    if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    [encoder setComputePipelineState:pipeline];
    encode(encoder);
    MTLSize grid_size = MTLSizeMake(1, 1, 1);
    MTLSize group_size = MTLSizeMake(thread_count, 1, 1);
    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
    ziggy_record_dispatch(state);
    ziggy_finish_compute_encoder(state, encoder, has_pending);

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

    id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
    if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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
    ziggy_record_dispatch(state);
    ziggy_finish_compute_encoder(state, encoder, has_pending);

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
        id<MTLComputePipelineState> matvec_q4k_dual_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_dual_f32", &pipeline_error);
        if (matvec_q4k_dual_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k dual pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_dual_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_dual_2048_f32", &pipeline_error);
        if (matvec_q4k_dual_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k dual 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_dual_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_dual_5632_f32", &pipeline_error);
        if (matvec_q4k_dual_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k dual 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_k_half_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_k_half_f32", &pipeline_error);
        if (matvec_q4k_k_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k k-half pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_k_half_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_k_half_2048_f32", &pipeline_error);
        if (matvec_q4k_k_half_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k k-half 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_k_half_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_k_half_5632_f32", &pipeline_error);
        if (matvec_q4k_k_half_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k k-half 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_q_rope_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_q_rope_f32", &pipeline_error);
        if (matvec_q4k_q_rope_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k q-rope pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_q_rope_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_q_rope_2048_f32", &pipeline_error);
        if (matvec_q4k_q_rope_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k q-rope 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_q_rope_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_q_rope_5632_f32", &pipeline_error);
        if (matvec_q4k_q_rope_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k q-rope 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_dual_kv_half_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_dual_kv_half_f32", &pipeline_error);
        if (matvec_q4k_dual_kv_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k dual kv-half pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_dual_kv_half_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_dual_kv_half_2048_f32", &pipeline_error);
        if (matvec_q4k_dual_kv_half_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k dual kv-half 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_dual_kv_half_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_dual_kv_half_5632_f32", &pipeline_error);
        if (matvec_q4k_dual_kv_half_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k dual kv-half 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_q6k_dual_kv_half_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_q6k_dual_kv_half_f32", &pipeline_error);
        if (matvec_q4k_q6k_dual_kv_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k/q6k dual kv-half pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_q6k_dual_kv_half_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_q6k_dual_kv_half_2048_f32", &pipeline_error);
        if (matvec_q4k_q6k_dual_kv_half_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k/q6k dual kv-half 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_q6k_dual_kv_half_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_q6k_dual_kv_half_5632_f32", &pipeline_error);
        if (matvec_q4k_q6k_dual_kv_half_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k/q6k dual kv-half 5632 pipeline");
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
        id<MTLComputePipelineState> matvec_q4k_argmax_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_argmax_f32", &pipeline_error);
        if (matvec_q4k_argmax_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k argmax pipeline");
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

        id<MTLComputePipelineState> matvec_moon_q4k_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_f32", &pipeline_error);
        if (matvec_moon_q4k_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k matvec pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_dual_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_dual_f32", &pipeline_error);
        if (matvec_moon_q4k_dual_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k dual pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_2048_f32", &pipeline_error);
        if (matvec_moon_q4k_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_dual_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_dual_2048_f32", &pipeline_error);
        if (matvec_moon_q4k_dual_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k dual 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_5632_f32", &pipeline_error);
        if (matvec_moon_q4k_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_dual_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_dual_5632_f32", &pipeline_error);
        if (matvec_moon_q4k_dual_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k dual 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_k_half_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_k_half_f32", &pipeline_error);
        if (matvec_moon_q4k_k_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k k-half pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_k_half_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_k_half_2048_f32", &pipeline_error);
        if (matvec_moon_q4k_k_half_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k k-half 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_k_half_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_k_half_5632_f32", &pipeline_error);
        if (matvec_moon_q4k_k_half_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k k-half 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_q_rope_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_q_rope_f32", &pipeline_error);
        if (matvec_moon_q4k_q_rope_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k q-rope pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_q_rope_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_q_rope_2048_f32", &pipeline_error);
        if (matvec_moon_q4k_q_rope_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k q-rope 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_q_rope_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_q_rope_5632_f32", &pipeline_error);
        if (matvec_moon_q4k_q_rope_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k q-rope 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_dual_kv_half_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_dual_kv_half_f32", &pipeline_error);
        if (matvec_moon_q4k_dual_kv_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k dual kv-half pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_dual_kv_half_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_dual_kv_half_2048_f32", &pipeline_error);
        if (matvec_moon_q4k_dual_kv_half_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k dual kv-half 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_dual_kv_half_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_dual_kv_half_5632_f32", &pipeline_error);
        if (matvec_moon_q4k_dual_kv_half_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k dual kv-half 5632 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_q6k_dual_kv_half_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_q6k_dual_kv_half_f32", &pipeline_error);
        if (matvec_moon_q4k_q6k_dual_kv_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k/q6k dual kv-half pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_q6k_dual_kv_half_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_q6k_dual_kv_half_2048_f32", &pipeline_error);
        if (matvec_moon_q4k_q6k_dual_kv_half_2048_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k/q6k dual kv-half 2048 pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_q6k_dual_kv_half_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_q6k_dual_kv_half_5632_f32", &pipeline_error);
        if (matvec_moon_q4k_q6k_dual_kv_half_5632_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal MoonQuant q4k/q6k dual kv-half 5632 pipeline");
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
        id<MTLComputePipelineState> build_ffn_gate_block_mask_pipeline = ziggy_pipeline(device, library, @"build_ffn_gate_block_mask_f32", &pipeline_error);
        if (build_ffn_gate_block_mask_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal FFN gate mask pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_gated_silu_down_add_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_gated_silu_down_add_f32", &pipeline_error);
        if (matvec_q4k_gated_silu_down_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal q4k gated silu down add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_q4k_gated_silu_down_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_gated_silu_down_add_2048_f32", &pipeline_error);
        id<MTLComputePipelineState> matvec_q4k_gated_silu_down_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_q4k_gated_silu_down_add_5632_f32", &pipeline_error);

        id<MTLComputePipelineState> matvec_moon_q4k_silu_down_add_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_silu_down_add_f32", &pipeline_error);
        if (matvec_moon_q4k_silu_down_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal moonq q4k silu down add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_silu_down_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_silu_down_add_2048_f32", &pipeline_error);
        id<MTLComputePipelineState> matvec_moon_q4k_silu_down_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_silu_down_add_5632_f32", &pipeline_error);
        id<MTLComputePipelineState> matvec_moon_q4k_gated_silu_down_add_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_gated_silu_down_add_f32", &pipeline_error);
        if (matvec_moon_q4k_gated_silu_down_add_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal moonq q4k gated silu down add pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> matvec_moon_q4k_gated_silu_down_add_2048_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_gated_silu_down_add_2048_f32", &pipeline_error);
        id<MTLComputePipelineState> matvec_moon_q4k_gated_silu_down_add_5632_pipeline = ziggy_pipeline(device, library, @"matvec_moonq_q4k_gated_silu_down_add_5632_f32", &pipeline_error);

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
        id<MTLComputePipelineState> rope_to_half_dst_pipeline = ziggy_pipeline(device, library, @"apply_rope_to_half_dst_f32", &pipeline_error);
        if (rope_to_half_dst_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal rope-to-half-dst pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> store_kv_half_pipeline = ziggy_pipeline(device, library, @"store_kv_half", &pipeline_error);
        if (store_kv_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal store_kv_half pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }
        id<MTLComputePipelineState> pack_kv_half_pipeline = ziggy_pipeline(device, library, @"pack_kv_half_f32", &pipeline_error);
        if (pack_kv_half_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal pack_kv_half pipeline");
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
        id<MTLComputePipelineState> sample_topk_pipeline = ziggy_pipeline(device, library, @"sample_topk_f32", &pipeline_error);
        if (sample_topk_pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal sample top-k pipeline");
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

        ZiggyMetalState *state = [ZiggyMetalState new];
        state.device = device;
        state.queue = queue;
        state.matvecPipeline = matvec_pipeline;
        state.matvecAddPipeline = matvec_add_pipeline;
        state.matvecAdd2048Pipeline = matvec_add_2048_pipeline;
        state.matvecAdd5632Pipeline = matvec_add_5632_pipeline;
        state.matvecQ4KPipeline = matvec_q4k_pipeline;
        state.matvecQ4KDualPipeline = matvec_q4k_dual_pipeline;
        state.matvecQ4KDual2048Pipeline = matvec_q4k_dual_2048_pipeline;
        state.matvecQ4KDual5632Pipeline = matvec_q4k_dual_5632_pipeline;
        state.matvecQ4KKHalfPipeline = matvec_q4k_k_half_pipeline;
        state.matvecQ4KKHalf2048Pipeline = matvec_q4k_k_half_2048_pipeline;
        state.matvecQ4KKHalf5632Pipeline = matvec_q4k_k_half_5632_pipeline;
        state.matvecQ4KQRopePipeline = matvec_q4k_q_rope_pipeline;
        state.matvecQ4KQRope2048Pipeline = matvec_q4k_q_rope_2048_pipeline;
        state.matvecQ4KQRope5632Pipeline = matvec_q4k_q_rope_5632_pipeline;
        state.matvecQ4KDualKvHalfPipeline = matvec_q4k_dual_kv_half_pipeline;
        state.matvecQ4KDualKvHalf2048Pipeline = matvec_q4k_dual_kv_half_2048_pipeline;
        state.matvecQ4KDualKvHalf5632Pipeline = matvec_q4k_dual_kv_half_5632_pipeline;
        state.matvecQ4KQ6KDualKvHalfPipeline = matvec_q4k_q6k_dual_kv_half_pipeline;
        state.matvecQ4KQ6KDualKvHalf2048Pipeline = matvec_q4k_q6k_dual_kv_half_2048_pipeline;
        state.matvecQ4KQ6KDualKvHalf5632Pipeline = matvec_q4k_q6k_dual_kv_half_5632_pipeline;
        state.matvecQ4KAddPipeline = matvec_q4k_add_pipeline;
        state.matvecQ4KAdd2048Pipeline = matvec_q4k_add_2048_pipeline;
        state.matvecQ4KAdd5632Pipeline = matvec_q4k_add_5632_pipeline;
        state.matvecQ4KArgmaxPipeline = matvec_q4k_argmax_pipeline;
        state.matvecQ6KPipeline = matvec_q6k_pipeline;
        state.matvecQ6KArgmaxPipeline = matvec_q6k_argmax_pipeline;
        state.matvecQ6KAddPipeline = matvec_q6k_add_pipeline;
        state.matvecQ6KAdd2048Pipeline = matvec_q6k_add_2048_pipeline;
        state.matvecQ6KAdd5632Pipeline = matvec_q6k_add_5632_pipeline;
        state.matvecQ80Pipeline = matvec_q80_pipeline;
        state.matvecQ80AddPipeline = matvec_q80_add_pipeline;
        state.matvecMoonQ4KPipeline = matvec_moon_q4k_pipeline;
        state.matvecMoonQ4KKHalfPipeline = matvec_moon_q4k_k_half_pipeline;
        state.matvecMoonQ4KKHalf2048Pipeline = matvec_moon_q4k_k_half_2048_pipeline;
        state.matvecMoonQ4KKHalf5632Pipeline = matvec_moon_q4k_k_half_5632_pipeline;
        state.matvecMoonQ4KQRopePipeline = matvec_moon_q4k_q_rope_pipeline;
        state.matvecMoonQ4KQRope2048Pipeline = matvec_moon_q4k_q_rope_2048_pipeline;
        state.matvecMoonQ4KQRope5632Pipeline = matvec_moon_q4k_q_rope_5632_pipeline;
        state.matvecMoonQ4KDualPipeline = matvec_moon_q4k_dual_pipeline;
        state.matvecMoonQ4K2048Pipeline = matvec_moon_q4k_2048_pipeline;
        state.matvecMoonQ4KDual2048Pipeline = matvec_moon_q4k_dual_2048_pipeline;
        state.matvecMoonQ4K5632Pipeline = matvec_moon_q4k_5632_pipeline;
        state.matvecMoonQ4KDual5632Pipeline = matvec_moon_q4k_dual_5632_pipeline;
        state.matvecMoonQ4KDualKvHalfPipeline = matvec_moon_q4k_dual_kv_half_pipeline;
        state.matvecMoonQ4KDualKvHalf2048Pipeline = matvec_moon_q4k_dual_kv_half_2048_pipeline;
        state.matvecMoonQ4KDualKvHalf5632Pipeline = matvec_moon_q4k_dual_kv_half_5632_pipeline;
        state.matvecMoonQ4KQ6KDualKvHalfPipeline = matvec_moon_q4k_q6k_dual_kv_half_pipeline;
        state.matvecMoonQ4KQ6KDualKvHalf2048Pipeline = matvec_moon_q4k_q6k_dual_kv_half_2048_pipeline;
        state.matvecMoonQ4KQ6KDualKvHalf5632Pipeline = matvec_moon_q4k_q6k_dual_kv_half_5632_pipeline;
        state.matvecMoonQ4KAddPipeline = matvec_moon_q4k_add_pipeline;
        state.matvecMoonQ4KAdd2048Pipeline = matvec_moon_q4k_add_2048_pipeline;
        state.matvecMoonQ4KAdd5632Pipeline = matvec_moon_q4k_add_5632_pipeline;
        state.matvecQ4KSiluDownAddPipeline = matvec_q4k_silu_down_add_pipeline;
        state.matvecQ4KSiluDownAdd2048Pipeline = matvec_q4k_silu_down_add_2048_pipeline;
        state.matvecQ4KSiluDownAdd5632Pipeline = matvec_q4k_silu_down_add_5632_pipeline;
        state.buildFfnGateBlockMaskPipeline = build_ffn_gate_block_mask_pipeline;
        state.matvecQ4KGatedSiluDownAddPipeline = matvec_q4k_gated_silu_down_add_pipeline;
        state.matvecQ4KGatedSiluDownAdd2048Pipeline = matvec_q4k_gated_silu_down_add_2048_pipeline;
        state.matvecQ4KGatedSiluDownAdd5632Pipeline = matvec_q4k_gated_silu_down_add_5632_pipeline;
        state.matvecMoonQ4KSiluDownAddPipeline = matvec_moon_q4k_silu_down_add_pipeline;
        state.matvecMoonQ4KSiluDownAdd2048Pipeline = matvec_moon_q4k_silu_down_add_2048_pipeline;
        state.matvecMoonQ4KSiluDownAdd5632Pipeline = matvec_moon_q4k_silu_down_add_5632_pipeline;
        state.matvecMoonQ4KGatedSiluDownAddPipeline = matvec_moon_q4k_gated_silu_down_add_pipeline;
        state.matvecMoonQ4KGatedSiluDownAdd2048Pipeline = matvec_moon_q4k_gated_silu_down_add_2048_pipeline;
        state.matvecMoonQ4KGatedSiluDownAdd5632Pipeline = matvec_moon_q4k_gated_silu_down_add_5632_pipeline;
        state.ropePipeline = rope_pipeline;
        state.ropeToDstPipeline = rope_to_dst_pipeline;
        state.ropeToHalfDstPipeline = rope_to_half_dst_pipeline;
        state.storeKvHalfPipeline = store_kv_half_pipeline;
        state.packKvHalfPipeline = pack_kv_half_pipeline;
        state.attentionFusedPipeline = attention_fused_pipeline;
        state.siluMulPipeline = silu_mul_pipeline;
        state.addInPlacePipeline = add_in_place_pipeline;
        state.rmsNormPipeline = rms_norm_pipeline;
        state.rmsNormPerHeadPipeline = rms_norm_per_head_pipeline;
        state.argmaxPipeline = argmax_pipeline;
        state.topKPipeline = topk_pipeline;
        state.sampleTopKPipeline = sample_topk_pipeline;
        state.batchArgmaxPipeline = batch_argmax_pipeline;
        state.batchMatvecAddPipeline = batch_matvec_add_pipeline;
        state.batchMatvecQ4KAddPipeline = batch_matvec_q4k_add_pipeline;
        state.batchMatvecQ4KAdd2048Pipeline = batch_matvec_q4k_add_2048_pipeline;
        state.batchMatvecQ4KAdd5632Pipeline = batch_matvec_q4k_add_5632_pipeline;
        state.batchMatvecQ4KPipeline = batch_matvec_q4k_pipeline;
        state.batchSiluMulPipeline = batch_silu_mul_pipeline;
        state.batchAddInPlacePipeline = batch_add_in_place_pipeline;
        state.batchRmsNormPipeline = batch_rms_norm_pipeline;
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
        wrapper.hostVisible = true;
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
        wrapper.hostVisible = true;
        *out_buffer = (__bridge_retained void *)wrapper;
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_create_empty_buffer_private(
    ZiggyMetalContext *ctx,
    size_t length,
    ZiggyMetalBuffer **out_buffer,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || out_buffer == NULL || length == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid private Metal buffer creation request");
        return ZIGGY_METAL_BUFFER_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        id<MTLBuffer> buffer = [state.device newBufferWithLength:length options:MTLResourceStorageModePrivate];
        if (buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate private Metal buffer");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        ZiggyMetalBufferState *wrapper = [ZiggyMetalBufferState new];
        wrapper.buffer = buffer;
        wrapper.length = length;
        wrapper.hostVisible = false;
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
        if (!wrapper.hostVisible) {
            ziggy_write_error(error_message, error_message_len, @"Metal buffer write requires a host-visible buffer");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
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
        if (!wrapper.hostVisible) {
            ziggy_write_error(error_message, error_message_len, @"Metal buffer read requires a host-visible buffer");
            return ZIGGY_METAL_BUFFER_FAILED;
        }
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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_dense_add_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_rowwise(encoder, pipeline, rows);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_q4k_add_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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

int ziggy_metal_run_matvec_q4k_dual_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix0,
    const ZiggyMetalBuffer *matrix1,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output0,
    ZiggyMetalBuffer *output1,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix0 == NULL || matrix1 == NULL || input == NULL || output0 == NULL || output1 == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q4k dual matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix0_buffer = ziggy_const_buffer(matrix0);
        const ZiggyMetalBufferState *matrix1_buffer = ziggy_const_buffer(matrix1);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output0_buffer = ziggy_buffer(output0);
        ZiggyMetalBufferState *output1_buffer = ziggy_buffer(output1);
        if (output0_buffer.length < (size_t)rows * sizeof(float) || output1_buffer.length < (size_t)rows * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal q4k dual matvec exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_q4k_dual_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix0_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:matrix1_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:output0_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:output1_buffer.buffer offset:0 atIndex:4];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:6];
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q4k dual matvec command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q4k_k_half_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *k_cache,
    size_t dst_offset_elements,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t cols,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || k_cache == NULL || head_count == 0 || head_dim == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q4k k-half request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (rope_style != 0 || pair_count == 0 || pair_count * 2 != head_dim) {
        ziggy_write_error(error_message, error_message_len, @"unsupported Metal q4k k-half rope configuration");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *k_cache_buffer = ziggy_buffer(k_cache);
        const uint32_t rows = head_count * head_dim;
        if (k_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal q4k k-half exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_q4k_k_half_pipeline(state, cols);
        uint32_t dst_base = (uint32_t)dst_offset_elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:k_cache_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&dst_base length:sizeof(dst_base) atIndex:3];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
        [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:6];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:7];
        [encoder setBytes:&position length:sizeof(position) atIndex:8];
        [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:9];
        ziggy_dispatch_q4k_rows(encoder, pipeline, head_count * pair_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q4k k-half command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q4k_q_rope_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t cols,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || head_count == 0 || head_dim == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q4k q-rope request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (pair_count == 0 || pair_count * 2 != head_dim) {
        ziggy_write_error(error_message, error_message_len, @"unsupported Metal q4k q-rope configuration");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        const uint32_t rows = head_count * head_dim;
        if (output_buffer.length < (size_t)rows * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal q4k q-rope exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_q4k_q_rope_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:3];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
        [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:5];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:6];
        [encoder setBytes:&position length:sizeof(position) atIndex:7];
        [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:8];
        [encoder setBytes:&rope_style length:sizeof(rope_style) atIndex:9];
        ziggy_dispatch_q4k_rows(encoder, pipeline, head_count * pair_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q4k q-rope command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_moonq_q4k_k_half_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *k_cache,
    size_t dst_offset_elements,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t cols,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || k_cache == NULL || head_count == 0 || head_dim == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal MoonQuant q4k k-half request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (rope_style != 0 || pair_count == 0 || pair_count * 2 != head_dim) {
        ziggy_write_error(error_message, error_message_len, @"unsupported Metal MoonQuant q4k k-half rope configuration");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *k_cache_buffer = ziggy_buffer(k_cache);
        const uint32_t rows = head_count * head_dim;
        if (k_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal MoonQuant q4k k-half exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_k_half_pipeline(state, cols);
        uint32_t dst_base = (uint32_t)dst_offset_elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:k_cache_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&dst_base length:sizeof(dst_base) atIndex:3];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
        [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:6];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:7];
        [encoder setBytes:&position length:sizeof(position) atIndex:8];
        [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:9];
        ziggy_dispatch_q4k_rows(encoder, pipeline, head_count * pair_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal MoonQuant q4k k-half command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_moonq_q4k_q_rope_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t cols,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix == NULL || input == NULL || output == NULL || head_count == 0 || head_dim == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal MoonQuant q4k q-rope request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (pair_count == 0 || pair_count * 2 != head_dim) {
        ziggy_write_error(error_message, error_message_len, @"unsupported Metal MoonQuant q4k q-rope configuration");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);
        const uint32_t rows = head_count * head_dim;
        if (output_buffer.length < (size_t)rows * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal MoonQuant q4k q-rope exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_q_rope_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:3];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
        [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:5];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:6];
        [encoder setBytes:&position length:sizeof(position) atIndex:7];
        [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:8];
        [encoder setBytes:&rope_style length:sizeof(rope_style) atIndex:9];
        ziggy_dispatch_q4k_rows(encoder, pipeline, head_count * pair_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal MoonQuant q4k q-rope command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q4k_dual_kv_half_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix_k,
    const ZiggyMetalBuffer *matrix_v,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *k_cache,
    ZiggyMetalBuffer *v_cache,
    size_t dst_offset_elements,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t cols,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix_k == NULL || matrix_v == NULL || input == NULL || k_cache == NULL || v_cache == NULL || head_count == 0 || head_dim == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q4k dual kv-half request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (rope_style != 0 || pair_count == 0 || pair_count * 2 != head_dim) {
        ziggy_write_error(error_message, error_message_len, @"unsupported Metal q4k dual kv-half rope configuration");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_k_buffer = ziggy_const_buffer(matrix_k);
        const ZiggyMetalBufferState *matrix_v_buffer = ziggy_const_buffer(matrix_v);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *k_cache_buffer = ziggy_buffer(k_cache);
        ZiggyMetalBufferState *v_cache_buffer = ziggy_buffer(v_cache);
        const uint32_t rows = head_count * head_dim;
        if (k_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t) ||
            v_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal q4k dual kv-half exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_q4k_dual_kv_half_pipeline(state, cols);
        uint32_t dst_base = (uint32_t)dst_offset_elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_k_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:matrix_v_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:k_cache_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:v_cache_buffer.buffer offset:0 atIndex:4];
        [encoder setBytes:&dst_base length:sizeof(dst_base) atIndex:5];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:8];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:9];
        [encoder setBytes:&position length:sizeof(position) atIndex:10];
        [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:11];
        ziggy_dispatch_q4k_rows(encoder, pipeline, head_count * pair_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q4k dual kv-half command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_moonq_q4k_dual_kv_half_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix_k,
    const ZiggyMetalBuffer *matrix_v,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *k_cache,
    ZiggyMetalBuffer *v_cache,
    size_t dst_offset_elements,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t cols,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix_k == NULL || matrix_v == NULL || input == NULL || k_cache == NULL || v_cache == NULL || head_count == 0 || head_dim == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal MoonQuant q4k dual kv-half request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (rope_style != 0 || pair_count == 0 || pair_count * 2 != head_dim) {
        ziggy_write_error(error_message, error_message_len, @"unsupported Metal MoonQuant q4k dual kv-half rope configuration");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_k_buffer = ziggy_const_buffer(matrix_k);
        const ZiggyMetalBufferState *matrix_v_buffer = ziggy_const_buffer(matrix_v);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *k_cache_buffer = ziggy_buffer(k_cache);
        ZiggyMetalBufferState *v_cache_buffer = ziggy_buffer(v_cache);
        const uint32_t rows = head_count * head_dim;
        if (k_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t) ||
            v_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal MoonQuant q4k dual kv-half exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_dual_kv_half_pipeline(state, cols);
        uint32_t dst_base = (uint32_t)dst_offset_elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_k_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:matrix_v_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:k_cache_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:v_cache_buffer.buffer offset:0 atIndex:4];
        [encoder setBytes:&dst_base length:sizeof(dst_base) atIndex:5];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:8];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:9];
        [encoder setBytes:&position length:sizeof(position) atIndex:10];
        [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:11];
        ziggy_dispatch_q4k_rows(encoder, pipeline, head_count * pair_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal MoonQuant q4k dual kv-half command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_q4k_q6k_dual_kv_half_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix_k,
    const ZiggyMetalBuffer *matrix_v,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *k_cache,
    ZiggyMetalBuffer *v_cache,
    size_t dst_offset_elements,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t cols,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix_k == NULL || matrix_v == NULL || input == NULL || k_cache == NULL || v_cache == NULL || head_count == 0 || head_dim == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q4k/q6k dual kv-half request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (rope_style != 0 || pair_count == 0 || pair_count * 2 != head_dim) {
        ziggy_write_error(error_message, error_message_len, @"unsupported Metal q4k/q6k dual kv-half rope configuration");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_k_buffer = ziggy_const_buffer(matrix_k);
        const ZiggyMetalBufferState *matrix_v_buffer = ziggy_const_buffer(matrix_v);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *k_cache_buffer = ziggy_buffer(k_cache);
        ZiggyMetalBufferState *v_cache_buffer = ziggy_buffer(v_cache);
        const uint32_t rows = head_count * head_dim;
        if (k_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t) ||
            v_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal q4k/q6k dual kv-half exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_q4k_q6k_dual_kv_half_pipeline(state, cols);
        uint32_t dst_base = (uint32_t)dst_offset_elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_k_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:matrix_v_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:k_cache_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:v_cache_buffer.buffer offset:0 atIndex:4];
        [encoder setBytes:&dst_base length:sizeof(dst_base) atIndex:5];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:8];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:9];
        [encoder setBytes:&position length:sizeof(position) atIndex:10];
        [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:11];
        ziggy_dispatch_q4k_rows(encoder, pipeline, head_count * pair_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q4k/q6k dual kv-half command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_moonq_q4k_q6k_dual_kv_half_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix_k,
    const ZiggyMetalBuffer *matrix_v,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *k_cache,
    ZiggyMetalBuffer *v_cache,
    size_t dst_offset_elements,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t cols,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix_k == NULL || matrix_v == NULL || input == NULL || k_cache == NULL || v_cache == NULL || head_count == 0 || head_dim == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal MoonQuant q4k/q6k dual kv-half request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;
    if (rope_style != 0 || pair_count == 0 || pair_count * 2 != head_dim) {
        ziggy_write_error(error_message, error_message_len, @"unsupported Metal MoonQuant q4k/q6k dual kv-half rope configuration");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix_k_buffer = ziggy_const_buffer(matrix_k);
        const ZiggyMetalBufferState *matrix_v_buffer = ziggy_const_buffer(matrix_v);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *k_cache_buffer = ziggy_buffer(k_cache);
        ZiggyMetalBufferState *v_cache_buffer = ziggy_buffer(v_cache);
        const uint32_t rows = head_count * head_dim;
        if (k_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t) ||
            v_cache_buffer.length < (dst_offset_elements + rows) * sizeof(uint16_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal MoonQuant q4k/q6k dual kv-half exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_q6k_dual_kv_half_pipeline(state, cols);
        uint32_t dst_base = (uint32_t)dst_offset_elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_k_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:matrix_v_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:k_cache_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:v_cache_buffer.buffer offset:0 atIndex:4];
        [encoder setBytes:&dst_base length:sizeof(dst_base) atIndex:5];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:8];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:9];
        [encoder setBytes:&position length:sizeof(position) atIndex:10];
        [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:11];
        ziggy_dispatch_q4k_rows(encoder, pipeline, head_count * pair_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal MoonQuant q4k/q6k dual kv-half command failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}

int ziggy_metal_run_matvec_moonq_q4k_dual_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix0,
    const ZiggyMetalBuffer *matrix1,
    const ZiggyMetalBuffer *input,
    ZiggyMetalBuffer *output0,
    ZiggyMetalBuffer *output1,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || matrix0 == NULL || matrix1 == NULL || input == NULL || output0 == NULL || output1 == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal MoonQuant q4k dual matvec request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *matrix0_buffer = ziggy_const_buffer(matrix0);
        const ZiggyMetalBufferState *matrix1_buffer = ziggy_const_buffer(matrix1);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output0_buffer = ziggy_buffer(output0);
        ZiggyMetalBufferState *output1_buffer = ziggy_buffer(output1);
        if (output0_buffer.length < (size_t)rows * sizeof(float) || output1_buffer.length < (size_t)rows * sizeof(float)) {
            ziggy_write_error(error_message, error_message_len, @"Metal MoonQuant q4k dual matvec exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
        const bool has_pending = command_buffer != nil;
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_dual_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix0_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:matrix1_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:output0_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:output1_buffer.buffer offset:0 atIndex:4];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:6];
        ziggy_dispatch_moonq_q4k_rows(encoder, pipeline, rows);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal MoonQuant q4k dual matvec command failed");
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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_q6k_add_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        if (input_buffer.length < ((size_t)cols * sizeof(float)) ||
            output_buffer.length < 3 * sizeof(uint32_t)) {
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
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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

int ziggy_metal_run_matvec_q4k_argmax_f32(
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
        ziggy_write_error(error_message, error_message_len, @"invalid Metal q4k argmax request");
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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        if (input_buffer.length < ((size_t)cols * sizeof(float)) ||
            output_buffer.length < 3 * sizeof(uint32_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal q4k argmax exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        [encoder setComputePipelineState:state.matvecQ4KArgmaxPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, state.matvecQ4KArgmaxPipeline, rows);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

        if (has_pending) return ZIGGY_METAL_OK;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal q4k argmax command buffer failed");
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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        [encoder setComputePipelineState:state.matvecQ80AddPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        ziggy_dispatch_q4k_rows(encoder, state.matvecQ80AddPipeline, rows);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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

    id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
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
    ziggy_record_dispatch(state);
    ziggy_finish_compute_encoder(state, encoder, has_pending);

    if (has_pending) return ZIGGY_METAL_OK;

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: command_error_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }
    return ZIGGY_METAL_OK;
}

static int ziggy_run_gated_fused_silu_down_matvec(
    ZiggyMetalState *state,
    id<MTLComputePipelineState> pipeline,
    const ZiggyMetalBufferState *matrix_buffer,
    const ZiggyMetalBufferState *gate_buffer,
    const ZiggyMetalBufferState *up_buffer,
    const ZiggyMetalBufferState *mask_buffer,
    ZiggyMetalBufferState *output_buffer,
    uint32_t rows,
    uint32_t cols,
    int dispatch_type,
    NSString *invalid_message,
    NSString *command_error_message,
    char *error_message,
    size_t error_message_len
) {
    if (matrix_buffer == NULL || gate_buffer == NULL || up_buffer == NULL || mask_buffer == NULL || output_buffer == NULL || rows == 0 || cols == 0) {
        ziggy_write_error(error_message, error_message_len, invalid_message);
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    id<MTLCommandBuffer> command_buffer = state.pendingCommandBuffer;
    const bool has_pending = command_buffer != nil;
    if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
    if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
    if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
    [encoder setBuffer:gate_buffer.buffer offset:0 atIndex:1];
    [encoder setBuffer:up_buffer.buffer offset:0 atIndex:2];
    [encoder setBuffer:mask_buffer.buffer offset:0 atIndex:3];
    [encoder setBuffer:output_buffer.buffer offset:0 atIndex:4];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:6];

    if (dispatch_type == 2) {
        ziggy_dispatch_moonq_q4k_rows(encoder, pipeline, rows);
    } else {
        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
    }
    ziggy_record_dispatch(state);
    ziggy_finish_compute_encoder(state, encoder, has_pending);

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

int ziggy_metal_build_ffn_gate_block_mask_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    ZiggyMetalBuffer *mask,
    ZiggyMetalBuffer *stats,
    uint32_t cols,
    float threshold,
    char *error_message,
    size_t error_message_len
) {
    @autoreleasepool {
        if (ctx == NULL || gate == NULL || up == NULL || mask == NULL || stats == NULL || cols == 0) {
            ziggy_write_error(error_message, error_message_len, @"invalid Metal FFN gate mask request");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        ZiggyMetalState *state = ziggy_state(ctx);
        return ziggy_run_compute(
            state,
            state.buildFfnGateBlockMaskPipeline,
            (cols + 255) / 256,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:ziggy_const_buffer(gate).buffer offset:0 atIndex:0];
                [encoder setBuffer:ziggy_const_buffer(up).buffer offset:0 atIndex:1];
                [encoder setBuffer:ziggy_buffer(mask).buffer offset:0 atIndex:2];
                [encoder setBuffer:ziggy_buffer(stats).buffer offset:0 atIndex:3];
                [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
                [encoder setBytes:&threshold length:sizeof(threshold) atIndex:5];
            },
            error_message,
            error_message_len
        );
    }
}

int ziggy_metal_run_matvec_q4k_gated_silu_down_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    const ZiggyMetalBuffer *mask,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        id<MTLComputePipelineState> pipeline = cols == 2048 ? state.matvecQ4KGatedSiluDownAdd2048Pipeline : (cols == 5632 ? state.matvecQ4KGatedSiluDownAdd5632Pipeline : state.matvecQ4KGatedSiluDownAddPipeline);
        return ziggy_run_gated_fused_silu_down_matvec(state, pipeline, ziggy_const_buffer(matrix), ziggy_const_buffer(gate), ziggy_const_buffer(up), ziggy_const_buffer(mask), ziggy_buffer(output), rows, cols, 1, @"invalid Metal q4k gated silu down add request", @"Metal command buffer failed", error_message, error_message_len);
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

int ziggy_metal_run_matvec_moonq_q4k_gated_silu_down_add_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *matrix,
    const ZiggyMetalBuffer *gate,
    const ZiggyMetalBuffer *up,
    const ZiggyMetalBuffer *mask,
    ZiggyMetalBuffer *output,
    uint32_t rows,
    uint32_t cols,
    char *error_message,
    size_t error_message_len
) {
    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        id<MTLComputePipelineState> pipeline = cols == 2048 ? state.matvecMoonQ4KGatedSiluDownAdd2048Pipeline : (cols == 5632 ? state.matvecMoonQ4KGatedSiluDownAdd5632Pipeline : state.matvecMoonQ4KGatedSiluDownAddPipeline);
        return ziggy_run_gated_fused_silu_down_matvec(state, pipeline, ziggy_const_buffer(matrix), ziggy_const_buffer(gate), ziggy_const_buffer(up), ziggy_const_buffer(mask), ziggy_buffer(output), rows, cols, 2, @"invalid Metal MoonQuant gated silu down add request", @"Metal command buffer failed", error_message, error_message_len);
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

    id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
    if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

	    id<MTLComputePipelineState> pipeline = ziggy_select_moonq_q4k_pipeline(state, cols, true);
	    [encoder setComputePipelineState:pipeline];
	    [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
	    [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
	    [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
	    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
	    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
	    ziggy_dispatch_moonq_q4k_rows(encoder, pipeline, rows);
	    ziggy_record_dispatch(state);
	    ziggy_finish_compute_encoder(state, encoder, has_pending);

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
            ziggy_end_pending_compute_encoder(state);
            id<MTLBlitCommandEncoder> encoder = ziggy_new_blit_encoder(state, state.pendingCommandBuffer, error_message, error_message_len);
            if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

            [encoder copyFromBuffer:src_buffer.buffer sourceOffset:src_offset toBuffer:dst_buffer.buffer destinationOffset:dst_offset size:length];
            [encoder endEncoding];
            return ZIGGY_METAL_OK;
        }

        id<MTLCommandBuffer> command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLBlitCommandEncoder> encoder = ziggy_new_blit_encoder(state, command_buffer, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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

int ziggy_metal_apply_rope_to_half_dst_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *src,
    ZiggyMetalBuffer *dst,
    size_t dst_offset_elements,
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
        ziggy_write_error(error_message, error_message_len, @"invalid Metal rope-to-half-dst request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    const uint32_t pair_count = (rope_dim < head_dim ? rope_dim : head_dim) / 2;

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *src_buffer = ziggy_const_buffer(src);
        ZiggyMetalBufferState *dst_buffer = ziggy_buffer(dst);
        const size_t element_count = (size_t)head_count * (size_t)head_dim;
        const size_t src_byte_count = element_count * sizeof(float);
        const size_t dst_byte_count = element_count * sizeof(uint16_t);
        const size_t dst_offset_bytes = dst_offset_elements * sizeof(uint16_t);
        if (src_byte_count == 0 || src_buffer.length < src_byte_count || dst_offset_bytes + dst_byte_count > dst_buffer.length) {
            ziggy_write_error(error_message, error_message_len, @"Metal rope-to-half-dst exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        const uint32_t dst_base = (uint32_t)dst_offset_elements;
        return ziggy_run_compute(
            state,
            state.ropeToHalfDstPipeline,
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

int ziggy_metal_pack_kv_half_f32(
    ZiggyMetalContext *ctx,
    const ZiggyMetalBuffer *k_src,
    const ZiggyMetalBuffer *v_src,
    ZiggyMetalBuffer *k_dst,
    ZiggyMetalBuffer *v_dst,
    size_t dst_offset_elements,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t position,
    float freq_base,
    uint32_t rope_style,
    char *error_message,
    size_t error_message_len
) {
    if (ctx == NULL || k_src == NULL || v_src == NULL || k_dst == NULL || v_dst == NULL || head_count == 0 || head_dim == 0) {
        ziggy_write_error(error_message, error_message_len, @"invalid Metal pack_kv_half request");
        return ZIGGY_METAL_EXECUTION_FAILED;
    }

    @autoreleasepool {
        ZiggyMetalState *state = ziggy_state(ctx);
        const ZiggyMetalBufferState *k_src_buffer = ziggy_const_buffer(k_src);
        const ZiggyMetalBufferState *v_src_buffer = ziggy_const_buffer(v_src);
        ZiggyMetalBufferState *k_dst_buffer = ziggy_buffer(k_dst);
        ZiggyMetalBufferState *v_dst_buffer = ziggy_buffer(v_dst);
        const uint32_t count = head_count * head_dim;

        if (k_src_buffer.length < count * sizeof(float) ||
            v_src_buffer.length < count * sizeof(float) ||
            k_dst_buffer.length < (dst_offset_elements + count) * sizeof(uint16_t) ||
            v_dst_buffer.length < (dst_offset_elements + count) * sizeof(uint16_t)) {
            ziggy_write_error(error_message, error_message_len, @"Metal pack_kv_half exceeded allocation");
            return ZIGGY_METAL_BUFFER_FAILED;
        }

        uint32_t dst_base = (uint32_t)dst_offset_elements;
        return ziggy_run_compute(
            state,
            state.packKvHalfPipeline,
            count,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:k_src_buffer.buffer offset:0 atIndex:0];
                [encoder setBuffer:v_src_buffer.buffer offset:0 atIndex:1];
                [encoder setBuffer:k_dst_buffer.buffer offset:0 atIndex:2];
                [encoder setBuffer:v_dst_buffer.buffer offset:0 atIndex:3];
                [encoder setBytes:&dst_base length:sizeof(dst_base) atIndex:4];
                [encoder setBytes:&head_count length:sizeof(head_count) atIndex:5];
                [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
                [encoder setBytes:&rope_dim length:sizeof(rope_dim) atIndex:7];
                [encoder setBytes:&position length:sizeof(position) atIndex:8];
                [encoder setBytes:&freq_base length:sizeof(freq_base) atIndex:9];
                [encoder setBytes:&rope_style length:sizeof(rope_style) atIndex:10];
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
        if (command_buffer == nil) command_buffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        if (command_buffer == nil) return ZIGGY_METAL_EXECUTION_FAILED;

    id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
    if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

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
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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
        state.pendingCommandBuffer = ziggy_new_command_buffer(state.queue, error_message, error_message_len);
        state.pendingComputeEncoder = nil;
        state.pendingEncoderCount = 0;
        state.pendingDispatchCount = 0;
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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        [encoder setComputePipelineState:state.rmsNormPerHeadPipeline];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:weights_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&head_count length:sizeof(head_count) atIndex:3];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:5];
        
        ziggy_dispatch_rowwise(encoder, state.rmsNormPerHeadPipeline, head_count);
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        [encoder setComputePipelineState:state.batchArgmaxPipeline];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:1];
        [encoder setBytes:&vocab_size length:sizeof(vocab_size) atIndex:2];
        [encoder setBytes:&batch_count length:sizeof(batch_count) atIndex:3];
        [encoder setBytes:&thread_count length:sizeof(thread_count) atIndex:4];

        MTLSize grid_size = MTLSizeMake(batch_count, 1, 1);
        MTLSize group_size = MTLSizeMake(thread_count, 1, 1);
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        [encoder setComputePipelineState:state.batchMatvecAddPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
	        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
	        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
	        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
	        [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:5];
	        ziggy_dispatch_rowwise(encoder, state.batchMatvecAddPipeline, rows);
	        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        id<MTLComputePipelineState> pipeline = ziggy_select_batch_q4k_add_pipeline(state, cols);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
	        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
	        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
	        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
	        [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:5];
	        ziggy_dispatch_q4k_rows(encoder, pipeline, rows);
	        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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
                [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:5];
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

        id<MTLComputeCommandEncoder> encoder = ziggy_acquire_compute_encoder(state, command_buffer, has_pending, error_message, error_message_len);
        if (encoder == nil) return ZIGGY_METAL_EXECUTION_FAILED;

        [encoder setComputePipelineState:state.batchMatvecQ4KPipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
	        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
	        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
	        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
	        [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:5];
	        ziggy_dispatch_q4k_rows(encoder, state.batchMatvecQ4KPipeline, rows);
	        ziggy_record_dispatch(state);
        ziggy_finish_compute_encoder(state, encoder, has_pending);

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
