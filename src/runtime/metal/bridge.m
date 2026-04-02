#import "bridge.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <stdio.h>
#import <string.h>

@interface ZiggyMetalState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
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

        id<MTLFunction> function = [library newFunctionWithName:@"matvec_f32"];
        if (function == nil) {
            ziggy_write_error(error_message, error_message_len, @"missing matvec_f32 Metal function");
            return ZIGGY_METAL_COMPILATION_FAILED;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal command queue");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        NSError *pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            ziggy_write_error(error_message, error_message_len, pipeline_error.localizedDescription ?: @"failed to create Metal pipeline");
            return ZIGGY_METAL_INITIALIZATION_FAILED;
        }

        ZiggyMetalState *state = [ZiggyMetalState new];
        state.device = device;
        state.queue = queue;
        state.pipeline = pipeline;
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
        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to allocate Metal command buffer");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            ziggy_write_error(error_message, error_message_len, @"failed to create Metal compute encoder");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }

        const ZiggyMetalBufferState *matrix_buffer = ziggy_const_buffer(matrix);
        const ZiggyMetalBufferState *input_buffer = ziggy_const_buffer(input);
        ZiggyMetalBufferState *output_buffer = ziggy_buffer(output);

        [encoder setComputePipelineState:state.pipeline];
        [encoder setBuffer:matrix_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];

        NSUInteger thread_width = state.pipeline.threadExecutionWidth;
        if (thread_width == 0) thread_width = 1;
        NSUInteger threads_per_group = thread_width < rows ? thread_width : rows;
        if (threads_per_group == 0) threads_per_group = 1;

        MTLSize grid_size = MTLSizeMake(rows, 1, 1);
        MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            ziggy_write_error(error_message, error_message_len, command_buffer.error.localizedDescription ?: @"Metal command buffer failed");
            return ZIGGY_METAL_EXECUTION_FAILED;
        }
        return ZIGGY_METAL_OK;
    }
}
