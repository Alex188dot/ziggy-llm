const std = @import("std");
const llama_runtime = @import("llama_runtime.zig");
const types = @import("types.zig");

pub fn generate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !types.GenerationReport {
    return llama_runtime.generate(allocator, model_path, prompt, options);
}

pub fn generateZiggy(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: types.GenerationOptions,
) !types.GenerationReport {
    return llama_runtime.generateZiggy(allocator, model_path, prompt, options);
}
