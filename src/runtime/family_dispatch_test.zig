const std = @import("std");
const types = @import("types.zig");

test "modelFamilyFromArchitecture routes llama" {
    try std.testing.expectEqual(types.ModelFamily.llama, try types.modelFamilyFromArchitecture("llama"));
}

test "modelFamilyFromArchitecture routes qwen variants" {
    try std.testing.expectEqual(types.ModelFamily.qwen, try types.modelFamilyFromArchitecture("qwen2"));
    try std.testing.expectEqual(types.ModelFamily.qwen, try types.modelFamilyFromArchitecture("qwen3"));
    try std.testing.expectEqual(types.ModelFamily.qwen, try types.modelFamilyFromArchitecture("qwen3moe"));
}

test "modelFamilyFromArchitecture rejects unsupported architecture" {
    try std.testing.expectError(error.UnsupportedArchitecture, types.modelFamilyFromArchitecture("gemma"));
}
