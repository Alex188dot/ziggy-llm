const std = @import("std");

pub fn qkHeadIndex(value_head_index: usize, qk_head_count: usize, value_head_count: usize) !usize {
    if (qk_head_count == 0 or value_head_count == 0) return error.InvalidMetadataValue;
    if (value_head_count % qk_head_count != 0) return error.InvalidMetadataValue;
    if (value_head_index >= value_head_count) return error.InvalidMetadataValue;

    const values_per_qk = value_head_count / qk_head_count;
    return @min(value_head_index / values_per_qk, qk_head_count - 1);
}

test "qwen35 grouped qk head mapping handles two value heads per qk head" {
    try std.testing.expectEqual(@as(usize, 0), try qkHeadIndex(0, 16, 32));
    try std.testing.expectEqual(@as(usize, 0), try qkHeadIndex(1, 16, 32));
    try std.testing.expectEqual(@as(usize, 1), try qkHeadIndex(2, 16, 32));
    try std.testing.expectEqual(@as(usize, 15), try qkHeadIndex(31, 16, 32));
}

test "qwen35 grouped qk head mapping rejects incompatible layouts" {
    try std.testing.expectError(error.InvalidMetadataValue, qkHeadIndex(0, 0, 32));
    try std.testing.expectError(error.InvalidMetadataValue, qkHeadIndex(0, 16, 30));
    try std.testing.expectError(error.InvalidMetadataValue, qkHeadIndex(32, 16, 32));
}
