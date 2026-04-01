pub const supported_versions = "TBD in v0.2.0";

pub const InspectSummary = struct {
    status: []const u8,
    message: []const u8,
};

pub fn inspectSummary(model_path: ?[]const u8) InspectSummary {
    if (model_path == null) {
        return .{
            .status = "missing model path",
            .message = "Pass -m or --model to inspect a GGUF file once the parser is implemented.",
        };
    }

    return .{
        .status = "parser not implemented",
        .message = "GGUF header and metadata parsing are scheduled for roadmap milestone v0.2.0.",
    };
}
