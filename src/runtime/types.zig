const std = @import("std");

pub const primary_target = "Apple Silicon + Metal";
pub const fallback_target = "Apple Silicon CPU";
pub const native_architecture = "ziggy-tiny";
pub const supported_architecture = "ziggy-tiny (native CPU + Metal) and llama (native CPU + Metal)";
pub const supported_model_family = "ziggy-tiny GGUF reference models through the CPU or Metal runtime and llama-family GGUF models through the native CPU or Metal runtime";
pub const supported_quantization = "ziggy-tiny: F16 only; llama CPU and Metal paths: F32, F16, Q4_K, and Q6_K";

pub const RuntimeError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedArtifactType,
    UnsupportedArchitecture,
    UnsupportedBackend,
    UnsupportedFileType,
    UnsupportedTensorType,
    UnsupportedTokenizer,
    MissingRequiredMetadata,
    MissingRequiredTensor,
    InvalidMetadataType,
    InvalidMetadataValue,
    InvalidTensorMetadata,
    InvalidPrompt,
    TruncatedFile,
    Overflow,
    UnknownToken,
    ContextOverflow,
    EmptyPrompt,
    MetalDisabled,
    MetalUnavailable,
    MetalInitializationFailed,
    MetalCompilationFailed,
    MetalBufferError,
    MetalExecutionFailed,
};

pub const BackendPreference = enum {
    auto,
    cpu,
    metal,

    pub fn parse(name: []const u8) ?BackendPreference {
        if (std.mem.eql(u8, name, "auto")) return .auto;
        if (std.mem.eql(u8, name, "cpu")) return .cpu;
        if (std.mem.eql(u8, name, "metal")) return .metal;
        return null;
    }

    pub fn label(self: BackendPreference) []const u8 {
        return switch (self) {
            .auto => "auto",
            .cpu => "cpu",
            .metal => "metal",
        };
    }
};

pub const BackendUsed = enum {
    cpu,
    metal,

    pub fn label(self: BackendUsed) []const u8 {
        return switch (self) {
            .cpu => "cpu",
            .metal => "metal",
        };
    }
};

pub const GenerationOptions = struct {
    max_tokens: usize = 16,
    seed: u64 = 0,
    temperature: f32 = 0.0,
    backend: BackendPreference = .auto,
};

pub const GenerationReport = struct {
    generated_text: []u8,
    prompt_token_count: usize,
    generated_token_count: usize,
    startup_ns: u64,
    prompt_ns: u64,
    ttft_ns: u64,
    decode_ns: u64,
    seed: u64,
    temperature: f32,
    backend: BackendUsed,

    pub fn deinit(self: *GenerationReport, allocator: std.mem.Allocator) void {
        allocator.free(self.generated_text);
        self.* = undefined;
    }

    pub fn decodeTokensPerSecond(self: GenerationReport) f64 {
        if (self.generated_token_count == 0 or self.decode_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.generated_token_count)) / (@as(f64, @floatFromInt(self.decode_ns)) / std.time.ns_per_s);
    }
};

pub fn deltaNs(start: i128, end: i128) u64 {
    return @intCast(@max(@as(i128, 0), end - start));
}

pub fn nsToMs(value: u64) f64 {
    return @as(f64, @floatFromInt(value)) / std.time.ns_per_ms;
}
