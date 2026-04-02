const std = @import("std");
const moon_quant = @import("../moon_quant.zig");

pub const primary_target = "Apple Silicon + Metal";
pub const fallback_target = "Apple Silicon CPU";
pub const native_architecture = "llama";
pub const supported_architecture = "llama (native CPU + Metal)";
pub const supported_model_family = "llama-family GGUF models through the native CPU or Metal runtime";
pub const supported_quantization = moon_quant.supported_quantization;

pub const RuntimeError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedArtifactType,
    UnsupportedArchitecture,
    UnsupportedBackend,
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
    UnsupportedSpeculativeSampling,
    InvalidSpeculativeConfig,
    SpeculativeDraftTokenizerMismatch,
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

pub const MoonQuantMode = enum {
    enabled,
    disabled,

    pub fn parse(name: []const u8) ?MoonQuantMode {
        if (std.mem.eql(u8, name, "enabled")) return .enabled;
        if (std.mem.eql(u8, name, "disabled")) return .disabled;
        return null;
    }

    pub fn label(self: MoonQuantMode) []const u8 {
        return switch (self) {
            .enabled => "enabled",
            .disabled => "disabled",
        };
    }
};

pub const GenerationOptions = struct {
    max_tokens: usize = 16,
    seed: u64 = 0,
    temperature: f32 = 0.0,
    repeat_penalty: f32 = 1.0,
    top_k: usize = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
    backend: BackendPreference = .auto,
    moon_quant: MoonQuantMode = .enabled,
    metal_profile: bool = false,
    speculative: SpeculativeConfig = .{},
};

pub const SpeculativeConfig = struct {
    draft_model_path: ?[]const u8 = null,
    draft_tokens: usize = 4,

    pub fn enabled(self: SpeculativeConfig) bool {
        return self.draft_model_path != null;
    }
};

pub const StartupBreakdown = struct {
    model_load_ns: u64 = 0,
    tensor_prepare_ns: u64 = 0,
    backend_init_ns: u64 = 0,
    metal_prewarm_ns: u64 = 0,
    session_init_ns: u64 = 0,
    first_decode_step_ns: u64 = 0,
};

pub const GenerationReport = struct {
    generated_text: []u8,
    prompt_token_count: usize,
    reused_prompt_token_count: usize = 0,
    generated_token_count: usize,
    startup_ns: u64,
    prompt_ns: u64,
    ttft_ns: u64,
    decode_ns: u64,
    seed: u64,
    temperature: f32,
    backend: BackendUsed,
    startup_breakdown: StartupBreakdown = .{},
    speculative: ?SpeculativeStats = null,
    metal_profile_summary: ?[]u8 = null,

    pub fn deinit(self: *GenerationReport, allocator: std.mem.Allocator) void {
        allocator.free(self.generated_text);
        if (self.metal_profile_summary) |summary| allocator.free(summary);
        self.* = undefined;
    }

    pub fn decodeTokensPerSecond(self: GenerationReport) f64 {
        if (self.generated_token_count == 0 or self.decode_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.generated_token_count)) / (@as(f64, @floatFromInt(self.decode_ns)) / std.time.ns_per_s);
    }
};

pub const SpeculativeStats = struct {
    draft_tokens: usize,
    rounds: usize,
    accepted_tokens: usize,
    rejected_tokens: usize,
    draft_decode_ns: u64,
    verifier_decode_ns: u64,
    resync_count: usize = 0,

    pub fn acceptanceRate(self: SpeculativeStats) f64 {
        const attempts = self.accepted_tokens + self.rejected_tokens;
        if (attempts == 0) return 0;
        return @as(f64, @floatFromInt(self.accepted_tokens)) /
            @as(f64, @floatFromInt(attempts));
    }
};

pub fn deltaNs(start: i128, end: i128) u64 {
    return @intCast(@max(@as(i128, 0), end - start));
}

pub fn nsToMs(value: u64) f64 {
    return @as(f64, @floatFromInt(value)) / std.time.ns_per_ms;
}
