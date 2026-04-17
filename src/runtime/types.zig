const std = @import("std");
const moon_quant = @import("../moon_quant.zig");

pub const primary_target = "Apple Silicon + Metal";
pub const fallback_target = "Apple Silicon CPU";
pub const native_architecture = "llama";
pub const supported_architecture = "llama, qwen2, qwen3, qwen3.5 (native CPU + Metal)";
pub const supported_model_family = "llama and qwen family GGUF models (including qwen3.5 variants) through the native CPU or Metal runtime";
pub const supported_quantization = moon_quant.supported_quantization;
pub const default_context_length: usize = 8192;

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

pub const SamplingStrategy = enum {
    auto,
    gpu_greedy,
    gpu_topk_sample,
    gpu_shortlist,
    cpu_full_logits,

    pub fn parse(name: []const u8) ?SamplingStrategy {
        if (std.mem.eql(u8, name, "auto")) return .auto;
        if (std.mem.eql(u8, name, "gpu-greedy")) return .gpu_greedy;
        if (std.mem.eql(u8, name, "gpu-topk-sample")) return .gpu_topk_sample;
        if (std.mem.eql(u8, name, "gpu-shortlist")) return .gpu_shortlist;
        if (std.mem.eql(u8, name, "cpu-full-logits")) return .cpu_full_logits;
        return null;
    }

    pub fn label(self: SamplingStrategy) []const u8 {
        return switch (self) {
            .auto => "auto",
            .gpu_greedy => "gpu-greedy",
            .gpu_topk_sample => "gpu-topk-sample",
            .gpu_shortlist => "gpu-shortlist",
            .cpu_full_logits => "cpu-full-logits",
        };
    }
};

pub const EffectiveSamplingPath = enum {
    cpu_logits,
    gpu_greedy_argmax,
    gpu_topk_sampler,
    gpu_shortlist_cpu_sampler,

    pub fn label(self: EffectiveSamplingPath) []const u8 {
        return switch (self) {
            .cpu_logits => "cpu-logits",
            .gpu_greedy_argmax => "gpu-greedy-argmax",
            .gpu_topk_sampler => "gpu-topk-sampler",
            .gpu_shortlist_cpu_sampler => "gpu-shortlist-cpu-sampler",
        };
    }
};

pub const ReadbackMode = enum {
    none,
    full_logits_f32,
    sampled_token_u32,
    shortlist_ids_scores,

    pub fn label(self: ReadbackMode) []const u8 {
        return switch (self) {
            .none => "none",
            .full_logits_f32 => "full-logits-f32",
            .sampled_token_u32 => "sampled-token-u32",
            .shortlist_ids_scores => "shortlist-ids-scores",
        };
    }
};

pub const GenerationOptions = struct {
    max_tokens: usize = 16,
    context_length: usize = default_context_length,
    seed: u64 = 0,
    temperature: f32 = 0.0,
    repeat_penalty: f32 = 1.0,
    top_k: usize = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
    backend: BackendPreference = .auto,
    moon_quant: MoonQuantMode = .enabled,
    metal_profile: bool = false,
    sampling_strategy: SamplingStrategy = .auto,
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
    sampling_strategy: SamplingStrategy = .auto,
    sampling_path: EffectiveSamplingPath = .cpu_logits,
    readback_mode: ReadbackMode = .none,
    startup_breakdown: StartupBreakdown = .{},
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

pub fn deltaNs(start: i128, end: i128) u64 {
    return @intCast(@max(@as(i128, 0), end - start));
}

pub fn nsToMs(value: u64) f64 {
    return @as(f64, @floatFromInt(value)) / std.time.ns_per_ms;
}

pub fn resolveSamplingPath(has_gpu_session: bool, temperature: f32, strategy: SamplingStrategy) EffectiveSamplingPath {
    if (!has_gpu_session) return .cpu_logits;
    return switch (strategy) {
        .cpu_full_logits => .cpu_logits,
        .gpu_greedy => if (temperature > 0) .cpu_logits else .gpu_greedy_argmax,
        // The stochastic GPU reduction path stays disabled until a hierarchical top-k reduction lands.
        .gpu_topk_sample => if (temperature > 0) .cpu_logits else .gpu_greedy_argmax,
        .gpu_shortlist => if (temperature > 0) .cpu_logits else .gpu_greedy_argmax,
        .auto => if (temperature > 0) .cpu_logits else .gpu_greedy_argmax,
    };
}

pub fn canUseGpuTopKSampling(options: GenerationOptions) bool {
    _ = options;
    return false;
}

pub fn readbackModeFor(backend: BackendUsed, sampling_path: EffectiveSamplingPath) ReadbackMode {
    if (backend != .metal) return .none;
    return switch (sampling_path) {
        .cpu_logits => .full_logits_f32,
        .gpu_greedy_argmax => .sampled_token_u32,
        .gpu_topk_sampler => .sampled_token_u32,
        .gpu_shortlist_cpu_sampler => .shortlist_ids_scores,
    };
}

test "sampling strategy parser accepts benchmark path values" {
    try std.testing.expectEqual(SamplingStrategy.auto, SamplingStrategy.parse("auto").?);
    try std.testing.expectEqual(SamplingStrategy.gpu_greedy, SamplingStrategy.parse("gpu-greedy").?);
    try std.testing.expectEqual(SamplingStrategy.gpu_topk_sample, SamplingStrategy.parse("gpu-topk-sample").?);
    try std.testing.expectEqual(SamplingStrategy.gpu_shortlist, SamplingStrategy.parse("gpu-shortlist").?);
    try std.testing.expectEqual(SamplingStrategy.cpu_full_logits, SamplingStrategy.parse("cpu-full-logits").?);
    try std.testing.expect(SamplingStrategy.parse("bogus") == null);
}

test "resolveSamplingPath keeps cpu backend on cpu logits and leaves gpu shortlist opt-in" {
    try std.testing.expectEqual(EffectiveSamplingPath.cpu_logits, resolveSamplingPath(false, 0, .auto));
    try std.testing.expectEqual(EffectiveSamplingPath.cpu_logits, resolveSamplingPath(true, 0.8, .gpu_greedy));
    try std.testing.expectEqual(EffectiveSamplingPath.cpu_logits, resolveSamplingPath(true, 0.8, .auto));
    try std.testing.expectEqual(EffectiveSamplingPath.cpu_logits, resolveSamplingPath(true, 0.8, .gpu_topk_sample));
    try std.testing.expectEqual(EffectiveSamplingPath.cpu_logits, resolveSamplingPath(true, 0.8, .gpu_shortlist));
    try std.testing.expectEqual(EffectiveSamplingPath.cpu_logits, resolveSamplingPath(true, 0, .cpu_full_logits));
    try std.testing.expectEqual(EffectiveSamplingPath.gpu_greedy_argmax, resolveSamplingPath(true, 0, .auto));
}

test "canUseGpuTopKSampling only accepts simple stochastic policy" {
    try std.testing.expect(!canUseGpuTopKSampling(.{ .temperature = 0.7 }));
    try std.testing.expect(!canUseGpuTopKSampling(.{ .temperature = 0.7, .top_k = 8 }));
    try std.testing.expect(!canUseGpuTopKSampling(.{ .temperature = 0.7, .repeat_penalty = 1.1 }));
    try std.testing.expect(!canUseGpuTopKSampling(.{ .temperature = 0.7, .top_p = 0.9 }));
    try std.testing.expect(!canUseGpuTopKSampling(.{ .temperature = 0.7, .min_p = 0.1 }));
    try std.testing.expect(!canUseGpuTopKSampling(.{ .temperature = 0.7, .top_k = 128 }));
}
