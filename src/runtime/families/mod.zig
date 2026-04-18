const std = @import("std");
const gguf = @import("../../gguf.zig");
const moon_quant = @import("../../moon_quant.zig");
const types = @import("../types.zig");

pub const supported_quantization = moon_quant.supported_quantization;

pub const ModelFamily = union(enum) {
    llama,
    qwen,
    qwen35,
    qwen35_text,
    mistral,
    mistral3_2512,
    gemma,
    custom: []const u8,

    pub fn label(self: ModelFamily) []const u8 {
        return switch (self) {
            .llama => "llama",
            .qwen => "qwen",
            .qwen35 => "qwen35",
            .qwen35_text => "qwen35_text",
            .mistral => "mistral",
            .mistral3_2512 => "ministral3_2512",
            .gemma => "gemma",
            .custom => |s| s,
        };
    }
};

pub const FamilyCapabilities = struct {
    supports_cpu: bool,
    supports_metal: bool,
    supported_quant_types: []const gguf.TensorType,
    max_context_length: usize,
};

pub const FamilyBackendPreference = enum {
    auto,
    cpu,
    metal,
};

pub const FamilyError = error{
    UnsupportedFamily,
    UnsupportedBackend,
    UnsupportedArchitecture,
    ModelLoadFailed,
    GenerationFailed,
    TokenizationFailed,
};

pub const FamilyGenerateOptions = struct {
    max_tokens: usize = 16,
    context_length: usize = 8192,
    seed: u64 = 0,
    temperature: f32 = 0.0,
    repeat_penalty: f32 = 1.0,
    top_k: usize = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
    backend: FamilyBackendPreference = .auto,
    moon_quant: types.MoonQuantMode = .enabled,
    metal_profile: bool = false,
    sampling_strategy: types.SamplingStrategy = .auto,
    exp_block_decode: bool = false,
    exp_block_k: usize = 2,
    exp_block_confidence_margin: f32 = 0.75,
    exp_block_cooldown_tokens: usize = 8,
};

pub const FamilyReport = struct {
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
    sampling_strategy: types.SamplingStrategy = .auto,
    sampling_path: types.EffectiveSamplingPath = .cpu_logits,
    readback_mode: types.ReadbackMode = .none,
    startup_breakdown: StartupBreakdown = .{},
    metal_profile_summary: ?[]u8 = null,
    exp_block_decode: bool = false,
    exp_block_k: usize = 0,
    exp_block_confidence_margin: f32 = 0.75,
    exp_block_cooldown_tokens: usize = 8,
    block_accepted_prefix_len: f64 = 0,
    block_rollback_count: usize = 0,
    block_confidence_gated_count: usize = 0,
    block_cooldown_active_count: usize = 0,
    block_verify_ns: u64 = 0,
    block_gpu_backup_ns: u64 = 0,
    block_gpu_restore_ns: u64 = 0,
    block_gpu_sequence_commits: usize = 0,
    block_gpu_fallback_count: usize = 0,

    pub fn deinit(self: *FamilyReport, allocator: std.mem.Allocator) void {
        allocator.free(self.generated_text);
        if (self.metal_profile_summary) |summary| allocator.free(summary);
        self.* = undefined;
    }

    pub fn decodeTokensPerSecond(self: FamilyReport) f64 {
        if (self.generated_token_count == 0 or self.decode_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.generated_token_count)) / (@as(f64, @floatFromInt(self.decode_ns)) / std.time.ns_per_s);
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

pub const StartupBreakdown = types.StartupBreakdown;

pub const FamilyRuntime = struct {
    ctx: ?*anyopaque,
    generate_fn: *const fn (?*anyopaque, std.mem.Allocator, []const u8, []const u8, FamilyGenerateOptions) anyerror!FamilyReport,
    deinit_fn: *const fn (?*anyopaque) void,

    pub fn generate(self: FamilyRuntime, allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8, options: FamilyGenerateOptions) anyerror!FamilyReport {
        return try self.generate_fn(self.ctx, allocator, model_path, prompt, options);
    }

    pub fn deinit(self: FamilyRuntime) void {
        self.deinit_fn(self.ctx);
    }
};

pub const FamilyHandler = struct {
    family: ModelFamily,
    capabilities: FamilyCapabilities,
    runtime: FamilyRuntime,
};

pub fn detectModelFamily(architecture: []const u8) ModelFamily {
    if (std.mem.eql(u8, architecture, "llama")) {
        return .llama;
    }
    if (std.mem.eql(u8, architecture, "qwen2") or std.mem.eql(u8, architecture, "qwen3")) {
        return .qwen;
    }
    if (std.mem.startsWith(u8, architecture, "qwen2_moe") or std.mem.startsWith(u8, architecture, "qwen3_moe")) {
        return .qwen35;
    }
    if (std.mem.eql(u8, architecture, "qwen35")) {
        return .qwen35_text;
    }
    if (std.mem.eql(u8, architecture, "qwen3_5_text")) {
        return .qwen35_text;
    }
    if (std.mem.eql(u8, architecture, "mistral3") or std.mem.eql(u8, architecture, "ministral3")) {
        return .mistral3_2512;
    }
    if (std.mem.startsWith(u8, architecture, "mistral")) {
        return .mistral;
    }
    if (std.mem.eql(u8, architecture, "gemma")) {
        return .gemma;
    }
    return ModelFamily{ .custom = architecture };
}

test "detectModelFamily recognizes llama" {
    try std.testing.expectEqual(ModelFamily.llama, detectModelFamily("llama"));
}

test "detectModelFamily recognizes qwen2 and qwen3" {
    try std.testing.expectEqual(ModelFamily.qwen, detectModelFamily("qwen2"));
    try std.testing.expectEqual(ModelFamily.qwen, detectModelFamily("qwen3"));
}

test "detectModelFamily recognizes qwen35 moe variants" {
    try std.testing.expectEqual(ModelFamily.qwen35, detectModelFamily("qwen2_moe"));
    try std.testing.expectEqual(ModelFamily.qwen35, detectModelFamily("qwen3_moe"));
}

test "detectModelFamily recognizes qwen35_text dense variant" {
    try std.testing.expectEqual(ModelFamily.qwen35_text, detectModelFamily("qwen3_5_text"));
    try std.testing.expectEqual(ModelFamily.qwen35_text, detectModelFamily("qwen35"));
}

test "detectModelFamily recognizes mistral" {
    try std.testing.expectEqual(ModelFamily.mistral, detectModelFamily("mistral"));
}

test "detectModelFamily recognizes mistral3_2512" {
    try std.testing.expectEqual(ModelFamily.mistral3_2512, detectModelFamily("mistral3"));
    try std.testing.expectEqual(ModelFamily.mistral3_2512, detectModelFamily("ministral3"));
}

test "detectModelFamily recognizes gemma" {
    try std.testing.expectEqual(ModelFamily.gemma, detectModelFamily("gemma"));
}

test "detectModelFamily falls back to custom for unknown architectures" {
    const custom = detectModelFamily("phi3");
    try std.testing.expect(std.mem.eql(u8, "phi3", custom.custom));
}
