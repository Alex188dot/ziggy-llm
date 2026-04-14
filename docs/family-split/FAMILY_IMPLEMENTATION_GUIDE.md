# Family Implementation Guide

This guide explains how to add a new model family to the ziggy-llm runtime.

## Overview

Each model family is implemented as a module under `src/runtime/families/<family_name>/`. The family module provides a `FamilyHandler` that the registry uses to dispatch model loading and inference to the correct runtime.

## Directory Structure

```
src/runtime/families/
├── mod.zig              # Common interface types and detectModelFamily()
├── registry.zig         # FamilyRegistry for dispatch
├── llama/              # Llama family implementation
├── qwen/               # Qwen family implementation
└── template/           # Template for adding new families
```

## Steps to Add a New Family

### 1. Copy the Template

```bash
cp -r src/runtime/families/template src/runtime/families/<your_family>
```

### 2. Implement the Family Runtime

Edit `src/runtime/families/<your_family>/runtime.zig`:

```zig
fn yourFamilyGenerate(
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    options: families_mod.FamilyGenerateOptions,
) !families_mod.FamilyReport {
    // Convert FamilyGenerateOptions to types.GenerationOptions
    const gen_opts = types.GenerationOptions{
        .max_tokens = options.max_tokens,
        .context_length = options.context_length,
        .seed = options.seed,
        .temperature = options.temperature,
        .repeat_penalty = options.repeat_penalty,
        .top_k = options.top_k,
        .top_p = options.top_p,
        .min_p = options.min_p,
        .backend = @enumFromInt(@intFromEnum(options.backend)),
        .moon_quant = options.moon_quant,
        .metal_profile = options.metal_profile,
        .sampling_strategy = options.sampling_strategy,
    };

    // Call your family-specific runtime
    const report = try your_runtime_generate(allocator, model_path, prompt, gen_opts);

    // Convert FamilyReport back
    return families_mod.FamilyReport{
        .generated_text = report.generated_text,
        // ... map other fields
    };
}
```

### 3. Update mod.zig

Edit `src/runtime/families/<your_family>/mod.zig`:

```zig
pub const runtime = @import("runtime.zig");
pub const FamilyHandler = runtime.createYourFamilyHandler();
```

### 4. Register the Family

In `src/runtime/mod.zig`:

```zig
const your_family = @import("families/<your_family>/mod.zig");

fn getRegistry() *registry_mod.FamilyRegistry {
    const reg = registry_mod.getGlobalRegistry();
    if (reg.count == 0) {
        reg.register(llama_family.FamilyHandler) catch unreachable;
        reg.register(qwen_family.FamilyHandler) catch unreachable;
        reg.register(your_family.FamilyHandler) catch unreachable;  // Add this
    }
    return reg;
}
```

### 5. Add Family Detection

In `src/runtime/families/mod.zig`, update `detectModelFamily()`:

```zig
pub fn detectModelFamily(architecture: []const u8) ModelFamily {
    if (std.mem.eql(u8, architecture, "llama")) {
        return .llama;
    }
    if (std.mem.eql(u8, architecture, "qwen2") or std.mem.eql(u8, architecture, "qwen3")) {
        return .qwen;
    }
    // Add your family detection:
    if (std.mem.eql(u8, architecture, "your_arch_name")) {
        return .your_family;
    }
    // ...
}
```

Also update the `ModelFamily` enum:

```zig
pub const ModelFamily = union(enum) {
    llama,
    qwen,
    qwen35,
    mistral,
    gemma,
    your_family,  // Add this
    custom: []const u8,
};
```

## Interface Reference

### FamilyRuntime

```zig
pub const FamilyRuntime = struct {
    ctx: ?*anyopaque,           // Runtime context (state)
    generate_fn: *const fn (
        ?*anyopaque,             // ctx
        std.mem.Allocator,       // allocator
        []const u8,              // model_path
        []const u8,              // prompt
        FamilyGenerateOptions     // generation options
    ) anyerror!FamilyReport,
    deinit_fn: *const fn (?*anyopaque) void,
};
```

### FamilyHandler

```zig
pub const FamilyHandler = struct {
    family: ModelFamily,                    // Family identifier
    capabilities: FamilyCapabilities,       // What the family supports
    runtime: FamilyRuntime,                 // The runtime implementation
};
```

### FamilyCapabilities

```zig
pub const FamilyCapabilities = struct {
    supports_cpu: bool,                    // CPU backend supported
    supports_metal: bool,                  // Metal backend supported
    supported_quant_types: []const gguf.TensorType,  // Quantization types
    max_context_length: usize,             // Maximum context length
};
```

### FamilyGenerateOptions

```zig
pub const FamilyGenerateOptions = struct {
    max_tokens: usize,
    context_length: usize,
    seed: u64,
    temperature: f32,
    repeat_penalty: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    backend: FamilyBackendPreference,
    moon_quant: MoonQuantMode,
    metal_profile: bool,
    sampling_strategy: SamplingStrategy,
};
```

### FamilyReport

```zig
pub const FamilyReport = struct {
    generated_text: []u8,
    prompt_token_count: usize,
    reused_prompt_token_count: usize,
    generated_token_count: usize,
    startup_ns: u64,
    prompt_ns: u64,
    ttft_ns: u64,
    decode_ns: u64,
    seed: u64,
    temperature: f32,
    backend: BackendUsed,
    sampling_strategy: SamplingStrategy,
    sampling_path: EffectiveSamplingPath,
    readback_mode: ReadbackMode,
    startup_breakdown: StartupBreakdown,
    metal_profile_summary: ?[]u8,
};
```

## Common Patterns

### Wrapping an Existing Runtime

If your family uses the same underlying implementation (like Qwen uses Llama's runtime):

```zig
fn yourFamilyGenerate(...) !FamilyReport {
    // Simply call the existing runtime
    const report = try llama_runtime.generate(allocator, model_path, prompt, gen_opts);
    return families_mod.FamilyReport{
        .generated_text = report.generated_text,
        // ... map all fields
    };
}
```

### Implementing a New Runtime

For families with different architectures (Mistral's Sliding Window, Gemma's tokenizer, etc.):

1. Implement the model loading in `cpu.zig`
2. Implement Metal kernels in `metal.zig` (if applicable)
3. Implement the forward pass with family-specific attention
4. Implement tokenization (may differ from Llama)

## Testing

After implementing a new family:

1. Build: `zig build`
2. Test CPU: `ziggy-llm run -m <model.gguf> -p "Hello" --backend cpu`
3. Test Metal: `ziggy-llm run -m <model.gguf> -p "Hello" --backend metal` (on Apple Silicon)
4. Run test suite: `zig build test`

## Notes

- The registry is a singleton - families are registered once on first use
- Family detection happens via GGUF `general.architecture` field
- The `custom` variant of `ModelFamily` handles unknown architectures
- MoonQuant support is per-family (some families may not support it)