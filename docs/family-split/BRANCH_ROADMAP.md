# Family-Split Branch Roadmap

This roadmap defines the phases to split the current uniformly Llama-based runtime into a modular, extensible architecture where each model family (Llama, Qwen, Mistral, Gemma, etc.) has its own specialized runtime. The goal is to make adding new model families in the future straightforward while maintaining high-performance CPU and Metal backends for each.

## How To Use This File

Every task in this file is a Markdown checkbox.

When a task is completed:
- replace `- [ ]` with `- [x]`
- keep the item in place instead of deleting it
- only mark it complete when the code, docs, and validation for that item are actually done

If a task turns out to be too large, split it into smaller checklist items rather than leaving a vague partially done item behind.

## Architecture Overview

The new architecture follows an interface-driven design:

```
src/runtime/families/
├── mod.zig                 # Common interface/trait definitions
├── registry.zig            # Family registration and dispatch
├── llama/
│   ├── cpu.zig            # Llama-specific CPU inference
│   ├── metal.zig         # Llama-specific Metal kernels
│   └── runtime.zig        # Llama runtime orchestration
├── qwen/
│   ├── cpu.zig
│   ├── metal.zig
│   └── runtime.zig
├── mistral/               # Future
├── gemma/                 # Future
└── qwen35/               # Future (MoE/DeltaNet variants)
```

Each family module implements the common interface:
- `loadModel()` - Load GGUF file and validate architecture
- `tokenize()` - Convert text to tokens
- `forward()` - Run inference forward pass
- `generate()` - End-to-end generation

## Phase 1: Foundation - Interface & Registry

Goal: Establish the common interface and family registry system. No new family implementations yet.

### Task 1.1: Define Common Family Interface

- [ ] Create `src/runtime/families/mod.zig` with:
  - `FamilyRuntime` trait/interface defining required methods
  - `FamilyBackend` enum (cpu, metal, auto)
  - `FamilyGenerateOptions` struct
  - `FamilyGenerateReport` struct
  - Error types specific to family runtime

- [ ] Define `ModelFamily` enum with variants:
  - `.llama`
  - `.qwen`
  - `.qwen35` (for MoE/DeltaNet variants)
  - `.mistral`
  - `.gemma`
  - `.custom([]const u8)` (for unknown architectures)

- [ ] Define `FamilyCapabilities` struct:
  - `supports_cpu: bool`
  - `supports_metal: bool`
  - `supported_quantizations: []const QuantType`
  - `max_context_length: usize`

### Task 1.2: Create Family Registry

- [ ] Create `src/runtime/families/registry.zig`:
  - `FamilyRegistry` struct mapping `ModelFamily` -> family handlers
  - `register(family: ModelFamily, handler: FamilyHandler)` function
  - `detectFamily(gguf_report: InspectReport) ModelFamily` function
  - `getRuntimeForFamily(family: ModelFamily) ?FamilyRuntime` function
  - Initialize registry with built-in families at startup

- [ ] Implement automatic family detection from GGUF architecture field:
  - "llama" -> `.llama`
  - "qwen2" -> `.qwen`
  - "qwen2_moe" or "qwen3" -> `.qwen35`
  - "mistral" -> `.mistral`
  - "gemma" -> `.gemma`
  - unknown -> `.custom(architecture_string)`

- [ ] Add registry lookup in `runtime/mod.zig`:
  - Replace direct `llama_runtime.generate()` call
  - Route to appropriate family runtime based on detected architecture

### Task 1.3: Refactor Existing Runtime Entry Point

- [ ] Update `src/runtime/mod.zig` `generate()` function:
  - Inspect GGUF to detect architecture
  - Use registry to dispatch to correct family runtime
  - Maintain backward compatibility for existing llama/qwen models

- [ ] Update `runCommand()` and `benchCommand()` to use dispatch

- [ ] Add comprehensive tests:
  - Test registry detects llama correctly
  - Test registry detects qwen correctly
  - Test unknown architecture returns custom variant
  - Test dispatch routes to correct runtime

Definition of done for Phase 1:

- [ ] Registry correctly dispatches to llama runtime for "llama" architecture
- [ ] Registry correctly dispatches to qwen runtime for "qwen2" architecture
- [ ] New families can be registered via registry without modifying core code
- [ ] All existing tests pass with the dispatch layer in place

## Phase 2: Extract Llama Family

Goal: Refactor current llama-specific code into dedicated family module while maintaining identical behavior.

### Task 2.1: Create Llama Family Module Structure

- [ ] Create `src/runtime/families/llama/mod.zig`:
  - Re-export all llama-specific types and functions
  - Implement `FamilyRuntime` interface
  - Define capabilities: CPU: true, Metal: true

- [ ] Create `src/runtime/families/llama/cpu.zig`:
  - Move CPU inference logic from `llama_cpu.zig`
  - Maintain identical API and behavior
  - Update imports

- [ ] Create `src/runtime/families/llama/metal.zig`:
  - Move Metal kernels from `llama_metal.zig`, `metal_backend.zig`
  - Maintain identical API and behavior

- [ ] Create `src/runtime/families/llama/runtime.zig`:
  - Move orchestration from `llama_runtime.zig`
  - Handle CPU/Metal backend selection
  - Implement interface methods

### Task 2.2: Update Registry and Imports

- [ ] Register llama family in `registry.zig`:
  - Add `.llama -> LlamaFamilyHandler` mapping
  - Point to new family module

- [ ] Update all internal imports:
  - `chat_runtime.zig` references
  - `resident_runtime.zig` references
  - `server_runtime.zig` references

- [ ] Ensure no breaking changes to external API

### Task 2.3: Validation

- [ ] Run existing tests to ensure identical behavior:
  - `zig build test` passes
  - CPU inference produces same outputs
  - Metal inference produces same outputs

- [ ] Benchmark to ensure no performance regression:
  - Startup time unchanged
  - Prompt processing time unchanged
  - Decode tok/s unchanged

Definition of done for Phase 2:

- [ ] All llama-specific code lives under `src/runtime/families/llama/`
- [ ] Llama family implements full `FamilyRuntime` interface
- [ ] All existing functionality preserved
- [ ] Zero test regressions

## Phase 3: Extract Qwen Family

Goal: Create dedicated qwen family module with full CPU + Metal support.

### Task 3.1: Create Qwen Family Module Structure

- [ ] Create `src/runtime/families/qwen/mod.zig`:
  - Implement `FamilyRuntime` interface
  - Define capabilities: CPU: true, Metal: true
  - Support both dense (Qwen 2, 2.5) and initial MoE variants

- [ ] Create `src/runtime/families/qwen/cpu.zig`:
  - Adapt from current qwen handling in `llama_cpu.zig`:
    - RoPE style: `.neox` (vs llama's `.interleaved`)
    - tokenizer differences
  - Handle quantization spread differences

- [ ] Create `src/runtime/families/qwen/metal.zig`:
  - Implement Metal kernels for qwen dense models
  - Start with base q4_k, q6_k support (matching llama)

- [ ] Create `src/runtime/families/qwen/runtime.zig`:
  - Orchestrate CPU/Metal backend selection
  - Implement interface methods

### Task 3.2: Extend Qwen35 Sub-Family (MoE / DeltaNet)

- [ ] Create `src/runtime/families/qwen35/mod.zig`:
  - Separate from dense qwen for clarity
  - Capabilities: CPU: partial, Metal: partial (start minimal)

- [ ] Implement MoE support:
  - Parse `ffn_gate_inp` router weights
  - Parse `ffn_down.N`, `ffn_up.N` expert arrays
  - Implement CPU MoE routing
  - Implement minimal Metal MoE kernel (top-k routing)

- [ ] Document limitations:
  - DeltaNet (linear attention) deferred
  - Only dense qwen35 models initially

### Task 3.3: Validation

- [ ] Test qwen2 models with CPU backend:
  - Verify tokenization correct
  - Verify generation matches reference (llama.cpp)

- [ ] Test qwen2 models with Metal backend:
  - Verify correctness vs CPU
  - Measure tok/s performance

- [ ] Test qwen35 dense models work (should share with qwen)

Definition of done for Phase 3:

- [ ] Qwen family fully implemented under `src/runtime/families/qwen/`
- [ ] Qwen35 family skeleton created for MoE
- [ ] Qwen models run on both CPU and Metal backends
- [ ] Zero regressions for existing qwen support

## Phase 4: Add Mistral Family

Goal: Implement mistral family with CPU + Metal support.

### Task 4.1: Mistral Family Implementation

- [ ] Create `src/runtime/families/mistral/mod.zig`:
  - Implement `FamilyRuntime` interface
  - Capabilities: CPU: true, Metal: true

- [ ] Create `src/runtime/families/mistral/cpu.zig`:
  - Mistral uses Sliding Window Attention (SWA)
  - RoPE style: similar to llama but with sliding window
  - Different tokenizer handling

- [ ] Create `src/runtime/families/mistral/metal.zig`:
  - Implement SWA Metal kernel (masked attention)
  - Support q4_k, q6_k quantization

- [ ] Create `src/runtime/families/mistral/runtime.zig`:
  - Orchestrate backends
  - Implement interface methods

### Task 4.2: Validation

- [ ] Test mistral-7b models (CPU):
  - Verify SWA behavior
  - Verify generation correctness

- [ ] Test mistral-7b models (Metal):
  - Verify correctness vs CPU
  - Measure performance

- [ ] Test Mixtral variants if available:
  - MoE handling (similar to qwen35 MoE approach)

Definition of done for Phase 4:

- [ ] Mistral family fully implemented
- [ ] Sliding Window Attention works on both backends
- [ ] Mixtral MoE support initiated

## Phase 5: Add Gemma Family

Goal: Implement gemma family with CPU + Metal support.

### Task 5.1: Gemma Family Implementation

- [ ] Create `src/runtime/families/gemma/mod.zig`:
  - Implement `FamilyRuntime` interface
  - Capabilities: CPU: true, Metal: true

- [ ] Create `src/runtime/families/gemma/cpu.zig`:
  - Gemma uses Gemma tokenizer (different from llama)
  - RoPE: similar to llama
  - Attention: standard (no sliding window)

- [ ] Create `src/runtime/families/gemma/metal.zig`:
  - Standard attention kernel
  - Support q4_k, q6_k quantization

- [ ] Create `src/runtime/families/gemma/runtime.zig`:
  - Orchestrate backends
  - Implement interface methods

### Task 5.2: Validation

- [ ] Test gemma-2b models (CPU):
  - Verify tokenizer
  - Verify generation correctness

- [ ] Test gemma-2b models (Metal):
  - Verify correctness vs CPU
  - Measure performance

Definition of done for Phase 5:

- [ ] Gemma family fully implemented
- [ ] Gemma tokenizer properly handled
- [ ] Both backends functional

## Phase 6: Extensibility Infrastructure

Goal: Make adding future families trivial.

### Task 6.1: Family Template/Scaffold

- [ ] Create `src/runtime/families/template/`:
  - Template family module with all required files
  - Comprehensive comments explaining what to implement
  - Ready-to-copy structure for new families

- [ ] Document the interface in `docs/FAMILY_IMPLEMENTATION_GUIDE.md`:
  - Step-by-step guide for adding new families
  - Example: "How to add Phi family"
  - Common pitfalls and how to avoid them

### Task 6.2: Automated Testing Infrastructure

- [ ] Create family-agnostic test suite:
  - Test interface compliance
  - Test dispatch mechanism
  - Test registry behavior

- [ ] Add golden output tests per family:
  - Each family has reference outputs
  - Verify CPU matches golden
  - Verify Metal matches CPU

### Task 6.3: Documentation

- [ ] Update README with supported families matrix
- [ ] Document architecture decision in `docs/`
- [ ] Add family-specific notes (e.g., Qwen3.5 MoE)

Definition of done for Phase 6:

- [ ] New family can be added in < 1 hour following template
- [ ] Clear documentation for future contributors
- [ ] Automated testing catches family-related regressions

## Deferred / Future Work

- [ ] DeltaNet (linear attention) for Qwen 3.5 sparse models
- [ ] Phi family support
- [ ] Stable Diffusion / vision models (separate runtime)
- [ ] Multi-modal support (Vision + Language)
- [ ] Linux GPU support (beyond Apple Silicon)

These are intentionally deferred. They should only move into active milestones after the family-split architecture is stable and tested across all initial families.

## Migration Notes

During implementation, maintain backward compatibility:
1. All existing commands continue to work
2. No breaking changes to CLI interface
3. Registry dispatch is transparent to users
4. Existing tests continue passing

When adding new families, follow the interface contract strictly to ensure the registry can correctly dispatch to the appropriate runtime.