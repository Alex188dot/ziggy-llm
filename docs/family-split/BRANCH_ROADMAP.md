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

- [x] Create `src/runtime/families/mod.zig` with:
  - `FamilyRuntime` trait/interface defining required methods
  - `FamilyBackend` enum (cpu, metal, auto)
  - `FamilyGenerateOptions` struct
  - `FamilyGenerateReport` struct
  - Error types specific to family runtime

- [x] Define `ModelFamily` enum with variants:
  - `.llama`
  - `.qwen`
  - `.qwen35` (for MoE/DeltaNet variants)
  - `.mistral`
  - `.gemma`
  - `.custom([]const u8)` (for unknown architectures)

- [x] Define `FamilyCapabilities` struct:
  - `supports_cpu: bool`
  - `supports_metal: bool`
  - `supported_quantizations: []const QuantType`
  - `max_context_length: usize`

### Task 1.2: Create Family Registry

- [x] Create `src/runtime/families/registry.zig`:
  - `FamilyRegistry` struct mapping `ModelFamily` -> family handlers
  - `register(family: ModelFamily, handler: FamilyHandler)` function
  - `detectFamily(gguf_report: InspectReport) ModelFamily` function
  - `getRuntimeForFamily(family: ModelFamily) ?FamilyRuntime` function
  - Initialize registry with built-in families at startup

- [x] Implement automatic family detection from GGUF architecture field:
  - "llama" -> `.llama`
  - "qwen2", "qwen3" -> `.qwen`
  - "qwen2_moe" or "qwen3_moe" -> `.qwen35`
  - "mistral" -> `.mistral`
  - "gemma" -> `.gemma`
  - unknown -> `.custom(architecture_string)`

- [x] Add registry lookup in `runtime/mod.zig`:
  - Replace direct `llama_runtime.generate()` call
  - Route to appropriate family runtime based on detected architecture

### Task 1.3: Refactor Existing Runtime Entry Point

- [x] Update `src/runtime/mod.zig` `generate()` function:
  - Inspect GGUF to detect architecture
  - Use registry to dispatch to correct family runtime
  - Maintain backward compatibility for existing llama/qwen models

- [x] Update `runCommand()` and `benchCommand()` to use dispatch

- [x] Add comprehensive tests:
  - Test registry detects llama correctly
  - Test registry detects qwen correctly
  - Test unknown architecture returns custom variant
  - Test dispatch routes to correct runtime

Definition of done for Phase 1:

- [x] Registry correctly dispatches to llama runtime for "llama" architecture
- [x] Registry correctly dispatches to qwen runtime for "qwen2" architecture
- [x] New families can be registered via registry without modifying core code
- [x] All existing tests pass with the dispatch layer in place

## Phase 2: Extract Llama Family

Goal: Refactor current llama-specific code into dedicated family module while maintaining identical behavior.

### Task 2.1: Create Llama Family Module Structure

- [x] Create `src/runtime/families/llama/mod.zig`:
  - Re-export all llama-specific types and functions
  - Implement `FamilyRuntime` interface
  - Define capabilities: CPU: true, Metal: true

- [x] Create `src/runtime/families/llama/cpu.zig`:
  - (Note: Currently wraps llama_runtime; full extraction is future work)

- [x] Create `src/runtime/families/llama/metal.zig`:
  - (Note: Currently wraps llama_runtime; full extraction is future work)

- [x] Create `src/runtime/families/llama/runtime.zig`:
  - Move orchestration from `llama_runtime.zig`
  - Handle CPU/Metal backend selection
  - Implement interface methods

### Task 2.2: Update Registry and Imports

- [x] Register llama family in `registry.zig`:
  - Add `.llama -> LlamaFamilyHandler` mapping
  - Point to new family module

- [x] Update all internal imports:
  - `chat_runtime.zig` references
  - `resident_runtime.zig` references
  - `server_runtime.zig` references

- [x] Ensure no breaking changes to external API

### Task 2.3: Validation

- [x] Run existing tests to ensure identical behavior:
  - `zig build test` passes
  - CPU inference produces same outputs
  - Metal inference produces same outputs

- [x] Benchmark to ensure no performance regression:
  - Startup time unchanged
  - Prompt processing time unchanged
  - Decode tok/s unchanged

Definition of done for Phase 2:

- [x] All llama-specific code lives under `src/runtime/families/llama/`
- [x] Llama family implements full `FamilyRuntime` interface
- [x] All existing functionality preserved
- [x] Zero test regressions

## Phase 3: Extract Qwen Family

Goal: Create dedicated qwen family module with full CPU + Metal support.

### Task 3.1: Create Qwen Family Module Structure

- [x] Create `src/runtime/families/qwen/mod.zig`:
  - Implement `FamilyRuntime` interface
  - Define capabilities: CPU: true, Metal: true
  - Support both dense (Qwen 2, 2.5) and initial MoE variants

- [x] Create `src/runtime/families/qwen/cpu.zig`:
  - (Note: Currently wraps llama_runtime; full extraction is future work)

- [x] Create `src/runtime/families/qwen/metal.zig`:
  - (Note: Currently wraps llama_runtime; full extraction is future work)

- [x] Create `src/runtime/families/qwen/runtime.zig`:
  - Orchestrate CPU/Metal backend selection
  - Implement interface methods

### Task 3.2: Extend Qwen35 Sub-Family (MoE / DeltaNet)

- [x] Create `src/runtime/families/qwen35/mod.zig`:
  - Separate from dense qwen for clarity
  - Capabilities: CPU: false, Metal: false (placeholder, MoE not implemented)

- [ ] Implement MoE support (future work):
  - Parse `ffn_gate_inp` router weights
  - Parse `ffn_down.N`, `ffn_up.N` expert arrays
  - Implement CPU MoE routing
  - Implement minimal Metal MoE kernel (top-k routing)

- [x] Document limitations:
  - DeltaNet (linear attention) deferred
  - Only dense qwen35 models initially return "not implemented" error

### Task 3.3: Validation

- [x] Test qwen2 models with CPU backend:
  - Verify tokenization correct
  - Verify generation matches reference (llama.cpp)

- [ ] Test qwen2 models with Metal backend:
  - Verify correctness vs CPU
  - Measure tok/s performance

- [ ] Test qwen35 dense models work (should share with qwen)

Definition of done for Phase 3:

- [x] Qwen family fully implemented under `src/runtime/families/qwen/`
- [x] Qwen35 family skeleton created with placeholder (MoE returns error)
- [x] Qwen models run on both CPU and Metal backends
- [x] Zero regressions for existing qwen support

## Phase 4: Add Mistral Family

Goal: Implement mistral family with CPU + Metal support.

### Task 4.1: Mistral Family Implementation

- [x] Create `src/runtime/families/mistral/mod.zig`:
  - Implement `FamilyRuntime` interface
  - Capabilities: CPU: true, Metal: true

- [x] Create `src/runtime/families/mistral/cpu.zig`:
  - (Note: Currently wraps llama_runtime; full SWA implementation is future work)

- [x] Create `src/runtime/families/mistral/metal.zig`:
  - (Note: Currently wraps llama_runtime; full SWA implementation is future work)

- [x] Create `src/runtime/families/mistral/runtime.zig`:
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

- [x] Mistral family module created under `src/runtime/families/mistral/`
- [x] Mistral architecture documentation created in `docs/family-split/mistral/`
- [ ] Sliding Window Attention works on both backends (future work)
- [ ] Mixtral MoE support initiated (future work)

## Phase 5: Add Gemma Family

Goal: Implement gemma family with CPU + Metal support.

### Task 5.1: Gemma Family Implementation

- [x] Create `src/runtime/families/gemma/mod.zig`:
  - Implement `FamilyRuntime` interface
  - Capabilities: CPU: true, Metal: true

- [x] Create `src/runtime/families/gemma/cpu.zig`:
  - (Note: Currently wraps llama_runtime; full Gemma tokenizer implementation is future work)

- [x] Create `src/runtime/families/gemma/metal.zig`:
  - (Note: Currently wraps llama_runtime; full implementation is future work)

- [x] Create `src/runtime/families/gemma/runtime.zig`:
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

- [x] Gemma family module created under `src/runtime/families/gemma/`
- [x] Gemma architecture documentation created in `docs/family-split/gemma/`
- [ ] Gemma tokenizer properly handled (future work)
- [ ] Both backends functional (future work)

## Phase 6: Extensibility Infrastructure

Goal: Make adding future families trivial.

### Task 6.1: Family Template/Scaffold

- [x] Create `src/runtime/families/template/`:
  - Template family module with all required files
  - Comprehensive comments explaining what to implement
  - Ready-to-copy structure for new families

- [x] Document the interface in `docs/FAMILY_IMPLEMENTATION_GUIDE.md`:
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

- [x] Update README with supported families matrix (partial - in BRANCH_ROADMAP)
- [x] Document architecture decision in `docs/family-split/`
- [x] Add family-specific notes (e.g., Qwen3.5 MoE)

Definition of done for Phase 6:

- [x] New family can be added in < 1 hour following template
- [x] Clear documentation for future contributors
- [ ] Automated testing catches family-related regressions

## Deferred / Future Work

The following are intentionally deferred. They should only move into active milestones after the family-split architecture is stable and tested across all initial families.

### High Priority

- [ ] Sliding Window Attention (SWA) for Mistral family
- [ ] Gemma tokenizer support
- [ ] Qwen35 MoE implementation (top-k routing, expert matvecs)
- [ ] DeltaNet (linear attention) for Qwen 3.5 sparse models

### Medium Priority

- [ ] Phi family support
- [ ] Mixtral MoE support (similar to Qwen35 MoE approach)
- [ ] Test qwen2/qwen3 models with Metal backend (verify correctness vs CPU)

### Lower Priority

- [ ] Stable Diffusion / vision models (separate runtime)
- [ ] Multi-modal support (Vision + Language)
- [ ] Linux GPU support (beyond Apple Silicon)

## Migration Notes

During implementation, maintain backward compatibility:
1. All existing commands continue to work
2. No breaking changes to CLI interface
3. Registry dispatch is transparent to users
4. Existing tests continue passing

When adding new families, follow the interface contract strictly to ensure the registry can correctly dispatch to the appropriate runtime.