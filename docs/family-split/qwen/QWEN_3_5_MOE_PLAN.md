# Qwen 3.5 MoE Implementation Plan

This document defines a staged implementation plan for Qwen 3.5 MoE support in ziggy-llm.

## Goal

Implement end-to-end support for Qwen 3.5 MoE GGUF models with a practical initial quantization matrix:

- `Q3_K`
- `IQ3_XXS`
- `IQ4_XS`

Optional later work:

- KV cache quantization

## Scope

This plan covers:

- family detection and routing
- GGUF metadata and tensor parsing for Qwen 3.5 MoE
- CPU runtime support
- Metal/runtime upload support for initial quantization targets
- validation and documentation

This plan does not assume support for all GGUF quantization formats. It intentionally prioritizes a narrow, shippable path before broader quant coverage.

## Current State

Current code already contains partial family scaffolding for Qwen 3.5:

- `src/runtime/families/qwen35/runtime.zig` exists but returns `Qwen35MoENotYetImplemented`
- `src/runtime/families/mod.zig` routes older `qwen2_moe` / `qwen3_moe` names, but not `qwen35moe`
- `src/model/loader.zig` is structured around dense FFN tensors and does not yet parse MoE expert arrays
- dequantization currently supports `q8_0`, `q4_k`, `q5_k`, and `q6_k`
- `Q3_K` has row-size metadata, but not runtime dequant support
- Metal tensor upload currently preserves raw quantized bytes only for `q4_k`, `q6_k`, and `q8_0`

## Design Principles

- land the smallest complete vertical slice first
- prefer explicit model-family handling over generic special cases
- keep MoE parsing separate from dense Qwen 3.5 logic
- add quantization support in usefulness order, not enum order
- avoid widening the supported quant matrix until the model path is correct

## Phase 1: Family Routing

### Goal

Route Qwen 3.5 MoE GGUFs into the correct family handler.

### Files

- `src/runtime/families/mod.zig`

### Tasks

- [x] update `detectModelFamily()` to recognize `qwen35moe`
- [x] keep support for existing `qwen2_moe` / `qwen3_moe` aliases if they are still encountered
- [x] add tests for:
  - [x] `qwen35moe`
  - [x] `qwen2_moe`
  - [x] `qwen3_moe`

### Acceptance Criteria

- `inspect` identifies the architecture and routes it to `.qwen35`
- runtime dispatch reaches the Qwen 3.5 MoE family instead of falling through to unsupported architecture handling

## Phase 2: Model Representation for MoE

### Goal

Extend the model loader to represent Qwen 3.5 MoE layer structure explicitly.

### Files

- `src/model/loader.zig`

### Tasks

- [x] define MoE-specific layer structures in the model representation
- [x] add metadata fields required for MoE, including:
  - [x] expert feed-forward length
  - [x] expert count
  - [x] experts-per-token / active expert count if present in GGUF metadata
  - [x] gating function metadata if present
- [x] add parsing for MoE tensors such as:
  - [x] router / gate input weights
  - [x] expert `ffn_up` tensors
  - [x] expert `ffn_down` tensors
  - [x] any shared-expert tensors if present in the target GGUF layout
- [x] preserve existing dense Qwen 3.5 handling without regression

### Notes

The current layer loader expects a single dense FFN triplet:

- `ffn_gate.weight`
- `ffn_up.weight`
- `ffn_down.weight`

Qwen 3.5 MoE requires a different representation:

- a router tensor
- arrays of expert FFN tensors per layer
- optional shared-expert tensors, depending on the GGUF layout

### Acceptance Criteria

- a Qwen 3.5 MoE GGUF can be loaded into an internal model structure without tensor-name mismatches
- missing MoE tensors fail with clear errors
- dense Qwen 3.5 models continue to load

## Phase 3: CPU MoE Execution

### Goal

Implement a correct CPU inference path for Qwen 3.5 MoE.

### Files

- `src/runtime/families/qwen35/runtime.zig`
- `src/runtime/llama_runtime.zig`
- `src/model/loader.zig`
- any new helper modules under `src/runtime/families/qwen35/`

### Tasks

- [x] replace the placeholder runtime with a real generation path
- [x] implement router evaluation for each token
- [x] implement top-k expert selection
- [x] implement expert weight normalization according to the GGUF metadata / architecture behavior
- [x] implement expert FFN evaluation:
  - [x] evaluate selected experts only
  - [x] combine outputs with routing weights
- [x] integrate the MoE layer path into the existing token generation loop
- [x] document and isolate any assumptions that are specific to Qwen 3.5 MoE

### Recommended Module Split

To keep the implementation modular and under file-size limits, split by responsibility:

- `src/runtime/families/qwen35/runtime.zig`
  - family entrypoint and capability declaration
- `src/runtime/families/qwen35/router.zig`
  - router logits, top-k selection, gating weights
- `src/runtime/families/qwen35/experts.zig`
  - expert FFN evaluation and weighted reduction
- `src/runtime/families/qwen35/cpu.zig`
  - CPU execution orchestration

### Current Assumptions

- explicit Metal support is still deferred; the Qwen 3.5 MoE family runtime currently forces the CPU path
- when expert gating metadata is absent, the implementation defaults to softmax gating with normalized top-k weights
- shared-expert routing is applied through `ffn_gate_inp_shexp.weight` when present

### Acceptance Criteria

- CPU generation runs for a Qwen 3.5 MoE model
- outputs are coherent enough to demonstrate correct routing and decode behavior
- runtime no longer returns `Qwen35MoENotYetImplemented`

## Phase 4: Initial Quantization Support

### Goal

Add the first quantization set required by this plan:

- `Q3_K`
- `IQ3_XXS`
- `IQ4_XS`

### Files

- `src/model/loader.zig`
- `src/model/quant_extra.zig`
- tests/integration validation through the local Qwen 3.5 MoE GGUF path

### Tasks

- [x] implement `dequantizeRowQ3K()`
- [x] implement `dequantizeRowIQ3XXS()`
- [x] implement `dequantizeRowIQ4XS()`
- [x] wire the new formats into `dequantizeRow()`
- [x] wire the new formats into the row-dot matvec path
- [x] verify `tensorRowByteSize()` and layout handling match the actual block definitions already encoded in the loader and GGUF inspector
- [x] validate the mixed-quant path through CPU model load and generation against the target GGUF

### Why This Quant Set First

`Q3_K`, `IQ3_XXS`, and `IQ4_XS` are the initial target because:

- they are already modeled in the tensor enum and row-size logic
- they are the exact formats required by the target Qwen 3.5 MoE GGUF
- they provide a narrow initial matrix that is useful without overcommitting to broad quant support

### Acceptance Criteria

- loader dequantizes `Q3_K` rows correctly
- unit tests cover the format

## Phase 5: Metal Tensor Upload Path for Q3

### Goal

Avoid forcing `Q3_K` tensors through dense `f32` expansion during Metal preparation.

### Files

- `src/runtime/gpu/metal/tensor_store.zig`
- backend code used by Metal matvec paths

### Tasks

- [x] extend raw quantized tensor handling to include `q3_k`
- [x] extend raw quantized tensor handling to include `iq3_xxs`
- [x] extend raw quantized tensor handling to include `iq4_xs`
- [x] ensure prewarm/caching logic treats the new raw quantized tensor types consistently
- [x] add tests or validation hooks showing the tensor store preserves raw quantized bytes for the initial target quant formats

### Notes

Current Metal tensor preparation treats these as raw quantized tensors:

- `q4_k`
- `q6_k`
- `q8_0`

Everything else is expanded to dense `f32`, which is not the right long-term path for initial Q3 support.

### Acceptance Criteria

- `Q3_K` tensors are stored in raw quantized form for Metal preparation
- quantized upload path remains correct for existing `q4_k`, `q6_k`, and `q8_0`

## Phase 6: Family Capabilities and Error Handling

### Goal

Advertise realistic family capabilities and fail clearly when unsupported combinations are requested.

### Files

- `src/runtime/families/qwen35/runtime.zig`
- `src/runtime/families/registry.zig`

### Tasks

- [x] update `supported_quant_types` to include the initial quant targets once implemented
- [x] enable CPU capability once the CPU path is working
- [x] keep Metal runtime capability disabled until the Qwen 3.5 MoE execution path is implemented
- [x] return precise unsupported backend errors for the current runtime boundary
- [x] return precise missing-feature errors if optional MoE behaviors are encountered but not implemented yet

### Acceptance Criteria

- capability reporting matches actual implementation state
- unsupported formats fail clearly and predictably

## Phase 7: Validation Matrix

### Goal

Validate correctness before widening support.

### Test Areas

- family routing
- model loading
- dequantization
- CPU generation
- Metal preparation
- output sanity

### Minimum Validation Cases

- [x] family detection for `qwen35moe`
- [x] model load for a Qwen 3.5 MoE GGUF
- [x] `Q3_K` tensor dequant fixture
- [x] CPU single-prompt generation
- [x] Metal startup / tensor prewarm planning without dense fallback for the initial quant targets

### Acceptance Criteria

- all targeted tests pass
- no regression in existing dense Qwen family handling

## Phase 8: Documentation Update

### Goal

Document the exact initial support boundary.

### Files

- `README.md`
- `docs/family-split/qwen/BRANCH_ROADMAP.md`
- this plan file

### Tasks

- [x] document that initial Qwen 3.5 MoE support targets `Q3_K`
- [x] document that initial Qwen 3.5 MoE support also covers `IQ3_XXS` and `IQ4_XS`
- [x] document any explicitly unsupported formats
- [x] document backend capability status
- [x] add example commands for inspect and run once support exists

### Acceptance Criteria

- support boundaries are explicit
- no ambiguity about which quants are expected to work

## Optional Phase 9: KV Cache Quantization

### Goal

Reduce KV cache footprint after the initial model path is stable.

### Files

- generation/session code
- backend cache structures
- CLI/config plumbing if exposed as a user option

### Tasks

- [ ] add KV cache quantization support behind an explicit option
- [ ] keep the feature independent from weight quantization support
- [ ] validate correctness against the unquantized KV cache path
- [ ] document tradeoffs and supported modes

### Notes

KV cache quantization should be treated as a follow-on optimization. It is not a substitute for initial weight quantization support and should not block the first Qwen 3.5 MoE milestone.

## Recommended Execution Order

1. Phase 1: Family routing
2. Phase 2: Model representation for MoE
3. Phase 3: CPU MoE execution
4. Phase 4: Initial quantization support
5. Phase 5: Metal tensor upload path for Q3
6. Phase 6: Family capabilities and error handling
7. Phase 7: Validation matrix
8. Phase 8: Documentation update
9. Optional Phase 9: KV cache quantization

## Definition of Done

Qwen 3.5 MoE support is considered initially implemented when all of the following are true:

- [x] `qwen35moe` routes to the Qwen 3.5 MoE family
- [x] a Qwen 3.5 MoE GGUF loads successfully
- [x] CPU generation works
- [x] `Q3_K` is supported end-to-end
- [x] Metal tensor preparation supports the initial quant targets without forced dense expansion
- [x] documentation clearly states the initial support matrix
