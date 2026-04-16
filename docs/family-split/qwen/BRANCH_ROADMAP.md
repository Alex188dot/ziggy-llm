# Qwen Family Implementation Roadmap

This roadmap defines the phases to implement full Qwen family support in ziggy-llm, focusing on Qwen3.5 dense models initially with MoE support staged for future implementation.

## How To Use This File

Every task in this file is a Markdown checkbox.

When a task is completed:
- replace `- [ ]` with `- [x]`
- keep the item in place instead of deleting it
- only mark it complete when the code, docs, and validation for that item are actually done

If a task turns out to be too large, split it into smaller checklist items rather than leaving a vague partially done item behind.

## Goal

**Primary**: Full Qwen3.5 dense support (at least the 2B model for testing)
**Secondary**: Stage infrastructure for Qwen3.5 MoE support

## Test Model

- `models/Qwen3.5-2B-Q4_K_M.gguf` - Qwen3.5 2B dense model (Gated DeltaNet architecture)
- Architecture string in GGUF: `qwen35`
- Context length: 262,144
- Quantization: Q4_K_M

---

## Phase 1: Fix Qwen3.5 Dense Architecture Detection

**Goal**: Fix the architecture detection so `qwen35` dense models are routed to the correct handler.

### Task 1.1: Diagnose Architecture Detection Issue

- [x] Confirm model architecture string is `qwen35` (not `qwen3_5_text`)
- [x] Confirm model is dense (not MoE) - has Gated DeltaNet layers, not expert routing
- [x] Identify routing issue: `qwen35` → `.custom()` → "UnsupportedArchitecture"

### Task 1.2: Fix Family Detection in mod.zig

- [x] Update `detectModelFamily()` to handle `qwen35` architecture string:
  - If architecture is `qwen35` without `moe` suffix, treat as `.qwen35_text` (dense)
  - If architecture contains `moe` (e.g., `qwen3_moe`), treat as `.qwen35` (MoE)

- [x] Update `loader.zig` if needed to properly parse `qwen35` dense models

### Task 1.3: Validate Fix

- [x] Run `./ziggy-llm inspect -m models/Qwen3.5-2B-Q4_K_M.gguf` - should work
- [x] Run `./ziggy-llm run -m models/Qwen3.5-2B-Q4_K_M.gguf -p "Hello" --max-tokens 20 --backend metal` - should not error with "UnsupportedArchitecture"
- [ ] Verify output is correct (decode works, not gibberish)

**Definition of Done**:
- [x] Qwen3.5-2B model no longer returns "UnsupportedArchitecture" error
- [x] Model can be loaded and basic generation works

**Notes**:
- Qwen3.5-2B uses Gated DeltaNet (Mamba/SSM-like) architecture with different tensor naming
- Standard dense attention tensors (attn_q, attn_k, attn_v) are not present
- Has attn_qkv, ssm_* (state space model) tensors instead
- This requires specialized runtime support for SSM/mamba layers
- Will need significant inference kernel changes to support

---

## Phase 2: Full CPU Inference Support

**Goal**: Ensure Qwen3 (non-SSM) models work on CPU backend.

### Task 2.1: Verify CPU Backend Path

- [x] Trace CPU inference path for qwen3 family
- [x] CPU inference runs but produces gibberish output for Qwen3 models
- [ ] Investigate tokenizer/chattemplate issues

### Task 2.2: Test CPU Inference

- [x] Run: `./ziggy-llm run -m models/Qwen3-1.7B-Q4_K_M.gguf -p "Hello" --max-tokens 20 --backend cpu`
- [ ] Verify output is coherent text - FAILS (gibberish)
- [ ] Measure tokens/second on CPU

### Task 2.3: Fix Any CPU Issues

- [ ] Fix tensor dequantization issues if any
- [ ] Fix tokenizer issues (Qwen uses BPE tokenizer differently)
- [ ] Fix chat template issues

**Definition of Done**:
- [ ] Qwen3 models generate correct output on CPU

---

## Phase 3: Full Metal/GPU Inference Support

**Goal**: Ensure Qwen3 (non-SSM) models work on Metal backend.

### Task 3.1: Verify Metal Backend Path

- [x] Metal inference runs successfully for Qwen3-1.7B
- [x] Measured performance: ~65 tok/s on Metal

### Task 3.2: Test Metal Inference

- [x] Run: `./ziggy-llm run -m models/Qwen3-1.7B-Q4_K_M.gguf -p "Hello" --max-tokens 20 --backend metal`
- [x] Output generated but has repetition issues
- [ ] Verify outputs are coherent

### Task 3.3: Fix Any Metal Issues

- [ ] Fix chat template/tokenizer issues causing repetition
- [ ] Add proper Qwen3 chat template support

**Definition of Done**:
- [x] Qwen3-1.7B generates tokens on Metal at ~65 tok/s
- [ ] Output is coherent (not gibberish)

**Definition of Done**:
- [ ] Qwen3.5-2B generates correct output on CPU
- [ ] No crashes or memory errors
- [ ] tokens/second measured and reasonable

---

## Phase 3: Full Metal/GPU Inference Support

**Goal**: Ensure Qwen3.5 dense models work on Metal backend.

### Task 3.1: Verify Metal Backend Path

- [ ] Trace Metal inference path for qwen35_text family
- [ ] Confirm tensor store uploads linear_attention tensors
- [ ] Check Metal shader compatibility with Gated DeltaNet architecture

### Task 3.2: Test Metal Inference

- [ ] Run: `./ziggy-llm run -m models/Qwen3.5-2B-Q4_K_M.gguf -p "Hello" --max-tokens 20 --backend metal`
- [ ] Verify output is coherent text
- [ ] Measure tokens/second on Metal

### Task 3.3: Verify CPU vs Metal Correctness

- [ ] Run same prompt on CPU and Metal
- [ ] Verify outputs match (within floating point tolerance)
- [ ] Document any differences

### Task 3.4: Fix Any Metal Issues

- [ ] Fix tensor upload issues if any
- [ ] Fix Metal kernel issues with linear_attention
- [ ] Fix memory management issues

**Definition of Done**:
- [ ] Qwen3.5-2B generates correct output on Metal
- [ ] No crashes or memory errors
- [ ] tokens/second measured and faster than CPU
- [ ] CPU and Metal outputs match

---

## Phase 4: Thinking Mode Support

**Goal**: Implement support for Qwen3.5's hybrid thinking/non-thinking modes.

### Task 4.1: Understand Thinking Mode Architecture

- [ ] Research how Qwen3.5 thinking mode works
- [ ] Identify required chat template changes
- [ ] Document expected behavior

### Task 4.2: Add Thinking Toggle Parameter

- [ ] Add `enable_thinking: bool` to `FamilyGenerateOptions`
- [ ] Wire through to generation options
- [ ] Document usage

### Task 4.3: Implement Chat Template for Thinking

- [ ] Verify current chat template handles thinking control
- [ ] Test with thinking enabled
- [ ] Test with thinking disabled
- [ ] Verify outputs differ appropriately

### Task 4.4: Document Thinking Mode

- [ ] Document recommended parameters for thinking vs non-thinking
- [ ] Add usage examples

**Definition of Done**:
- [ ] Can toggle thinking mode via CLI parameter
- [ ] Thinking mode produces reasoning tokens
- [ ] Non-thinking mode produces direct response
- [ ] Documentation updated

---

## Phase 5: Additional Qwen3.5 Dense Model Sizes

**Goal**: Support other Qwen3.5 dense sizes (0.8B, 4B, 9B, 27B).

### Task 5.1: Test 0.8B Model

- [ ] Obtain Qwen3.5-0.8B GGUF if available
- [ ] Test CPU and Metal inference
- [ ] Verify outputs correct

### Task 5.2: Test 4B Model

- [ ] Obtain Qwen3.5-4B GGUF if available
- [ ] Test CPU and Metal inference
- [ ] Verify outputs correct

### Task 5.3: Test 9B Model

- [ ] Obtain Qwen3.5-9B GGUF if available
- [ ] Test CPU and Metal inference
- [ ] Verify outputs correct

### Task 5.4: Test 27B Model

- [ ] Obtain Qwen3.5-27B GGUF if available
- [ ] Test CPU and Metal inference
- [ ] Verify outputs correct
- [ ] Measure performance on Metal

**Definition of Done**:
- [ ] All tested Qwen3.5 dense sizes work
- [ ] Performance acceptable for each size

---

## Phase 6: MoE Infrastructure (Staged for Future)

**Goal**: Stage infrastructure for Qwen3.5 MoE models (35B-A3B, 122B-A10B, 397B-A17B).

### Task 6.1: Document MoE Architecture Requirements

- [ ] Document Qwen3.5 MoE architecture (expert routing, top-k selection)
- [ ] Identify required tensor names (ffn_gate_inp, ffn_down.N, ffn_up.N, etc.)
- [ ] Create design document for MoE implementation

### Task 6.2: Add MoE Tensor Parsing (Skeleton)

- [ ] Add placeholder for MoE tensor detection
- [ ] Add expert count metadata parsing
- [ ] Document what would need to be implemented

### Task 6.3: Reserve Code Paths

- [ ] Ensure MoE models return clear "not implemented" error
- [ ] Add helpful error message pointing to future support

**Definition of Done**:
- [ ] MoE infrastructure staged
- [ ] Clear error for MoE models
- [ ] Documentation for future implementation

---

## Phase 7: Quantization Verification

**Goal**: Ensure all common quantization types work for Qwen3.5 dense.

### Task 7.1: Test Q4_K_M

- [x] Model is Q4_K_M - verify works

### Task 7.2: Test Other Quantizations

- [ ] Test Q6_K quantization if available
- [ ] Test Q8_0 quantization if available
- [ ] Test F16 quantization if available

### Task 7.3: Document Supported Quantizations

- [ ] Update README with supported quantizations for Qwen3.5
- [ ] Document any known issues

**Definition of Done**:
- [ ] Q4_K_M fully verified
- [ ] Other quantizations tested and documented

---

## Validation Checklist

For each phase, verify:

1. Code compiles without warnings
2. `zig build test` passes
3. Model loads without errors
4. Generation produces coherent output
5. No memory leaks
6. Performance acceptable

## Deferred / Future Work

The following are intentionally deferred until dense support is stable:

- **High Priority**:
  - Qwen3.5 MoE implementation (expert routing, top-k kernel)
  - Vision/multimodal support
  - KV cache quantization

- **Medium Priority**:
  - MXFP4_MOE quantization support
  - Long context optimization (YaRN)
  - Multi-GPU inference

- **Lower Priority**:
  - AWQ quantization support
  - GGUF creation from HuggingFace models
