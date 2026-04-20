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

---

## FROM HERMES AGENT, FINDINGS ON QWEN 3.5 IMPLEMENTATION

**Source**: llama.cpp master branch (latest), scraped from GitHub raw content on 2026-04-19
**Source URLs**:
- `src/llama-arch.h` - Architecture enum, tensor/KV key definitions
- `src/llama-arch.cpp` - Architecture name mapping, classification functions
- `src/llama-hparams.h` - Hyperparameter struct
- `src/llama-vocab.h` - Vocab pre-type enum
- `src/llama-model.h` - Layer struct, LLM_TYPE enum
- `src/llama-model.cpp` (partial) - hparams loading, tensor split logic, model type classification

---

### 1. ARCHITECTURE ENUM AND NAME MAPPING

**In `llama-arch.h`**:
```c
enum llm_arch {
    // ... existing arches ...
    LLM_ARCH_QWEN3NEXT,
    LLM_ARCH_QWEN3VL,
    LLM_ARCH_QWEN3VLMOE,
    LLM_ARCH_QWEN35,        // <-- Qwen3.5 dense
    LLM_ARCH_QWEN35MOE,     // <-- Qwen3.5 MoE
    // ...
};
```

**In `llama-arch.cpp`**:
```c
{LLM_ARCH_QWEN3NEXT, "qwen3next"},
{LLM_ARCH_QWEN35,   "qwen35"},     // architecture string in GGUF = "qwen35"
{LLM_ARCH_QWEN35MOE, "qwen35moe"}, // architecture string in GGUF = "qwen35moe"
```

**Key insight**: Qwen3.5 (dense) maps to `LLM_ARCH_QWEN35` with architecture name `"qwen35"`. Qwen3.5 MoE maps to `LLM_ARCH_QWEN35MOE` with name `"qwen35moe"`.

---

### 2. VOCABULARY PRETYPE

**In `llama-vocab.h`**:
```c
LLAMA_VOCAB_PRE_TYPE_QWEN35 = 46,
```

This is a dedicated vocab pretokenization type for Qwen3.5 (distinct from `LLAMA_VOCAB_PRE_TYPE_QWEN2 = 11`). The tokenizer uses SPM-style BPE with spaces replaced by `▁` before BPE merges.

---

### 3. ARCHITECTURE CLASSIFICATION

**In `llama-arch.cpp`**:

```c
// llm_arch_is_hybrid()
case LLM_ARCH_QWEN3NEXT:
case LLM_ARCH_KIMI_LINEAR:
case LLM_ARCH_QWEN35:
case LLM_ARCH_QWEN35MOE:
    return true;
```

**Critical**: Qwen3.5 is classified as **hybrid** (not purely recurrent, not purely dense attention). This means it mixes standard attention layers with SSM/linear attention layers.

**Not recurrent** (unlike Qwen3Next which is also hybrid):
```c
// llm_arch_is_recurrent() - QWEN35 is NOT listed here
case LLM_ARCH_MAMBA:
case LLM_ARCH_MAMBA2:
case LLM_ARCH_RWKV6:
case LLM_ARCH_RWKV6QWEN2:
case LLM_ARCH_RWKV7:
case LLM_ARCH_ARWKV7:
    return true;
```

---

### 4. HYPERPARAMETERS (llama-hparams.h)

Qwen3.5 uses these SSM-related hyperparameters (same system as Mamba/Kimi Linear/Granite Hybrid):

```c
struct llama_hparams {
    // SSM parameters (used by Qwen3.5):
    uint32_t ssm_d_conv = 0;       // SSM conv kernel size
    uint32_t ssm_d_inner = 0;      // SSM inner size
    uint32_t ssm_d_state = 0;      // SSM state size (aka head_dim for SSM)
    uint32_t ssm_dt_rank = 0;      // Delta-t rank
    uint32_t ssm_n_group = 0;      // SSM group count (for Qwen3.5: n_k_heads)
    // ...
};
```

**Also notable in hparams** - `n_embd_per_layer` (used by Gemma4, may be relevant for Qwen3.5):
```c
uint32_t n_embd_per_layer = 0;  // gemma4 per-layer embedding
```

---

### 5. LAYER STRUCTURE (llama-model.h)

Per-layer tensors for Qwen3.5 include **both standard attention AND SSM tensors**:

```c
struct llama_layer {
    // SSM / State Space Model tensors (Qwen3.5):
    struct ggml_tensor * ssm_in = nullptr;
    struct ggml_tensor * ssm_x = nullptr;
    struct ggml_tensor * ssm_dt = nullptr;
    struct ggml_tensor * ssm_out = nullptr;
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a = nullptr;
    struct ggml_tensor * ssm_d = nullptr;
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b = nullptr;
    struct ggml_tensor * ssm_beta_alpha = nullptr;  // qwen3next
    struct ggml_tensor * ssm_alpha = nullptr;       // qwen3.5
    struct ggml_tensor * ssm_norm = nullptr;
    struct ggml_tensor * ssm_dt_norm = nullptr;
    struct ggml_tensor * ssm_b_norm = nullptr;
    struct ggml_tensor * ssm_c_norm = nullptr;

    // Standard attention tensors:
    struct ggml_tensor * attn_norm = nullptr;
    struct ggml_tensor * wq = nullptr;
    struct ggml_tensor * wk = nullptr;
    struct ggml_tensor * wv = nullptr;
    struct ggml_tensor * wo = nullptr;
    struct ggml_tensor * wqkv = nullptr;  // combined QKV
    struct ggml_tensor * attn_out_norm = nullptr;

    // FFN:
    struct ggml_tensor * ffn_norm = nullptr;
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
};
```

**Tensor naming** (from `LLM_TENSOR_NAMES` in `llama-arch.cpp`):
```
blk.%d.ssm_conv1d      - SSM conv1d
blk.%d.ssm_dt          - Delta-t
blk.%d.ssm_a           - SSM A matrix
blk.%d.ssm_d           - SSM D vector
blk.%d.ssm_out         - SSM output projection
blk.%d.ssm_alpha       - Qwen3.5 alpha (Gated DeltaNet gate)
blk.%d.ssm_beta        - Qwen3.5 beta (shared with Kimi)
blk.%d.ssm_ba          - Qwen3Next beta-alpha combined
blk.%d.ssm_norm        - SSM normalization
blk.%d.ssm_b_norm      - SSM B normalization
blk.%d.ssm_c_norm      - SSM C normalization
blk.%d.ssm_dt_norm     - SSM dt normalization
blk.%d.attn_qkv        - Combined QKV projection
blk.%d.attn_output     - Output projection
blk.%d.ffn_gate        - FFN gate (SwiGLU)
blk.%d.ffn_down        - FFN down projection
blk.%d.ffn_up          - FFN up projection
blk.%d.attn_norm       - Attention RMSNorm
blk.%d.ffn_norm        - FFN RMSNorm
```

---

### 6. TENSOR OPERATIONS AND INFERENCE KERNELS

**From `LLM_TENSOR_INFOS` in `llama-arch.cpp`**:
```c
{LLM_TENSOR_SSM_CONV1D,  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
{LLM_TENSOR_SSM_A,       {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_SCAN}},
{LLM_TENSOR_SSM_A_NOSCAN,{LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},  // MUL variant for Qwen3Next
{LLM_TENSOR_SSM_DT,      {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_SSM_OUT,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_SSM_ALPHA,   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_SSM_BETA_ALPHA,{LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_SSM_X,       {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_SSM_D,       {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
{LLM_TENSOR_SSM_DT_NORM, {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
{LLM_TENSOR_SSM_B_NORM,  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
{LLM_TENSOR_SSM_C_NORM,  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
{LLM_TENSOR_SSM_NORM,    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
```

**Critical inference operations needed**:
- `GGML_OP_SSM_CONV` - 1D convolution for SSM input
- `GGML_OP_SSM_SCAN` - Selective scan operation (the core SSM computation)
- Standard `GGML_OP_MUL_MAT` for projections
- Standard `GGML_OP_MUL` for elementwise operations

---

### 7. MULTI-DEVICE TENSOR SPLIT LOGIC (CRITICAL FOR GPU OFFLOAD)

**From `llama_meta_device_get_split_state()` in `llama-model.cpp`**, Qwen3.5 has **special multi-device split handling**:

```cpp
if (ud->model->arch == LLM_ARCH_QWEN3NEXT ||
    ud->model->arch == LLM_ARCH_QWEN35 ||
    ud->model->arch == LLM_ARCH_QWEN35MOE) {
    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t head_v_dim = hparams.ssm_d_state;
    const int64_t n_k_heads = hparams.ssm_n_group;
    const int64_t n_v_heads = hparams.ssm_dt_rank;
    const int64_t key_dim = head_k_dim * n_k_heads;
    const int64_t value_dim = head_v_dim * n_v_heads;
```

**Key split difference between Qwen3Next and Qwen3.5**:

```cpp
// Qwen3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] pattern
if (ud->model->arch == LLM_ARCH_QWEN3NEXT) {
    if (regex_match(pattern_qkv_weight) || regex_match(pattern_ssm_conv1d)) {
        return {key_dim, key_dim, value_dim};
    }
}
// Qwen3.5: [k0_v0, k1_v1, k0_v2, k1_v3] pattern (different broadcasting)
// needs segmenting of V on the scale of K to get the correct pattern
else {
    const int64_t head_ratio = n_v_heads / n_k_heads;
    if (regex_match(pattern_qkv_weight) || regex_match(pattern_ssm_conv1d)) {
        return std::vector(2 + head_ratio, key_dim);  // e.g., [k, k, v0, v1]
    }
    if (regex_match(pattern_attn_gate_weight) || regex_match(pattern_ssm_out_weight)) {
        return std::vector(head_ratio, key_dim);
    }
    if (regex_match(pattern_ssm_dt) || regex_match(pattern_ssm_a) ||
        regex_match(pattern_ssm_alpha) || regex_match(pattern_ssm_beta)) {
        return std::vector(head_ratio, n_k_heads);
    }
    if (regex_match(pattern_r_cache)) {
        return std::vector(2 + head_ratio, key_dim * (hparams.ssm_d_conv - 1));
    }
    if (regex_match(pattern_s_cache)) {
        return std::vector(head_ratio, n_k_heads * head_v_dim * head_v_dim);
    }
}
```

**Also notable**: For Qwen3.5, Q gate tensor granularity is doubled:
```cpp
if (ud->model->arch == LLM_ARCH_QWEN3NEXT ||
    ud->model->arch == LLM_ARCH_QWEN35 ||
    ud->model->arch == LLM_ARCH_QWEN35MOE) {
    return {std::lcm(2*n_embd_q, blck_size)};
}
```

---

### 8. SOFTMAX GATING FOR MoE

For **Qwen3.5 MoE** (not dense):
```c
{LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func};
```

```c
enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID = 2,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3, // applied to router weights
};
```

---

### 9. MODEL TYPE CLASSIFICATION (LLM_TYPE)

From `llama-model.h` and the hparams loading in `llama-model.cpp`:
```c
// Qwen3.5 model type classification from n_layer:
// Note: Qwen3.5 dense uses standard switch-case based on n_layer
// MoE types from llm_type enum:
LLM_TYPE_35B_A3B,        // Qwen3.5 dense (35B, 3B active?)
LLM_TYPE_48B_A3B,        // Another Qwen3.5 variant
LLM_TYPE_122B_A10B,      // Qwen3.5 MoE
LLM_TYPE_397B_A17B,      // Qwen3.5 MoE large
```

---

### 10. QWEN3NEXT vs QWEN3.5 KEY DIFFERENCES

| Feature | Qwen3Next | Qwen3.5 |
|---------|-----------|---------|
| Architecture | Hybrid SSM + attention | Hybrid SSM + attention |
| SSM variant | Uses `ssm_ba` (beta-alpha fused) | Uses `ssm_alpha` + `ssm_beta` separate |
| V broadcasting | [k0_v0, k0_v1, k1_v2, k1_v3] | [k0_v0, k1_v1, k0_v2, k1_v3] |
| head_ratio handling | Not needed | n_v_heads / n_k_heads |
| R/S cache split | 3 segments | 2 + head_ratio segments |
| Model sizes | 80B-A3B (Qwen3Next) | 35B, 48B dense; 122B, 397B MoE |

---

### 11. IMPLEMENTATION CHECKLIST FOR ZIGGY-LLM

Based on the llama.cpp analysis, here's what's needed to implement Qwen3.5:

#### A. Architecture Registration
- [ ] Map `"qwen35"` → `LLM_ARCH_QWEN35` in `mod.zig`
- [ ] Map `"qwen35moe"` → `LLM_ARCH_QWEN35MOE` in `mod.zig`
- [ ] Register as hybrid architecture (not recurrent)

#### B. Vocabulary
- [ ] Add `LLAMA_VOCAB_PRE_TYPE_QWEN35` tokenizer handling
- [ ] SPM-style BPE with `▁` replacement before merges
- [ ] BOS/EOS token handling (standard Qwen2/Qwen3 style)

#### C. Hyperparameters to Parse from GGUF
- [ ] `ssm_d_conv` → SSM conv kernel
- [ ] `ssm_d_state` → state dimension
- [ ] `ssm_dt_rank` → delta-t rank
- [ ] `ssm_n_group` → number of K heads in SSM
- [ ] Standard: `n_embd`, `n_layer`, `n_head`, `n_head_kv`, `n_ff`

#### D. Tensor Name Mapping
Required tensor names for dense Qwen3.5:
```
token_embd
blk.{i}.ssm_conv1d
blk.{i}.ssm_dt
blk.{i}.ssm_a
blk.{i}.ssm_alpha   (Qwen3.5 specific!)
blk.{i}.ssm_beta    (Qwen3.5 specific!)
blk.{i}.ssm_norm
blk.{i}.ssm_b_norm
blk.{i}.ssm_c_norm
blk.{i}.ssm_dt_norm
blk.{i}.ssm_out
blk.{i}.ssm_x       (input for SSM)
blk.{i}.ssm_d       (D vector, elementwise)
blk.{i}.attn_qkv    (combined QKV)
blk.{i}.attn_output
blk.{i}.ffn_gate
blk.{i}.ffn_down
blk.{i}.ffn_up
blk.{i}.attn_norm
blk.{i}.ffn_norm
output
output_norm
```

#### E. Compute Kernels Needed
1. **SSM Conv** (`GGML_OP_SSM_CONV`) - 1D causal convolution
2. **SSM Scan** (`GGML_OP_SSM_SCAN`) - Selective scan / state space computation
3. **SSM Elementwise** - for `ssm_d`, `ssm_dt_norm`, `ssm_b_norm`, `ssm_c_norm`, `ssm_norm`
4. **Standard MatMul** for projections (`attn_qkv`, `attn_output`, `ffn_*`)
5. **RMSNorm** for `attn_norm`, `ffn_norm`, SSM norms
6. **Silu** activation for FFN gate

#### F. Multi-Device Split Logic
- [ ] Qwen3.5 V broadcasting: segments = `[key_dim, key_dim, v0, v1, ...]` pattern
- [ ] Q gate granularity doubled (like Qwen3Next)
- [ ] head_ratio = n_v_heads / n_k_heads for split calculations
- [ ] R cache and S cache split accounting for head_ratio

#### G. KV Cache
- Qwen3.5 uses standard attention layers interleaved with SSM layers
- KV cache format: standard for attention, SSM state cache for SSM layers
- `n_layer_kv_from_start` may be relevant if some layers share KV

#### H. RoPE
- Standard YaRN scaling for extended context
- Check `rope_scaling_type_train`, `rope_freq_base_train`
- Qwen3.5 supports up to 262,144 context (per test model metadata)

---

### 12. SIMILAR ARCHITECTURES FOR REFERENCE

The following architectures share similar patterns that can inform implementation:

1. **Qwen3Next** (`LLM_ARCH_QWEN3NEXT`) - Most similar, uses `ssm_ba` instead of separate `ssm_alpha`/`ssm_beta`
2. **GraniteHybrid** (`LLM_ARCH_GRANITE_HYBRID`) - SSM + attention hybrid
3. **Mamba2** (`LLM_ARCH_MAMBA2`) - Pure SSM (non-hybrid)
4. **KimiLinear** (`LLM_ARCH_KIMI_LINEAR`) - Also hybrid SSM+attention with similar SSM tensor names

---

### 13. GGUF KV KEYS FOR QWEN3.5

From `LLM_KV_NAMES` (formatted with arch prefix):
```
qwen35.vocab_size
qwen35.context_length
qwen35.embedding_length
qwen35.block_count
qwen35.feed_forward_length
qwen35.expert_feed_forward_length (MoE)
qwen35.attention.head_count
qwen35.attention.head_count_kv
qwen35.attention.layer_norm_rms_epsilon
qwen35.ssm.inner_size
qwen35.ssm.state_size
qwen35.ssm.time_step_rank
qwen35.ssm.conv_kernel
qwen35.ssm.group_count
qwen35.rope.dimension_count
qwen35.rope.freq_base
qwen35.rope.scaling.type
qwen35.rope.scaling.factor
qwen35.rope.scaling.original_context_length
```

---

### 14. QUANTIZATION CONSIDERATIONS

Qwen3.5 uses standard quantization paths for the attention layers. SSM layers (ssm_conv1d, ssm_a, ssm_dt, etc.) are typically quantized with the same type as other linear layers. The Q4_K_M quantization of the test model should work for both attention and SSM projections.

Key: `ssm_alpha` and `ssm_beta` are elementwise tensors (1D) that may need special handling if they appear in quantized form.

---

### 15. RUNTIME MEMORY ESTIMATION

For Qwen3.5 2B model (rough estimates):
- Embeddings: ~16MB
- Per SSM layer: ~2B params × 2 (SSM + FFN) / n_layers
- Attention projections: ~same
- KV cache: standard attention + SSM state
- Total: ~2.4GB for Q4_K_M

---

### 16. REFERENCES

- llama.cpp repo: https://github.com/ggml-org/llama.cpp
- Qwen3.5 HF page: https://huggingface.co/Qwen/Qwen3.5-2B
- Related architectures for cross-reference: Granite Hybrid, Kimi Linear, Mamba2

