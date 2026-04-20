# Gemma 3 Implementation Guide

This guide covers implementation details for Gemma 3 (1B, 4B, 12B, 27B) for GGUF inference support.

## Model Identification

- **Architecture string in GGUF**: `gemma3` (likely)
- **HF Model IDs**: 
  - 1B: `google/gemma-3-1b`
  - 4B: `google/gemma-3-4b`
  - 12B: `google/gemma-3-12b`
  - 27B: `google/gemma-3-27b-it`
- **Parameters**: 1B / 4B / 12B / 27B
- **Release date**: March 2025

## Key Architecture Differences from Gemma 2

| Feature | Gemma 3 | Gemma 2 |
|---------|---------|---------|
| Type | Multimodal | Text-only |
| Context | 128K (1B,4B,12B), 32K (27B) | 8K |
| Vision | SigLIP encoder | N/A |
| Attention | GQA | GQA |

## Current Status

**TEXT-ONLY SUPPORT EXPERIMENTAL** - Several fixes applied, but Gemma 3 still produces garbage output.

### Fixes Applied (2026-04-19)

1. **embedding_scale** - Fixed to apply `sqrt(embedding_length)` for gemma3 (was missing)
2. **rope_scaling_factor** - Added metadata parsing for `gemma3.rope.scaling.factor`
3. **effective_freq_base** - Computed as `rope_freq_base / rope_scaling_factor` for linear scaling
4. **attn_scale** - Applied post-RoPE to Q: `1/sqrt(head_dimension)` (~0.062 for 4B model)
5. **Metal scale kernel** - Added `scale_in_place_f32` kernel and bridge function

### Known Issues (Gemma 3 Broken - Gemma 2 Works)

| Property | Gemma 2 (Working) | Gemma 3 (Broken) |
|----------|-------------------|-------------------|
| rope_freq_base | 10000 | 1,000,000 |
| rope_scaling_factor | none | 8.0 (linear) |
| effective_freq_base | 10000 | 125,000 |
| attn_logit_softcapping | 50.0 | 50.0 |
| attn_q_norm / attn_k_norm | no | yes (256-dim per-head norms) |
| embedding_scale | sqrt(2304)=48 | sqrt(2560)=50.6 |

### Observations

1. **CPU backend**: Produces garbage but logs show valid logits (top_token ~20000+, scores ~20-25)
2. **Metal backend**: Produces `<unused*>` tokens
3. **Output weight**: tensor is tied/identical to token_embd.weight (2560 x 262208, type=14 q6_k)
4. **Token indices**: `<unused0>` = index 6, `<unused7>` = index 13, etc.

### Suspected Root Causes

1. **Per-head Q/K normalization** (`attn_q_norm`, `attn_k_norm`) - These are new in Gemma 3 with 256-dim per-head normalization. May need specific initialization or handling in the Metal attention kernel.

2. **RoPE scaling mismatch** - The effective_freq_base (125K) may not be the correct way to handle linear scaling. Perhaps the original freq_base should be used directly with position-dependent scaling.

3. **High freq_base numerical precision** - 1M base with position 0 gives theta = 0 / pow(1M, 0) = 0/1 = 0, but floating point division of small numbers by large bases may cause issues.

### Next Steps

1. Investigate per-head normalization handling in Metal backend
2. Compare with llama.cpp's Gemma 3 implementation for Metal
3. Consider alternative RoPE scaling approaches (position scaling vs frequency scaling)

### Metadata Verified from GGUF

```
gemma3.context_length: 131072
gemma3.embedding_length: 2560
gemma3.block_count: 34
gemma3.feed_forward_length: 10240
gemma3.attention.head_count: 8
gemma3.attention.head_count_kv: 4
gemma3.attention.key_length: 256
gemma3.attention.value_length: 256
gemma3.rope.freq_base: 1000000.0
gemma3.rope.scaling.type: linear
gemma3.rope.scaling.factor: 8.0
gemma3.attn_logit_softcapping: 50.0
gemma3.final_logit_softcapping: 30.0
```

## Reference

- [Gemma 3 Report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
- [HF Gemma3 Docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3)

## Multimodal Notes (Future Phase)

Gemma 3 adds:
- **Vision Encoder**: SigLIP-based
- **Multi-modal Fusion**: Vision + Language
- **Extended Context**: Up to 128K

For full multimodal support, separate vision path required.

## FROM HERMES AGENT, FINDINGS ON GEMMA 3 IMPLEMENTATION

Date: 2026-04-19
Source: Direct analysis of ggml-org/llama.cpp master branch source code

---

### Executive Summary

Gemma 3 IS FULLY IMPLEMENTED in llama.cpp as of the current master branch. The architecture has dedicated implementation files (`gemma3.cpp`) with full support for both standard attention and sliding window attention (SWA/ISWA) variants. The implementation supports both text-only and multimodal variants through template parameters.

---

### Architecture Registration in llama.cpp

**File: `src/llama-arch.h`** (llm_arch enum)
```
LLM_ARCH_GEMMA  = 47
LLM_ARCH_GEMMA2
LLM_ARCH_GEMMA3
LLM_ARCH_GEMMA3N  (Gemma 3 with NPU support)
LLM_ARCH_GEMMA4
LLM_ARCH_GEMMA_EMBEDDING
```

**File: `src/llama-arch.cpp`** (LLM_ARCH_NAMES map)
```
{ LLM_ARCH_GEMMA,     "gemma" },
{ LLM_ARCH_GEMMA2,    "gemma2" },
{ LLM_ARCH_GEMMA3,    "gemma3" },
{ LLM_ARCH_GEMMA3N,   "gemma3n" },
{ LLM_ARCH_GEMMA4,    "gemma4" },
{ LLM_ARCH_GEMMA_EMBEDDING, "gemma-embedding" },
```

---

### Gemma 3 Specific Implementation Files

**File: `src/models/gemma3.cpp`** (3,709 bytes)

This is a template-based implementation supporting two variants via compile-time parameters:
- `iswa = false`: Standard attention
- `iswa = true`: Sliding Window Attention (SWA/ISWA)

Key template instantiations:
```cpp
template struct llm_build_gemma3<false>;  // standard
template struct llm_build_gemma3<true>;   // ISWA variant
```

**Architecture Classification Functions** (in `llama-arch.cpp`):
- `llm_arch_is_recurrent()`: Gemma models return `false` (not recurrent)
- `llm_arch_is_hybrid()`: Gemma models return `false` (not hybrid)
- `llm_arch_is_diffusion()`: Gemma models return `false` (not diffusion)
- `llm_arch_supports_sm_tensor()`: Gemma3N returns `false`, other Gemma variants return `true`

---

### Gemma 3 Forward Pass Implementation

The `llm_build_gemma3` template constructs the computation graph with the following structure:

#### Per-Layer Computation:

1. **Input Norm**: `build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il)`
   - Uses RMSNorm (no bias)

2. **QKV Projection** with per-head normalization:
   ```cpp
   auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);
   Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
   Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
   ```

3. **RoPE Application** with per-layer frequency scaling:
   ```cpp
   Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, ...);
   Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, ...);
   ```

4. **Attention Scale** (Gemma-specific):
   ```cpp
   // Reference: https://github.com/google/gemma_pytorch/blob/014acb7/model.py#L315
   Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);
   ```

5. **Attention Output**: `build_attn(inp_attn, model.layers[il].wo, ..., Qcur, Kcur, Vcur, ...)`

6. **Post-Attention Norm**: `build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il)`

7. **Residual Connection**: `sa_out = ggml_add(ctx0, cur, inpL)`

8. **FFN with GELU**: 
   ```cpp
   cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, 
                   NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, 
                   LLM_FFN_GELU, LLM_FFN_PAR, il);
   ```

9. **FFN Post-Norm**: `build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, -1)`

10. **Final Residual**: `cur = ggml_add(ctx0, cur, sa_out)`

#### Final Output:
```cpp
cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
res->t_embd = cur;

// lm_head with softcapping
cur = build_lora_mm(model.output, cur);
if (hparams.f_final_logit_softcapping) {
    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
    cur = ggml_tanh(ctx0, cur);
    cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
}
```

---

### Key Gemma 3 Hyperparameters (from `llama-hparams.h`)

```cpp
struct llama_hparams {
    // ... standard fields ...
    
    // Gemma-specific softcapping
    float f_attn_logit_softcapping = 50.0f;   // attention softcap
    float f_router_logit_softcapping = 30.0f; // router softcap (MoE)
    float f_final_logit_softcapping = 30.0f;  // final output softcap
    
    // Gemma 4 per-layer embedding
    uint32_t n_embd_per_layer = 0;
};
```

---

### Tensor Naming Convention (from `LLM_TENSOR_NAMES`)

Per-layer tensors use the `blk.%d.` prefix pattern:
```
blk.%d.attn_norm      -> layer normalization
blk.%d.attn_q         -> query projection
blk.%d.attn_k         -> key projection
blk.%d.attn_v         -> value projection
blk.%d.attn_output    -> output projection (wo)
blk.%d.attn_q_norm    -> per-head Q normalization (Gemma 3 new)
blk.%d.attn_k_norm    -> per-head K normalization (Gemma 3 new)
blk.%d.post_attention_norm -> post-attention RMSNorm
blk.%d.ffn_norm       -> FFN input norm
blk.%d.post_ffw_norm   -> FFN post-norm
blk.%d.ffn_gate       -> FFN gate (w1)
blk.%d.ffn_up         -> FFN up (w3)
blk.%d.ffn_down        -> FFN down (w2)
```

Global tensors:
```
token_embd            -> embedding table
output_norm           -> final RMSNorm
output                -> lm_head weight
```

---

### Attention Types Supported

**Standard Attention** (`build_attn_inp_kv()`):
- Uses standard KV cache
- Supports grouped query attention (GQA)

**Sliding Window Attention (ISWA)** (`build_attn_inp_kv_iswa()`):
- Per-layer configurable RoPE frequency bases
- Supports hybrid dense/SWA layer patterns

---

### Key Implementation Details for Gemma 3

1. **Per-Head Q/K Norm**: Gemma 3 introduces `attn_q_norm` and `attn_k_norm` with 256-dim per-head normalization. This is applied BEFORE RoPE.

2. **Attention Scale**: Reference to official Gemma implementation applies `hparams.f_attention_scale` to Q after RoPE.

3. **Final Logit Softcapping**: Applied at output before computing logits:
   ```
   logits = tanh(logits / softcap) * softcap
   ```

4. **GELU Activation**: FFN uses standard GELU (not SwiGLU like Llama).

5. **RMSNorm**: All normalizations use RMSNorm without bias terms.

6. **Parallel FFN**: FFN gate and up projections run in parallel (`LLM_FFN_PAR`).

---

### Related Gemma Variants in llama.cpp

| Variant | File | Notes |
|---------|------|-------|
| Gemma 1 | `gemma.cpp` | Original Gemma |
| Gemma 2 | `gemma2-iswa.cpp` | Gemma 2 with ISWA |
| Gemma 3 | `gemma3.cpp` | Gemma 3 standard + SWA |
| Gemma 3N | `gemma3n-iswa.cpp` | Gemma 3N with NPU/ISWA |
| Gemma 4 | `gemma4-iswa.cpp` | Gemma 4 with ISWA |
| Gemma Embedding | `gemma-embedding.cpp` | Sentence transformer variant |

---

### Quantization Support

Gemma 3 uses standard quantization formats supported by llama.cpp:
- Q4_K_M, Q5_K_S, Q6_K (recommended)
- Q8_0, F16, BF16 (full precision)
- IQ2, IQ3, IQ4 (low-bit formats)

The model architecture itself is quantization-friendly due to:
- No layer norms with bias
- Standard attention mechanism
- GELU activation (not MoE)

---

### SWA (Sliding Window Attention) Pattern

Gemma 3 supports per-layer SWA through the `iswa` template parameter:

```cpp
if constexpr (iswa) {
    freq_base_l = model.get_rope_freq_base(cparams, il);
    freq_scale_l = model.get_rope_freq_scale(cparams, il);
} else {
    freq_base_l = freq_base;
    freq_scale_l = freq_scale;
}
```

The `llama_hparams::swa_layers` array determines which layers use SWA.

---

### Critical Fields in llama_layer for Gemma 3

```cpp
struct llama_layer {
    // Gemma 3 attention norms
    struct ggml_tensor * attn_norm = nullptr;      // input attn norm
    struct ggml_tensor * attn_q_norm = nullptr;   // per-head Q RMSNorm
    struct ggml_tensor * attn_k_norm = nullptr;   // per-head K RMSNorm
    struct ggml_tensor * attn_post_norm = nullptr; // post-attn RMSNorm
    
    // Attention projections
    struct ggml_tensor * wq = nullptr;  // query
    struct ggml_tensor * wk = nullptr;  // key
    struct ggml_tensor * wv = nullptr;  // value
    struct ggml_tensor * wo = nullptr;  // output
    
    // FFN
    struct ggml_tensor * ffn_norm = nullptr;      // pre-FFN norm
    struct ggml_tensor * ffn_post_norm = nullptr; // post-FFN norm
    struct ggml_tensor * ffn_gate = nullptr;       // w1
    struct ggml_tensor * ffn_up = nullptr;         // w3
    struct ggml_tensor * ffn_down = nullptr;      // w2
};
```

---

### KV Metadata Keys for Gemma 3

From `LLM_KV_NAMES`:
```
%s.vocab_size
%s.context_length
%s.embedding_length
%s.block_count
%s.feed_forward_length
%s.attention.head_count
%s.attention.head_count_kv
%s.attention.key_length
%s.attention.value_length
%s.attention.layer_norm_rms_epsilon
%s.rope.dimension_count
%s.rope.freq_base
%s.rope.scaling.type
%s.rope.scaling.factor
%s.attn_logit_softcapping
%s.final_logit_softcapping
```

---

### Implementation Verification Checklist

Based on source code analysis:

- [x] Architecture enum defined (LLM_ARCH_GEMMA3 = ~50)
- [x] String mapping registered ("gemma3")
- [x] Template implementation in gemma3.cpp
- [x] Forward pass implemented with all Gemma 3 features
- [x] Per-head Q/K norm support
- [x] Attention softcapping support
- [x] Final logit softcapping support
- [x] GELU FFN activation
- [x] RMSNorm throughout
- [x] RoPE with frequency scaling
- [x] ISWA/SWA template variant
- [x] Tensor naming conventions match HF format
- [x] KV metadata keys defined
- [x] llm_build_gemma3 template instantiated for both iswa variants

---

### Conclusion

**Gemma 3 is FULLY IMPLEMENTED in llama.cpp** at the master branch. The implementation is a template-based approach supporting both standard and sliding window attention variants. Key differentiating features from Gemma 2:

1. **Per-head Q/K RMSNorm** (256-dim)
2. **Attention scale** applied post-RoPE
3. **Higher rope_freq_base** (up to 1M)
4. **RoPE scaling** support
5. **Extended context** (up to 128K)

For any GGUF engine implementation, the llama.cpp source at `ggml-org/llama.cpp` should be considered the canonical reference for correct Gemma 3 behavior.

---

### Source Files Analyzed

- `src/llama-arch.h` (15,705 bytes) - Architecture enum definitions
- `src/llama-arch.cpp` (45,084 bytes) - Architecture mappings and helpers
- `src/llama-hparams.h` (9,409 bytes) - Hyperparameters including Gemma-specific fields
- `src/models/gemma3.cpp` (3,709 bytes) - Gemma 3 forward pass implementation
- `src/llama-graph.h` (29,301 bytes) - Graph building infrastructure
- `src/llama-model.h` (18,799 bytes) - Model and layer structures
- `src/llama-model-loader.cpp` (53,590 bytes) - GGUF loading infrastructure

---

## Reference

- [Gemma 3 Report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
- [HF Gemma3 Docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3)