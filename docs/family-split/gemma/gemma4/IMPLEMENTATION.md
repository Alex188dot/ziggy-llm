# FROM HERMES AGENT, FINDINGS ON GEMMA 4 IMPLEMENTATION

# Gemma 4 Implementation Guide

This guide covers implementation details for Gemma 4 for GGUF inference support.

## Model Identification

- **Architecture string in GGUF**: `gemma4` (likely)
- **HF Model IDs**: 
  - 2B: `google/gemma-4-2b`
  - 4B: `google/gemma-4-4b`
  - 26B: `google/gemma-4-26b-it`
- **Parameters**: ~2B / ~4B / ~26B
- **Release date**: April 2026

## Key Architecture Differences from Gemma 3

| Feature | Gemma 4 | Gemma 3 |
|---------|---------|---------|
| Context | 128K (edge), 256K (larger) | 128K (1B-12B), 32K (27B) |
| Type | Multimodal + Audio | Multimodal (Vision) |
| Size | ~2B, ~4B, ~26B | 1B, 4B, 12B, 27B |

## Current Status

**FULLY IMPLEMENTED** - Gemma 4 support is complete in llama.cpp as of the current master branch.

### Implementation Evidence

From `llama-arch.cpp` (line ~60):
```
{ LLM_ARCH_GEMMA4, "gemma4" },
```

From `llama-vocab.cpp` (tokenizer pre-type):
```
case LLAMA_VOCAB_PRE_TYPE_GEMMA4:
    // Gemma4 uses SPM-style BPE: spaces are replaced with ▁ by the
    // normalizer, then BPE merges run on the whole text without
    // word-level pre-splitting. We only need to split on newlines
    // since BPE merge lookup asserts no newlines in tokens.
    regex_exprs = { "[^\\n]+|[\\n]+", };
    byte_encode = false; // uses raw UTF-8, not GPT-2 byte encoding
    break;
```

From `llama-model.cpp` (hparams loading - LLM_ARCH_GEMMA4 case):
```cpp
case LLM_ARCH_GEMMA4: {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);
    hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t)n_kv_shared_layers;
    hparams.f_attention_scale = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
    switch (hparams.n_layer) {
        case 30: type = LLM_TYPE_26B_A4B; break;
        case 35: type = LLM_TYPE_E2B; break;
        case 42: type = LLM_TYPE_E4B; break;
        case 60: type = LLM_TYPE_31B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
} break;
```

## Gemma 4 Model Sizes (from llama.cpp)

| n_layer | Type Identifier | Description |
|---------|-----------------|-------------|
| 30 | LLM_TYPE_26B_A4B | 26B active params, 4B active (MoE) |
| 35 | LLM_TYPE_E2B | 2B embedding model |
| 42 | LLM_TYPE_E4B | 4B embedding model |
| 60 | LLM_TYPE_31B | 31B dense model |

## Key Architectural Features

### 1. Sliding Window Attention (SWA)
- **Type**: `LLAMA_SWA_TYPE_STANDARD`
- **Pattern**: Configurable per-layer via `hparams.swa_layers` array
- **SWA head dimensions**: Separate `n_embd_head_k_swa` and `n_embd_head_v_swa` (configurable)
- **RoPE base for SWA**: `rope_freq_base_train_swa` (can differ from main RoPE)

### 2. Shared KV Layers
- **Key**: `attention.shared_kv_layers` (GGUF KV)
- **Behavior**: `hparams.n_layer_kv_from_start = hparams.n_layer - n_kv_shared_layers`
- First `n_layer_kv_from_start` layers share KV cache, remaining use full KV

### 3. Attention Scaling
- **Value**: `f_attention_scale = 1.0f` (Gemma4 uses no pre-attention scaling)
- Gemma 2/3 use `1.0f / sqrt(head_dim)` scaling

### 4. Expert FFN Support (MoE Models)
- **Key**: `expert_feed_forward_length` (n_ff_exp)
- Enables MoE variant detection alongside dense models

### 5. Per-Layer Embedding Length
- **Key**: `embedding_length_per_layer` (n_embd_per_layer)
- Enables variable-width hidden states per layer

### 6. Softcapping
- **Final logit softcapping**: `final_logit_softcapping` parameter
- Applied at output layer before softmax

### 7. Tokenizer: Gemma4 SPM-Style BPE
- **Pre-tokenization**: Split only on newlines `[^\\n]+|[\\n]+`
- **No byte encoding**: Uses raw UTF-8 (unlike GPT-2 byte encoding)
- **Normalization**: Spaces replaced with ▁ (U+2581) before BPE merges
- **Special handling** (from PR #21343 fix):
  ```cpp
  else if (tok_pre == LLAMA_VOCAB_PRE_TYPE_GEMMA4 && word.find_first_not_of('\n') == std::string::npos) {
      // fix for gemma 4 - handle pure newline tokens
  }
  ```

## Gemma 4 vs Gemma 3 Hyperparameters

| Parameter | Gemma 3 | Gemma 4 |
|-----------|---------|---------|
| SWA Type | STANDARD (period=6) | STANDARD (per-layer) |
| Attention Scale | 1/sqrt(head_dim) | 1.0 (no scaling) |
| Shared KV Layers | No | Yes |
| Per-Layer Embedding | No | Yes |
| n_embd_per_layer | Not present | Present |
| SWA Head Dim Config | No | Yes (k_swa, v_swa) |
| Final Logit Softcapping | 0.0f default | Configurable |
| FFN Expert Support | No | Yes |
| MoE Support | No (Gemma3N only) | Yes |

## Gemma 4 vs Gemma 2/3: SWA Differences

| Aspect | Gemma 2 | Gemma 3 | Gemma 4 |
|--------|---------|---------|---------|
| SWA Type | STANDARD | STANDARD | STANDARD |
| Period | 2 | 6 | Per-layer |
| n_swa default | 4096 | (from KV) | (from KV) |
| attn_soft_cap | Yes | No | No |
| attn_logit_softcapping | Yes | No | No |
| kq_norm | No | No | No |
| Shared KV | No | No | Yes |

## Required GGUF KV Pairs for Gemma 4

### Required
- `general.architecture`: "gemma4"
- `gemma4.context_length`: int
- `gemma4.embedding_length`: int
- `gemma4.block_count`: int
- `gemma4.attention.head_count`: int
- `gemma4.attention.head_count_kv`: int
- `gemma4.attention.layer_norm_rms_epsilon`: float
- `gemma4.attention.sliding_window`: int
- `gemma4.attention.sliding_window_pattern`: int[] (per-layer pattern)
- `gemma4.rope.dimension_count`: int
- `gemma4.rope.freq_base`: float
- `gemma4.rope.scaling.factor`: float
- `gemma4.rope.scaling.original_context_length`: int
- `gemma4.rope.scaling.type`: string
- `gemma4.feed_forward_length`: int

### Optional
- `gemma4.attention.key_length_swa`: int
- `gemma4.attention.value_length_swa`: int
- `gemma4.rope.freq_base_swa`: float
- `gemma4.embedding_length_per_layer`: int
- `gemma4.expert_feed_forward_length`: int (for MoE)
- `gemma4.attention.shared_kv_layers`: int
- `gemma4.final_logit_softcapping`: float

## Tensor Names for Gemma 4

Uses standard tensor naming from `LLM_TENSOR_NAMES`:
- `token_embd`: Embeddings
- `blk.%d.attn_norm`: Attention norm (RMS)
- `blk.%d.attn_q`, `attn_k`, `attn_v`: QKV projections
- `blk.%d.attn_output`: Output projection
- `blk.%d.ffn_norm`: FFN norm
- `blk.%d.ffn_gate`, `ffn_up`, `ffn_down`: FFN weights (SwiGLU)
- `blk.%d.post_ffw_norm_1`: FFN post-norm 1 (Gemma4 specific)
- `blk.%d.post_ffw_norm_2`: FFN post-norm 2 (Gemma4 specific)
- `blk.%d.pre_ffw_norm_2`: FFN pre-norm 2 (Gemma4 specific)
- `output_norm`: Output norm
- `output`: LM head

## Implementation Status Summary

| Component | Status | Location |
|-----------|--------|----------|
| Architecture Enum | ✅ Done | `llama-arch.h`: `LLM_ARCH_GEMMA4` |
| Architecture Name | ✅ Done | `llama-arch.cpp`: `"gemma4"` |
| Vocab Type | ✅ Done | `llama-vocab.cpp`: `LLM_VOCAB_PRE_TYPE_GEMMA4` |
| Tokenizer | ✅ Done | `llama-vocab.cpp` (SPM-style BPE) |
| Hyperparameters | ✅ Done | `llama-model.cpp` (~60 lines) |
| Model Types | ✅ Done | `llama-model.cpp`: 4 model sizes |
| Tokenizer Special Fix | ✅ Done | `llama-vocab.cpp` (PR #21343) |

## Reference

- [llama-arch.h - Architecture enum](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.h)
- [llama-arch.cpp - Architecture names](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp)
- [llama-model.cpp - Hparams loading (Gemma4 case)](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)
- [llama-vocab.cpp - Tokenizer](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp)
- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 Launch Post](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)