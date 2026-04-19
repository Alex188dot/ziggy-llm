# Gemma 2 Implementation Guide

This guide covers implementation details for Gemma 2 (2B, 9B, 27B) for GGUF inference support.

## Model Identification

- **Architecture string in GGUF**: `gemma2`
- **HF Model IDs**: 
  - 2B: `google/gemma-2-2b`
  - 9B: `google/gemma-2-9b`
  - 27B: `google/gemma-2-27b`
- **Parameters**: 2B / 9B / 27B
- **Release date**: June 2024

## Key Architecture Differences from Gemma 1

| Feature | Gemma 2 | Gemma 1 |
|---------|---------|---------|
| Attention | GQA (all sizes) | MQA (2B), MHA (7B) |
| Sliding Window | 4096 (local) | None |
| Global Attention | Alternating layers | None |
| Head Size | 128 (27B), 256 (2B/9B) | 256 |
| Norm | Pre + Post | Pre only |

## Implementation Checklist

### Metadata to Parse

```zig
// Gemma 2 2B
attention.head_count           // 8
attention.head_count_kv        // 4 (GQA)
hidden_size                    // 2304
num_hidden_layers             // 26
intermediate_size             // 18432
head_dim                      // 256
attention.sliding_window       // 4096

// Gemma 2 9B
attention.head_count           // 16
attention.head_count_kv        // 8 (GQA)
hidden_size                    // 3584
num_hidden_layers             // 42
intermediate_size             // 28672
head_dim                      // 256

// Gemma 2 27B
attention.head_count           // 32
attention.head_count_kv       // 16 (GQA)
hidden_size                   // 4608
num_hidden_layers             // 46
intermediate_size             // 73728
head_dim                      // 128
```

### Sliding Window + Global Attention

```zig
// Gemma 2 uses hybrid attention:
// - Local: Sliding window of 4096 tokens
// - Global: Full attention on alternating layers

const SLIDING_WINDOW: usize = 4096;

// Pattern (for 46 layers):
// Layer 0: Global
// Layer 1: Local (window=4096)
// Layer 2: Global
// Layer 3: Local
// ...

// Cache management:
// - Local layers: Cache size limited to 4096
// - Global layers: Full cache (context length)
```

### GQA Implementation

All Gemma 2 variants use GQA:

```zig
// Gemma 2 27B Example:
// 32 query heads, 16 KV heads, 128 head size

const NUM_QUERY_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 16;
const HEAD_SIZE: usize = 128;

// Each group of 2 query heads shares 1 KV head
// Query: [seq_len, 32 * 128] = [seq_len, 4096]
// Key: [seq_len, 16 * 128] = [seq_len, 2048]
// Value: [seq_len, 16 * 128] = [seq_len, 2048]
```

### Normalization

```zig
// Gemma 2 uses both pre-norm and post-norm:
// (Unlike Gemma 1 which uses only pre-norm)

model.layers[i].input_layernorm.weight    // Pre-norm
model.layers[i].post_attention_layernorm.weight  // Post-norm
```

## Quantization Support

### Tested Working (expected)
- Q4_K_M
- Q5_K_M
- Q6_K
- Q8_0

### Notes
- Same quantization approach as Gemma 1
- Smaller head size (128 for 27B) may affect accuracy

## Current Status

**IMPLEMENTATION NOT STARTED** - Requires:
1. Sliding window attention implementation
2. Global/local layer alternation
3. GQA kernels (similar to Mistral)
4. Dual normalization (pre + post)

## Testing Checklist

- [ ] Verify hybrid local/global attention
- [ ] Test GQA attention outputs
- [ ] Validate both pre-norm and post-norm
- [ ] Compare head size 128 vs 256 quantization
- [ ] Benchmark vs Gemma 1 and Llama 3

## Reference Implementation

- [Gemma 2 Report](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)
- [HF Transformers Gemma2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py)

## 2026-04-19 Fix Plan (Gibberish Output Investigation)

### Problem Statement

Gemma 2 and Gemma 3 produce gibberish/unintelligible output. Investigation shows likely issues:

1. **Critical Bug - `embedding_scale` not applied to Gemma 3**
   - Location: `src/model/loader.zig:2278-2281`
   - Issue: Gemma 3 uses `embedding_scale = sqrt(embedding_length)` but this is only applied to Gemma 1 and 2
   - Fix: Add `"gemma3"` to the condition

2. **Sliding Window Attention Implementation**
   - Gemma 2 uses hybrid local/global attention (alternating layers)
   - Global attention layers: full context
   - Local attention layers: 4096 token window
   - Need to verify `global_attention_interval = 2` is correct

3. **Post-Norm Application**
   - Gemma 2 has `post_attention_norm` and `post_ffw_norm`
   - Must verify these are being applied in the attention and FFN blocks

### Implementation Steps

1. **Step 1: Add Diagnostic Tool**
   - Enhance existing `inspect` command to dump all GGUF metadata
   - Add tensor listing capability
   - Focus on Gemma-specific metadata fields

2. **Step 2: Run Diagnostics**
   - Inspect `gemma-2-2b-it-Q4_K_M.gguf` (bartowski)
   - Capture all metadata fields
   - Verify tensor shapes and names

3. **Step 3: Fix Critical Bug (Gemma 3 embedding_scale)**
   - File: `src/model/loader.zig`
   - Line ~2278: Add `gemma3` to embedding_scale condition

4. **Step 4: Validate Prompt Format**
   - Gemma 2 chat format:
     ```
     <bos><start_of_turn>user
     {prompt}<end_of_turn>
     <start_of_turn>model
     <end_of_turn>
     <start_of_turn>model
     ```
   - Verify `chat_prompt.zig` handles this correctly

5. **Step 5: Compare with llama.cpp Reference**
   - Generate same text with same seed/temperature
   - Compare output token-by-token
   - Narrow down which computation step is wrong

### Files to Modify

| File | Change |
|------|--------|
| `src/model/loader.zig` | Fix embedding_scale for Gemma 3; verify RoPE params |
| `src/commands.zig` | Ensure inspect command dumps all metadata |
| `src/runtime/families/gemma2/runtime.zig` | Add validation/debug capabilities |
| `src/chat_prompt.zig` | Verify Gemma 2 prompt format handling |

### Validation Plan

1. Run inspect on Gemma 2 GGUF and document all metadata
2. Fix embedding_scale bug
3. Test with simple prompt "Hello" using both zig_ and llama.cpp
4. Compare outputs with identical seed/temperature settings
5. If mismatch persists, add intermediate debug outputs (layer-wise activations)