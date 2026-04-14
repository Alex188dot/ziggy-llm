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