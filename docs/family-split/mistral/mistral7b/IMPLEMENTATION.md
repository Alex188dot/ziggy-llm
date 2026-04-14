# Mistral 7B Implementation Guide

This guide covers implementation details specific to Mistral 7B v0.1 for GGUF inference support.

## Model Identification

- **Architecture string in GGUF**: `mistral`
- **HF Model ID**: `mistralai/Mistral-7B-v0.1`
- **Parameters**: 7B
- **Release date**: September 2023

## Key Architecture Differences from Llama

| Feature | Mistral 7B | Llama 2 7B |
|---------|------------|------------|
| Attention | GQA (8 KV heads) | MHA (32 KV heads) |
| Sliding Window | 4096 tokens | None |
| RoPE Style | Neox | Interleaved |
| FFN | SwiGLU | SwiGLU |
| Context | 8K | 4K |

## Implementation Checklist

### Metadata to Parse

```zig
// Required configuration fields
attention.head_count           // 32
attention.head_count_kv        // 8 (GQA)
attention.sliding_window        // 4096
hidden_size                    // 4096
num_hidden_layers             // 32
intermediate_size              // 14336
rms_norm_eps                   // 1e-5
vocab_size                     // 32000
rope_theta                     // 10000.0
max_position_embeddings       // 32768 (but effective 8K)
```

### Tensor Mapping

```zig
// Embedding layer
model.embed_tokens.weight -> [32000, 4096]

// Attention layers (32 total)
model.layers[i].input_layernorm.weight -> [4096]
model.layers[i].self_attn.q_proj.weight -> [4096, 4096]
model.layers[i].self_attn.k_proj.weight -> [1024, 4096]  // 8 heads * 128
model.layers[i].self_attn.v_proj.weight -> [1024, 4096]
model.layers[i].self_attn.o_proj.weight -> [4096, 4096]
model.layers[i].post_attention_layernorm.weight -> [4096]

// FFN (SwiGLU)
model.layers[i].mlp.gate_proj.weight -> [14336, 4096]
model.layers[i].mlp.up_proj.weight -> [14336, 4096]
model.layers[i].mlp.down_proj.weight -> [4096, 14336]
```

### Sliding Window Attention Implementation

```zig
// Sliding window attention key aspects:
// 1. Query attends to [i - sliding_window, i] tokens
// 2. Cache size limited to sliding_window
// 3. Rotating buffer for KV cache management
// 4. Higher layers see beyond window through stacking

const SLIDING_WINDOW: usize = 4096;

// Cache management:
// - Store only last sliding_window tokens in KV cache
// - Rotate entries as new tokens are generated
// - Position encoding handles window boundaries
```

### GQA Implementation

```zig
// Grouped-Query Attention:
// - 32 query heads, 8 KV heads
// - Each group of 4 query heads shares 1 KV head
// - This reduces KV cache size significantly

const NUM_QUERY_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_SIZE: usize = 128;

// For each query head group:
// q[i * HEAD_SIZE .. (i+1) * HEAD_SIZE]
// attends to kv[(i / 4) * HEAD_SIZE .. (i/4 + 1) * HEAD_SIZE]
```

### RoPE Implementation

```zig
// Mistral uses Neox-style RoPE (same as Llama, different from Qwen)

const ROPE_THETA: f32 = 10000.0;
const MAX_POS_EMBEDDINGS: usize = 32768;

// RoPE applied to each attention layer
// Unlike Qwen's specialized RoPE, Mistral uses standard Neox
```

## Quantization Support

### Tested Working
- Q4_K_M
- Q5_K_M  
- Q6_K
- Q8_0
- F16
- F32

### Notes
- MoonQuant Q4_K_M path available
- Same quantization handling as Llama family
- SWA doesn't affect quantization

## CPU Inference Notes

1. **Sliding Window Impact**:
   - Memory usage similar to Llama 7B
   - Cache size: 8 * 128 * 4096 * 2 (K+V) = ~8MB per layer
   - Total cache: ~256MB for 32 layers

2. **GQA Benefits**:
   - 4x fewer KV cache entries than MHA
   - Faster attention computation
   - Reduced memory bandwidth

3. **Position Encoding**:
   - RoPE handles positions up to 32768
   - Effective 8K context through SWA
   - No special handling needed beyond SWA

## Metal Backend Notes

1. **Kernel Changes**:
   - SWA requires masked attention kernel
   - Rotating buffer management for KV cache
   - GQA requires grouped projection kernels

2. **Performance**:
   - SWA provides 2x speedup at 16K sequence length
   - GQA reduces memory bandwidth
   - Similar performance profile to Llama 7B

## Testing Checklist

- [ ] Verify SWA produces correct outputs vs full attention
- [ ] Test GQA attention outputs match MHA reference
- [ ] Validate RoPE position encoding across context lengths
- [ ] Compare generation against llama.cpp for identical inputs
- [ ] Benchmark startup time and tok/s vs Llama 7B

## Reference Implementation

- [HF Transformers Mistral](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [Mistral Reference](https://github.com/mistralai/mistral-src)