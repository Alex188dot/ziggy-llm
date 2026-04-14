# Gemma 1 Implementation Guide

This guide covers implementation details specific to Gemma 1 (2B and 7B) for GGUF inference support.

## Model Identification

- **Architecture string in GGUF**: `gemma`
- **HF Model IDs**: 
  - 2B: `google/gemma-2b`
  - 7B: `google/gemma-7b`
- **Parameters**: 2B / 7B
- **Release date**: February 2024

## Key Architecture Differences from Llama

| Feature | Gemma 1 2B | Gemma 1 7B | Llama 2 7B |
|---------|------------|------------|------------|
| Attention | MQA | MHA | MHA |
| KV Heads | 1 | 16 | 32 |
| Vocab Size | 256128 | 256128 | 32000 |
| Activations | GeGLU | GeGLU | SwiGLU |
| Norm | RMSNorm | RMSNorm | RMSNorm |
| Embeddings | Tied | Tied | Untied |

## Implementation Checklist

### Metadata to Parse

```zig
// Gemma 2B
attention.head_count           // 8
attention.head_count_kv        // 1 (MQA)
hidden_size                    // 2048
num_hidden_layers             // 18
intermediate_size             // 32768
head_dim                      // 256
vocab_size                    // 256128

// Gemma 7B
attention.head_count           // 16
attention.head_count_kv       // 16 (MHA)
hidden_size                   // 3072
num_hidden_layers             // 28
intermediate_size             // 49152
head_dim                      // 256
vocab_size                    // 256128
```

### Tensor Mapping

```zig
// Gemma 1 7B Example:
// Embedding layer (tied)
model.embed_tokens.weight -> [256128, 3072]

// Attention (32 layers)
model.layers[i].input_layernorm.weight -> [3072]
model.layers[i].self_attn.q_proj.weight -> [3072, 4096]  // 16 heads * 256
model.layers[i].self_attn.k_proj.weight -> [4096, 3072]  // 16 heads * 256
model.layers[i].self_attn.v_proj.weight -> [4096, 3072]
model.layers[i].self_attn.o_proj.weight -> [3072, 4096]
model.layers[i].post_attention_layernorm.weight -> [3072]

// FFN - GeGLU (not SwiGLU!)
model.layers[i].mlp.gate_proj.weight -> [49152, 3072]
model.layers[i].mlp.up_proj.weight -> [49152, 3072]
model.layers[i].mlp.down_proj.weight -> [3072, 49152]
```

### MQA vs MHA Implementation

**Gemma 1 2B (MQA)**:
```zig
// 8 query heads, 1 KV head, 256 head size
// KV projection: [256, 2048] (not [4096, 2048])
// Broadcast same KV across all query heads

const NUM_QUERY_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 1;
const HEAD_SIZE: usize = 256;

// Q: [seq_len, 8 * 256] = [seq_len, 2048]
// K: [seq_len, 1 * 256] = [seq_len, 256]
// V: [seq_len, 1 * 256] = [seq_len, 256]

// Broadcast K,V to all query heads:
// K_expanded: [seq_len, 8 * 256] - repeat each KV 8 times
// V_expanded: [seq_len, 8 * 256] - repeat each VV 8 times
```

**Gemma 1 7B (MHA)**:
```zig
// 16 query heads, 16 KV heads, 256 head size
// Standard multi-head attention

const NUM_QUERY_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 16;
const HEAD_SIZE: usize = 256;
```

### GeGLU Implementation

```zig
// GeGLU(x) = GeLU(xW_gate) * xW_up
// Not SwiGLU which uses SiLU(xW_gate) * xW_up

// gate_proj: linear(d_model, ffn_hidden)
// up_proj: linear(d_model, ffn_hidden)
// down_proj: linear(ffn_hidden, d_model)

// GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

## Quantization Support

### Tested Working (expected)
- Q4_K_M
- Q5_K_M
- Q6_K
- Q8_0

### Notes
- Large vocab (256K) affects embedding quantization
- Tied embeddings - only one embedding matrix to quantize

## CPU Inference Notes

1. **MQA Benefits** (2B):
   - 16x fewer KV cache entries than MHA
   - Significant memory savings
   - Faster attention computation

2. **Large Vocabulary**:
   - 256K vs 32K (Llama)
   - Embedding layer: ~1GB for F32
   - Needs careful quantization

3. **GeGLU**:
   - Different from SwiGLU
   - Need separate kernel or wrapper
   - GeLU approximation is efficient

## Metal Backend Notes

1. **Kernel Changes**:
   - GeGLU kernel needed (vs SwiGLU)
   - MQA broadcast kernel for 2B
   - MHA is standard for 7B

2. **Performance**:
   - MQA much faster for 2B
   - Large vocab affects embedding lookup

## Testing Checklist

- [ ] Verify MQA attention outputs (2B)
- [ ] Verify MHA attention outputs (7B)
- [ ] Test GeGLU activation vs SwiGLU
- [ ] Validate large vocab tokenization
- [ ] Compare generation against llama.cpp
- [ ] Test tied embedding handling

## Reference Implementation

- [Gemma Paper](https://arxiv.org/abs/2403.08295)
- [HF Transformers Gemma](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py)