# Gemma Family Architecture

This document provides a high-level overview of the Gemma model family architecture, covering all variants and their key differences.

## Family Overview

| Model | Type | Parameters | Context | Key Features |
|-------|------|------------|---------|--------------|
| Gemma 1 2B | Dense | 2B | 8K | MQA, GeGLU |
| Gemma 1 7B | Dense | 7B | 8K | MHA, GeGLU |
| Gemma 2 2B | Dense | 2B | 8K | GQA, SWA, GeGLU |
| Gemma 2 9B | Dense | 9B | 8K | GQA, SWA, GeGLU |
| Gemma 2 27B | Dense | 27B | 8K | GQA, SWA, GeGLU |
| Gemma 3 1B | Dense+Vision | 1B | 128K | Multimodal, GQA |
| Gemma 3 4B | Dense+Vision | 4B | 128K | Multimodal, GQA |
| Gemma 3 12B | Dense+Vision | 12B | 128K | Multimodal, GQA |
| Gemma 3 27B | Dense+Vision | 27B | 32K | Multimodal, GQA |
| Gemma 4 2B | Dense+Vision | ~2B | 128K | Multimodal |
| Gemma 4 4B | Dense+Vision | ~4B | 128K | Multimodal |
| Gemma 4 26B | Dense+Vision | ~26B | 256K | Multimodal |

## Core Architectural Components

### 1. Attention Types

**Gemma 1 2B**: Multi-Query Attention (MQA)
- 8 query heads, 1 KV head
- Memory efficient

**Gemma 1 7B**: Multi-Head Attention (MHA)
- 16 query heads, 16 KV heads

**Gemma 2 (all sizes)**: Grouped-Query Attention (GQA)
- 2B: 8 heads, 4 KV heads
- 9B: 16 heads, 8 KV heads
- 27B: 32 heads, 16 KV heads

**Gemma 3+**: Grouped-Query Attention (GQA)

### 2. Sliding Window Attention (Gemma 2 only)

Gemma 2 introduced local sliding window attention:
- **Window size**: 4096 tokens
- **Global attention**: Alternating layers use global attention
- **Hybrid approach**: Local + global attention pattern

### 3. GeGLU Activations

All Gemma variants use GeGLU (Gated GeLU) instead of standard ReLU:
- `GeGLU(x) = GeLU(xW_1) * xW_2`
- Approximated GeLU for efficiency

### 4. RoPE Position Embeddings

All Gemma variants use Rotary Position Embeddings:
- Neox-style (same as Llama, Mistral)
- Not Qwen-style specialized RoPE

### 5. RMSNorm

All variants use RMSNorm (not LayerNorm):
- More memory efficient
- Similar performance

## Model-Specific Details

### Gemma 1 (2B, 7B)

**Parameters**:
| | 2B | 7B |
|---|---|---|
| d_model | 2048 | 3072 |
| Layers | 18 | 28 |
| FFN hidden | 32768 | 49152 |
| Heads | 8 | 16 |
| KV Heads | 1 | 16 |
| Head size | 256 | 256 |
| Vocab | 256128 | 256128 |

**Key Features**:
- First Gemma release
- Standard transformer (no SWA)
- Different attention per size

### Gemma 2 (2B, 9B, 27B)

**Parameters**:
| | 2B | 9B | 27B |
|---|---|---|---|
| d_model | 2304 | 3584 | 4608 |
| Layers | 26 | 42 | 46 |
| FFN hidden | 18432 | 28672 | 73728 |
| Heads | 8 | 16 | 32 |
| KV Heads | 4 | 8 | 16 |
| Head size | 256 | 256 | 128 |
| Window | 4096 | 4096 | 4096 |

**Key Features**:
- Sliding window + global attention
- GQA for all sizes
- Post-norm in addition to pre-norm

### Gemma 3 (1B, 4B, 12B, 27B)

**Key Features**:
- **Multimodal**: SigLIP vision encoder
- **Long context**: 128K (except 27B: 32K)
- **GQA**: All sizes
- **Text-only support**: Language backbone only

### Gemma 4 (2B, 4B, 26B)

**Key Features**:
- **Multimodal**: Vision + audio
- **Extended context**: 128K (edge), 256K (larger)
- **Latest release**: 2026

## GGUF-Specific Considerations

### Tensor Naming Conventions

Standard HuggingFace naming:
- `model.embed_tokens.weight`
- `model.layers.*.self_attn.q_proj.weight`
- `model.layers.*.self_attn.k_proj.weight`
- `model.layers.*.self_attn.v_proj.weight`
- `model.layers.*.self_attn.o_proj.weight`
- `model.layers.*.mlp.gate_proj.weight` (GeGLU gate)
- `model.layers.*.mlp.up_proj.weight` (GeGLU up)
- `model.layers.*.mlp.down_proj.weight`

### Metadata Fields

```zig
hidden_size                    // d_model
num_hidden_layers             // layers
num_attention_heads            // heads
num_key_value_heads            // KV heads (for GQA)
head_dim                       // head size
intermediate_size              // FFN hidden
vocab_size                     // 256128
max_position_embeddings       // 8192 (G1/G2), 32K-128K (G3/G4)
rope_theta                    // 10000.0
rms_norm_eps                   // 1e-6
```

### Vocabulary

- **256,128 tokens** - Large Gemini vocabulary
- **Tied embeddings** - Input/output embeddings shared
- SentencePiece tokenizer

## Implementation Priority

1. **Gemma 1 (2B, 7B)** - Simplest, MQA/MHA attention
2. **Gemma 2 (2B, 9B, 27B)** - Add GQA + SWA
3. **Gemma 3 text-only** - Same as G2 with extended context
4. **Gemma 3+ multimodal** - Vision encoder (future phase)

## References

- [Gemma 1 Paper](https://arxiv.org/abs/2403.08295)
- [Gemma 2 Technical Report](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)
- [Gemma 3 Report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
- [Gemma Explained Blog](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/)