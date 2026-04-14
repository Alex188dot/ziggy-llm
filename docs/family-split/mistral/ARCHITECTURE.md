# Mistral Family Architecture

This document provides a high-level overview of the Mistral model family architecture, covering all variants and their key differences.

## Family Overview

The Mistral family includes multiple model variants with different architectural features:

| Model | Type | Parameters | Context | Key Features |
|-------|------|------------|---------|--------------|
| Mistral 7B | Dense | 7B | 8K | GQA, SWA |
| Mixtral 8x7B | MoE | 8x7B | 32K | MoE, SWA |
| Ministral 3B | Dense | 3B | 32K | GQA (newer) |
| Ministral 8B | Dense | 8B | 32K | GQA (newer) |

## Core Architectural Components

### 1. Sliding Window Attention (SWA)

Mistral uses sliding window attention where each layer attends to the previous 4,096 hidden states. This provides:
- **Linear compute cost** O(sliding_window * seq_len) 
- **Cache size** limited to sliding_window tokens
- **Long context handling** through stacked layers (attention spans beyond window size)

Implementation notes:
- Window size: 4096 tokens
- Cache management using rotating buffers
- Half the cache memory for sequence length of 8192

### 2. Grouped-Query Attention (GQA)

Mistral 7B uses grouped-query attention:
- Fewer KV heads than query heads
- Shared KV projections across query head groups
- Faster inference with minimal quality loss

### 3. Rotary Position Embeddings (RoPE)

- Used in all Mistral variants
- Neox-style (as opposed to Llama's interleaved style)
- Base frequency needs to be checked per model

## Model-Specific Details

### Mistral 7B (v0.1)

- **Architecture**: Decoder-only transformer
- **Hidden size**: 4096
- **Layers**: 32
- **Attention heads**: 32
- **KV heads**: 8 (GQA)
- **Head size**: 128
- **FFN hidden**: 14336
- **Vocabulary**: ~32K ( SentencePiece)
- **Context**: 8K tokens

### Mixtral 8x7B

- **Type**: Mixture of Experts (MoE)
- **Experts**: 8, with 2 active per token
- **Expert hidden**: 14336
- **Routing**: Top-2 gating
- **Context**: 32K (extended from 8K)

### Ministral 3 / 8B (Newer)

- **Architecture**: Similar to Mistral 7B with improvements
- **Context**: Extended to 32K
- **KV heads**: 8 (for 8B), varies for 3B
- **Position**: Replacement for Mistral 7B (retired)

## GGUF-Specific Considerations

### Tensor Naming Conventions

GGUF files follow HuggingFace naming:
- `model.embed_tokens.weight` - Embedding layer
- `model.layers.*.self_attn.q_proj.weight` - Query projection
- `model.layers.*.self_attn.k_proj.weight` - Key projection  
- `model.layers.*.self_attn.v_proj.weight` - Value projection
- `model.layers.*.self_attn.o_proj.weight` - Output projection
- `model.layers.*.mlp.gate_proj.weight` - FFN gate
- `model.layers.*.mlp.up_proj.weight` - FFN up
- `model.layers.*.mlp.down_proj.weight` - FFN down

### Metadata Fields

Key metadata to parse from GGUF:
- `attention.sliding_window` - Window size (typically 4096)
- `attention.head_count` - Number of heads
- `attention.head_count_kv` - Number of KV heads
- `hidden_size` - Hidden dimension
- `num_hidden_layers` - Layer count
- `rms_norm_eps` - RMSNorm epsilon
- `rope_theta` - RoPE frequency base

### Tokenizer

Mistral uses SentencePiece tokenizer with:
- Vocabulary size: ~32K
- Special tokens: BOS, EOS, UNK, PAD

## Implementation Priority

For GGUF inference support, prioritize in this order:

1. **Mistral 7B** - Original, simplest architecture
2. **Mixtral 8x7B** - MoE added, requires expert routing
3. **Ministral 3/8B** - Newer, longer context (future)

## References

- [Mistral 7B Paper](https://arxiv.org/abs/2310.06825)
- [Mistral 7B Release Post](https://mistral.ai/news/announcing-mistral-7b)
- [HuggingFace Transformers Mistral Config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/configuration_mistral.py)