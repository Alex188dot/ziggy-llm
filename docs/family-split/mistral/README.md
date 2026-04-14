# Mistral Family Documentation

## Overview

This directory contains architecture and implementation documentation for the Mistral model family.

## Directory Structure

```
mistral/
├── ARCHITECTURE.md          # High-level family overview
├── mistral7b/
│   └── IMPLEMENTATION.md    # Mistral 7B v0.1 details
├── mixtral/
│   └── IMPLEMENTATION.md    # Mixtral 8x7B MoE details
└── ministral3/
    └── IMPLEMENTATION.md   # Ministral 3/8B details
```

## Quick Reference

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| Mistral 7B v0.1 | Not Started | 1 | SWA + GQA |
| Mixtral 8x7B | Not Started | 2 | MoE routing |
| Ministral 3B/8B | Not Started | 3 | Extended context |

## Implementation Order

1. **Mistral 7B** - Original, most similar to existing Llama support
2. **Mixtral** - Adds MoE routing
3. **Ministral** - Extended context

## Key Architecture Features

- **Sliding Window Attention**: 4096 token window
- **Grouped-Query Attention**: 8 KV heads (vs 32 for Llama)
- **Neox RoPE**: Standard rotary embeddings
- **SwiGLU**: Gated linear unit activations

## Resources

- [Mistral 7B Paper](https://arxiv.org/abs/2310.06825)
- [Official Model Page](https://docs.mistral.ai/models/mistral-7b-0-1)
- [HF Transformers Config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/configuration_mistral.py)