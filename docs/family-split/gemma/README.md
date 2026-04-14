# Gemma Family Documentation

## Overview

This directory contains architecture and implementation documentation for the Gemma model family.

## Directory Structure

```
gemma/
├── ARCHITECTURE.md     # High-level family overview
├── gemma1/
│   └── IMPLEMENTATION.md  # Gemma 1 (2B, 7B) details
├── gemma2/
│   └── IMPLEMENTATION.md  # Gemma 2 (2B, 9B, 27B) details
├── gemma3/
│   └── IMPLEMENTATION.md  # Gemma 3 (multimodal) details
└── gemma4/
    └── IMPLEMENTATION.md  # Gemma 4 (latest) details
```

## Quick Reference

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| Gemma 1 2B | Not Started | 1 | MQA, GeGLU, 8K |
| Gemma 1 7B | Not Started | 1 | MHA, GeGLU, 8K |
| Gemma 2 2B | Not Started | 2 | GQA, SWA |
| Gemma 2 9B | Not Started | 2 | GQA, SWA |
| Gemma 2 27B | Not Started | 2 | GQA, SWA, 128 head |
| Gemma 3 text | Not Started | 3 | Extended context |
| Gemma 3+ | Future | - | Multimodal |
| Gemma 4 | Not Started | 4 | Extended context |

## Implementation Order

1. **Gemma 1 (2B, 7B)** - Simplest, MQA/MHA, no SWA
2. **Gemma 2 (all)** - Adds GQA + sliding window
3. **Gemma 3 text-only** - Extended context, similar to G2
4. **Gemma 4** - Latest, extended context

## Key Architecture Features

- **GeGLU Activations**: Gated GeLU (different from SwiGLU)
- **Large Vocabulary**: 256,128 tokens
- **Tied Embeddings**: Input/output share same embeddings
- **RMSNorm**: Without bias
- **RoPE**: Neox-style rotary embeddings
- **Gemma 2+**: GQA + sliding window attention

## Resources

- [Gemma Paper](https://arxiv.org/abs/2403.08295)
- [Gemma 2 Report](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)
- [Gemma 3 Report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
- [Gemma Explained Blog](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/)