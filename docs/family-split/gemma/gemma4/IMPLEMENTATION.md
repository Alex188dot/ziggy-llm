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

**NOT YET RESEARCHED** - Gemma 4 is the newest release (April 2026).

### Notes

- Latest Gemma model
- Adds audio support in addition to vision
- Extended context up to 256K
- For text-only GGUF inference:
  - Likely similar to Gemma 3 text backbone
  - Extended position embeddings needed
  - GQA attention

## Reference

- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 Launch Post](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)