# Ministral 3 Implementation Guide

This guide covers implementation details for Ministral 3B (and 8B) variants.

## Model Identification

- **Architecture string in GGUF**: `ministral`
- **HF Model ID**: `mistralai/Ministral-3-3B-Base-2512`, `mistralai/Ministral-3-8B-Instruct-2512`
- **Parameters**: 3B / 8B
- **Release date**: 2025

## Key Architecture Differences from Mistral 7B

| Feature | Ministral 3/8B | Mistral 7B |
|---------|----------------|------------|
| Context | 32K | 8K |
| Position Embeddings | Extended RoPE | Standard RoPE |
| KV Heads | 8 (similar) | 8 (GQA) |

## Implementation Checklist

### Metadata to Parse

```zig
// Required configuration fields
attention.head_count           // 32 (for 8B), varies for 3B
attention.head_count_kv        // 8
attention.sliding_window       // 4096 (likely)
hidden_size                    // 3584 (8B), TBD (3B)
num_hidden_layers             // TBD
intermediate_size             // TBD
max_position_embeddings       // 131072 (32K)
rope_theta                    // likely 10000.0
```

### Current Status

**IMPLEMENTATION NOT STARTED** - More research needed on architecture specifics.

### Notes

- Extended context length (32K) may require special RoPE handling
- Similar SWA to Mistral 7B likely
- Check for any new metadata fields

## References

- [Ministral 3 Documentation](https://docs.mistral.ai/models/ministral-3-8b-25-12)
- [HF Model Card](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)