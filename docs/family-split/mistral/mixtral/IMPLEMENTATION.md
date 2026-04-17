# Mixtral 8x7B Implementation Guide

This guide covers implementation details for Mixtral 8x7B MoE variant.

## Model Identification

- **Architecture string in GGUF**: `mixtral`
- **HF Model ID**: `mistralai/Mixtral-8x7B-v0.1`
- **Parameters**: 8 experts × 7B = 56B total, ~12B active
- **Release date**: December 2023

## Key Architecture Differences from Mistral 7B

| Feature | Mixtral 8x7B | Mistral 7B |
|---------|--------------|------------|
| Type | Mixture of Experts | Dense |
| Experts | 8 (2 active per token) | N/A |
| Context | 32K | 8K |
| Expert FFN | 14336 | 14336 |

## Implementation Checklist

### Metadata to Parse

```zig
// Required configuration fields
num_local_experts            // 8
num_experts_per_token        // 2
attention.head_count         // 32
attention.head_count_kv      // 8
attention.sliding_window      // 4096
hidden_size                  // 4096
num_hidden_layers            // 32
intermediate_size           // 14336
```

### Tensor Mapping

```zig
// Additional tensors for MoE:
// (Same as Mistral 7B, plus expert routing)

model.layers[i].block_sparse_moe.gate.weight -> [8, 4096]  // Router
model.layers[i].block_sparse_moe.experts[0].gate_proj.weight -> [14336, 4096]
model.layers[i].block_sparse_moe.experts[0].up_proj.weight -> [14336, 4096]
model.layers[i].block_sparse_moe.experts[0].down_proj.weight -> [4096, 14336]
// ... experts 1-7
```

### MoE Routing Implementation

```zig
// Top-2 gating implementation:
// 1. Compute gating scores: gate(x) = linear(x)
// 2. Select top-2 experts: top_k(gating_scores, k=2)
// 3. Normalize weights: softmax(top_2)
// 4. Compute output: sum(expert_i(x) * weight_i) for i in top_2

const NUM_EXPERTS: usize = 8;
const ACTIVE_EXPERTS: usize = 2;

// Router:
// gate_logits = gate_proj(x)  // [seq_len, num_experts]
// top_k_indices = argtopk(gate_logits, k=2)
// top_k_weights = softmax(gate_logits[top_k_indices])
```

## Current Status

**IMPLEMENTATION NOT STARTED** - This model is not yet implemented in the runtime.

### Required Work

1. Parse additional expert tensors
2. Implement MoE routing kernel
3. Implement expert parallel matvec
4. Add Metal kernel for MoE
5. Update cache management

## References

- [Mixtral Release Post](https://mistral.ai/news/mixtral-of-experts)
- [MoE Paper](https://arxiv.org/abs/2401.04088)