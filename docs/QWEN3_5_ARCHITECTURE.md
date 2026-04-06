# Qwen 3.5 Architecture Notes

## Dense Models (0.5B, 1.5B, 2B, 3B, 7B)
The smaller variants of the Qwen family (often seen as Qwen 2, 2.5, and the smaller 3.5 models) are standard dense transformers. They rely on the identical computational blocks to LLaMA:
- RMSNorm
- RoPE (with a base frequency typically expanded to 1,000,000)
- Q/K/V Projections
- SwiGLU FFN (`gate` and `up` combined via SiLU, multiplied, and passed to `down`)

These can be natively executed by `llama_metal.zig` and the `llama_cpu.zig` pipelines without architectural changes. The only requirement is parsing their specific quantization spreads (e.g., `Q5_K` usage in `Q4_K_M` files).

## Sparse / Large Models (Qwen 3.5 MoE / DeltaNet)
Larger variants introduce two massive architectural deviations from LLaMA:

1. **Mixture-of-Experts (MoE):**
   Instead of a single massive FFN layer, tokens are dynamically routed.
   *What we need to implement:* 
   - Parse `ffn_gate_inp` (the router weights).
   - Parse arrays of `ffn_down.N` and `ffn_up.N` weights.
   - Implement a Metal kernel to compute Top-K routing logits per token and dispatch those tokens dynamically to specific expert matvecs.

2. **Gated Delta Networks (DeltaNet):**
   Instead of standard quadratic dot-product Attention, long-context Qwen 3.5 models use linear RNN-style attention (DeltaNet).
   *What we need to implement:*
   - Replace standard K/V caches with continuous recurrent state caches.
   - Write custom Metal shaders for the linear DeltaNet absorption formulas.

*Conclusion:* For now, dense variants run via the `llama` backend paths. Sparse MoE models will require a dedicated `qwen35_metal.zig` execution graph.
