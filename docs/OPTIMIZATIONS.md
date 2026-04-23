# Performance Optimizations Roadmap

This document catalogs every known inefficiency in `ziggy-llm` that costs tokens-per-second (TPS). Each item is organized by the architectural layer it touches, with a conservative TPS speedup estimate and a concrete checklist.

> **Conventions**
> - Conservative estimates assume an M3 MacBook Pro, 7B Q4_K_M model, 128-token prompt.
> - Checklists follow the same style as other docs in this repo: `- [ ]` for pending, `- [x]` for done.

---

## Phase 1: GPU Synchronization & Command Buffer Batching

### 1.1 Immediate Commit/Wait in Kernel Helpers
**What it does:** `bridge.m` helpers like `ziggy_run_compute`, `ziggy_run_rowwise_matvec`, and `ziggy_run_single_threadgroup` commit and wait for the command buffer to complete when `state.pendingCommandBuffer == nil`. This means *every individual kernel* that is not explicitly wrapped in `beginSequence`/`commitSequence` forces a full GPU round-trip.

**Where:** `bridge.m` (lines 355-427, 429-492, and every kernel wrapper that calls them without a pending buffer).

**How to correct:**
- Ensure `beginSequence` is called at the start of every decode step.
- Route *all* kernel dispatches through the pending command buffer path.
- Defer `commitSequence` until the very end of the token (after sampling readback).
- For kernels that currently use `ziggy_run_compute` (which auto-commits when no pending buffer exists), switch them to the pending-buffer path or add a `force_batch` flag.

**Benefits:** Eliminates ~20-40 microsecond CPU-GPU ping-pong per kernel. On a 32-layer model this is 60-100 kernels per token.

**Expected conservative TPS speedup:** +8-15 %

- [ ] Audit every kernel wrapper in `bridge.m` to ensure it prefers the pending command buffer.
- [ ] Add a runtime assertion that `pendingCommandBuffer != nil` inside `ziggy_run_compute` when compiled in debug mode.
- [ ] Remove the `has_pending ? return : commit+wait` fallback from all non-terminal kernels.
- [ ] Document the rule: "Only the final sampling readback may trigger a commit."

---

### 1.2 Attention Block Host Readbacks
**What it does:** `runAttentionBlock` in `session.zig` (lines 448-453) reads back the attention output to CPU, applies a sigmoid gate element-wise, then writes it back to GPU. This inserts a full GPU sync mid-layer.

**Where:** `src/runtime/gpu/session.zig` lines 448-453.

**How to correct:** Move the `q_gate` sigmoid scale into a dedicated Metal kernel (or fuse it into the existing `sigmoid_scale_add_f32` kernel) so it stays on GPU.

**Benefits:** Removes one full readback+writeback per layer per token.

**Expected conservative TPS speedup:** +3-6 %

- [ ] Implement a GPU-side `sigmoid_scale_add_f32` variant that accepts a packed q/gate buffer.
- [ ] Remove `host_attn_values` readback path for gated attention.
- [ ] Validate correctness against CPU reference.

---

### 1.3 RoPE Style 2 Host Fallback
**What it does:** When `rope_style == 2` (imrope), `applyRoPEBuffer` reads the entire Q/K buffer back to CPU, runs `applyImrope` in Zig, then writes back. This is a full sync per head per layer per token.

**Where:** `src/runtime/gpu/session.zig` lines 768-772; `applyImrope` in same file.

**How to correct:** Implement `apply_rope_imrope_f32` as a Metal kernel.

**Benefits:** Removes readback/writeback for Qwen-style 3D rope.

**Expected conservative TPS speedup:** +2-4 % (only affects Qwen models).

- [ ] Write `apply_rope_imrope_f32` kernel with section-based theta scaling.
- [ ] Add pipeline state compilation in `bridge.m`.
- [ ] Remove host fallback path for `rope_style == 2`.
- [ ] Test on Qwen 2.5 / Qwen3 fixtures.

---

### 1.4 Linear Attention CPU Bottleneck
**What it does:** `runLinearAttention` runs the convolution, recurrence, and normalization entirely on the CPU after reading back five projection buffers.

**Where:** `src/runtime/gpu/session.zig` lines 1169-1305.

**How to correct:**
- Implement the 1D convolution, state update, and per-head normalization as Metal kernels.
- Keep only the small scalar parameter reads (`dt_bias`, `A_log`) on CPU.
- The recurrent state can stay in a GPU buffer and be updated in-place.

**Benefits:** Moves the bulk of linear-attention work back to the GPU where memory bandwidth is higher.

**Expected conservative TPS speedup:** +10-20 % (dominant for linear-attention models like Ministral 3B).

- [ ] Implement `linear_conv1d_f32` kernel.
- [ ] Implement `linear_recurrent_update_f32` kernel.
- [ ] Implement `linear_norm_silu_gate_f32` kernel.
- [ ] Remove `host_linear_*` readback arrays from the hot path.
- [ ] Add end-to-end correctness test for linear attention on GPU.

---

## Phase 2: Memory Layout & Coalescing

### 2.1 Column-Major Matrix Storage on GPU
**What it does:** All dense matrices are stored column-major (`matrix[row + col * rows]`). In `matvec_f32`, each thread works on one row and threads in a threadgroup access strided columns. This destroys GPU memory coalescing—neighboring threads read memory hundreds or thousands of bytes apart.

**Where:** `matvec.metal` (all matvec kernels); `tensor_store.zig` (dequantization loop).

**How to correct:**
- **Option A (preferred):** Transpose all dense matrices to row-major at load time (`tensor_store.zig`). Update all dense matvec kernels to read contiguous `float4` blocks per row.
- **Option B:** Change the thread mapping so threads within a simdgroup work on adjacent columns of the same row. This is less efficient because it requires intra-simdgroup reduction.

**Benefits:** Coalesced loads can saturate ~80 % of GPU memory bandwidth versus ~20-30 % today.

**Expected conservative TPS speedup:** +15-25 %

- [ ] Add a row-major transpose step in `DenseTensorStore.addTensor` for `tensor_type_f32`.
- [ ] Update `matvec_f32`, `matvec_add_f32`, `batch_matvec_add_f32` to read `float4` from row-major.
- [ ] Update `matvec_q4k_f32` and variants to load Q4_K blocks in row-major order (may require transposing block layout).
- [ ] Benchmark memory bandwidth before/after with a standalone matvec harness.
- [ ] Validate all model families still produce identical outputs.

---

### 2.2 Strided Quantized Access in Q4_K / Q6_K / Q8_0
**What it does:** Quantized matvec kernels read `row_bytes = matrix + row * row_stride`. Within a threadgroup, each thread reads a different row, so the access pattern is naturally coalesced *per row* but not across rows. The bigger issue is that inside the loop, the Q4_K kernel reads individual `uchar4` and `float4` blocks that may not be aligned for vector loads.

**Where:** `matvec.metal` lines 220-292 (Q4K), 370-434 (Q6K), 522-565 (Q8_0).

**How to correct:**
- Use `device const uint4*` or `device const packed_float4*` for input vectors when aligned.
- Prefetch scales into a small `threadgroup` cache to avoid repeated `read_half_le` calls.
- For MoonQuant Q4_K, the 4-row grouping is good; replicate that pattern for generic Q4_K.

**Benefits:** Fewer scalar ALU ops, better cache utilization inside the shader.

**Expected conservative TPS speedup:** +5-10 %

- [ ] Ensure all quantized input buffers are 16-byte aligned.
- [ ] Add `const device float4*` vector loads for input vectors in Q4_K/Q6_K/Q8_0.
- [ ] Prefetch `d`, `dmin`, and scales into registers before the inner dot loop.
- [ ] Profile with Metal System Trace to verify reduced scalar ALU percentage.

---

### 2.3 Per-Row Heap Allocation During Dequantization
**What it does:** `DenseTensorStore.addTensor` allocates `row_dense` for every single row when converting quantized tensors to dense. On a 7B model this is tens of thousands of tiny allocations, causing allocator contention and slow startup.

**Where:** `src/runtime/gpu/metal/tensor_store.zig` lines 208-216.

**How to correct:**
- Allocate a single scratch buffer of `cols` floats outside the row loop.
- Reuse it for every row.
- Or, better: skip dense dequantization entirely for GPU-bound models and keep everything raw/quantized on GPU.

**Benefits:** Faster model loading (lower TTFT) and reduced memory fragmentation.

**Expected conservative TPS speedup:** +1-3 % (mainly TTFT improvement).

- [ ] Replace per-row `allocator.alloc(f32, cols)` with one pre-allocated scratch buffer.
- [ ] Measure startup time difference on a 7B Q4_K_M model.

---

## Phase 3: Kernel Micro-Optimizations

### 3.1 Attention Kernel Register Pressure & Occupancy
**What it does:** `attention_fused_f32` declares `float local_out[ZIGGY_MAX_HEAD_DIM];` where `ZIGGY_MAX_HEAD_DIM = 256`. This is 1 KB of register memory per thread, limiting threadgroup occupancy to at most ~2-4 warps/simdgroups per core. It also does a scalar loop over `head_dim` for the Q·K dot product.

**Where:** `matvec.metal` lines 938-1053.

**How to correct:**
- Reduce `ZIGGY_MAX_HEAD_DIM` to the actual head dimension at dispatch time, or use a dynamically sized threadgroup allocation.
- Vectorize the Q·K dot product with `simd_shuffle_xor` or `simd_sum` across the simdgroup, with each lane handling a slice of `head_dim`.
- Use `half` precision for the softmax accumulator when possible.

**Benefits:** Higher occupancy means more heads run in parallel; vectorized dot products use SIMD units efficiently.

**Expected conservative TPS speedup:** +8-12 %

- [ ] Change `local_out` from fixed array to `threadgroup` shared memory sized by actual `head_dim`.
- [ ] Vectorize Q·K dot with simdgroup reductions.
- [ ] Profile occupancy with Metal System Trace before/after.
- [ ] Verify numerical stability with `half` softmax for models that use it.

---

### 3.2 Q8_0 Scalar Inner Loop
**What it does:** The Q8_0 kernel has an inner loop `for (uint value_index = 0; value_index < 32; value_index += 1)` doing scalar multiply-accumulate. This is 32 scalar FMULs where a single `dot(float4, float4)` or SIMD shuffle could do the work.

**Where:** `matvec.metal` lines 543-550 and 588-595.

**How to correct:**
- Load the 32 `char` values into a `char32` or four `float8` vectors.
- Use SIMD-group reductions to sum partial results.

**Benefits:** ~4x fewer ALU instructions per block.

**Expected conservative TPS speedup:** +3-5 %

- [ ] Rewrite Q8_0 inner loop with `char4`/`float4` vector loads.
- [ ] Add a micro-benchmark kernel that measures Q8_0 matvec throughput in isolation.

---

### 3.3 RoPE `pow` Per Pair
**What it does:** Both `apply_rope_f32` and `apply_rope_to_dst_f32` compute `pow(freq_base, exponent)` inside the kernel for every pair, every head, every token. `pow` is expensive in Metal.

**Where:** `matvec.metal` lines 860-878, 907-935.

**How to correct:** Precompute a theta-frequency buffer on CPU once per model load, upload it as a small constant buffer, and index into it. Or compute `theta` incrementally with multiplication instead of `pow`.

**Benefits:** Replaces ~100 `pow` calls per token with simple multiplications.

**Expected conservative TPS speedup:** +2-4 %

- [ ] Precompute `1.0 / pow(freq_base, exponent)` table on CPU.
- [ ] Pass it as `constant float *theta_inv_table` to RoPE kernels.
- [ ] Update `apply_rope_f32`, `apply_rope_to_dst_f32`, and `apply_rope_at_offset_f32`.

---

### 3.4 GELU `tanh` Approximation
**What it does:** `gelu_mul_f32` uses the exact `tanh` formulation. `tanh` is among the most expensive scalar ops in Metal.

**Where:** `matvec.metal` lines 1066-1078.

**How to correct:** Use the fast sigmoid/GELU approximation from llama.cpp or the Hugging Face `fast_gelu`:
```metal
float gelu_fast(float x) {
    const float c = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + c * x3)));
}
```
Alternatively, use a rational approximation or a lookup table.

**Benefits:** ~2x faster GELU evaluation.

**Expected conservative TPS speedup:** +1-2 % (small because GELU is only one kernel per layer).

- [ ] Replace `tanh` GELU with a fast polynomial approximation.
- [ ] Verify maximum relative error < 1e-4 against reference.

---

### 3.5 RMSNorm Threadgroup Under-utilization
**What it does:** `rms_norm_f32` uses a single threadgroup of size `thread_width * 4` (usually 128 or 256) to process one vector. For `embedding_length` of 4096, each thread only does ~16 FMAs. The reduction barrier cost dominates.

**Where:** `matvec.metal` lines 1089-1129.

**How to correct:**
- Process multiple rows (or multiple heads) per threadgroup to amortize barrier cost.
- For `rms_norm_per_head_f32`, dispatch one threadgroup per head but with more lanes active.

**Benefits:** Better ALU utilization, fewer idle lanes.

**Expected conservative TPS speedup:** +2-4 %

- [ ] Fused batch RMSNorm: process `N` rows per threadgroup where `N = max_total_threads / embedding_threads`.
- [ ] Benchmark with `embedding_length` values of 2048, 4096, and 8192.

---

## Phase 4: CPU Path Inefficiencies

### 4.1 Naive Single-Threaded CPU MatVec
**What it does:** `cpu_backend.zig` implements a literal textbook matvec with zero vectorization, zero threading, and column-major access (`matrix[row + col * rows]`).

**Where:** `src/runtime/cpu_backend.zig`.

**How to correct:**
- Use Zig's `@Vector` types for 4- or 8-wide FMA.
- Add a simple thread-pool for rows > `parallel_matvec_min_rows` (the constant already exists in `loader.zig`).
- Transpose to row-major on CPU for better cache locality.

**Benefits:** CPU fallback becomes usable for small models and prompt processing.

**Expected conservative TPS speedup:** +20-40 % on CPU path (affects prompt processing and TTFT).

- [ ] Rewrite `cpuMatVec` with `@Vector(8, f32)` dot accumulation.
- [ ] Spawn `std.Thread` workers when `rows * cols > parallel_matvec_min_work`.
- [ ] Add a CPU micro-benchmark to `bench_runner.zig`.

---

### 4.2 Prompt Processing Not Batched
**What it does:** The CPU path processes prompt tokens one at a time. There is no batched prompt-processing kernel.

**Where:** `src/model/loader.zig` (inference loop).

**How to correct:** Implement a batched matmul for the prompt phase so all prompt positions are computed in parallel. On Metal this means a `matmul_f32` kernel instead of repeated `matvec` calls.

**Benefits:** Prompt processing (TTFT) can be 10-50x faster.

**Expected conservative TPS speedup:** -50 % TTFT (not decode TPS, but user-perceived latency).

- [ ] Add a `matmul_f32` kernel for prompt batching.
- [ ] Use it for the first `prompt_len` tokens before switching to matvec decode.
- [ ] Ensure KV cache is filled correctly for all positions.

---

## Phase 5: Sampling & Output Reduction

### 5.1 Full Logits Readback for Sampling
**What it does:** When `temperature > 0`, the code falls back to `.cpu_logits`, which reads the entire `vocab_size` logits buffer back to CPU for softmax + multinomial sampling.

**Where:** `src/runtime/types.zig` `resolveSamplingPath`; `src/runtime/gpu/session.zig` `runOutput`.

**How to correct:**
- Implement GPU-side top-p / min-p filtering and a GPU multinomial sampler.
- Only read back the sampled token index.

**Benefits:** Removes a ~32KB-128KB readback per token.

**Expected conservative TPS speedup:** +5-10 % for `temperature > 0` use cases.

- [ ] Implement `top_p_filter_f32` and `multinomial_sample_f32` kernels.
- [ ] Wire them into `resolveSamplingPath` for `.gpu_topk_sample` when temperature > 0.
- [ ] Remove `cpu_logits` fallback for standard sampling configs.

---

### 5.2 TopK Kernel Insertion Sort
**What it does:** `topk_f32` and `sample_topk_f32` use an insertion-sort-like loop (`for (uint slot = 0; slot < top_k; slot++)`) inside each thread. For `top_k = 64` and `vocab_size = 128256`, this is `O(64 * 128256)` comparisons per token.

**Where:** `matvec.metal` lines 1232-1315, 1362-1470.

**How to correct:**
- Use a bitonic or radix selection algorithm.
- Or, use a two-stage reduction: first stage finds local top-k per simdgroup, second stage merges.

**Benefits:** Reduces top-k from ~8M comparisons to ~200K.

**Expected conservative TPS speedup:** +3-6 %

- [ ] Implement a two-stage bitonic top-k.
- [ ] Benchmark against current `topk_f32` with `vocab_size` 32000 and 128256.

---

### 5.3 Argmax Atomic Contention
**What it does:** `matvec_q6k_argmax_f32` uses a global `atomic_uint` compare-exchange loop across all threadgroups. At high row counts, threads spin-wait on the same atomic.

**Where:** `matvec.metal` lines 441-520.

**How to correct:**
- First reduce within each simdgroup, then within each threadgroup, then do a single atomic update per threadgroup.
- Or, write partial maxima to a threadgroup buffer and do a final serial reduction in lane 0.

**Benefits:** Eliminates global atomic contention.

**Expected conservative TPS speedup:** +2-4 % (only when using Q6_K argmax path).

- [ ] Replace global atomic loop with simdgroup+threadgroup staged reduction.
- [ ] Add a correctness test that argmax still returns the correct token on ties.

---

## Phase 6: Startup & Memory Overhead

### 6.1 60+ Pipeline States at Init
**What it does:** `bridge.m` compiles ~65 individual compute pipelines at context creation. Each pipeline consumes Metal driver memory and increases initialization time.

**Where:** `bridge.m` lines 494-878.

**How to correct:**
- Lazily compile pipelines on first use.
- Group related kernels into a single dynamic-dispatch kernel (e.g., one `matvec` kernel that switches on quantization type via a constant buffer).

**Benefits:** Faster startup (lower TTFT), lower static memory footprint.

**Expected conservative TPS speedup:** N/A on decode TPS, but -20 % TTFT.

- [ ] Implement lazy pipeline compilation: store pipeline names in a lookup table and compile on first dispatch.
- [ ] Measure `ziggy_metal_create_context` duration before/after.

---

### 6.2 Metal Buffer Address-Key Caching
**What it does:** `metal_backend.zig` caches matrix buffers by `(@intFromPtr(matrix.ptr), matrix.len)`. If the model is reloaded or the OS moves memory, the cache becomes invalid but the key stays the same. More importantly, the cache lookup is a hashmap operation on every matvec dispatch.

**Where:** `src/runtime/metal_backend.zig` lines 120-134.

**How to correct:**
- Pre-upload all tensors to Metal during `prewarm` and store their `BufferHandle` directly in `DenseTensorStore` or `Session`.
- Remove the per-dispatch hashmap lookup entirely.

**Benefits:** Removes a Zig-side hashmap lookup (~50-100 ns) from every matvec call.

**Expected conservative TPS speedup:** +1-2 %

- [ ] Add a `metal_buffer: ?BufferHandle` field to `TensorDesc` or `DenseTensorStore`.
- [ ] Populate it during `prewarm`.
- [ ] Update `runProjection` to use the pre-cached handle directly.

---

### 6.3 Profiler `nanoTimestamp` Syscalls in Hot Path
**What it does:** `session.zig` calls `std.time.nanoTimestamp()` before and after every kernel dispatch when profiling is enabled. `nanoTimestamp` is a syscall (`mach_absolute_time`) which has non-trivial overhead.

**Where:** `src/runtime/gpu/session.zig` (throughout `runProjection`, `runRmsNorm`, etc.).

**How to correct:**
- Use `metal_backend.commitSequenceTimed` GPU timestamps for kernel duration instead of CPU timestamps.
- Or, read `mach_absolute_time` once per token, not per kernel.

**Benefits:** Eliminates ~20-40 syscalls per token when profiling.

**Expected conservative TPS speedup:** +1-3 % (only when profiling is on, but profiling is needed for optimization work).

- [ ] Add a compile-time flag to replace per-kernel timestamps with GPU timer queries.
- [ ] Batch profile events into a ring buffer and flush once per token.

---

## Phase 7: MoE & Advanced Architectures

### 7.1 Serial Expert Execution
**What it does:** `runMoeFfnBlock` loops over `expert_used_count` experts sequentially, dispatching one indexed matvec at a time. There is no parallelism between experts.

**Where:** `src/runtime/gpu/session.zig` lines 539-572.

**How to correct:**
- Dispatch all expert gate projections in parallel using the batch kernels.
- For the top-k selected experts, use a single kernel that indexes and accumulates all selected experts.

**Benefits:** MoE layers become roughly as fast as dense FFN layers.

**Expected conservative TPS speedup:** +15-30 % on MoE models.

- [ ] Implement `batch_indexed_matvec_iq3_xxs_f32` for parallel expert dispatch.
- [ ] Fused MoE output accumulation into one kernel.
- [ ] Benchmark on a Mixtral/Qwen-MoE fixture.

---

### 7.2 Dual-Indexed Matvec Barrier
**What it does:** `runDualIndexedProjectionIQ3XXS` computes gate and up projections in one kernel, but the result is still stored to separate buffers before the SiLU multiplication.

**Where:** `src/runtime/gpu/session.zig` lines 989-1024; `matvec.metal` lines 2165-2228.

**How to correct:** Fuse gate+up+SiLU into a single kernel that writes the final activated output directly.

**Benefits:** Removes one kernel launch and one buffer write per MoE expert.

**Expected conservative TPS speedup:** +3-5 % on MoE models.

- [ ] Write `dual_indexed_matvec_iq3_xxs_silu_f32` kernel.
- [ ] Integrate into `runMoeFfnBlock`.

---

## Summary of Expected Gains

If all phases are completed, the **cumulative conservative speedup** on a 7B Q4_K_M model on Apple M3 is estimated at:

| Phase | Conservative TPS Gain |
|-------|----------------------|
| Phase 1 (Sync) | +8-15 % |
| Phase 2 (Memory) | +15-25 % |
| Phase 3 (Kernels) | +12-18 % |
| Phase 4 (CPU) | +20-40 % (CPU path) |
| Phase 5 (Sampling) | +5-10 % |
| Phase 6 (Overhead) | +2-5 % |
| Phase 7 (MoE) | +15-30 % (MoE models) |

**Combined (multiplicative, not additive):** roughly **+40-70 %** faster decode TPS for standard dense models, and **+60-100 %** for MoE/linear-attention models.

---

## How to Use This File

1. Pick one phase that matches your current bottleneck (check `metal_decode_profile_enabled` output).
2. Complete every checklist item in that phase before moving to the next.
3. After each item, run `zig build test` and the benchmark fixture to ensure no regressions.
4. Update the checkbox from `- [ ]` to `- [x]` and commit.

---

*Last updated: 2026-04-22*
