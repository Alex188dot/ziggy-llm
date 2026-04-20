# ziggy-llm Optimization Checklist

This document contains all identified optimization opportunities organized by priority and category. Tackle items one by one in the recommended order.

---

## Section 1: Obsolete/Unused Code Removal (Do First)

### Immediate Cleanup

- [ ] Remove `.tmp_llama_debug.zig` - Temporary debug test harness (16 lines, not referenced anywhere)
- [ ] Remove `new.zig` - Completely empty file (1 byte)
- [ ] Evaluate `AGENTS.md` - Gitignored file, determine if needed or can be removed

### Legacy Code Cleanup

- [ ] Review legacy quantization formats in `moon_quant.zig`:
  - [ ] Lines 19-20: `legacy_q4_k` and `legacy_q6_k` enum values
  - [ ] Lines 31-32, 41, 48: Legacy format labels and descriptions
  - [ ] Lines 265-277: Legacy format mapping logic
  - [ ] Lines 287, 294, 303, 316-317: Legacy format handling
  - [ ] Remove if no longer used by current models, or add documentation for why they're kept

---

## Section 2: GPU/Metal Kernel Optimizations

**SKIPPED** - Bottleneck analysis showed `commit_wait` is 70-80x larger than shader compute time. Shader barrier optimizations would provide <1% TPS improvement. Real optimization opportunity is Section 5 (commit frequency optimization).

### Metal Shader: Reduce Barrier Overhead (`src/runtime/metal/matvec.metal`)

- [x] Audit all `threadgroup_barrier` usage (lines 50, 149, 226, 299, 368, 439, 499, 544, 610, 751, 947, 957, 964, 973, 986, 1057, 1067, 1106, 1116, 1148, 1161, 1225, 1311, 1402, 1415, 1468, 1544, 1623, 1689)
- [x] Identify barriers that can be eliminated by fusing operations
- [x] Evaluate using `simd_shuffle` for small reductions instead of threadgroup barriers
- [x] Benchmark and tune `ZIGGY_MAX_ROW_SIMDGROUPS = 8` constant
- [x] Benchmark and tune `ZIGGY_MAX_Q4K_SIMDGROUPS = 8` constant
- [x] Evaluate if `ZIGGY_MAX_HEAD_DIM = 256` is optimal or can be dynamic

### Metal Shader: Attention Kernel Optimization (lines 882-997)

- [x] Analyze the 3-stage reduction pattern (local max → threadgroup max → global max)
- [x] Evaluate using `simd_max` and `simd_sum` more aggressively
- [x] Test `threadgroup_barrier(mem_flags::mem_none)` where memory ordering isn't critical
- [x] Consider making `local_out[ZIGGY_MAX_HEAD_DIM]` dynamically sized based on actual `head_dim`
- [x] Profile attention kernel to identify actual bottleneck stages

### Metal Shader: RoPE Kernel Optimization (lines 800-880)

- [x] Analyze `apply_rope_to_dst_f32` kernel per-thread trig computation
- [x] Implement precomputation of `cos(theta)` and `sin(theta)` in shared memory per head
- [x] Evaluate fast math approximations for trig functions if precision allows
- [x] Benchmark interleaved vs neox rope styles for performance differences

---

## Section 3: Metal Bridge Optimizations (`src/runtime/metal/bridge.m`)

### Command Buffer Batching

- [ ] Review `pendingCommandBuffer` usage (line 61)
- [ ] Analyze commit frequency patterns in current code
- [ ] Evaluate opportunities for more aggressive batching
- [ ] Review `ziggy_commit_pending` (lines 281-298) for async commit opportunities
- [ ] Test asynchronous commit patterns where possible
- [ ] Benchmark different batch sizes for optimal throughput

### Thread Group Sizing

- [ ] Centralize repeated thread group sizing logic (lines 111, 171, 191, 251, 1899, 2260, 2307, 2359, 2410)
- [ ] Create helper function for thread group sizing
- [ ] Benchmark `threads_per_group = thread_width * 4` heuristic
- [ ] Test alternative heuristics for different kernel types
- [ ] Evaluate the 256-thread cap (lines 179, 199) for different GPU generations
- [ ] Add GPU generation-specific tuning if beneficial

### Pipeline State Management

- [ ] Review all pipeline state objects (lines 13-60)
- [ ] Evaluate if any pipelines are unused or redundant
- [ ] Consider lazy-loading rarely-used pipelines

---

## Section 4: Inference Code Optimizations

### Dense Tensor Store (`src/runtime/gpu/metal/tensor_store.zig`)

#### Row-by-Row Dequantization Optimization (lines 194-202)

- [ ] Eliminate per-row `row_dense` allocation
- [ ] Allocate entire `dense` array once upfront
- [ ] Dequantize directly into column-major layout
- [ ] Implement SIMD/vectorized dequantization if possible
- [ ] Benchmark before/after performance impact

#### Prewarm Optimization (lines 98-130)

- [ ] Profile prewarm timing for large models
- [ ] Evaluate parallelizing prewarm across multiple threads
- [ ] Implement priority-based prewarm (frequently-accessed tensors first)
- [ ] Add prewarm progress reporting for better UX

#### Tensor Population (lines 38-75)

- [ ] Review tensor population order
- [ ] Consider reordering for better cache locality
- [ ] Evaluate batch tensor additions

### GPU Session Buffer Management (`src/runtime/gpu/session.zig`)

#### Buffer Allocation (lines 46-142)

- [ ] Audit all scratch buffer allocations
- [ ] Implement buffer pooling across sessions
- [ ] Add buffer reuse logic when dimensions match
- [ ] Review `linear_conv_state` and `linear_recurrent_state` allocation (lines 108-111)
- [ ] Evaluate if per-block allocation is optimal

#### Linear Attention CPU Fallback (lines 690-733)

- [ ] Profile CPU readback overhead (lines 712-723)
- [ ] Profile CPU memcpy operations (lines 783-785)
- [ ] Design GPU-side linear attention implementation
- [ ] Implement GPU-side linear attention
- [ ] Benchmark GPU vs CPU implementation

#### Sequence Management

- [ ] Audit all `beginSequence` and `commitSequence` calls (lines 175, 363, 847, 851, 875, 992)
- [ ] Evaluate opportunities to batch layer operations into single commit
- [ ] Use `commitSequenceTimed` stats to tune commit frequency
- [ ] Test different commit strategies for optimal TPS

### Model Loader (`src/model/loader.zig`)

#### Tokenizer Encoding Optimization (lines 370-607)

- [ ] Audit DP algorithm allocations (lines 463-468: `best_scores`, `best_tokens`, `best_next`)
- [ ] Implement array reuse across chunks with proper resizing
- [ ] Evaluate if chunking is optimal or can be larger
- [ ] Benchmark tokenizer encoding performance

#### KV Cache Operations (lines 1227-1228)

- [ ] Review CPU-side memcpy operations for KV cache
- [ ] Ensure GPU path using `storeKvHalf` is always preferred
- [ ] Add assertions or warnings if CPU path is used

#### General Memory Allocation

- [ ] Audit all allocator usage throughout loader
- [ ] Identify hot allocation paths
- [ ] Implement arena allocators for temporary allocations where appropriate

---

## Section 5: Max TPS (Tokens Per Second) Optimization

**SKIPPED** - Requires major architectural refactoring. commit_wait bottleneck (7-8ms per token) is 70-80x larger than shader compute time. All attempted approaches failed:

### Phase 2: Prompt Token Batching - FAILED

- Only affects initial prompt processing via `runPrompt()`, not generation loop
- Generation loop uses different path that still commits after each token
- Benchmarking showed no reduction in commit count
- Root cause: commit_wait bottleneck is in generation path, not prompt processing

### Phase 3B: Async Commit with CPU-GPU Overlap - FAILED

- Implemented async commit mechanism in Metal bridge
- Encountered "Metal sequence already active" error
- Root cause: Current architecture processes tokens sequentially with dependencies
- Metal command buffer system doesn't support multiple concurrent sequences
- Requires fundamental restructuring of inference pipeline

### Phase 1: Batch Kernel Integration - NOT ATTEMPTED

- Batch kernels exist (batchSiluMul, batchAddInPlace, batchRmsNorm)
- Current architecture processes one token at a time through layers sequentially
- To use batch kernels effectively would require restructuring similar to Phase 2
- Would not address the dominant commit_wait bottleneck

**Conclusion**: Section 5 requires major architectural refactoring that doesn't meet the "efficiency evidence + no regressions" requirement. The commit_wait bottleneck is fundamental to the current synchronous token-by-token design.

### Profiling-Driven Optimization

#### Leverage Existing Profiling (`src/runtime/metal_profile.zig`)

- [x] Run `ziggy-llm bench --metal-profile` on target models
- [x] Establish baseline TPS measurements
- [x] Identify top 3 bottleneck categories from profile output
- [ ] Focus optimization efforts on identified bottlenecks
- [ ] Re-profile after each optimization to measure impact

**Findings:**

- commit_wait: 7-8ms per token (dominant bottleneck)
- projections: 0.1-0.11ms per token
- attention: 0.01-0.02ms per token
- kv_writes: 0.06-0.1ms per token
- normalization: 0.035-0.05ms per token
- elementwise_ops: 0.05-0.065ms per token
- output_reduce: 0.001-0.002ms per token
- host_readback: 0.001ms per token
- cpu_sampling: 0ms (GPU sampling path)

#### Profile Categories to Monitor

- [x] projections - Matrix-vector multiplication operations
- [x] attention - Self-attention computation
- [x] kv_writes - Key-value cache writes
- [x] normalization - RMS norm operations
- [x] elementwise_ops - Element-wise operations (SiLU, GeLU, etc.)
- [x] output_reduce - Output layer reductions (argmax, sampling)
- [x] commit_wait - GPU command buffer commit wait time
- [x] host_readback - GPU to CPU memory transfers
- [x] cpu_sampling - CPU-side sampling operations

### Batch Operations

#### Existing Batch Kernels

- [x] Audit usage of `batchArgmaxPipeline` (bridge.m line 46)
- [x] Audit usage of `batchMatvecAddPipeline` (bridge.m line 47)
- [x] Audit usage of `batchSiluMulPipeline` (bridge.m line 58)
- [x] Verify these are being used effectively
- [ ] If underutilized, identify opportunities to use them

**Findings:**

- batchArgmaxPipeline: Used once in session.zig for speculative decoding
- batchMatvecAddPipeline: Defined in bridge.m but not used in Zig runtime
- batchSiluMulPipeline: Defined in bridge.m but not used in Zig runtime
- Other batch kernels (batchRmsNorm, batchAddInPlace): Defined but unused
- Batch kernels exist but are not integrated into the main inference path

#### Speculative Decoding

- [ ] Research speculative decoding approach
- [ ] Design batch token generation using existing batch kernels
- [ ] Implement speculative decoding prototype
- [ ] Validate correctness with comprehensive tests
- [ ] Benchmark TPS improvement

### Commit Frequency Optimization

#### Current Pattern Analysis

- [x] Map all commit points in inference path
- [x] Measure time between commits
- [x] Identify commits that can be deferred
- [ ] Test batching multiple layer operations
- [ ] Find optimal batch size through benchmarking

**Findings:**

- Current pattern: commit after each token (beginToken → runTokenCore → commitToken)
- Prompt processing: commits after each prompt token via stepNoOutput
- Generation: commits after each generated token
- Commit overhead is fundamental to autoregressive generation design
- Batching commits would require redesigning the token generation loop

#### Commit Statistics

- [x] Use `commitSequenceTimed` stats (lines 308-324 in metal_backend.zig)
- [x] Track CPU wait time vs GPU elapsed time
- [x] Identify commits with high CPU wait overhead
- [ ] Optimize those commit points

**Findings:**

- CPU wait time ~7-8ms per commit
- GPU elapsed time is negligible compared to wait time
- Bottleneck is synchronous commit pattern, not GPU execution time
- Optimizing individual commits won't address the fundamental issue

### Memory Transfer Reduction

#### Host Readback Optimization

- [ ] Audit all `readBufferF32` calls (session.zig line 357 and others)
- [ ] Identify readbacks that can be avoided
- [ ] Keep logits on GPU for sampling when possible
- [ ] Leverage GPU-side argmax/sampling (already implemented via `sampleTopK`)
- [ ] Only readback when necessary (final output, CPU sampling)

#### Write Optimization

- [ ] Audit all `writeBufferF32` calls
- [ ] Identify redundant writes
- [ ] Batch writes where possible
- [ ] Use write-combining strategies

---

## Section 6: High-Impact, Low-Risk Changes (Implement First)

### File Cleanup

- [ ] Delete `.tmp_llama_debug.zig`
- [ ] Delete `new.zig`
- [ ] Update `.gitignore` if needed

### Dense Tensor Store Fix

- [ ] Eliminate per-row allocation in `tensor_store.zig:194-202`
- [ ] Dequantize directly into column-major layout
- [ ] Test with existing test suite
- [ ] Benchmark performance improvement

### Attention Kernel Barrier Reduction

- [ ] Evaluate necessity of all 3 barrier stages in `matvec.metal:944-974`
- [ ] Test removing or combining barriers
- [ ] Validate correctness with tests
- [ ] Benchmark improvement

### Thread Group Sizing Centralization

- [ ] Create helper function for thread group sizing
- [ ] Replace all repeated logic with helper
- [ ] Add documentation for tuning parameters
- [ ] Test no performance regression

---

## Section 7: Medium-Impact, Medium-Risk Changes

### Buffer Pooling Implementation

- [ ] Design buffer pool architecture
- [ ] Implement buffer pooling across sessions
- [ ] Add buffer reuse logic for matching dimensions
- [ ] Test with resident runtime
- [ ] Benchmark memory usage and TPS

### Thread Group Sizing Tuning

- [ ] Benchmark different `threads_per_group` heuristics
- [ ] Test GPU generation-specific tuning
- [ ] Add runtime detection for GPU capabilities
- [ ] Implement adaptive sizing based on workload
- [ ] Document optimal settings per GPU generation

### Batch Operation Improvements

- [ ] Reduce commit frequency by batching layer operations
- [ ] Use profiling data to find optimal batch size
- [ ] Implement adaptive batching based on workload
- [ ] Test with different model sizes
- [ ] Document trade-offs

---

## Section 8: High-Impact, Higher-Risk Changes

### GPU-Side Linear Attention

- [ ] Design GPU implementation of linear attention
- [ ] Implement GPU kernels for linear attention operations
- [ ] Remove CPU fallback paths
- [ ] Eliminate CPU readbacks (session.zig:712-723, 783-785)
- [ ] Comprehensive testing and validation
- [ ] Benchmark TPS improvement

### Speculative Decoding Implementation

- [ ] Research and design speculative decoding approach
- [ ] Implement batch token generation
- [ ] Use existing batch kernels where possible
- [ ] Implement validation logic
- [ ] Add comprehensive tests
- [ ] Benchmark TPS improvement

---

## Section 9: Recommended Benchmarking Strategy

### Baseline Measurement

- [ ] Run `ziggy-llm bench --metal-profile` on all target models
- [ ] Document baseline TPS for each model
- [ ] Document baseline memory usage
- [ ] Document baseline power consumption if possible
- [ ] Save baseline profiles for comparison

### Profile Analysis

- [ ] Analyze profile output for each model
- [ ] Identify top 3 bottleneck categories per model
- [ ] Document common bottlenecks across models
- [ ] Prioritize optimizations that help multiple models

### Incremental Optimization

- [ ] Apply one change at a time
- [ ] Measure impact with benchmarks
- [ ] Document before/after metrics
- [ ] Run full test suite after each change
- [ ] Revert if regression detected

### Regression Testing

- [ ] Ensure all existing tests pass (`zig build test`)
- [ ] Add performance regression tests
- [ ] Test on multiple GPU generations if possible
- [ ] Validate correctness with model outputs

### Hardware-Specific Tuning

- [ ] Test on M1, M2, M3 if available
- [ ] Adjust thread group sizes per generation
- [ ] Adjust barrier strategies per generation
- [ ] Document optimal settings per generation
- [ ] Implement runtime detection if beneficial

---

## Section 10: Documentation and Maintenance

### Code Documentation

- [ ] Add comments explaining optimization decisions
- [ ] Document tuning parameters and their effects
- [ ] Add benchmarking guide for contributors
- [ ] Document GPU-specific optimizations

### Performance Monitoring

- [ ] Add performance counters to critical paths
- [ ] Implement ongoing performance monitoring
- [ ] Add performance regression tests to CI
- [ ] Document expected TPS ranges per model

### Future Research Areas

- [ ] Explore advanced Metal features (mesh shaders if available)
- [ ] Investigate model-specific optimizations
- [ ] Consider multi-GPU support for larger models
- [ ] Research quantization-aware training integration
- [ ] Explore compiler optimizations (Zig 0.15.2+ features)

---

## Section 11: Validation Checklist

Before considering optimization complete for any change:

- [ ] All existing tests pass (`zig build test`)
- [ ] No performance regression in other models
- [ ] Correctness validated with multiple prompts
- [ ] Memory usage not significantly increased
- [ ] Benchmark shows measurable improvement
- [ ] Code is well-documented
- [ ] Change is committed with clear message
- [ ] Performance documented in benchmark results

---

## Section 12: Tracking Progress

### Completed Optimizations

- [ ] (Add completed items here with date and notes)

### In Progress

- [ ] (Add items currently being worked on)

### Blocked

- [ ] (Add items that are blocked with reason)

### Deferred

- [ ] (Add items deferred for future consideration)

---

## Appendix: File Reference Map

### Key Files for Optimization

- `src/runtime/metal/matvec.metal` - Metal shader kernels
- `src/runtime/metal/bridge.m` - Metal Objective-C bridge
- `src/runtime/metal/bridge.h` - Metal bridge header
- `src/runtime/metal_backend.zig` - Metal backend Zig interface
- `src/runtime/gpu/session.zig` - GPU session management
- `src/runtime/gpu/metal/tensor_store.zig` - Tensor storage and caching
- `src/runtime/metal_profile.zig` - Profiling infrastructure
- `src/model/loader.zig` - Model loading and inference
- `src/runtime/llama_runtime.zig` - LLaMA runtime
- `src/runtime/resident_runtime.zig` - Resident runtime with caching

### Test Files

- `src/runtime/metal_backend_test.zig` - Metal backend tests
- `src/runtime/bench_runner.zig` - Benchmark runner

### Build Files

- `build.zig` - Build configuration
- `build.zig.zon` - Dependencies

---

## Notes

- Always run `zig build test` after changes
- Use `--metal-profile` flag for benchmarking
- Benchmark on target hardware, not just development machine
- Consider power consumption in addition to TPS
- Document all tuning parameters and their rationale
- Keep changes minimal and focused
- Use git branches for experimental changes
