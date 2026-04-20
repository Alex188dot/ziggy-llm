# Section 5 Refactoring Plan: Commit Frequency + Batch Kernel Integration

## Current State Analysis

### Bottleneck Identification

- **commit_wait**: 7-8ms per token (dominant bottleneck)
- **Shader compute**: 0.1-0.4ms per token
- **Ratio**: commit_wait is 70-80x larger than compute time

### Current Commit Pattern

```
for each token:
  beginToken() → beginSequence()
  runTokenCore() → runs all layers
  commitToken() → commitSequence() + waitUntilCompleted()
```

### Batch Kernel Status

- **batchArgmaxPipeline**: Used once for speculative decoding
- **batchMatvecAddPipeline**: Defined but unused in Zig runtime
- **batchSiluMulPipeline**: Defined but unused in Zig runtime
- **batchRmsNormPipeline**: Defined but unused in Zig runtime
- **batchAddInPlacePipeline**: Defined but unused in Zig runtime

## Refactoring Plan

### Phase 1: Low-Risk Batch Kernel Integration (Safe, Measurable)

#### Objective

Integrate existing batch kernels into the main inference path to reduce kernel launch overhead.

#### Target Operations

1. **batchSiluMul** - Replace individual SiLU + multiply operations in FFN
   - Current: gate → SiLU → multiply with up → add
   - Batched: Can process multiple FFN layers in single dispatch
   - Risk: Low - element-wise operations, no cross-layer dependencies

2. **batchAddInPlace** - Replace individual add operations
   - Current: Multiple addInPlace calls for residual connections
   - Batched: Can batch multiple adds across layers
   - Risk: Low - commutative operation

3. **batchRmsNorm** - Replace individual RMS norm operations
   - Current: RMS norm called before each attention/FFN block
   - Batched: Can batch normalization across layers
   - Risk: Medium - need to ensure correctness with different parameters per layer

#### Implementation Strategy

1. Add Zig wrapper functions for batch kernels in metal_backend.zig
2. Modify gpu/session.zig to use batch kernels where applicable
3. Add configuration flag to enable/disable batch kernels (for A/B testing)
4. Benchmark with and without batch kernels
5. Test correctness with existing test suite

#### Expected Impact

- Reduce kernel launch overhead (fewer dispatches)
- Estimated improvement: 5-15% reduction in non-commit time
- Risk: Very low (can be disabled if issues arise)

---

### Phase 2: Prompt Token Batching (Safe, High Impact)

#### Objective

Batch multiple prompt tokens into a single commit to reduce commit frequency during prompt processing.

#### Current Pattern

```zig
for (prompt_tokens[0..prompt_tokens.len - 1]) |token_id| {
    try session.stepNoOutput(token_id);  // Commits after each token
}
```

#### Optimized Pattern

```zig
try session.beginPromptBatch();
for (prompt_tokens[0..prompt_tokens.len - 1]) |token_id| {
    try session.stepNoOutputNoCommit(token_id);  // No commit
}
try session.commitPromptBatch();  // Single commit for all prompt tokens
```

#### Why This Is Safe

- Prompt tokens don't require sampling
- Processing is deterministic
- No dependencies between prompt tokens
- Only the last prompt token needs output for generation

#### Implementation Strategy

1. Add `beginPromptBatch()` and `commitPromptBatch()` methods to Session
2. Add `stepNoOutputNoCommit()` variant that doesn't commit
3. Modify loader.zig to use batching for prompt processing
4. Benchmark prompt processing time
5. Test correctness with existing test suite

#### Expected Impact

- Reduce commits from N to 1 for N prompt tokens
- For 100-token prompt: 100 commits → 1 commit
- Estimated improvement: 90% reduction in prompt processing commit overhead
- Risk: Low (isolated to prompt processing, well-defined semantics)

#### Actual Results

**FAILED** - Prompt batching only affects initial prompt processing via `runPrompt()`, not the generation/decode loop. The generation loop uses a different path that still commits after each token. Benchmarking showed no reduction in commit count (commit_wait.calls remained at 16 for 9 decode tokens).

**Root Cause**: The commit_wait bottleneck is in the generation path, not the initial prompt processing. Batching prompt tokens provides minimal benefit for typical workloads where generation dominates.

---

### Phase 3: Commit Frequency Optimization for Generation (Complex, High Reward)

#### Objective

Reduce commit frequency during generation while maintaining correctness.

#### Challenge

Autoregressive generation requires previous token's output to generate next token.

#### Options

**Option A: Speculative Decoding (Already Partially Implemented)**

- Generate multiple draft tokens in parallel
- Verify and accept/reject in batches
- Uses existing batchArgmaxPipeline
- Complexity: Medium (already has foundation)
- Expected improvement: 1.5-2x TPS for typical workloads

**Option B: Async Commit with CPU-GPU Overlap**

- Commit asynchronously and continue CPU work
- Overlap commit wait with next token's CPU-side work
- Complexity: Very High (requires major architectural changes)
- Expected improvement: 30-50% reduction in perceived latency

**Option C: Batch Layer Operations Within Single Token**

- Batch multiple layer operations into single commit
- Still commit after each token, but fewer commits per token
- Complexity: Medium (need to identify independent operations)
- Expected improvement: 10-20% reduction in commit overhead

#### Recommended Approach

Start with Option A (Speculative Decoding) since it's already partially implemented and uses existing batch kernels.

---

### Phase 3B: Async Commit with CPU-GPU Overlap - FAILED

#### Attempt

Implemented async commit mechanism in Metal bridge:

- Added `ziggy_commit_async` to commit without waiting
- Added `ziggy_wait_for_completion` to wait when needed
- Added C API functions and Zig wrappers
- Attempted to use in runtime for prompt token processing

#### Results

**FAILED** - Encountered "Metal sequence already active" error when trying to begin new sequence while previous async commit was still pending.

**Root Cause**: The current architecture processes tokens sequentially with each token depending on the previous token's hidden state. The Metal command buffer system doesn't support multiple concurrent sequences in the current design. To achieve true CPU-GPU overlap, would require:

1. Restructuring the generation loop to interleave CPU work with GPU execution
2. Decoupling token encoding from commit
3. Careful synchronization to ensure dependencies are respected
4. Major architectural changes to the token generation pipeline

**Complexity**: Very high - requires fundamental restructuring of the inference pipeline, not just adding async calls.

**Conclusion**: Async commit with CPU-GPU overlap is not feasible without major architectural refactoring that doesn't meet the "efficiency evidence + no regressions" requirement.

---

## Implementation Order

1. **Phase 1** (1-2 days)
   - Implement batchSiluMul integration
   - Implement batchAddInPlace integration
   - Benchmark and test
   - Commit if efficiency gain + no regressions

2. **Phase 2** (1-2 days)
   - Implement prompt token batching
   - Benchmark and test
   - Commit if efficiency gain + no regressions

3. **Phase 3A** (2-3 days)
   - Enhance speculative decoding
   - Benchmark and test
   - Commit if efficiency gain + no regressions

## Success Criteria

Each phase must meet:

- **Efficiency Gain**: Measurable improvement in TPS or latency
- **No Regressions**: All existing tests pass
- **Correctness**: Output identical to baseline (within floating-point tolerance)
- **Configurability**: Can be disabled if issues arise

## Rollback Plan

Each phase will:

- Add feature flags to disable optimizations
- Preserve original code paths
- Allow A/B testing
- Easy to revert if issues arise
