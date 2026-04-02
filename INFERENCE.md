# INFERENCE.md

## Summary

This is a new repo-root document at `INFERENCE.md` that serves as the performance execution checklist for the Apple Silicon Metal inference path. The document is internal-facing in intent but safe for a public repository: it must describe the goal as increasing decode throughput as much as materially possible, with explicit milestone targets of `100+ TPS` as the primary threshold and `200+ TPS` as the stretch target (for our specific use case, tiny llama 1.1b Q_4_K_M quantized model, on a Macbook Pro M3 18GB Hardware).

## Key Changes

### 1. Add `INFERENCE.md` at the repo root

Use this exact document shape:

1. `# Inference Performance Plan`
2. Short intro paragraph:
   State that this file is the working plan for materially increasing decode throughput on the Apple Silicon Metal path, that the target is `100+ TPS` with a stretch target of `200+ TPS`, and that the plan is intentionally checklist-driven.
3. `## How To Use This File`
4. `## Current State`
5. `## Throughput Targets`
6. `## Benchmark Discipline`
7. `## Phase 1: Measurement Before More Optimization`
8. `## Phase 2: Memory Residency And Buffer Ownership`
9. `## Phase 3: Decode Graph Fusion`
10. `## Phase 4: Decode-Specialized Kernels`
11. `## Phase 5: KV Cache And Attention Path`
12. `## Phase 6: CPU-Side Overhead Removal`
13. `## Phase 7: Quantization Coverage And Bandwidth Reduction`
14. `## Phase 8: Validation, Regression Control, And Performance Gates`
15. `## Exit Criteria`

### 2. Make the checklist usage rules explicit

In `## How To Use This File`, include these checklist rules as checkbox items or short bullets:

- Every implementation task in this file is a Markdown checkbox.
- Replace `- [ ]` with `- [x]` only after code, validation, and benchmark evidence are complete.
- Do not delete completed items.
- If an item is too large, split it into smaller checkbox items before starting.
- Do not mark performance work complete based on theory alone; attach before/after measurements.
- Re-run the canonical benchmark after every material optimization.
- If a change improves one metric but regresses correctness or stability, leave the item unchecked until resolved.

### 3. Ground the document in the current repo/runtime shape

In `## Current State`, describe the current implementation based on the actual codebase:

- The active fast path is the Apple Silicon Metal decode runtime for `llama` GGUF.
- GPU-side execution already includes RMSNorm, RoPE, attention, FFN, residual adds, dense matvec, and `Q4_K` matvec.
- Per-token sequencing is already batched through `beginSequence` / `commitSequence`.
- Buffers are still primarily created with `MTLResourceStorageModeShared`.
- The decode session still reads logits back to CPU for sampling.
- The current pipeline still consists of many small per-layer operations even after recent kernel work.

Do not put hardcoded current TPS numbers in this section. Keep it evergreen and methodology-oriented.

### 4. Include explicit throughput milestones

In `## Throughput Targets`, use checkboxes for milestone goals:

- [ ] Reach and sustain `100+ TPS` on the primary Apple Silicon benchmark machine for the canonical decode benchmark.
- [ ] Reach and sustain `150+ TPS` on the same benchmark without correctness regressions.
- [ ] Reach and sustain `200+ TPS` as a stretch target if memory bandwidth, residency, and launch overhead improvements make it realistic.
- [ ] Keep TTFT, startup time, and output correctness tracked alongside decode TPS so throughput gains do not hide regressions elsewhere.

### 5. Make benchmark discipline a first-class checklist section

In `## Benchmark Discipline`, include concrete checklist items that define how optimization work is measured:

- [ ] Define one canonical decode benchmark command and keep it stable across optimization work.
- [ ] Record machine, OS, Zig version, model, quantization, prompt length, generated length, temperature, seed, and backend for every published result.
- [ ] Add per-stage timing for startup, prompt processing, TTFT, and decode tok/s.
- [ ] Add per-kernel or per-operation timing around dense matvec, quantized matvec, attention, RMSNorm, RoPE, KV writes, residual adds, and CPU sampling.
- [ ] Distinguish prompt processing throughput from decode throughput in all measurements.
- [ ] Keep a simple benchmark table in the document with date and result snapshots.
- [ ] Treat any optimization without benchmark evidence as incomplete.

### 6. Define the optimization checklist in priority order

Use the following exact substance for the main checklist phases.

#### `## Phase 1: Measurement Before More Optimization`

- [ ] Add low-overhead instrumentation around the Metal decode path so each token step can report time spent in projections, attention, KV writes, normalization, elementwise ops, readback, and CPU sampling.
- [ ] Add a benchmark mode or debug flag that can print aggregated per-op timings without changing correctness behavior.
- [ ] Identify the top three decode bottlenecks from measured time share before starting the next major optimization pass.
- [ ] Record which tensor shapes dominate real runs so later kernel specialization targets real workloads rather than synthetic shapes.

#### `## Phase 2: Memory Residency And Buffer Ownership`

This section should explicitly target the current `Shared` allocation strategy in `src/runtime/metal/bridge.m` and the session buffers in `src/runtime/llama_gpu.zig`.

Checklist items:

- [ ] Separate long-lived model weights from CPU-visible scratch/output buffers in the Metal buffer API.
- [ ] Add support for GPU-resident or device-private weight buffers with explicit upload/staging only where needed.
- [ ] Keep CPU-written and CPU-read buffers shared only when readback or host writes are actually required.
- [ ] Evaluate moving the KV cache to GPU-owned residency instead of always treating it like generic shared scratch.
- [ ] Ensure buffer creation APIs make residency an explicit choice rather than an implicit global default.
- [ ] Benchmark memory residency changes independently from kernel rewrites so gains can be attributed correctly.
- [ ] Keep fallback behavior and correctness intact when a residency mode is unsupported or not worth using.

#### `## Phase 3: Decode Graph Fusion`

Ground this section in the actual call flow in `src/runtime/llama_gpu.zig`.

Checklist items:

- [ ] Reduce per-layer compute encoder count by fusing operations that always occur back to back during single-token decode.
- [ ] Evaluate fusing RMSNorm with immediately following projection input handling where data reuse is available.
- [ ] Evaluate fusing Q/K/V projection preparation steps to reduce launch overhead and intermediate traffic.
- [ ] Evaluate fusing RoPE application with KV write or adjacent attention setup work when the data lifetime allows it.
- [ ] Evaluate fusing FFN gate and up-path work more aggressively than the current separate matvec plus SiLU multiply flow.
- [ ] Evaluate fusing residual add work into adjacent producer kernels where it removes a full pass over memory.
- [ ] Prefer decode-specific fusion for batch size `1` rather than general kernels that preserve unnecessary flexibility.
- [ ] Re-benchmark after each fusion stage to verify that reduced launch count translates into higher TPS.

#### `## Phase 4: Decode-Specialized Kernels`

This section should be explicitly about making the hot path less generic.

Checklist items:

- [ ] Add single-token decode-specialized kernels rather than relying only on shape-generic kernels.
- [ ] Specialize dense and quantized matvec kernels for the dominant `head_dim`, `embedding_length`, and `feed_forward_length` shapes seen in benchmarked models.
- [ ] Add fast paths for the most common quantized tensor layouts used by supported GGUF models.
- [ ] Avoid runtime branching in hot kernels when shape-specialized pipelines can remove it.
- [ ] Cache specialized pipeline variants so compile or selection overhead does not erase runtime gains.
- [ ] Keep a generic fallback kernel path for unsupported shapes, but ensure specialized paths are preferred automatically when shapes match.

#### `## Phase 5: KV Cache And Attention Path`

This section should acknowledge that attention is improved but still likely memory-heavy.

Checklist items:

- [ ] Rework the attention path to reduce full-score materialization and repeated passes where possible.
- [ ] Investigate a streaming or tiled softmax formulation that keeps running max and denominator instead of treating the entire score vector as the default working set.
- [ ] Reduce threadgroup memory pressure in the attention path where it limits occupancy.
- [ ] Revisit KV cache layout for decode access locality rather than only for implementation simplicity.
- [ ] Minimize bytes moved during KV append for every generated token.
- [ ] Benchmark attention separately at short, medium, and long context lengths so improvements are not tuned only for one regime.
- [ ] Confirm that any new attention strategy preserves numerical stability and output consistency against the CPU reference path.

#### `## Phase 6: CPU-Side Overhead Removal`

Ground this in the current `runOutput` path and CPU-side orchestration.

Checklist items:

- [ ] Measure how much decode time is spent outside GPU kernel execution, especially final readback and CPU sampling.
- [ ] Reduce command-buffer churn and host synchronization points that are not strictly necessary for single-token decode.
- [ ] Keep intermediate activations on GPU until the final logits are actually needed by CPU-side sampling.
- [ ] Investigate moving more of the output selection path onto the GPU if CPU readback becomes a measurable throughput limiter.
- [ ] Ensure per-token orchestration does not allocate, rebuild, or rebind avoidable state on the CPU.
- [ ] Reuse command encoders, pipeline states, and scratch resources wherever the Metal API allows it cleanly.

#### `## Phase 7: Quantization Coverage And Bandwidth Reduction`

Tie this to the current support matrix in `README.md` and `src/runtime/types.zig`.

Checklist items:

- [ ] Identify whether the benchmarked models are still spending too much decode bandwidth in dense `F16` weights.
- [ ] Expand optimized GPU support for the quantized formats that matter most for the target models.
- [ ] Prioritize formats that materially reduce bytes moved per token without breaking output quality expectations for the intended model sizes.
- [ ] Add format-specific correctness tests and performance tests for every new optimized quantized path.
- [ ] Prefer quantization work that lowers end-to-end decode bandwidth over low-impact micro-optimizations in already-fast dense paths.

#### `## Phase 8: Validation, Regression Control, And Performance Gates`

Checklist items:

- [ ] Keep CPU-vs-Metal correctness tests for dense matvec, quantized matvec, and fused attention.
- [ ] Add regression tests for any new fused kernels or residency modes.
- [ ] Add benchmark guardrails so major TPS regressions are visible before merge.
- [ ] Record whether each optimization changes startup time, TTFT, decode TPS, memory footprint, or correctness.
- [ ] Do not close an optimization item unless both functional validation and benchmark validation are attached.
- [ ] Keep public benchmark claims tied to documented commands and exact machine details.

### 7. End with strict exit criteria

In `## Exit Criteria`, include a final checklist such as:

- [ ] The canonical Apple Silicon Metal benchmark reaches `100+ TPS` reproducibly.
- [ ] The path remains stable across repeated benchmark runs.
- [ ] CPU and Metal outputs remain acceptably aligned on the correctness suite.
- [ ] No major optimization item is marked complete without benchmark evidence.
- [ ] The document reflects the current highest-priority bottlenecks rather than stale guesses.
- [ ] Stretch work toward `150+` and `200+ TPS` is clearly separated from the minimum `100+ TPS` success line.

## Test Plan

The document itself should instruct future implementers to validate each item with:

- `zig build test`
- the canonical `bench` command on the primary Apple Silicon target machine
- before/after TPS, TTFT, startup, and decode metrics
- focused correctness checks for dense matvec, quantized matvec, attention, and any new fused or residency-specific paths
- context-length-sensitive attention benchmarks so short-context wins do not mask long-context regressions

## Assumptions And Defaults

- The file lives at the repo root as `INFERENCE.md`.
- The tone should match the existing docs: direct, technical, honest, and benchmark-oriented.
- The file is public-safe and must not mention other inference projects by name.
- The document is a working optimization plan, not a marketing page.
- All action items are checkboxes; no major work item should be written as plain prose only.
- The plan should prioritize end-to-end decode TPS over isolated microbenchmark wins, while still requiring correctness and TTFT tracking.
