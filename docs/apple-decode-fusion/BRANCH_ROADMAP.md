# Branch Roadmap — Apple decode fusion

## Branch goal

Push Apple Silicon decode throughput materially higher by specializing the Metal execution path for narrow hot decode workloads instead of attempting a single monolithic whole-model megakernel first.

This branch is about:

- measuring the real decode bottlenecks precisely
- fusing the decode graph in modular steps
- reducing CPU orchestration and readback overhead
- aligning kernel work with MoonQuant and compiled-layout execution

This branch is not about:

- building a one-dispatch whole-model shader immediately
- broad server batching work
- generic multi-platform optimization
- expanding model-family scope before the llama path is stronger

---

## How to use this file

- [ ] Every implementation task in this file is a Markdown checkbox.
- [ ] Replace `- [ ]` with `- [x]` only after code, validation, and benchmark evidence are complete.
- [ ] Do not delete completed items.
- [ ] If an item is too large, split it before starting.
- [ ] Do not mark performance work complete from theory alone; attach before/after measurements.
- [ ] Re-run the canonical benchmark after every material optimization.
- [ ] If a change improves TPS but regresses correctness, stability, or startup path behavior, leave the item unchecked until resolved.

---

## Current thesis

The repo already has the right ingredients for aggressive Apple-first decode optimization:

- Metal is already the hot path for supported llama-family decode.
- Quant-aware kernels and MoonQuant packing already exist.
- Per-token work is already batched more intelligently than a naive op-by-op runtime.
- The current path still pays for too many small operations, too many per-token kernel/encoder boundaries, and too much memory traffic inside the steady-state decode step.

The working thesis for this branch is:

- a strict CUDA-style whole-model megakernel is not the best first move on Apple Silicon
- decode-step graph fusion, residency, and orchestration cleanup should come first
- once steady-state decode reaches one command buffer per token with low non-GPU wait, further wins must come primarily from reducing real GPU work per token rather than from more command-submission cleanup
- if those improvements stall, a narrower persistent decode-step kernel can be tested later

---

## Success criteria

- [ ] Raise TinyLlama 1.1B Q4_K_M warm decode throughput materially beyond the current documented `~123 tok/s` baseline on the primary benchmark machine.
- [ ] Raise Llama 3.2 3B Q4_K_M warm decode throughput materially beyond the current documented `~40 tok/s` baseline on the primary benchmark machine.
- [ ] Preserve or improve output correctness relative to the current CPU reference path.
- [ ] Keep startup time and TTFT tracked so decode wins do not hide regressions elsewhere.
- [ ] End the branch with one clear answer on whether a narrower persistent decode-step kernel is justified.

---

## Benchmark discipline

- [ ] Keep one canonical benchmark command for TinyLlama 1.1B and one for Llama 3.2 3B.
- [ ] Record machine, OS, Zig version, model, quantization, prompt length, generated length, temperature, seed, backend, and run mode for every published number.
- [ ] Record cold and warm results separately.
- [ ] Add or preserve timings for startup, prompt processing, TTFT, and decode TPS.
- [ ] Add per-stage decode timing for:
- [ ] attention
- [ ] dense matvec and quantized matvec
- [ ] RMSNorm
- [ ] RoPE
- [ ] residual-add work
- [ ] FFN work
- [ ] KV writes
- [ ] logits readback
- [ ] CPU sampling
- [ ] Keep benchmark tables in-repo as the branch evolves.

---

## Phase 1 — Decode bottleneck instrumentation

### Goal

Stop guessing where decode time goes.

### Deliverables

- [x] Add per-op timing instrumentation inside the Metal decode path.
- [x] Separate GPU execution time from CPU wait time where the Metal API allows it.
- [x] Attribute decode cost at least at the layer-subgraph level.
- [x] Add a stable benchmark report section for decode stage timings.
- [ ] Confirm whether current decode is matvec-bound, attention-bound, or synchronization-bound on:
- [x] TinyLlama 1.1B Q4_K_M
- [x] Llama 3.2 3B Q4_K_M

### Success criteria

- [x] We can rank the top decode bottlenecks with measured evidence.
- [x] We know whether launch overhead and readback overhead are large enough to justify deeper fusion work.

---

## Phase 2 — Residency and resource ownership

### Goal

Reduce bytes moved and resource churn before deeper kernel work.

### Deliverables

- [x] Audit which decode-path buffers remain host-visible by default and which can be made more GPU-friendly.
- [x] Reuse scratch buffers, pipeline state, and encoder setup more aggressively across tokens.
- [x] Remove avoidable per-token resource setup in the llama Metal runtime.
- [x] Keep more decode intermediates resident on GPU through the token step.
- [x] Measure whether logits readback volume can be reduced before CPU sampling.

### Success criteria

- [x] Lower CPU-side decode overhead without correctness regressions.
- [ ] Lower command-buffer or encoder churn measurably on the canonical benchmark runs.

---

## Phase 3 — Layer-local decode fusion

### Goal

Fuse the hot decode graph in modular pieces instead of jumping directly to a single giant shader, explicitly prioritizing fewer dispatches and lower `commit_wait` before kernel math tuning.

### Deliverables

- [x] Identify the highest-value subgraph to fuse first for llama decode.
- [x] Add one fused decode-first kernel family for a stable high-frequency shape that reduces steady-state dispatch count.
- [ ] Fuse at least one residual-add-adjacent path with its producer operation where it reduces extra writes.
- [x] Fuse at least one FFN-local sequence where shape stability makes the tradeoff favorable.
- [x] Add shape-specialized kernel variants only where benchmark evidence justifies them.

### Success criteria

- [ ] At least one fused decode subgraph shows a measurable warm decode gain.
- [ ] Kernel count or dispatch count per generated token is reduced on the instrumented path.

---

## Phase 4 — Attention and KV decode path

### Goal

Tighten the attention-side hot path without overfitting only to matvec work.

### Deliverables

- [ ] Measure the attention share of decode time across short and medium contexts.
- [x] Reduce unnecessary writes or format conversions around K and V handling.
- [ ] Revisit KV append layout for decode locality.
- [x] Reduce synchronization around attention-adjacent kernels where safe.
- [x] Validate numerical stability against the CPU reference path.

### Success criteria

- [x] Attention and KV work become a smaller fraction of per-token decode time on at least one canonical model.

---

## Phase 5 — CPU overhead removal

### Goal

Cut host orchestration overhead that does not contribute useful model work, but stop treating it as the default explanation once measurements show the steady-state token loop is already mostly GPU-bound.

### Deliverables

- [x] Measure time spent in final readback and CPU-side sampling.
- [ ] Reduce host synchronization points in the steady-state token loop.
- [ ] Investigate shortlist or top-k/top-p preparation on GPU before final CPU token choice.
- [ ] Ensure the decode loop performs no avoidable steady-state allocations.
- [ ] Keep the resident runtime path as the primary benchmark target for optimization work.
- [ ] Document when command-buffer structure stops being the top limiter so later work shifts toward kernel count, memory traffic, and decode-step specialization.

### Success criteria

- [ ] CPU overhead is no longer one of the top decode bottlenecks on the benchmarked llama path.
- [ ] The roadmap clearly marks when command-buffer churn is no longer the primary decode bottleneck.

---

## Phase 6 — MoonQuant and compiled-layout alignment

### Goal

Make weight layout and kernel design reinforce each other.

### Deliverables

- [ ] Identify which llama decode tensors benefit most from tighter compiled-layout specialization.
- [ ] Ensure MoonQuant layout decisions are driven by decode bandwidth and kernel shape, not only packing convenience.
- [ ] Expand compiled-layout metadata only where it enables cleaner or faster decode kernels.
- [ ] Avoid generic fallback paths inside the hottest decode loop where a narrow optimized path is available.

### Success criteria

- [ ] Quant/layout specialization reduces decode bandwidth or intermediate work on measured runs.

---

## Phase 7 — Persistent decode-step experiment

### Goal

Only after the prior phases, test whether a narrower megakernel-like approach is actually worth it.

### Scope

Strictly limited to:

- one model family
- one decode-step slice
- one or two dominant quant/layout paths
- benchmarked against the modular fused baseline

### Deliverables

- [ ] Design one narrow persistent decode-step experiment for llama.
- [ ] Keep the implementation modular enough that it can be removed cleanly if it loses.
- [ ] Compare it against the best modular fused path on:
- [ ] TinyLlama 1.1B Q4_K_M
- [ ] Llama 3.2 3B Q4_K_M
- [ ] Record wins, losses, maintenance cost, and portability cost.

### Success criteria

- [ ] The experiment shows a material gain beyond the modular fused path.
- [ ] Or the branch closes with a clear measured decision not to pursue a larger megakernel direction.

---

## Reuse across model families

### Reusable work

- [ ] Benchmark instrumentation should be backend-wide and reusable.
- [ ] Buffer residency improvements should be reusable across llama-like and qwen-like paths.
- [ ] CPU orchestration cleanup should be reusable across all Metal-backed decode runtimes.
- [ ] Generic utility kernels and command-submission improvements should be reusable where tensor shapes are compatible.

### Family-specific work

- [ ] Layer-subgraph fusion is architecture-specific.
- [ ] Persistent decode-step kernels are strongly architecture-specific.
- [ ] Quant-layout choices are partly reusable, but the best packing strategy depends on tensor roles and shapes.
- [ ] Attention optimizations may split into family-specific variants once rope/head/layout details differ.

### Working rule

- [ ] Implement shared infrastructure once.
- [ ] Implement fused hot kernels per architecture family.
- [ ] Do not block llama wins waiting for a perfectly generic abstraction.

---

## Explicit non-goals for this branch

- [ ] No attempt to support every GGUF quantization in the hot path.
- [ ] No broad server batching or multi-tenant scheduling work.
- [ ] No Linux or CUDA portability work.
- [ ] No assumption that one fused kernel design should cover llama, qwen, mistral, and hybrid DeltaNet models equally well.
- [ ] No whole-model single-dispatch Metal design unless the narrower persistent experiment proves clearly worthwhile.

---

## Recommended implementation order

1. [ ] Instrument the decode path properly.
2. [ ] Remove avoidable residency and orchestration overhead.
3. [ ] Fuse one high-value llama decode subgraph.
4. [ ] Tighten the attention and KV decode path.
5. [ ] Align MoonQuant and compiled layout with the hottest kernels.
6. [x] Re-benchmark TinyLlama 1.1B and Llama 3.2 3B.
7. [ ] Decide whether a persistent decode-step experiment is justified.

### Current evidence notes

- 2026-04-09 TinyLlama 1.1B Q4_K_M warm decode measured `125.185 tok/s` with GPU greedy argmax after commit-path instrumentation updates. Warm decode still spent `996.927 ms` of `1023.226 ms` in `commit_wait` (`97.430%`), but the new split showed this was overwhelmingly real GPU work: `968.863 ms` GPU time vs `28.064 ms` non-GPU wait. The steady-state path is already at `1.000` command buffer per generated token, so command-buffer structure is no longer the main frontier. The remaining hot-path shape was `26624` encoders and `23808` dispatches total, or `208` encoders and `186` dispatches per generated token.
- 2026-04-09 Llama 3.2 3B Q4_K_M warm decode measured `41.790 tok/s` with CPU logits sampling. This path also remained dominated by `commit_wait`, but CPU sampling plus readback were still secondary to the GPU-side decode workload. The working conclusion after the TinyLlama command-buffer split is that future wins are more likely to come from reducing real GPU work per token than from more submission-structure cleanup alone.
- Fused decode profiling on both canonical llama-family models showed `Q+RoPE` active on all attempted layers. Fused `KV half-write` coverage was partial (`54.545%` on TinyLlama, `50.000%` on Llama 3.2), and the remaining cases fell back because `attn_v` was not `tensor_type=12` while the single fused `K` path still succeeded on all remaining attempts.

---

## Definition of done

- [ ] The branch produces a clearly faster llama Apple decode path with benchmark evidence.
- [ ] The branch leaves behind reusable instrumentation and runtime structure, not only one-off shader experiments.
- [ ] The branch documents which wins generalize across model families and which do not.
- [ ] The branch ends with a measured yes or no on deeper megakernel-style work.
