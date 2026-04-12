# Branch Roadmap — DFlash speculative decoding

## Branch goal

Add a real Apple Silicon DFlash path to `zig_` for one tightly scoped model family first, then expand only after the core runtime architecture is proven.

This branch is about:

- building a working dual-model speculative runtime in Zig
- supporting one DFlash-backed Qwen path end to end on Apple Silicon
- widening the current speculative verification path from toy scale to DFlash-sized blocks
- keeping the implementation modular enough to survive later model-family expansion

This branch is not about:

- supporting every published DFlash checkpoint immediately
- building a generic multi-backend speculative framework before the Metal path works
- matching vLLM or SGLang serving features on day one
- solving long-context degradation before the short and medium context path is solid

---

## How to use this file

- [ ] Every implementation task in this file is a Markdown checkbox.
- [ ] Replace `- [ ]` with `- [x]` only after code, validation, and benchmark evidence are complete.
- [ ] Do not delete completed items.
- [ ] If an item is too large, split it before starting.
- [ ] Do not mark speculative decoding work complete from theory alone; attach correctness and throughput evidence.
- [ ] Re-run the canonical benchmark after every material runtime or kernel change.
- [ ] If a change improves throughput but breaks exact-match greedy verification, leave the item unchecked until resolved.

---

## Current thesis

The repo already has the beginnings of a speculative decode path:

- `llama_cpu.zig` already has draft discovery and verification flow.
- `llama_gpu.zig` already has a batched speculative verification kernel path.
- the resident runtime already caches one loaded model and one reusable session.

The core gap is that DFlash is not a normal autoregressive draft model:

- it is a second model, not a heuristic draft source
- it conditions on selected hidden states from the target model
- it drafts a block in parallel and then verifies that block with the target

The working thesis for this branch is:

- a Qwen-only Apple-first MVP is feasible in `zig_`
- the first real blocker is runtime architecture, not kernel math
- the second blocker is hidden-state extraction from selected target layers
- only after those are solved does widening the speculative verify path to block size `16` become worth doing

---

## Success criteria

- [ ] Run one DFlash-backed Qwen model end to end in `zig_` on Apple Silicon.
- [ ] Support greedy exact-match speculative decoding with block size `16`.
- [ ] Beat the current non-DFlash greedy decode path on the primary benchmark machine for the chosen model.
- [ ] Preserve output correctness relative to the target-only greedy path.
- [ ] End the branch with a clear answer on whether DFlash is a practical Apple-first acceleration path for `zig_`.

---

## Scope lock for v1

### Supported in MVP

- [ ] Apple Silicon Metal backend only
- [ ] one Qwen DFlash pair only
- [ ] greedy decoding only
- [ ] single sequence only
- [ ] short and medium context benchmarking

### Explicitly deferred

- [ ] top-k or temperature sampling for DFlash
- [ ] multi-request server batching
- [ ] generic support for all DFlash architectures
- [ ] quantized DFlash draft weights
- [ ] long-context optimization past the initial working path

---

## Benchmark discipline

- [ ] Keep one canonical benchmark command for target-only greedy decode.
- [ ] Keep one canonical benchmark command for DFlash greedy decode.
- [ ] Record machine, OS, Zig version, model pair, prompt length, generated length, backend, and run mode for every published number.
- [ ] Record cold and warm results separately.
- [ ] Track TTFT, per-token decode TPS, acceptance length, and acceptance rate.
- [ ] Track separate timings for:
- [ ] target prefill
- [ ] draft block generation
- [ ] target block verification
- [ ] hidden-state extraction
- [ ] CPU orchestration overhead
- [ ] Keep benchmark tables in-repo as the branch evolves.

---

## Phase 1 — Model and checkpoint reconnaissance

### Goal

Lock the first supported DFlash pair before changing runtime architecture.

### Deliverables

- [ ] Pick one concrete target and draft checkpoint pair for the branch MVP.
- [ ] Record the exact draft architecture details needed by runtime code:
- [ ] hidden size
- [ ] head count
- [ ] KV head count
- [ ] draft layer count
- [ ] block size
- [ ] target layer IDs used for context features
- [ ] Confirm whether the chosen target architecture already fits the repo's Qwen execution assumptions.
- [ ] Document unsupported checkpoint families and why they are out of scope for v1.

### Success criteria

- [ ] There is one fixed checkpoint pair for implementation and benchmarking.
- [ ] Runtime work is driven by one known-good architecture instead of abstract generalization.

---

## Phase 2 — Dual-model runtime foundation

### Goal

Teach `zig_` to load and manage a target model and a draft model together.

### Deliverables

- [ ] Add a dual-model runtime container without breaking the current single-model path.
- [ ] Define modular ownership for:
- [ ] target model resources
- [ ] draft model resources
- [ ] shared tokenizer assumptions
- [ ] reusable sessions and caches
- [ ] Extend generation options and CLI plumbing for an optional draft model path.
- [ ] Keep existing non-speculative code paths unchanged when no draft model is provided.
- [ ] Ensure model unload and resident-runtime reuse work correctly for both models.

### Success criteria

- [ ] The runtime can load, hold, and unload both models cleanly.
- [ ] Existing single-model generation remains stable.

---

## Phase 3 — DFlash model ingestion

### Goal

Make one DFlash checkpoint executable in `zig_`.

### Deliverables

- [ ] Define a modular execution model for the DFlash draft architecture.
- [ ] Add loader support for the chosen draft checkpoint format or a branch-local converted format.
- [ ] Map draft tensors into internal runtime structures without polluting the base llama/qwen model path.
- [ ] Keep file and module boundaries small enough that the draft path remains maintainable.
- [ ] Add inspect or debug output for the draft model structure.

### Success criteria

- [ ] The chosen DFlash draft checkpoint can be loaded and validated locally.
- [ ] The draft path does not require invasive hacks in unrelated runtime modules.

---

## Phase 4 — Target hidden-state taps

### Goal

Expose the selected target hidden states required by DFlash conditioning.

### Deliverables

- [ ] Add a way to capture hidden states from selected target layers during prefill and block verification.
- [ ] Keep this feature optional so the normal decode path does not pay for it when disabled.
- [ ] Decide whether hidden taps stay GPU-resident, CPU-visible, or mixed based on measured cost.
- [ ] Add a compact structure for passing extracted context features into the draft model.
- [ ] Validate that tapped hidden states match the intended layer indices for the chosen checkpoint.

### Success criteria

- [ ] The draft model receives correct target context features.
- [ ] Hidden-state extraction overhead is measured rather than guessed.

---

## Phase 5 — Block drafting path

### Goal

Run the DFlash draft model to produce one speculative block.

### Deliverables

- [ ] Implement the draft forward path for one block using the chosen DFlash architecture.
- [ ] Support mask-token initialization and block-size `16` drafting.
- [ ] Reuse existing Qwen-style attention and FFN code where possible instead of forking unrelated logic.
- [ ] Keep draft KV cache management separate from target KV cache management.
- [ ] Add focused tests for:
- [ ] block shape correctness
- [ ] position handling
- [ ] cache growth and crop behavior

### Success criteria

- [ ] The draft model can emit a full speculative block on Apple Silicon.
- [ ] Draft execution is deterministic under greedy settings.

---

## Phase 6 — Target verification path

### Goal

Verify DFlash blocks efficiently with the target model.

### Deliverables

- [ ] Expand the current speculative verification path from `max_draft_len = 4` to DFlash-sized blocks.
- [ ] Support acceptance-length calculation for full-block verification.
- [ ] Preserve exact greedy semantics against target-only decoding.
- [ ] Ensure accepted tokens and fallback tokens update the target KV cache correctly.
- [ ] Record acceptance statistics in benchmark output.

### Success criteria

- [ ] Verification correctness matches target-only greedy decoding.
- [ ] The verify path handles block size `16` without brittle fixed-size assumptions.

---

## Phase 7 — Decode-loop integration

### Goal

Replace the current toy speculative loop with a real DFlash decode loop behind a controlled path.

### Deliverables

- [ ] Add a dedicated DFlash decode mode instead of overloading the existing heuristic draft flow.
- [ ] Integrate:
- [ ] target prefill
- [ ] first next-token seed
- [ ] draft block generation
- [ ] target block verification
- [ ] accepted-token commit
- [ ] stop-token handling
- [ ] Keep streaming behavior coherent when more than one token is accepted per iteration.
- [ ] Ensure resident-runtime session reuse does not corrupt draft or target caches across turns.

### Success criteria

- [ ] The end-to-end DFlash loop runs in CLI and resident-runtime flows.
- [ ] Chat, run, and bench entry points can exercise the path without architectural duplication.

---

## Phase 8 — Apple Silicon optimization

### Goal

Recover enough performance that DFlash is not merely correct, but useful.

### Deliverables

- [ ] Measure whether the main bottleneck is:
- [ ] target verification cost
- [ ] hidden-state extraction
- [ ] draft compute
- [ ] command submission overhead
- [ ] Reduce avoidable host-visible transfers in the DFlash path.
- [ ] Keep draft embeddings and intermediate buffers GPU-resident where practical.
- [ ] Revisit command-buffer structure for draft plus verify loops.
- [ ] Avoid steady-state allocations inside the speculative decode loop.

### Success criteria

- [ ] DFlash beats the target-only baseline for the chosen model on the benchmark machine.
- [ ] Profiling shows where remaining Apple-specific bottlenecks actually are.

---

## Phase 9 — Long-context and stability follow-up

### Goal

Decide whether the DFlash path remains useful beyond the shortest contexts.

### Deliverables

- [ ] Benchmark at multiple context lengths, including a point beyond `4K` tokens.
- [ ] Measure how acceptance rate and throughput shift with KV growth.
- [ ] Record whether hidden-state extraction or KV pressure dominates long-context degradation.
- [ ] Decide whether long-context work belongs in this branch or a follow-up branch.

### Success criteria

- [ ] The branch closes with a measured statement about short, medium, and longer context behavior.

---

## Validation matrix

- [ ] Exact greedy output match against target-only decode on a fixed prompt suite.
- [ ] EOS and stop-token handling remain correct.
- [ ] Resident runtime reuse across multiple prompts remains correct.
- [ ] Unload and reload logic remain correct.
- [ ] Benchmark output reports acceptance metrics and throughput cleanly.
- [ ] Existing non-DFlash tests continue to pass.

---

## Recommended implementation order

1. [ ] Lock one Qwen target + DFlash draft pair.
2. [ ] Add dual-model runtime ownership.
3. [ ] Add draft model ingestion.
4. [ ] Add target hidden-state taps.
5. [ ] Implement block drafting.
6. [ ] Widen verification to block size `16`.
7. [ ] Integrate the end-to-end DFlash decode loop.
8. [ ] Benchmark and optimize the Apple path.

---

## Definition of done

- [ ] One DFlash-backed Qwen path works end to end on Apple Silicon.
- [ ] Greedy exact-match correctness is preserved.
- [ ] The branch produces benchmark evidence showing whether DFlash is a worthwhile direction for `zig_`.
