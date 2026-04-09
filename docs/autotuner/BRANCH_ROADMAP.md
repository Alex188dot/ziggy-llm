# Branch Roadmap — Metal autotuner

## Branch goal

Build a practical Apple-first autotuning system for the Metal inference path so the runtime can discover better kernel configurations and tensor packing choices per machine, model family, quantization, and context regime instead of relying only on hand-tuned defaults.

This branch is about:

- defining a stable tuning surface for the current Metal runtime
- benchmarking candidate configurations automatically
- storing and reusing winning profiles per target regime
- supporting both manual exploration and one-click optimization

This branch is not about:

- replacing the runtime with a new inference engine immediately
- exposing every possible low-level knob before the search space is understood
- claiming wins without reproducible benchmark evidence
- tuning broad multi-batch server workloads first

---

## How to use this file

- [ ] Every implementation task in this file is a Markdown checkbox.
- [ ] Replace `- [ ]` with `- [x]` only after code, validation, and benchmark evidence are complete.
- [ ] Do not delete completed items.
- [ ] If an item is too large, split it before starting.
- [ ] Do not mark tuning work complete from theory alone; attach before/after measurements.
- [ ] Re-run the canonical benchmark set after every material tuning-system change.
- [ ] If a tuned profile improves TPS but regresses correctness, stability, or startup path behavior, leave the item unchecked until resolved.

---

## Current thesis

- Apple Metal performance is sensitive to configuration choices that are currently only partially encoded in the runtime.
- Small models make launch overhead, kernel shape choices, and packing decisions much more visible.
- A good autotuner can expose whether a current performance ceiling is caused by weak defaults, a weak kernel family, or a deeper architectural limit.
- The autotuner should become part of the import and compile pipeline later, but it should start as a benchmark-driven system around the current runtime.

---

## Success criteria

- [ ] The runtime can benchmark multiple candidate kernel and layout configurations automatically for at least one canonical tiny-model workload.
- [ ] The autotuner can persist and reload the best known profile for a specific Apple machine and benchmark regime.
- [ ] The tuned path materially improves warm decode TPS on at least one canonical Metal benchmark over the default configuration.
- [ ] Manual tuning and automatic tuning produce identical runtime behavior when given the same selected configuration.
- [ ] The branch ends with a clear answer on which knobs matter, which knobs are noise, and which bottlenecks require runtime or kernel redesign instead of more tuning.

---

## Benchmark discipline

- [ ] Keep one canonical benchmark command for TinyLlama 1.1B and at least one larger comparison model.
- [ ] Record machine, OS, Zig version, backend, model, quantization, prompt length, generated length, temperature, seed, and power mode for every published tuning result.
- [ ] Record cold and warm results separately.
- [ ] Record startup, prompt processing, TTFT, decode TPS, and key decode-profile counters when available.
- [ ] Distinguish default runtime results from autotuned results clearly.
- [ ] Keep benchmark tables in-repo as tuning profiles evolve.
- [ ] Do not count a profile as winning unless it beats the default across repeated runs, not just a single lucky result.

---

## Phase 1 — Tuning surface definition

### Goal

Expose only the knobs that can plausibly move performance.

### Deliverables

- [ ] Enumerate the current Metal kernel and runtime decisions that can be varied without changing model correctness.
- [ ] Separate cheap runtime knobs from expensive offline layout or repacking knobs.
- [ ] Define a structured tuning config schema that can describe:
- [ ] threadgroup sizes
- [ ] tile sizes
- [ ] SIMD-group mappings
- [ ] staging or prefetch variants
- [ ] accumulator formats where numerically safe
- [ ] tensor packing or layout variants
- [ ] Mark each knob with applicability constraints by op family, quant format, and model family.
- [ ] Reject invalid knob combinations before benchmark execution.

### Success criteria

- [ ] The tuning surface is explicit, versioned, and small enough to search without chaos.

---

## Phase 2 — Benchmark harness and profile capture

### Goal

Make tuning runs reproducible and easy to compare.

### Deliverables

- [ ] Add a benchmark harness that can run a single candidate config repeatedly and emit structured results.
- [ ] Save per-run metadata, measured timings, and decode-profile counters to a machine-readable artifact.
- [ ] Key benchmark results by:
- [ ] chip family
- [ ] OS version
- [ ] model family
- [ ] quant format
- [ ] prompt/context regime
- [ ] generated length
- [ ] Add profile hashing so reused results are tied to exact kernel and layout choices.
- [ ] Make failed or unstable candidates visible instead of silently discarding them.

### Success criteria

- [ ] Tuning runs are reproducible enough that profile comparisons are trustworthy.

---

## Phase 3 — Search strategy

### Goal

Find good configurations efficiently without brute-forcing everything.

### Deliverables

- [ ] Implement a baseline search strategy for a bounded candidate set.
- [ ] Add early-pruning rules for clearly losing candidates.
- [ ] Support search modes for:
- [ ] quick local search
- [ ] deeper offline tuning
- [ ] focused retuning after a kernel change
- [ ] Capture not only the winner, but also the top-N near-best candidates for analysis.
- [ ] Record which knobs actually changed the outcome and by how much.

### Success criteria

- [ ] The autotuner can reach a stable best-known profile in a practical amount of time on a laptop.

---

## Phase 4 — Profile storage and runtime integration

### Goal

Make winning configs reusable automatically.

### Deliverables

- [ ] Define a profile database or file format for saved winning configs.
- [ ] Load the best matching profile automatically at runtime when one exists.
- [ ] Fall back cleanly to defaults when no profile exists or compatibility checks fail.
- [ ] Surface the active profile choice in logs or benchmark output.
- [ ] Version profiles so old entries can be invalidated after kernel or layout changes.

### Success criteria

- [ ] A tuned configuration can be reused without manual intervention and without hidden correctness risk.

---

## Phase 5 — Manual tuning UI

### Goal

Make low-level tuning explorable without editing code for every experiment.

### Deliverables

- [ ] Build a developer UI that can select model, backend, quant regime, prompt length, generated length, and run count.
- [ ] Expose the highest-value tuning knobs first, not the entire internal state space.
- [ ] Show benchmark outputs that include startup, TTFT, decode TPS, and key decode-profile counters.
- [ ] Show diffs against the current default config and against the best saved profile.
- [ ] Allow saving a manual configuration as a named profile candidate.

### Success criteria

- [ ] Manual tuning is materially faster than changing constants and recompiling for every experiment.

---

## Phase 6 — One-click optimize flow

### Goal

Provide an `Optimize` path that searches for the best config automatically for the selected workload.

### Deliverables

- [ ] Add an `Optimize` action that runs the autotuner against the current machine and workload selection.
- [ ] Let users choose quick, balanced, or deep tuning budgets.
- [ ] Persist the best found profile automatically when validation passes.
- [ ] Show the default result, best found result, and net gain clearly.
- [ ] Keep a log of tried candidates and rejected candidates for debugging.

### Success criteria

- [ ] A user can get a tuned profile for their Mac without touching low-level code.

---

## Phase 7 — Offline packing and import-pipeline alignment

### Goal

Expand beyond runtime-only tuning once the basic system is proven.

### Deliverables

- [ ] Identify which current bottlenecks cannot be fixed by runtime knobs alone.
- [ ] Add offline experiments for pack layouts and compiled tensor formats where gains justify the extra scope.
- [ ] Tie packing choices to model family and quant format rather than treating them as universal defaults.
- [ ] Reuse the autotuner evidence to decide which layouts deserve long-term support.

### Success criteria

- [ ] The autotuner helps justify which deeper runtime or compiler changes are worth implementing next.

---

## Risks and working rules

- [ ] Do not let the search space explode before the benchmark harness is stable.
- [ ] Do not add UI-only knobs that are not backed by runtime support and validation.
- [ ] Do not treat autotuning as a substitute for better kernels; use it to expose where kernels or layouts are the real limit.
- [ ] Prefer narrow, high-value tuning dimensions over a giant configuration matrix with noisy results.
- [ ] Keep the autotuner useful for tiny-model decode first, then expand only with evidence.
