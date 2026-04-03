# GPU-Side Sampling / Tiny Readback Only

## Summary

This is the first moonshot optimization track for the Apple Silicon Metal decode path.

The goal is simple: stop reading the full logits vector back to CPU on every generated token.

For TinyLlama `1.1B`, this is the most realistic first bet to push decode throughput toward and past `250+ TPS` on target hardware. It also compounds well with every other future optimization, because removing per-token CPU readback and CPU-side sampling overhead helps every model size and every faster kernel path.

The practical strategy is:

- keep logits on GPU
- do token selection on GPU
- read back only the selected token, or at most a tiny shortlist
- preserve CPU fallback and correctness validation while the GPU path matures

## Why This Is The Best First Bet

Right now the Metal path already keeps most decode math on GPU:

- RMSNorm
- RoPE
- attention
- FFN
- residual adds
- output projection
- quantized matvec fast paths

But the decode loop still pays a synchronization tax whenever CPU needs the full logits buffer for sampling.

That is a bad trade for fast decode:

- the logits vector is large relative to the one value we actually need next
- readback forces GPU/CPU synchronization
- CPU sampling time does not shrink just because the GPU got faster
- every future kernel improvement becomes less visible if token selection stays host-bound

This project already contains the seed of the solution:

- [`src/runtime/llama_gpu.zig`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/llama_gpu.zig) already supports GPU-side greedy argmax via `runOutputArgmax`
- the current general path still falls back to full-logits readback through `runOutput`
- CPU sampling logic still lives in [`src/llama_cpu.zig`](/Users/alessioleodori/HelloWorld/zig_/src/llama_cpu.zig)

So this is not a speculative architecture rewrite. It is an expansion of an existing fast path into the default decode strategy.

## Current Repo Reality

The design should start from the actual codebase rather than from a generic inference-engine sketch.

Current relevant facts:

- GPU decode state and token-step execution live in [`src/runtime/llama_gpu.zig`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/llama_gpu.zig)
- Metal backend entry points live in [`src/runtime/metal_backend.zig`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal_backend.zig)
- the Metal shader set already includes an `argmax_f32` kernel in [`src/runtime/metal/matvec.metal`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal/matvec.metal)
- greedy decode with `temperature <= 0` already uses GPU argmax, then reads back a single `u32`
- non-greedy sampling still depends on CPU-side `sampleToken`
- profiling already tracks `readback` and `cpu_sampling`, which makes this optimization measurable

This means Phase 1 is not “invent GPU sampling.” Phase 1 is “turn the existing GPU argmax path into a benchmarked, explicit, default-first throughput strategy.”

## Target End State

The intended end state is a staged sampling stack:

1. Greedy decode: GPU computes argmax, CPU reads back one token id.
2. Tiny-shortlist sampling: GPU computes top-k shortlist and shortlist metadata, CPU reads back only a small candidate set and performs the final stochastic choice.
3. Full GPU sampling: GPU applies temperature and filtering, draws the token, CPU reads back only the chosen token id and optional debug stats.

This order matters.

The first two stages are much lower risk than jumping straight to full stochastic GPU sampling. They also preserve a clean validation story against the existing CPU implementation.

## Architecture Plan

### Stage A: Productize GPU Greedy Sampling

This should become the first hard win.

Implementation direction:

- keep `runOutputArgmax` as the decode fast path for `temperature <= 0`
- make the no-full-logits-readback property explicit in code and profiling output
- ensure the prompt path and decode path do not accidentally force logits readback for greedy decode
- treat this as the canonical “decode ceiling” benchmark for the current runtime

Expected outcome:

- immediate reduction in per-token synchronization cost
- a cleaner measurement baseline for later stochastic sampling work

### Stage B: Add GPU Top-K / Shortlist Extraction

This is the most important bridge step.

Instead of reading back `vocab_size` logits, the GPU should produce a tiny candidate bundle:

- top token ids
- top logits or normalized weights
- optional cumulative mass hints

CPU then performs repeat penalty, top-p, min-p, and RNG-based final choice on a small shortlist rather than on the full vocabulary.

This gives most of the readback win without forcing the entire sampling policy onto GPU at once.

Recommended first target:

- fixed `top_k = 32` or `64` shortlist kernel
- one compact GPU output buffer per token
- read back only shortlist ids and scores

### Stage C: Move Sampling Policy Onto GPU

After shortlist mode is correct and benchmarked, move more of the policy to GPU:

- temperature scaling
- softmax or stable weight computation
- top-k truncation
- optional top-p truncation
- final categorical sample

CPU should read back:

- chosen token id
- optional score/probability for debugging
- optional fallback reason if a debug mode triggers CPU verification

This stage should be optional at first, behind a flag, until correctness confidence is high.

## Design Rules

- Do not remove the CPU sampling path until GPU sampling is benchmarked and validated.
- Do not expand scope to every sampler feature at once.
- Do not chase elegant abstraction if it hides GPU/CPU synchronization points.
- Do not treat prompt processing and decode sampling as the same problem.
- Prefer tiny deterministic GPU outputs over large “just in case” readbacks.
- Keep per-token allocations at zero on both GPU and CPU paths.

## Main Technical Risks

### 1. Sampler Feature Parity

The CPU sampler currently handles:

- temperature
- top-k
- top-p
- min-p
- repeat penalty

A full GPU port of all of that is feasible, but it is not the right first milestone.

Risk control:

- first win is greedy
- second win is shortlist readback
- only then move policy details to GPU

### 2. GPU Reduction Complexity

Efficient `argmax` is easy enough.

Efficient `top-k` across vocabulary is more involved:

- reduction shape matters
- shared/threadgroup memory use matters
- stable handling of equal logits matters
- bandwidth can still dominate if the kernel is naïve

Risk control:

- start with fixed-size shortlist kernels
- optimize for TinyLlama-class vocabulary first
- benchmark kernel cost versus saved readback cost

### 3. Repeat Penalty On GPU

Repeat penalty depends on token history, which currently lives naturally on CPU.

Risk control:

- keep repeat penalty on CPU during shortlist mode
- only move it to GPU if it shows up as a real bottleneck
- consider uploading a compact recent-token ring buffer rather than mirroring all decode state

### 4. Numerical Drift

Sampling is sensitive to ordering, truncation, and floating-point differences.

Risk control:

- preserve the current CPU sampler as the reference
- add deterministic tests with fixed seeds
- validate distributional behavior, not just single-token equality, for stochastic modes

## Proposed Implementation Order

1. Benchmark and isolate current greedy GPU path versus full-logits CPU path.
2. Make greedy GPU sampling the clearly documented fast path.
3. Add a Metal top-k or shortlist kernel and backend bridge API.
4. Add a decode mode that reads back only shortlist ids and scores.
5. Reuse existing CPU sampler logic over the shortlist where possible.
6. Compare TPS, TTFT, and output quality against the current full-logits path.
7. Only after that, prototype full GPU stochastic sampling.

## Benchmark Standard

Every sub-step in this track should be judged on the canonical Apple Silicon benchmark with exact machine and command details recorded.

Minimum metrics to track:

- decode TPS
- decode ms per token
- TTFT
- `readback` time
- `cpu_sampling` time
- correctness versus the CPU reference path

The first success condition is not “GPU sampling exists.”

The first success condition is:

- greedy decode no longer reads back full logits
- the gain is measurable
- the new path is stable and reproducible

## Current Isolation Harness

The repo now has an explicit benchmark switch to isolate the two current `temperature <= 0` Metal decode paths:

- `--sampling-path gpu-greedy` keeps the existing GPU argmax fast path active
- `--sampling-path cpu-full-logits` forces the Metal decode path to read back full logits and sample on CPU even for greedy decode

Canonical isolation commands:

```sh
zig build run -- bench -m /path/to/model.gguf -p "a" --backend metal --temperature 0 --max-tokens 128 --bench-runs 6 --sampling-path gpu-greedy --metal-profile
zig build run -- bench -m /path/to/model.gguf -p "a" --backend metal --temperature 0 --max-tokens 128 --bench-runs 6 --sampling-path cpu-full-logits --metal-profile
```

Benchmark output now prints:

- `sampling_strategy`
- `sampling_path`
- `readback_mode`

That makes the greedy A/B comparison explicit in both normal bench output and profiled runs.

For Metal profiling, the distinction is visible in two places:

- `readback_mode=sampled-token-u32` versus `readback_mode=full-logits-f32`
- `profile.shape_*` entries for `readback`, where greedy GPU argmax shows `cols=1` and full-logits mode shows `cols=vocab_size`

## Current TinyLlama Baseline

Measured on `2026-04-03` on:

- model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- machine: `Mac15,6`
- CPU/GPU class: `Apple M3 Pro`
- memory: `18 GB`
- OS: `macOS 26.0.1`

Prompt and generation settings used for all runs:

```sh
./zig-out/bin/ziggy-llm bench \
  --model /Users/alessioleodori/HelloWorld/zig_/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal \
  --max-tokens 128 \
  --seed 42 \
  --bench-runs 5 \
  --metal-profile
```

Warm baseline snapshots:

| Mode | Extra flags | warm prompt ms avg | warm TTFT ms avg | warm TPS avg | readback mode |
| --- | --- | ---: | ---: | ---: | --- |
| Greedy GPU argmax | `--temperature 0 --sampling-path gpu-greedy` | `561.975` | `562.125` | `33.474` | `sampled-token-u32` |
| Greedy forced full logits | `--temperature 0 --sampling-path cpu-full-logits` | `551.710` | `551.928` | `33.484` | `full-logits-f32` |
| Non-greedy current path | `--temperature 0.7` | `548.177` | `548.892` | `33.506` | `full-logits-f32` |

Current profiling snapshots from those same runs:

- Greedy GPU argmax: `profile.readback.ns=3744686000` and `profile.cpu_sampling.ns=0`
- Greedy forced full logits: `profile.readback.ns=3653074000` and `profile.cpu_sampling.ns=6675000`
- Non-greedy current path: `profile.readback.ns=3648060000` and `profile.cpu_sampling.ns=24190000`

What that means right now:

- the fast path isolation is real and measurable in output
- greedy GPU sampling is functionally on the tiny-readback path
- on this benchmark, readback plus synchronization still dominates both greedy modes, so the throughput gain is not yet material

## Shortlist API Status

The repo now contains the first fixed-size shortlist extraction slice:

- a Metal `topk_f32` kernel with `top_k <= 64`
- Objective-C bridge entrypoint `ziggy_metal_topk_f32`
- Zig backend wrapper `topKShortlist`
- GPU session method `runOutputShortlist`

This is an API and kernel landing, not a full decode-path switch yet. CPU shortlist sampling reuse still remains as the next step.

The repo now also contains the first end-to-end decode mode for that path:

- `--sampling-path gpu-shortlist` forces stochastic decode to read back only shortlist ids and scores
- CPU sampler policy is reused over the shortlist rather than over the full vocabulary
- `auto` still stays on full-logits CPU sampling for stochastic decode until the Metal top-k kernel is fast enough to win on benchmark

## Implementation Notes

The current code now makes the greedy fast path explicit instead of implicit:

- `temperature <= 0` still defaults to GPU greedy argmax when a Metal session is active
- the old full-logits path can be forced only for benchmarking and validation with `--sampling-path cpu-full-logits`
- greedy prompt ingestion no longer reads back full logits for every prompt token
- prompt prefix tokens now advance with no output readback, and the final prompt token uses GPU argmax on the fast path

## Checklist

- [x] Document the current baseline for greedy decode and non-greedy decode on TinyLlama `1.1B`.
- [x] Confirm from profiling output how much time is currently spent in `readback` and `cpu_sampling`.
- [x] Make GPU greedy sampling the explicit default fast path for `temperature <= 0`.
- [x] Verify the greedy Metal path reads back only one token id, not the full logits vector.
- [x] Add benchmark output that clearly distinguishes full-logits readback from tiny-token readback.
- [x] Add a Metal backend API for shortlist extraction rather than full-logits readback.
- [x] Implement a first fixed-size GPU `top-k` or shortlist kernel in [`src/runtime/metal/matvec.metal`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal/matvec.metal).
- [x] Add the Objective-C bridge plumbing in [`src/runtime/metal/bridge.m`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal/bridge.m) and [`src/runtime/metal/bridge.h`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal/bridge.h).
- [x] Add Zig backend wrappers in [`src/runtime/metal_backend.zig`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal_backend.zig) for shortlist extraction and tiny readback.
- [x] Extend [`src/runtime/llama_gpu.zig`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/llama_gpu.zig) with a `runOutputShortlist` path.
- [x] Reuse the existing CPU sampling logic over shortlist candidates before attempting full GPU stochastic sampling.
- [ ] Prove shortlist mode is faster than full-logits readback on the canonical TinyLlama benchmark.
- [ ] Add correctness tests that compare shortlist-assisted sampling against the full CPU sampler under fixed seeds.
- [ ] Add stress tests for edge cases such as `top_k = 0`, `top_p = 1`, tiny candidate sets, and EOS-heavy outputs.
- [ ] Add a guarded experimental mode for full GPU stochastic sampling.
- [ ] Implement GPU-side temperature scaling and categorical sampling only after shortlist mode is benchmarked and validated.
- [ ] Decide whether repeat penalty stays on CPU with shortlist mode or moves to GPU in a later phase.
- [ ] Keep CPU fallback intact until GPU sampling passes both correctness and benchmark gates.
- [ ] Record before/after TPS snapshots in this file as the work lands.

## Current Shortlist Decode Result

Shortlist decode is now functionally wired, but it is not yet the fast path on the current Metal kernel.

Measured on `2026-04-03` with the same TinyLlama benchmark:

| Mode | Extra flags | warm TPS avg | readback mode | Notes |
| --- | --- | ---: | --- | --- |
| Full logits CPU sampler | `--temperature 0.7` | `71.768` | `full-logits-f32` | current default |
| GPU shortlist + CPU sampler reuse | `--temperature 0.7 --sampling-path gpu-shortlist --top-k 8` | `24.488` | `shortlist-ids-scores` | works, but Metal top-k kernel dominates |
| GPU shortlist + CPU sampler reuse | `--temperature 0.7 --sampling-path gpu-shortlist` | `4.760` | `shortlist-ids-scores` | fixed shortlist `64` is much too slow right now |

That means steps 4 and 5 are implemented, but the kernel from step 3 still needs a substantial rewrite before this path can replace full-logits readback in `auto`.

## Definition Of Done For This Moonshot Track

This track is successful when all of the following are true:

- TinyLlama `1.1B` decode no longer depends on full-logits CPU readback on the fast path.
- Greedy decode uses GPU-side token selection by default.
- At least one stochastic mode uses tiny shortlist readback instead of full-logits readback.
- Benchmark data shows a material TPS improvement on the primary Apple Silicon machine.
- The path remains reproducible, deterministic where expected, and validated against the CPU reference implementation.

If this lands cleanly, it becomes the foundation for the bigger wins after it:

- deeper decode graph fusion
- more aggressive residency work
- more quant-specific fast paths
- speculative decoding on top of a stronger base decode engine
