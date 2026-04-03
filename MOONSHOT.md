# Moonshot Checklist

This file tracks the moonshot execution path. Every time an item is completed, it should be checked here in the same change.

## Pillar 1: MoonQuant

- [x] Create a dedicated `MoonQuant` subsystem and make it a first-class project concept.
- [x] Define a fixed Apple-first quant family: `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`.
- [x] Surface `MoonQuant` status in `inspect` so supported-vs-planned formats are explicit.
- [x] Define the first Metal-native packed layout contract for `Q4_K_M`.
- [x] Implement a loader-side packer that turns GGUF `Q4_K` rows into the fixed-stride `MoonQuant` `Q4_K_M` layout.
- [x] Thread the packed `MoonQuant` tensors through the full runtime hot path.
- [x] Add a dedicated Metal kernel that consumes the packed `MoonQuant` `Q4_K_M` layout directly.
- [x] Specialize that kernel for static extents that matter on Apple Silicon decode.
- [x] Add postfix fusion around the packed `MoonQuant` path where it measurably reduces memory traffic.
- [x] Implement per-layer or non-uniform bit allocation instead of one global quant choice.
- [x] Implement imatrix-style calibration to drive the per-layer quant decisions.
- [x] Add end-to-end correctness tests for GGUF-to-MoonQuant packing and execution parity.
- [x] Benchmark `MoonQuant` against current generic `Q4_K` / `Q6_K` baselines on Apple Silicon.

### MoonQuant Next Pass

- [x] Add real-model benchmark coverage for more than one prompt and generation-length pair.
- [x] Add benchmark guardrails so MoonQuant regressions are visible in CI or in a scripted local check.
- [x] Expand optimized paths beyond the current `Q4_K` packing focus.
- [x] Verify whether `Q6_K` should get a real Metal fast path instead of serving only as a baseline.
- [x] Profile tensor-prepare and Metal prewarm cost and reduce the largest startup regressions.
- [x] Record per-op deltas specifically attributable to MoonQuant in the real decode loop.
- [x] Tighten benchmark discipline around one canonical MoonQuant comparison workflow.
- [x] Run one more optimization pass on real-model decode hotspots before shifting the main focus to speculative decoding.

## Pillar 2: Speculative Decoding

- [ ] Keep speculative decoding scoped to a narrow llama-first path.
- [ ] Build decode-first benchmarks for acceptance rate, verifier overhead, and throughput.
- [ ] Implement low-overhead verification on top of the strong base runtime.
- [ ] Validate single-user local inference before any serving-oriented work.

## Pillar 3: Fused Apple-Specific Kernels

- [x] Prioritize Apple Silicon and Metal as the first-class optimization target.
- [x] Add fused decode-first kernels for the dominant llama shapes.
- [ ] Add quant-specific fast paths for the MoonQuant target family.
- [ ] Minimize intermediate memory traffic in the decode loop.
- [ ] Minimize kernel launch overhead in the decode loop.
- [ ] Record benchmark deltas after each meaningful fused-kernel step.
