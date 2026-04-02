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
- [ ] Specialize that kernel for static extents that matter on Apple Silicon decode.
- [ ] Add postfix fusion around the packed `MoonQuant` path where it measurably reduces memory traffic.
- [ ] Implement per-layer or non-uniform bit allocation instead of one global quant choice.
- [ ] Implement imatrix-style calibration to drive the per-layer quant decisions.
- [ ] Add end-to-end correctness tests for GGUF-to-MoonQuant packing and execution parity.
- [ ] Benchmark `MoonQuant` against current generic `Q4_K` / `Q6_K` baselines on Apple Silicon.

## Pillar 2: Speculative Decoding

- [ ] Keep speculative decoding scoped to a narrow llama-first path.
- [ ] Build decode-first benchmarks for acceptance rate, verifier overhead, and throughput.
- [ ] Implement low-overhead verification on top of the strong base runtime.
- [ ] Validate single-user local inference before any serving-oriented work.

## Pillar 3: Fused Apple-Specific Kernels

- [ ] Prioritize Apple Silicon and Metal as the first-class optimization target.
- [ ] Add fused decode-first kernels for the dominant llama shapes.
- [ ] Add quant-specific fast paths for the MoonQuant target family.
- [ ] Minimize intermediate memory traffic in the decode loop.
- [ ] Minimize kernel launch overhead in the decode loop.
- [ ] Record benchmark deltas after each meaningful fused-kernel step.
