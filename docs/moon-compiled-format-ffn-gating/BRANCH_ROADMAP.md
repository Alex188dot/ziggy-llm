# Branch Roadmap — MoonQuant compiled format + gated FFN experiment

## Branch goal

Make MoonQuant a real Apple-first compiled execution format, then add one experimental approximate optimization on top.

---

## Phase 1 — Compiled format first

### Goal

Treat GGUF as the import format and `.ziggy` as the execution format.

### Deliverables

- [x] Define compiled container format (`.ziggy`)
- [x] Add manifest/header for:
  - [x] architecture
  - [x] quantization version
  - [x] tensor count
  - [x] compiled tensor records
  - [x] alignment
  - [x] runtime compatibility version
- [x] Add tensor record entries for:
  - [x] tensor name
  - [x] original GGUF type
  - [x] compiled layout kind
  - [x] rows / cols
  - [x] byte offset
  - [x] byte length
  - [x] row stride / tile stride
- [x] Implement `GGUF -> compiled` offline conversion
- [x] Reuse existing MoonQuant Q4_K repacking path
- [x] Add compiled-path loader
- [x] Keep GGUF loader as fallback/reference path
- [x] Add inspect command for compiled files
- [x] Add tests for round-trip metadata + tensor lookup

### Success criteria

- [x] Can load compiled Q4_K_M path without GGUF runtime repacking
- [x] Same outputs as current reference path within expected tolerance
- [x] Lower runtime pointer/layout work in Metal path
- [x] Foundation is ready for more aggressive kernel-specific formats

---

## Phase 2 — Expand compiled layout coverage

### Goal

Move from one packed path to a real compiled-layout family.

### Deliverables

- [ ] Add compiled layouts for:
  - [ ] Q4_K_M
  - [ ] Q6_K
  - [ ] Q8_0
  - [ ] optional Q5_K_M later
- [ ] Add per-layout decode metadata
- [ ] Add kernel-family tags:
  - [ ] attention
  - [ ] FFN
  - [ ] output
- [ ] Allow layout selection per tensor role
- [ ] Reuse calibration output to select format/layout policy

### Success criteria

- [ ] Multiple tensor families load directly from compiled format
- [ ] No repacking needed at runtime for supported paths
- [ ] Cleaner separation between import path and execution path

---

## Phase 3 — Experimental activation-gated FFN skipping

### Goal

Test one approximate optimization that may reduce memory traffic.

### Scope

Only:

- `ffn_down`
- selected layers
- optional flag
- calibrated thresholds

### Deliverables

- [ ] Add calibration output for FFN gate sparsity stats
- [ ] Define threshold policy per layer
- [ ] Emit optional compiled metadata for gated FFN path
- [ ] Implement Metal kernel variant for gated `ffn_down`
- [ ] Add runtime flag:
  - [ ] `--experimental-gated-ffn`
- [ ] Add validation harness:
  - [ ] perplexity delta
  - [ ] token output diff
  - [ ] throughput change
  - [ ] memory bandwidth / occupancy observations

### Success criteria

- [ ] Throughput win on at least some prompts/models
- [ ] Quality delta remains acceptable
- [ ] Easy to disable if regressions appear

---

## Explicit non-goals for this branch

- [ ] No ASTC texture-weight experiment
- [ ] No SRAM LUT arithmetic
- [ ] No dense->MoE conversion
- [ ] No semantic slicing
- [ ] No recurrent layer folding

These are later-stage research ideas, not next-branch priorities.

---

## Recommended implementation order

1. [ ] Define `.moon` / `.ziggy` file structure
2. [ ] Build offline compiler from GGUF
3. [ ] Load compiled format directly in runtime
4. [ ] Expand compiled layouts across tensor families
5. [ ] Add calibration hooks for FFN gating
6. [ ] Implement gated FFN experimental kernel
7. [ ] Benchmark correctness + speed

---

## Definition of done

- [ ] MoonQuant is a true compiled execution pipeline
- [ ] GGUF is treated as source format, not final runtime format
- [ ] At least one experimental gated FFN optimization is benchmarkable behind a flag
