# Branch Roadmap — MoonQuant compiled format + gated FFN experiment

## Branch goal

Make MoonQuant a real Apple-first compiled execution format, then add one experimental approximate optimization on top.

---

## Phase 1 — Compiled format first

### Goal

Treat GGUF as the import format and `.moon` / `.ziggy` as the execution format.

### Deliverables

- [ ] Define compiled container format (`.moon` or `.ziggy`)
- [ ] Add manifest/header for:
  - [ ] architecture
  - [ ] quantization version
  - [ ] tensor count
  - [ ] compiled tensor records
  - [ ] alignment
  - [ ] runtime compatibility version
- [ ] Add tensor record entries for:
  - [ ] tensor name
  - [ ] original GGUF type
  - [ ] compiled layout kind
  - [ ] rows / cols
  - [ ] byte offset
  - [ ] byte length
  - [ ] row stride / tile stride
- [ ] Implement `GGUF -> compiled` offline conversion
- [ ] Reuse existing MoonQuant Q4_K repacking path
- [ ] Add compiled-path loader
- [ ] Keep GGUF loader as fallback/reference path
- [ ] Add inspect command for compiled files
- [ ] Add tests for round-trip metadata + tensor lookup

### Success criteria

- [ ] Can load compiled Q4_K_M path without GGUF runtime repacking
- [ ] Same outputs as current reference path within expected tolerance
- [ ] Lower runtime pointer/layout work in Metal path
- [ ] Foundation is ready for more aggressive kernel-specific formats

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
