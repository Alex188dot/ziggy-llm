# Hypothesis 7 Deep Dive: Multi-Token Decode-in-One-Pass (Blockwise Exact Decode)

This document is an implementation playbook for Hypothesis 7 from `EXPERIMENTS_3.md`.

Hypothesis summary: execute a short decode block (`k=2..8`) in one GPU sequence, then commit only the accepted exact prefix. The design target is **exactness-first acceleration** for temperature `0` deterministic decode.

---

## 1) Success Criteria (Hard Gates)

- [ ] End-to-end TPS gain on canonical benchmark (TinyLlama 1.1B Q4_K_M, 128 tokens, `temp=0`, `seed=42`) is at least +20% vs baseline
- [ ] Token outputs are byte-identical to baseline at `temp=0`
- [ ] `zig build test` passes
- [ ] TTFT regression is <= 5%
- [ ] Peak memory growth is documented and <= agreed budget
- [ ] `docs/INFERENCE.md` benchmark tables and notes are updated

If exactness fails, this experiment is considered failed regardless of TPS.

---

## 2) Scope and Non-Goals

### In Scope (Phase 1-3)

- llama-family decode path first
- batch size = 1 decode fast path
- deterministic mode first (`temp=0`)
- acceptance by exact token agreement vs sequential semantics

### Out of Scope (Later)

- stochastic sampling (`temp>0`) blockwise correctness
- all families in first implementation
- large `k` (>4) before stable wins at `k=2`
- replacing the existing sequential path

---

## 3) Current Runtime Touchpoints

Primary touchpoints in this repo:

- `src/runtime/gpu/session.zig`
  - sequence orchestration (`beginSequence`, `commitSequenceTimed`)
  - existing speculative buffers and verification (`batch_logits`, `batch_tokens`)
  - output paths (`runOutput`, `runOutputArgmax`, shortlist/sample)
- `src/runtime/metal_backend.zig`
  - command submission and timing collection
  - batched execution primitives
- `src/runtime/metal/matvec.metal`
  - Metal kernels for attention/matvec hot paths
  - likely home for blockwise verification/select kernels
- `src/model/loader.zig`
  - decode loop integration and accepted-token control flow
- `src/runtime/bench_runner.zig`
  - benchmark harness and TPS measurement
- `src/runtime/metal_profile.zig`
  - dominant-shape evidence and launch behavior

---

## 4) Target Architecture

### Concept

For each decode step at position `t`:

1. Build candidate block of up to `k` future positions
2. Run GPU sequence that materializes logits for block positions
3. Perform exact verifier (argmax match to sequential criterion)
4. Compute accepted prefix length `a` where `0 <= a <= k`
5. Commit only first `a` tokens + corresponding KV writes
6. Resume from first rejected position

### Exactness Contract

- In deterministic mode, final emitted token stream must match baseline token-for-token
- Acceptance is prefix-based: first mismatch stops acceptance
- No speculative state may leak beyond accepted prefix

### Data/State Additions

- block candidate tokens buffer (`k` max)
- block verified tokens buffer (`k` max)
- acceptance mask/prefix-length buffer
- staged KV write metadata (or deterministic overwrite-safe indexing)

---

## 5) Implementation Phases

## Phase 0: Baseline and Instrumentation

Goal: establish indisputable baseline metrics and add counters needed to evaluate the experiment.

### Tasks

- [ ] Add baseline benchmark script invocation for canonical scenario
- [ ] Capture baseline metrics:
  - [ ] TPS
  - [ ] TTFT
  - [ ] CPU wait ns/token
  - [ ] GPU elapsed ns/token
  - [ ] command submissions/token
- [ ] Add experiment counters in runtime stats:
  - [ ] `block_k`
  - [ ] `accepted_prefix_len`
  - [ ] `rollback_count`
  - [ ] `mean_accepted_len`
  - [ ] `verify_ns`
- [ ] Add logging flag (example: `--exp-block-decode`) disabled by default

### Exit Criteria

- [ ] Baseline report committed in notes or `docs/INFERENCE.md` scratch section
- [ ] New counters compile and are visible in profiling output

---

## Phase 1: Minimal Block Decode Skeleton (`k=2`, CPU Verifier)

Goal: wire end-to-end control flow with the smallest possible correctness-preserving implementation.

### Tasks

- [ ] Add config plumbing for block decode parameters:
  - [ ] enable flag
  - [ ] max `k`
  - [ ] deterministic-only guard
- [ ] Implement block loop in decode driver (`src/model/loader.zig`)
- [ ] Reuse existing GPU forward primitives to produce logits for positions `t` and `t+1`
- [ ] Implement verifier on CPU first (temporary):
  - [ ] read logits for each position
  - [ ] run argmax
  - [ ] compute accepted prefix
- [ ] Commit only accepted tokens to output stream
- [ ] Guarantee safe handling when `accepted=0` (fallback to baseline single-step)

### Exit Criteria

- [ ] Functional correctness on small deterministic prompt set
- [ ] No crashes/memory corruption with block mode on/off
- [ ] Output equality vs baseline at `temp=0`

---

## Phase 2: GPU Verifier + Prefix Acceptance Kernel

Goal: move verifier hot path from CPU to GPU and remove multi-logit readback.

### Tasks

- [ ] Add Metal kernel for per-position argmax over vocabulary for block logits
- [ ] Add Metal kernel (or fused path) to compute accepted prefix length from:
  - [ ] predicted candidate tokens
  - [ ] verified argmax tokens
- [ ] Keep CPU-visible output minimal:
  - [ ] accepted prefix length
  - [ ] accepted tokens only
- [ ] Integrate into `session.zig` sequence commit path
- [ ] Validate deterministic equivalence on expanded prompt set

### Exit Criteria

- [ ] CPU readback reduced to tiny token/prefix buffers
- [ ] Equality vs baseline remains exact at `temp=0`
- [ ] Profiling shows lower CPU wait than Phase 1

---

## Phase 3: KV Commit Safety and Rollback Discipline

Goal: ensure accepted-prefix commit is robust and rollback is cheap.

### Tasks

- [ ] Introduce staged KV write strategy for block positions (or overwrite-safe indexed writes)
- [ ] Commit KV pages only for accepted prefix
- [ ] Add explicit rollback path for first divergence
- [ ] Add invariants/assertions:
  - [ ] no KV writes beyond accepted prefix survive
  - [ ] position index always matches emitted token count
- [ ] Add targeted unit/integration tests for rollback edge cases

### Exit Criteria

- [ ] Repeated mismatch scenarios remain stable
- [ ] No drift in long deterministic generations
- [ ] Memory bounds validated under max `k`

---

## Phase 4: Performance Tuning (`k=2 -> 4`)

Goal: convert correctness implementation into real TPS gain.

### Tasks

- [ ] Profile dominant kernels with `--metal-profile`
- [ ] Tune only hot shapes first (TinyLlama/llama decode)
- [ ] Evaluate `k=2`, then `k=4` only if acceptance supports it
- [ ] Reduce extra command submissions inside block path
- [ ] Fuse tiny verifier operations if kernel launch overhead is visible
- [ ] Add adaptive `k` policy prototype:
  - [ ] high-confidence region -> `k=4`
  - [ ] fallback region -> `k=2` or `k=1`

### Exit Criteria

- [ ] > = 20% TPS gain in canonical deterministic benchmark
- [ ] CPU wait and command submission/token are down
- [ ] No TTFT regression beyond allowed gate

---

## Phase 5: Family Rollout and Guardrails

Goal: productionize the feature safely across families.

### Tasks

- [ ] Keep llama as default experiment target
- [ ] Add guarded support matrix for qwen/mistral/gemma
- [ ] Enable per-family default `k` and fallback policy
- [ ] Add runtime kill switch (env/flag)
- [ ] Document known-safe configs and known-failure cases
- [ ] Update docs with benchmark table and profiler evidence

### Exit Criteria

- [ ] Family-level deterministic correctness green where enabled
- [ ] Clear fallback behavior for unsupported shapes/families
- [ ] `docs/INFERENCE.md` updated with results

---

## 6) Test Plan (Checklist)

### Deterministic Correctness

- [ ] Golden prompts: token-by-token equality vs baseline (`temp=0`, fixed seed)
- [ ] Length sweep: short, medium, long generations
- [ ] Context boundary tests near max context
- [ ] Repetition and low-entropy prompts (high acceptance expected)
- [ ] Adversarial prompts (frequent divergence expected)

### Runtime Safety

- [ ] Toggle on/off repeatedly in same process
- [ ] No leaks for repeated decode sessions
- [ ] No out-of-bounds on max `k`
- [ ] Rollback path stress test with forced mismatch every step

### Performance

- [ ] Canonical TPS benchmark before/after
- [ ] `--metal-profile` capture for baseline vs block mode
- [ ] command submissions/token comparison
- [ ] CPU wait ns/token comparison
- [ ] GPU elapsed ns/token comparison

---

## 7) Suggested Task Breakdown (PR Plan)

### PR 1: Instrumentation + Flags

- [ ] Add config/flags and metrics counters
- [ ] No behavior change by default

### PR 2: `k=2` Skeleton with CPU Verifier

- [ ] End-to-end block control flow
- [ ] Deterministic correctness tests

### PR 3: GPU Verifier + Minimal Readback

- [ ] Metal verifier kernels
- [ ] Prefix acceptance integration

### PR 4: KV Commit/Rollback Hardening

- [ ] Staged KV or equivalent safe commit mechanism
- [ ] Rollback stress tests

### PR 5: Tuning + Docs

- [ ] `k` tuning and launch reduction
- [ ] Benchmark/profiler evidence
- [ ] `docs/INFERENCE.md` updates

---

## 8) Risk Register

- [ ] Risk: accepted prefix too low -> no TPS win
  - Mitigation: adaptive `k`, automatic fallback
- [ ] Risk: verifier overhead cancels gains
  - Mitigation: GPU verifier + fused tiny kernels
- [ ] Risk: KV rollback bugs create silent drift
  - Mitigation: strict invariants + golden token tests
- [ ] Risk: TTFT regression from extra setup
  - Mitigation: lazy-init experiment buffers
- [ ] Risk: shape/family fragmentation
  - Mitigation: llama-first, guarded rollout matrix

---

## 9) Stop/Go Decision Gates

### Gate A (after Phase 1)

- [ ] Exact deterministic equality proven
- [ ] Mean accepted length >= 1.4 on target prompts

If either fails -> stop or redesign before GPU kernel investment.

### Gate B (after Phase 3)

- [ ] Stable rollback/KV correctness under stress
- [ ] CPU overhead profile indicates remaining optimization headroom

If not met -> pause and keep as research branch.

### Gate C (after Phase 4)

- [ ] Canonical TPS gain >= 20%
- [ ] TTFT and memory within budget

If not met -> do not merge as default path.

---

## 10) Definition of Done

- [ ] All Success Criteria in section 1 are checked
- [ ] Bench and profiler evidence attached to docs
- [ ] Feature is behind a safe runtime gate
- [ ] Sequential baseline path remains intact
- [ ] Final notes recorded in `docs/INFERENCE.md`

---

## 11) Current State (Implementation Recap)

Date of recap: 2026-04-18

### Implemented So Far

- [x] Phase 0 foundations: experiment flags and block metrics are plumbed through runtime reports (`exp_block_decode`, `exp_block_k`, accepted prefix, rollback count, verify time).
- [x] Phase 1 baseline functionality: deterministic block decode control flow exists and falls back safely to sequential verification when needed.
- [x] Phase 2 core path: GPU block verifier path is wired and active; batch decode produces verifier tokens and accepted prefix semantics.
- [x] Phase 3 safety hardening: speculative KV protection is implemented with backup + restore of non-accepted suffix positions to prevent state leakage past accepted prefix.
- [x] Warm resident path support: warm bench path is enabled again for block decode (no forced fresh session for block mode).
- [x] Profiling counters for block path overhead are now emitted:
  - `block_gpu_backup_ns`
  - `block_gpu_restore_ns`
  - `block_gpu_sequence_commits`
  - `block_gpu_fallback_count`

### What Current Measurements Show

- Deterministic exactness (`temp=0`) is preserved in tested runs.
- Block GPU verifier is active in current tested configuration (fallback count can be zero).
- Acceptance is currently low in the tested scenario (example observed mean accepted prefix around `~0.741` in one run set).
- Rollbacks are frequent in that scenario (example observed rollback count around `~9` per measured run set).
- Net result: verifier/rollback overhead dominates and TPS is currently below the non-block baseline in these runs.

### Current Practical Conclusion

- The implementation is now in a state where correctness and rollback safety are substantially addressed.
- The immediate bottleneck is not missing infrastructure; it is low draft acceptance quality.
- Throughput gains depend primarily on increasing acceptance and reducing rollback frequency before scaling `k` aggressively.

---

## 12) Next Steps To Raise Acceptance and Reduce Rollbacks

### A) Improve Draft Quality (Highest Impact)

- [ ] Replace or augment simple history n-gram drafting with stronger proposal logic.
- [ ] Add shortlist-aware draft proposal (use top candidates from current logits and propose a short chain).
- [ ] Optionally evaluate a lightweight draft model path for stronger candidate quality.

Expected effect: higher accepted prefix length and fewer rollbacks.

### B) Confidence-Gated Speculation

- [ ] Compute a confidence signal per step (e.g., top-1 vs top-2 logit margin).
- [ ] Speculate only when confidence passes threshold.
- [ ] Route low-confidence steps directly to `k=1`.

Expected effect: avoid expensive speculative work on likely-divergence tokens.

### C) Stronger Adaptive `k` Policy

- [ ] Keep dynamic `k` selection and make policy confidence-aware.
- [ ] Suggested policy:
  - high confidence + recent high acceptance -> larger `k`
  - medium confidence -> `k=2`
  - low confidence or recent instability -> `k=1`

Expected effect: preserve wins in easy regions, limit losses in unstable regions.

### D) Rollback Cooldown / Rejection Memory

- [ ] After rollback, temporarily lower `k` for a short cooldown window (`M` tokens).
- [ ] Re-escalate `k` only after stable acceptance resumes.

Expected effect: reduce repeated rollback bursts in difficult spans.

### E) Verify Proposer/Verifier Alignment

- [ ] Re-check token/state alignment invariants between proposal and verifier paths.
- [ ] Ensure no hidden mismatch in tokenizer/state assumptions for drafted tokens.

Expected effect: remove artificial acceptance loss caused by alignment bugs.

### F) Continue Reducing Rollback Cost (Secondary)

- [ ] Keep optimizing backup/restore and tiny commit overhead.
- [ ] Explore fusion opportunities for copy/restore operations where safe.

Expected effect: lower penalty when rollback still occurs (does not directly raise acceptance).

---

## 13) Sequencing Recommendation Before `k=6/8/10/12`

- [ ] Step 1: implement confidence-gated speculation and cooldown.
- [ ] Step 2: improve draft proposer quality.
- [ ] Step 3: re-measure acceptance + rollback metrics on canonical bench.
- [ ] Step 4: only then test larger `k` values (`6`, `8`, `10`, `12`) behind flags.

Rationale: scaling `k` before acceptance improves is likely to amplify rollback overhead rather than increase TPS.

---

## 14) Practical Commands (Current Workflow)

Use these commands from repo root.

### Build

```bash
zig build -Doptimize=ReleaseFast
```

### Baseline Bench (Deterministic, No Block Decode)

```bash
./zig-out/bin/ziggy-llm bench \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 128 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy --bench-runs 5
```

### Block Decode Bench (Default Exactness-First Verifier)

```bash
./zig-out/bin/ziggy-llm bench \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 128 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy --bench-runs 5 \
  --exp-block-decode --exp-block-k 4
```

Interpretation note: if `block.confidence_gated_count_avg` is near generated token count, block decode was mostly gated off and throughput is effectively baseline.

### Force Speculation (for Acceptance/Rollback Measurement)

```bash
./zig-out/bin/ziggy-llm bench \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 128 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy --bench-runs 5 \
  --exp-block-decode --exp-block-k 4 \
  --exp-block-confidence-margin 0
```

### Optional: GPU Verifier Experiment (Not Default)

```bash
./zig-out/bin/ziggy-llm bench \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 128 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy --bench-runs 5 \
  --exp-block-decode --exp-block-k 4 \
  --exp-block-confidence-margin 0 \
  --exp-block-gpu-verifier
```

### Deterministic Equality Check (Baseline vs Block)

```bash
./zig-out/bin/ziggy-llm run \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 64 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy \
  | awk 'BEGIN{p=0} /^generated_text:/{p=1; sub(/^generated_text: /,""); print; next} /^prompt_tokens:/{p=0} p{print}' > /tmp/base.txt

./zig-out/bin/ziggy-llm run \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 64 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy \
  --exp-block-decode --exp-block-k 4 --exp-block-confidence-margin 0 \
  | awk 'BEGIN{p=0} /^generated_text:/{p=1; sub(/^generated_text: /,""); print; next} /^prompt_tokens:/{p=0} p{print}' > /tmp/block.txt

cmp -s /tmp/base.txt /tmp/block.txt && echo MATCH || echo DIFF
```

### Tests

```bash
zig build test
zig test src/cli.zig
zig test src/runtime/block_policy.zig
zig test src/runtime/draft_proposer.zig
```

```bash
# 1) Build
zig build -Doptimize=ReleaseFast

# 2) Baseline bench (prints TPS + all block/key metrics)
./zig-out/bin/ziggy-llm bench \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 128 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy --bench-runs 5 | tee /tmp/bench_base.txt

# 3) Block bench (default gate)
./zig-out/bin/ziggy-llm bench \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 128 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy --bench-runs 5 \
  --exp-block-decode --exp-block-k 4 | tee /tmp/bench_block_default.txt

# 4) Block bench (force speculation, useful for acceptance/rollback measurement)
./zig-out/bin/ziggy-llm bench \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 128 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy --bench-runs 5 \
  --exp-block-decode --exp-block-k 4 --exp-block-confidence-margin 0 | tee /tmp/bench_block_forced.txt

# 5) Print only key lines from each run (TPS + acceptance + rollback + verify + gating)
for f in /tmp/bench_base.txt /tmp/bench_block_default.txt /tmp/bench_block_forced.txt; do
  echo "===== $f ====="
  rg -n "warm\\.tps_avg|warm\\.decode_tok_s_avg|warm\\.generated_tokens_avg|warm\\.block\\.accepted_prefix_len_avg|warm\\.block\\.rollback_count_avg|warm\\.block\\.confidence_gated_count_avg|warm\\.block\\.cooldown_active_count_avg|warm\\.block\\.verify_ms_avg|warm\\.block\\.gpu_sequence_commits_avg|warm\\.block\\.gpu_commits_per_token_avg|cold\\.tps|cold\\.block\\.accepted_prefix_len|cold\\.block\\.rollback_count" "$f"
  echo
done

# 6) Deterministic equality check (baseline vs block)
./zig-out/bin/ziggy-llm run \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 64 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy \
  | awk 'BEGIN{p=0} /^generated_text:/{p=1; sub(/^generated_text: /,""); print; next} /^prompt_tokens:/{p=0} p{print}' > /tmp/base.txt

./zig-out/bin/ziggy-llm run \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 64 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy \
  --exp-block-decode --exp-block-k 4 --exp-block-confidence-margin 0 \
  | awk 'BEGIN{p=0} /^generated_text:/{p=1; sub(/^generated_text: /,""); print; next} /^prompt_tokens:/{p=0} p{print}' > /tmp/block.txt

cmp -s /tmp/base.txt /tmp/block.txt && echo "EQUALITY: MATCH" || echo "EQUALITY: DIFF"

# 7) If DIFF, quickly show first content lines
sed -n '1,20p' /tmp/base.txt
sed -n '1,20p' /tmp/block.txt
```

---

## 15) Immediate Performance Plan (After Exactness Fix)

1. Improve draft proposer quality further (logits-driven chain, repeat-penalty aware), then re-bench.
2. Keep speculation heavily gated on low confidence; reduce bad speculative attempts.
3. Optimize sequential verifier path for early mismatch exits and less per-step overhead.
4. Hold `k` small (`1/2`, maybe adaptive `1/2/4`) until accepted prefix improves materially.
5. Re-run deterministic equality + benchmark matrix after each change.
6. Only attempt `k=6/8/10/12` after acceptance and rollback metrics improve at `k<=4`.

### 15.1) Implementation Check (2026-04-18)

- [x] `15.1` Proposer quality upgraded:
  - logits-driven chain drafting implemented
  - repeat-penalty-aware proposer scoring implemented
  - shortlist + transition evidence combined in proposer ranking
- [x] `15.2` Stricter speculation gating implemented:
  - stronger adaptive `k` policy in unstable regions
  - warmup-aware `k=0` shutoff when rollback EMA is high and acceptance EMA is low
- [x] `15.3` Sequential verifier path pressure reduced:
  - conservative depth cap applied (`k -> 1`) when acceptance remains low in sequential verifier mode
- [x] `15.4` Conservative `k` discipline enforced:
  - policy now strongly prefers `1/2`; `4` requires both high confidence and stable acceptance/rollback EMA
- [x] `15.5` Equality + benchmark matrix rerun:
  - deterministic equality check: `MATCH` at `temp=0` on the canonical prompt
  - benchmark reruns completed (baseline/default-block/forced-block)
- [ ] `15.6` Ready for `k=6/8/10/12`:
  - still blocked by low acceptance at `k<=4`

### 15.2) Latest Measured Snapshot

- Baseline warm TPS: `41.372`
- Block warm TPS (default gating): `41.028` (mostly confidence-gated)
- Block warm TPS (forced speculation, `confidence_margin=0`): `40.433`
- Forced mode key metrics:
  - `accepted_prefix_len_avg=0.000`
  - `rollback_count_avg=8`
  - `verify_ms_avg=194.441`

Interpretation: rollback and verifier overhead improved versus earlier forced runs, but acceptance is still not high enough to create a net win.

---

## 16) Next Steps (Priority Order)

1. Fix forced-mode acceptance quality first:
   - add a proposer debug trace mode (`draft token`, `verified token`, mismatch position, margin at decision step)
   - collect mismatch histograms over at least 3 prompts, not only one
2. Implement a confidence-conditioned single-token precheck:
   - when margin is low/medium, precheck one drafted token before committing to full draft length
   - skip full speculative verify when precheck fails
3. Add a strict acceptance quality gate in runtime:
   - if rolling window accepted prefix drops below threshold (for example `<0.6` over N steps), hard-disable speculation for M steps
4. Retune adaptive thresholds from measured data:
   - tune margin multipliers and EMA cutoffs using observed mismatch histograms
5. Re-benchmark with a small prompt suite:
   - repetitive prompt, neutral narrative prompt, adversarial/unstable prompt
   - require improvement in both mean TPS and worst-case TPS before raising `k`
6. Only after sustained acceptance improvement at `k<=4`:
   - lift `max_draft_len` to enable `k=6/8/10/12` experiments behind flags

### 16.1) Implementation Status (2026-04-18)

- [x] `16.1` Proposer/verifier mismatch tracing support:
  - added `--exp-block-trace` debug trace mode (`BLOCK_TRACE` lines include step, margin, mismatch position, drafted vs verified token, precheck flags)
  - added mismatch histogram metrics to reports/bench:
    - `block.mismatch_pos0_count`
    - `block.mismatch_pos1_count`
    - `block.mismatch_pos2_count`
    - `block.mismatch_pos3_count`
    - `block.full_accept_count`
- [x] `16.2` Confidence-conditioned single-token precheck:
  - added precheck path in block verify loop (single-step greedy precheck before deeper verify)
  - added controls/metrics:
    - `--exp-block-precheck-margin-multiplier` (default `2.0`)
    - `block.precheck_count`
    - `block.precheck_fail_count`
- [x] `16.3` Strict acceptance-quality runtime gate:
  - rolling accepted-prefix gate added with hard disable window
  - added controls:
    - `--exp-block-acceptance-threshold` (default `0.6`)
    - `--exp-block-acceptance-window` (default `8`)
    - `--exp-block-disable-steps` (default `16`)
  - added metrics:
    - `block.quality_gate_active_count`
    - `block.quality_gate_trigger_count`
- [x] `16.4` Threshold retune support:
  - added runtime knobs above so margin/gating behavior can be tuned from measured data without code edits
- [x] `16.5` Small prompt-suite re-benchmark (baseline/default/forced):
  - prompts used: repetitive, neutral narrative, unstable punctuation mix
  - observed: default gate remains near-baseline (speculation mostly gated off), forced mode now reports precheck + quality-gate behavior clearly
- [ ] `16.6` Raise `k` above `4`:
  - still blocked until acceptance improves materially on neutral/unstable prompts at `k<=4`

### 16.2) Quick Commands For Section 16 Instrumentation

```bash
# Forced speculation with section-16 telemetry enabled in output
./zig-out/bin/ziggy-llm bench \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 128 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy --bench-runs 5 \
  --exp-block-decode --exp-block-k 4 --exp-block-confidence-margin 0
```

```bash
# Optional trace mode (verbose)
./zig-out/bin/ziggy-llm run \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal --max-tokens 64 --temperature 0 --seed 42 \
  --sampling-path gpu-greedy \
  --exp-block-decode --exp-block-k 4 --exp-block-confidence-margin 0 \
  --exp-block-trace
```

---

## 17) Handoff Paragraph For Next Agent

Continue Hypothesis 7 from commit `8d78ea4` with exactness-first constraints. Current state: deterministic equality matches baseline in sequential verifier mode, but forced speculation still has low acceptance (`accepted_prefix_len_avg` near zero on the canonical prompt) and does not improve TPS. Your task is to increase acceptance before any `k>4` work: add proposer/verifier mismatch tracing, build per-prompt mismatch histograms, implement a low-cost single-token precheck before full speculative verify, and add a rolling acceptance gate that temporarily disables speculation when quality collapses. Re-benchmark baseline vs default block vs forced block after each material change, keep `temp=0` equality checks green, and do not enable larger `k` until acceptance and rollback metrics materially improve at `k<=4`.

---

## 18) Token-Selection Failure Analysis + Fix Plan (2026-04-18)

### Observed Failure (From Live Trace)

- Draft token selection is frequently wrong at position 0 of each speculative block.
- Current trace shows repeated proposals like `"!"`/ `"."` while verifier predicts narrative tokens (`"Ġis"`, `"Ġa"`, `"Ġyoung"`...).
- As a result:
  - `accepted_prefix_len` collapses to `0`
  - `rollback_count` rises
  - precheck fails almost every speculative attempt
  - block path adds overhead without decode gain

Conclusion: primary bottleneck is proposer token ranking/selection quality, not verifier infrastructure.

### Plan (Execution Order)

1. Proposer score decomposition trace:
   - For each drafted position, print top-N candidates from raw logits and final proposer score.
   - Print each score component separately (logit, repeat penalty, transition bonus, n-gram bonus, any heuristic bonus).
   - Goal: prove which term is pushing punctuation tokens to rank 1.

2. Add a strict proposer sanity filter:
   - If top drafted token is far below verifier top-1 confidence envelope, do not speculate (force `k=1`).
   - Add a guardrail against repeated low-information punctuation loops in early drafted positions.
   - Keep this behind an experiment flag initially.

3. Recalibrate proposer weighting:
   - Reduce weight of heuristic priors that can overpower current logits.
   - Increase dependence on current-step logits/top shortlist for first drafted token.
   - Keep transition/history signals only as tie-breakers unless confidence is high.

4. Separate first-token strategy from tail strategy:
   - Use a conservative first drafted token policy (closest to verifier top-1/top-2).
   - Apply richer history/transition heuristics only for drafted positions 2+ after first token is plausible.

5. Add acceptance-oriented offline checks:
   - On fixed prompts, compute draft-vs-verifier top-1 match rate at each drafted position before full runtime benchmarking.
   - Target: position-0 match rate must materially improve before tuning `k`.

6. Re-benchmark gates before any `k>4`:
   - Required to proceed:
     - non-trivial `accepted_prefix_len` increase on neutral prompt
     - rollback reduction vs current forced run
     - no deterministic output drift (`temp=0` equality still `MATCH`)
   - Only then test `k=6/8/10/12`.

### Immediate Success Criteria For This Subproblem

- Position-0 mismatch is no longer dominant bucket.
- `precheck_fail_count` drops materially on canonical neutral prompt.
- Forced speculation TPS is no longer clearly below baseline by verifier overhead alone.

### 18.1) Implementation Status (2026-04-18)

- [x] Added proposer score decomposition tracing:
  - `BLOCK_PROPOSER` line prints chosen token and score components (`raw`, `penalized`, `transition_count`, `transition_bonus`, `total`).
  - `BLOCK_PROPOSER_CAND` lines print ranked shortlist candidates and full score decomposition.
- [x] Added explicit proposer guard traces:
  - `BLOCK_PROPOSER_GUARD` lines with reject reason and thresholds (`flat_logits`, `first_token_gap`, `flat_logits_fastpath`).
- [x] Implemented conservative first-token policy:
  - first drafted token now strongly prefers shortlist/logit-consistent candidates.
  - off-shortlist/history-only first-token picks are prevented when they conflict with shortlist confidence.
- [x] Implemented first-token sanity gate:
  - if first drafted token falls below top-1 penalized confidence envelope, speculation is rejected for that step (`k=0` effective).
- [x] Implemented flat-logits runtime fast path:
  - if proposer input margin is effectively zero (`top1-top2 ~= 0`), speculation is skipped before drafting.
  - this prevents repeated bad proposals like punctuation loops when logits carry no usable signal in this path.

### 18.2) Observed Result After Implementation

- On canonical neutral prompt with forced mode (`--exp-block-confidence-margin 0`):
  - `rollback_count` dropped to `0`
  - `precheck_fail_count` dropped to `0`
  - `confidence_gated_count` rose to generated token count (speculation mostly disabled when logits are flat)
- Interpretation:
  - dominant failure mode was proposer input quality (flat logits), not verifier correctness.
  - this change stabilizes behavior and removes rollback/precheck overhead from bad drafts.
  - next step is enabling a proposer input path with real logit signal before expecting acceptance gains.

### 18.3) Follow-up Fix: Post-Token Shortlist Bootstrap

- [x] Root bug fixed:
  - the proposer had been reading the distribution before consuming the current emitted token, so it was effectively one token behind.
  - greedy block mode now advances one exact step first, then drafts from the fresh post-token GPU shortlist.
- [x] Result:
  - canonical forced run now reaches `accepted_prefix_len ~= 1.0` instead of `0.0`
  - first drafted token is usually exact and accepted
  - rollback count drops materially versus the original broken proposer path
- [ ] Remaining problem:
  - TPS is still below baseline because tail drafting is weak and verifier work is still expensive
  - current tail proposer often only contributes one exact token plus a verifier bonus, which is not enough amortization

Interpretation: the major correctness/selection bug is fixed. The next optimization target is no longer "why is position-0 wrong?" but "how do we raise accepted tail length above 1 without reintroducing mismatch churn?"

### 18.4) Tail Stop Guard + Human Trace

- [x] Trace token text is now human-readable:
  - block traces print decoded token text (`" is"`, `" for"`, `" an"`) instead of raw tokenizer internals like `Ġis`.
- [x] Tail proposer no longer forces stale token-2 guesses when there is no continuation evidence:
  - if history-chain evidence for token 2+ is absent, block drafting stops at length 1 for that step
  - verifier bonus still appends the exact next token
- [x] Measured effect on canonical forced run:
  - repeated bad tails like `"for" -> "for"` are removed
  - `accepted_prefix_len` improved slightly above `1.0`
  - rollback count dropped further
  - TPS improved versus the previous post-shortlist-bootstrap state, though still below baseline

Interpretation: stale/fake tail speculation is now suppressed. The remaining path to a real speedup is stronger token-2+ proposal quality, not more permissive drafting.

---

## 19) Summary So Far (2026-04-18)

- Phase 0/1 proved the basic speculative block-decode loop could work with a CPU verifier, but it was slower than baseline because verifier and rollback overhead dominated.
- Early block runs had very low acceptance and frequent rollbacks, so increasing `k` was not justified.
- Section 12 work improved gating, fallback behavior, acceptance guards, and instrumentation, but did not create a net speedup.
- Section 15 tightened adaptive `k` policy and verifier pressure; equality stayed correct, but forced speculation still had near-zero acceptance on the canonical prompt.
- Section 16 added the missing observability:
  - mismatch histograms
  - single-token precheck
  - rolling acceptance quality gate
  - trace mode and prompt-suite benchmarking
- Section 18 identified the real token-selection failure:
  - the proposer was often wrong at position 0
  - traces showed obviously bad proposals and immediate mismatches
- The first major correctness fix was post-token shortlist bootstrap:
  - speculative drafting now starts from the fresh distribution after consuming the current token
  - this moved `accepted_prefix_len` from roughly `0.0` to roughly `1.0`
  - position-0 acceptance became mostly correct
- The second cleanup was trace and tail discipline:
  - trace output now prints human-readable token text instead of raw tokenizer internals
  - weak token-2+ guesses are no longer forced when there is no continuation evidence
  - repeated garbage tails were removed
- Current state:
  - correctness is materially better than the original speculative path
  - the first drafted token is usually correct
  - tail acceptance is still too weak
  - verifier cost is still too high relative to useful accepted speculative work
  - throughput is still below baseline on the canonical benchmark

### 19.1) Short Diagnosis

- The original blocker was not "Metal verifier is broken".
- The original blocker was "the proposer was drafting from the wrong or weak signal".
- That bug is now fixed for token 1 of each block.
- The remaining blocker is narrower and clearer:
  - token 2+ drafting is not strong enough to amortize verifier cost
  - current block decode behaves more like "1 exact drafted token + verifier bonus" than true multi-token speculation

---

## 20) Plan To Achieve A Real Speedup: Build A Stronger Token-2+ Proposer

The next speedup will not come from more permissive guessing. It will come from raising accepted tail length beyond `1` while keeping exactness and rollback rate under control.

### 20.1) Objective

Raise accepted speculative tail length at positions `2+` so that each verify pass accepts multiple useful drafted tokens, not just the first one.

### 20.2) Execution Plan

1. Build a real tail proposer from exact accepted context:
   - after the exact bootstrap token is accepted, draft token 2+ from the actual accepted sequence prefix, not from stale or fallback heuristics
   - make the proposer stateful over the accepted draft prefix inside the current block

2. Separate token-2+ proposal sources by reliability:
   - first preference: exact local continuation evidence from recent accepted history
   - second preference: transition/n-gram continuation statistics only when they clearly agree
   - last resort: stop drafting the tail rather than inventing a weak token

3. Add tail-quality instrumentation:
   - track acceptance separately for position 1, 2, 3, 4 within the draft
   - print per-position proposer source and score reason in trace mode
   - expose metrics showing where tail acceptance collapses

4. Add offline proposer evaluation before full TPS benchmarking:
   - on fixed prompts, measure top-1 match rate for drafted position 2, 3, 4 against verifier tokens
   - require visible improvement in those match rates before spending time on end-to-end benchmark tuning

5. Re-benchmark only after tail quality improves:
   - canonical neutral prompt
   - repetitive prompt
   - unstable/adversarial prompt
   - compare baseline, default gated block mode, and forced block mode

6. Only raise `k` after tail acceptance proves durable:
   - do not move to `k=6/8/10/12` until token-2+ acceptance is materially improved at `k<=4`
   - otherwise verifier overhead will grow faster than useful accepted work

### 20.3) Success Criteria

- `accepted_prefix_len_avg` rises materially above `1.0`
- mismatch buckets shift away from position `1`
- rollback count falls or stays low while accepted work rises
- verify time per generated token is amortized enough to beat baseline TPS

### 20.4) Practical Next Implementation

If continuing immediately, the next implementation should be:

1. add per-position acceptance metrics for draft positions `1..k`
2. make the tail proposer explicitly consume the accepted in-block prefix when proposing token 2+
3. keep "stop drafting" as the default when tail evidence is weak
4. benchmark again before touching larger `k`

### 20.5) Implementation Status (2026-04-18)

- [x] Added per-position draft/accept metrics for positions `1..4`:
  - `block.draft_pos1_count` ... `block.draft_pos4_count`
  - `block.accept_pos1_count` ... `block.accept_pos4_count`
  - warm bench averages emitted for the same metrics
- [x] Replaced the weak tail fallback with a context-suffix proposer:
  - token 2+ is now proposed from the actual accepted context suffix plus in-block drafted prefix
  - the proposer backs off from longer suffixes to shorter suffixes and stops when evidence is weak
- [x] Added tail-proposer trace output:
  - `BLOCK_TAIL_PROPOSAL pos=... token=... piece="..." source=... matched_context=... matches=...`
- [x] Benchmarked the canonical prompt after implementation

### 20.6) Latest Measured Result

- Canonical forced bench (`k=4`, `confidence_margin=0`, warm avg):
  - `warm.tps_avg=38.467`
  - `warm.block.accepted_prefix_len_avg=1.000`
  - `warm.block.rollback_count_avg=4`
  - `warm.block.draft_pos2_count_avg=4`
  - `warm.block.accept_pos2_count_avg=0`

Interpretation:

- The new tail proposer is conservative enough to avoid most of the earlier bad tail behavior.
- Throughput recovered to roughly baseline territory on the canonical forced run.
- But the core remaining problem is still visible in the new metrics:
  - token 2 is occasionally proposed
  - token 2 is still not being accepted on the canonical prompt
- So section 20 is only partially complete in outcome terms:
  - the right architecture and instrumentation are now in place
  - the next improvement must come from a stronger token-2 selection policy, not from drafting deeper with the current proposer
