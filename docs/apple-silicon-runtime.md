# Apple Silicon Runtime Notes

This document records the current Apple Silicon runtime notes for the `llama` Metal path.

## Target Machine

Measurements recorded on April 2, 2026:

- MacBook Pro (`Mac15,6`)
- Apple M3 Pro
- 18 GB unified memory
- macOS 26.0.1 (`25A362`)
- Zig 0.15.2

## Reproducible Benchmark Setup

Build the default release binary:

```bash
zig build
```

Benchmark the CPU and Metal backends on the same llama-family GGUF model:

```bash
./zig-out/bin/ziggy-llm bench -m /path/to/model.gguf -p "Hello" --max-tokens 256 --seed 7 --backend cpu
./zig-out/bin/ziggy-llm bench -m /path/to/model.gguf -p "Hello" --max-tokens 256 --seed 7 --backend metal
```

To force the legacy generic `Q4_K` Metal path for comparison, disable MoonQuant packing explicitly:

```bash
./zig-out/bin/ziggy-llm bench -m /path/to/model.gguf -p "Hello" --max-tokens 256 --seed 7 --backend metal --moon-quant disabled
```

To generate synthetic `Q4_K` and `Q6_K` benchmark fixtures with TinyLlama-like decode shapes:

```bash
zig build bench-fixture -- --format q4_k --output /tmp/moonbench-q4.gguf
zig build bench-fixture -- --format q6_k --output /tmp/moonbench-q6.gguf
```

Those fixtures use a short `16`-token context, so the reproducible comparison command uses `--max-tokens 12`:

```bash
./zig-out/bin/ziggy-llm bench -m /tmp/moonbench-q4.gguf -p a --max-tokens 12 --seed 0 --temperature 0 --bench-runs 5 --backend metal
./zig-out/bin/ziggy-llm bench -m /tmp/moonbench-q4.gguf -p a --max-tokens 12 --seed 0 --temperature 0 --bench-runs 5 --backend metal --moon-quant disabled
./zig-out/bin/ziggy-llm bench -m /tmp/moonbench-q6.gguf -p a --max-tokens 12 --seed 0 --temperature 0 --bench-runs 5 --backend cpu
```

Recent local llama Metal work on the primary machine reached roughly `27.2` warm decode tok/s on the current MoonQuant-enabled TinyLlama path.

## MoonQuant Benchmark Results

Synthetic benchmark fixtures measured on April 2, 2026:

| Model | Backend | MoonQuant | Warm decode tok/s | Warm decode ms |
| --- | --- | --- | ---: | ---: |
| `/tmp/moonbench-q4.gguf` | `metal` | `enabled` | `574.046` | `20.904` |
| `/tmp/moonbench-q4.gguf` | `metal` | `disabled` | `421.241` | `28.487` |
| `/tmp/moonbench-q6.gguf` | `cpu` | `n/a` | `187.014` | `64.166` |

Real-model `bench --bench-runs 5` spot check on April 2, 2026:

| Model | Backend | MoonQuant | Warm decode tok/s | Warm decode ms |
| --- | --- | --- | ---: | ---: |
| `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | `metal` | `enabled` | `39.194` | `816.453` |
| `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | `metal` | `disabled` | `35.215` | `908.705` |

On this machine and with these commands, the packed MoonQuant `Q4_K` path is about `36.3%` faster than the generic `Q4_K` baseline on the synthetic fixture and about `11.3%` faster on the local TinyLlama decode run.

## Current Metal Notes

The active Metal path is the llama decode runtime:

- hidden-state, RMSNorm, residual adds, RoPE, attention, FFN, and output projection run on the GPU
- command submission is batched so a token step waits at final readback rather than after each tiny op
- CPU remains responsible for prompt orchestration, model loading, and token sampling

Current limitations:

- Apple Silicon macOS builds only
- llama-family GGUF models only
- sampling still happens on CPU
- more kernel fusion is still available, especially around projection and sampling work

## Running Inference On GPU

Example:

```bash
zig build
./zig-out/bin/ziggy-llm run -m /path/to/model.gguf -p "Hello" --max-tokens 32 --seed 7 --backend metal
```

Notes:

- `--backend auto` will try Metal first on Apple Silicon and fall back to CPU if Metal initialization fails
- `--backend cpu` remains the correctness and comparison path
- `--moon-quant disabled` keeps the same Metal runtime but bypasses the packed `Q4_K` MoonQuant path, which makes it the direct comparison switch for benchmark and regression work
- prefer `bench` over `run` for published numbers, because `bench --bench-runs N` separates cold startup from warm reused-runtime measurements
- compare CPU and Metal with the same prompt, token count, seed, and model when tracking regressions
