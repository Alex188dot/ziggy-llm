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

Recent local llama Metal work on the primary machine reached roughly `21.6` decode tok/s in the current path.

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
- compare CPU and Metal with the same prompt, token count, seed, and model when tracking regressions
