# Apple Silicon Runtime Notes

This document records the first measured `ziggy-tiny` Metal results on the primary target machine and the current optimization notes for the macOS path.

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

Create the looping `ziggy-tiny` benchmark fixture used for the Metal path:

```bash
zig build tiny-fixture -- --loop /tmp/ziggy-tiny-loop.gguf
```

Benchmark the CPU and Metal backends:

```bash
./zig-out/bin/ziggy-llm bench -m /tmp/ziggy-tiny-loop.gguf -p a --max-tokens 256 --seed 7 --backend cpu
./zig-out/bin/ziggy-llm bench -m /tmp/ziggy-tiny-loop.gguf -p a --max-tokens 256 --seed 7 --backend metal
```

Average of 5 runs on this machine:

| Backend | Avg TTFT | Avg decode tok/s |
| --- | ---: | ---: |
| CPU | 0.293 ms | 591139.158 |
| Metal | 61.382 ms | 1135.149 |

Interpretation:

- the Metal path is functionally correct on Apple Silicon
- the current `ziggy-tiny` Metal implementation is not yet performance-competitive with the CPU reference path
- TTFT is dominated by Metal device and pipeline setup plus per-dispatch synchronization overhead

## macOS Profiling Notes

Profile command used on macOS:

```bash
./zig-out/bin/ziggy-llm bench -m /tmp/ziggy-tiny-loop.gguf -p a --max-tokens 300 --seed 7 --backend metal
```

Host-side profile captured with `sample` because `xcrun xctrace` was not installed in the active command-line tools environment:

```bash
zsh -lc './zig-out/bin/ziggy-llm bench -m /tmp/ziggy-tiny-loop.gguf -p a --max-tokens 300 --seed 7 --backend metal >/tmp/ziggy-metal-debug-bench.out & pid=$!; sleep 0.3; sample $pid 1 1 -mayDie; wait $pid'
```

Key findings from `/tmp/ziggy-llm_2026-04-02_085337_W1V3.sample.txt`:

- the hottest host stack repeatedly lands in [`bridge.m`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal/bridge.m) `ziggy_metal_run_matvec_f32`, especially the `waitUntilCompleted` call at line 288
- command-buffer creation and compute-encoder creation also appear repeatedly in the sampled stacks
- the Zig-side call path is [`tiny_runtime.zig`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/tiny_runtime.zig) `Session.step` -> [`metal_backend.zig`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal_backend.zig) `State.matVec` -> [`bridge.m`](/Users/alessioleodori/HelloWorld/zig_/src/runtime/metal/bridge.m) `ziggy_metal_run_matvec_f32`
- `writeBuffer` and `readBuffer` show up in the sampled call tree, confirming the current path pays CPU-GPU transfer overhead for every matvec invocation

Optimization notes recorded from the current profile:

- stop synchronizing each matvec independently; batch multiple ops into a single command buffer per token
- stop round-tripping activations through shared CPU memory after every kernel; keep intermediate activations resident on the GPU
- fuse the current small-matrix sequence where practical instead of issuing many tiny command buffers from the host
- keep the one-time Metal context creation cost out of steady-state benchmarking when measuring decode throughput
- add a richer GPU profiler pass with Instruments once `xctrace` or the full Xcode toolchain is available

## Running Inference On GPU

Today, GPU inference is available only for the `ziggy-tiny` Metal path.

Example:

```bash
zig build
zig build tiny-fixture -- --loop /tmp/ziggy-tiny-loop.gguf
./zig-out/bin/ziggy-llm run -m /tmp/ziggy-tiny-loop.gguf -p a --max-tokens 32 --seed 7 --backend metal
```

Notes:

- `--backend auto` will try Metal first on Apple Silicon and fall back to CPU if Metal initialization fails
- real `llama` GGUF models still run on the native CPU backend only
- for example, `./zig-out/bin/ziggy-llm run -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello" --backend auto` currently resolves to CPU, not Metal
