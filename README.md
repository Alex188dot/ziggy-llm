# ziggy-llm

![ziggy-llm logo](assets/ziggy-llm-logo.png)

A Mac-first, Zig-native GGUF inference engine with first-class Apple Metal support.

`ziggy-llm` is a deliberately narrow local LLM inference engine:

- Apple Silicon first
- Metal first
- GGUF only
- single binary
- CLI first
- tiny OpenAI-compatible server second

The goal is to build a small, understandable, high-performance inference engine that feels native to Apple hardware and is easy to benchmark honestly.

ziggy-llm is currently the fastest Zig GGUF inference engine on Apple Silicon.

## Performance Comparison

The following table compares end-to-end decode throughput on Apple Silicon (MacBook Pro M3 18GB) across ziggy-llm and llama.cpp using identical prompts and generation parameters. ZINC, another Zig GGUF inference engine (tested on M1 Max 32 GB, according to their docs) is also included for reference, although the prompt used is unknown.

| Model              | GGUF   | ziggy-llm (Metal) | ZINC (Metal) | llama.cpp (Metal) |
| ------------------ | ------ | ----------------- | ------------ | ----------------- |
| **TinyLlama 1.1B** | Q4_K_M | ~123 tok/s        | —            | 151.4 tok/s       |
| **Llama 3.2 3B**   | Q4_K_M | ~40 tok/s         | —            | 53.5 tok/s        |
| **Llama 3.1 8B**   | Q4_K_M | ~18 tok/s         | ~10 tok/s    | 23.1 tok/s        |
| **Qwen3 1.7B**     | Q4_K_M | ~65 tok/s         | —            | 92.0 tok/s        |
| **Qwen3 8B**       | Q4_K_M | ~17.5 tok/s       | ~8 tok/s     | 25.0 tok/s        |

Note: ZINC's supported models are limited to the models listed in their documentation and the hardware they tested on (M1 Max 32 GB).

## Quick Start

Prerequisite:

- Zig 0.15.2 or newer

Clone:

```bash
git clone https://github.com/Alex188dot/ziggy-llm.git
cd ziggy-llm
```

Build:

```bash
zig build -Doptimize=ReleaseFast
```

Chat:

```bash
./zig-out/bin/ziggy-llm chat \
  --model path/to/model.gguf \
  --backend metal \
  --temperature 0 \
  --seed 42
```

Run 1 prompt:

```bash
./zig-out/bin/ziggy-llm run \
  --model path/to/model.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal \
  --max-tokens 128 \
  --temperature 0.7 \
  --seed 42
```

Benchmark:

```bash
./zig-out/bin/ziggy-llm bench \
  --model path/to/model.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal \
  --max-tokens 128 \
  --temperature 0.7 \
  --seed 42 \
  --bench-runs 5
```

With `--bench-runs N`, the first run is cold and the remaining runs use the resident runtime path.

Run tests:

```bash
zig build test
```

Update:

```bash
./zig-out/bin/ziggy-llm update
```

## Product Shape

Planned first version:

- GGUF-only model loading
- Metal backend for Apple Silicon
- Apple Silicon CPU fallback for validation and correctness
- single-binary CLI
- optional tiny HTTP server
- narrow supported model matrix
- narrow supported quantization matrix
- benchmark-friendly workflow

## Current models and quantizations support

Qwen (2 and 3) and LLama model families, in particular these have been tested:

- Qwen 3
- Llama 3.1
- Llama 3.2
- TinyLlama

Quantization support: Q4_K_M, Q6_K, Q8_0, F16, F32.

## CLI

Current commands:

```bash
zig build run
zig build run -- inspect -m /path/to/model.gguf
zig build run -- run -m /path/to/model.gguf -p "What is the meaning of life?" --max-tokens 8 --seed 7 --backend auto
zig build run -- run -m /path/to/model.gguf -p "What is the meaning of life?" --max-tokens 8 --seed 7 --backend metal
zig build run -- bench -m /path/to/model.gguf -p "What is the meaning of life?" --max-tokens 8 --seed 7 --backend metal
```

Right now, `inspect`, `run`, and `bench` are native Zig code. `chat` and `serve` are still scaffold commands.

`--backend auto` is the default. On Apple Silicon builds with Metal enabled, the `llama` path will use Metal when it can initialize and fall back to CPU otherwise.

## GGUF Support

`ziggy-llm inspect` currently supports:

- architecture
- tensor count
- metadata count
- file alignment
- GGUF file-type quantization when present
- dominant tensor type across the tensor table
- tokenizer model and pre-tokenizer metadata when present
- tokenizer token count and common special-token ids when present

## Planned HTTP API

The server should stay small.

Initial target endpoints:

- `/health`
- `/v1/completions`
- `/v1/chat/completions`

The API exists to make testing and integration easy. It should not drag the project into becoming a giant orchestration platform.

## Architecture Direction

Core technical choices:

- GGUF-only ingestion
- `mmap` model loading where it improves startup and memory behavior
- explicit allocator strategy
- decode path with zero allocations where possible
- a narrow backend abstraction centered on actual inference hot paths
- reproducible benchmark workflows

Current implemented quantization support:

- `F32`, `F16`, `Q4_K`, and `Q6_K` for the native `llama` CPU and Metal paths

Planned broader quantization target set after the reference path stabilizes:

- `Q4_K_M`
- `Q5_K_M`
- `Q6_K`
- `Q8_0`
- `F16`

Initial implementation order:

1. GGUF inspect and validation
2. Apple Silicon CPU correctness path
3. Metal decode path
4. tiny HTTP server
5. launch-quality benchmarks and demos

Please note ziggy-llm is still in active development, things may change, could break or be unstable.

## Want to participate? Here is a TODO list of things that need immediate attention:

- [ ] Implement OpenAI compatible server
- [ ] Add support for Qwen 3.5 (MoE and DeltaNet variants), Gemma and Mistral model families
- [ ] Make chat more robust
- [ ] Test all quants (currently tested only Q4_K_M)
- [ ] Test bigger models (of Qwen 3 and Llama families) with higher end hardware, bigger context sizes and benchmark performance

## Support

Found the repo interesting? Star it ⭐️, it will help us grow! 🌱

## License

Apache-2.0
