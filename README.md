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

The goal is to build a small, understandable, high-performance inference runner that feels native to a MacBook Pro and is easy to benchmark honestly.

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

## Why Zig

Zig is a strong fit for this project because it makes the important tradeoffs visible:

- explicit allocators
- direct control over memory layout
- low-overhead C interop where needed
- straightforward single-binary distribution
- clear systems code without heavyweight abstraction layers

For local inference, hidden allocations and accidental complexity matter. Zig keeps both under pressure.

## Performance Comparison

The following table compares end-to-end decode throughput on Apple Silicon (MacBook Pro M3 18GB) across ziggy-llm and llama.cpp using identical prompts and generation parameters. ZINC (tested on M1 Max 32 GB, according to their docs) is also included for reference, although the prompt used is unknown.

| Model              | GGUF   | ziggy-llm (Metal) | ZINC (Metal) | llama.cpp (Metal) |
| ------------------ | ------ | ----------------- | ------------ | ----------------- |
| **TinyLlama 1.1B** | Q4_K_M | ~120 tok/s        | —            | 151.4 tok/s       |
| **Llama 3.2 3B**   | Q4_K_M | ~40 tok/s         | —            | 53.5 tok/s        |
| **Llama 3.1 8B**   | Q4_K_M | ~18 tok/s         | ~10 tok/s    | 23.1 tok/s        |
| **Qwen3 1.7B**     | Q4_K_M | —                 | —            | 92.0 tok/s        |
| **Qwen3 8B**       | Q4_K_M | ~17.5 tok/s       | ~8 tok/s     | 25.0 tok/s        |

Note: ZINC's supported models are limited to the models listed in their documentation and the hardware they tested on (M1 Max 32 GB).

## Planned CLI

Target command surface:

```bash
ziggy-llm run -m /path/to/model.gguf -p "Write a haiku about compilers"
ziggy-llm chat -m /path/to/model.gguf
ziggy-llm inspect -m /path/to/model.gguf
ziggy-llm bench -m /path/to/model.gguf
ziggy-llm serve -m /path/to/model.gguf --port 8080
```

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

- GGUF `v2` and `v3`
- `general.type=model` artifacts
- required `general.architecture` metadata
- standard inspection fields from `general.*` and `tokenizer.ggml.*`
- tensor-table validation for name, dimension count, dimension sizes, tensor type, alignment, and data extents

The current inspect output reports:

- architecture
- tensor count
- metadata count
- file alignment
- GGUF file-type quantization when present
- dominant tensor type across the tensor table
- tokenizer model and pre-tokenizer metadata when present
- tokenizer token count and common special-token ids when present

Unsupported or rejected today:

- GGUF versions other than `v2` and `v3`
- big-endian GGUF files
- non-model artifacts such as adapters or auxiliary blobs
- malformed tensor metadata
- truncated metadata or tensor payloads

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

## Quick Start

Prerequisite:

- Zig 0.15.2 or newer

Build:

```bash
zig build
```

Run the CLI scaffold:

```bash
zig build run
```

Run an implemented CPU reference command:

```bash
zig build run -- run -m /path/to/model.gguf -p "abc" --max-tokens 8 --seed 7
```

Run the implemented Metal path on Apple Silicon:

```bash
./zig-out/bin/ziggy-llm run -m /path/to/llama-model.gguf -p "Hello" --max-tokens 32 --seed 7 --backend metal
```

Benchmark CPU vs Metal on the same model:

```bash
./zig-out/bin/ziggy-llm bench -m /path/to/llama-model.gguf -p "Hello" --max-tokens 256 --seed 7 --backend cpu
./zig-out/bin/ziggy-llm bench -m /path/to/llama-model.gguf -p "Hello" --max-tokens 256 --seed 7 --backend metal
```

With `--bench-runs N`, the first run is cold and the remaining runs use the resident runtime path. Warm output now reports `warm.reused_prompt_tokens_avg` so prompt-prefix reuse is visible instead of being inferred from TTFT alone.

Run tests:

```bash
zig build test
```

## Want to participate? Here is a TODO list of things that need immediate attention:

- [ ] Implement OpenAI compatible server
- [ ] Add support for Gemma and Mistral model families
- [ ] Make chat more robust
- [ ] Test all quants (currently tested only Q4_K_M)
- [ ] Test bigger models (of Qwen 3 and Llama families) with higher end hardware, bigger context sizes and benchmark performance

## License

Apache-2.0
