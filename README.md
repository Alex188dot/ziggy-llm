# ziggy-llm

A Mac-first, Zig-native GGUF inference engine with first-class Apple Metal support.

`ziggy-llm` is a deliberately narrow local LLM inference engine:

- Apple Silicon first
- Metal first
- GGUF only
- single binary
- CLI first
- tiny OpenAI-compatible server second

The project is not trying to be Zig vLLM or a broad Ollama replacement. The goal is to build a small, understandable, high-performance inference runner that feels native to a MacBook Pro and is easy to benchmark honestly.

## Status

This repository is in the first CPU correctness stage.

Today, the codebase provides:

- a working Zig build
- a CLI surface for the core commands
- a working `inspect` command for GGUF metadata and tensor-table validation
- a narrow native CPU reference runtime for `ziggy-tiny` GGUF fixtures
- a native CPU `llama` GGUF runtime in Zig
- deterministic `run` and `bench` execution with seed and timing output
- a module layout for CLI, commands, runtime, GGUF, and server code
- project docs, scope, and roadmap

Metal acceleration, interactive chat, and the HTTP server are not implemented yet.

## Repo Description

Suggested short repo description:

> A Mac-first, Zig-native GGUF inference engine with first-class Apple Metal support

Yes, "inference engine" is the right term here. It is accurate, concise, and strong enough for a public repository description without overclaiming.

## Why This Exists

Most local LLM tools optimize for breadth:

- many backends
- many model families
- many deployment modes
- many integrations

That breadth is useful, but it also creates complexity, larger binaries, more moving parts, and less room to optimize deeply for one machine class.

`ziggy-llm` goes the other direction. It optimizes for depth:

- one model container format
- one primary platform
- one primary acceleration path
- one primary use case

The thesis is simple: a narrow Mac-first inference runner can ship faster, feel cleaner, and make more credible performance claims than a general-purpose platform built too early.

## Target Machine

Primary development target:

- Apple Silicon MacBook Pro
- Apple M3
- 18 GB unified memory

That target shapes the project:

- 1B to 3B instruct models should feel great
- 7B quantized models should be a realistic stretch target
- startup time and TTFT matter more than feature sprawl
- Metal is the first-class runtime, not an afterthought

Linux support is intentionally left open for future CPU support, but macOS on Apple Silicon is the only first-class target in the early versions.

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

Explicit non-goals for v0:

- CUDA
- ROCm
- Vulkan
- distributed inference
- multimodal support
- support for every GGUF model family
- broad quantization coverage from day one
- fine-tuning or training

## Why Zig

Zig is a strong fit for this project because it makes the important tradeoffs visible:

- explicit allocators
- direct control over memory layout
- low-overhead C interop where needed
- straightforward single-binary distribution
- clear systems code without heavyweight abstraction layers

For local inference, hidden allocations and accidental complexity matter. Zig keeps both under pressure.

## Differentiation

There are related Zig projects, but this combination is still open:

- `cgbur/llama2.zig` proves there is interest in pure Zig inference, but it is closer to a focused educational runtime than a Mac-first GGUF + Metal product
- `jaco-bro/MLX.zig` is relevant to Apple Silicon, but it is built around MLX rather than a Zig-native GGUF runtime
- `zolotukhin/zinc` is the closest serious Zig inference engine in product ambition, but it is AMD GPU + Vulkan + Linux-first rather than Mac-first + Metal-first

The gap this repo is trying to own is:

> a small-surface, Zig-native, GGUF-first, Metal-first local inference runner for Apple Silicon

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
zig build run -- run -m /path/to/model.gguf -p "abc" --max-tokens 8 --seed 7
zig build run -- bench -m /path/to/model.gguf -p "abc" --max-tokens 8 --seed 7
zig build run -- serve -m /path/to/model.gguf --port 8080
```

Right now, `inspect`, `run`, and `bench` are native Zig code. `chat` and `serve` are still scaffold commands.

## GGUF Support

`ziggy-llm inspect` currently supports:

- GGUF `v2` and `v3`
- little-endian files only
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

## CPU Reference Path

The first implemented runtime path is intentionally narrow:

- architecture: `ziggy-tiny`
- model family: small GGUF reference fixtures used for CPU validation
- quantization: `F16` tensors only with `general.file_type=1`
- tokenizer source: `tokenizer.ggml.tokens`
- runtime surface: `run` and `bench`

This path exists to make prompt processing, decode behavior, seeded sampling, and timing instrumentation testable before the Metal backend lands.

There is also a pragmatic real-model path:

- architecture: `llama`
- tokenizer: native `llama` tokenizer path with GGUF vocab, scores, and byte fallback
- forward pass: native CPU-only incremental decode with RMSNorm, RoPE, GQA attention, SiLU-gated FFN, and KV cache
- currently implemented tensor types: `F32`, `F16`, `Q4_K`, and `Q6_K`
- intended use: real TinyLlama/LLaMA-family GGUF execution without `ollama` or `llama.cpp`

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

- `F16` for `ziggy-tiny`
- `F32`, `F16`, `Q4_K`, and `Q6_K` for the native `llama` CPU path

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

Run tests:

```bash
zig build test
```

## Repository Docs

- [PROJECT_OUTLINE.md](/Users/alessioleodori/HelloWorld/zig_/PROJECT_OUTLINE.md)
- [ROADMAP.md](/Users/alessioleodori/HelloWorld/zig_/ROADMAP.md)

## Error Handling Conventions

Early scaffold conventions:

- CLI parsing returns explicit errors for unknown commands, unknown flags, missing values, and invalid ports
- command handlers print scaffold status instead of pretending unimplemented features already exist
- unsupported functionality should fail clearly and specifically rather than silently degrade
- initialization-time failures and runtime hot-path behavior should stay conceptually separate as the engine grows

## Success Criteria

Technical success:

- a compact, understandable codebase
- stable inference on Apple Silicon
- low startup time
- competitive TTFT on small and medium GGUF models
- benchmark methodology that other people can reproduce

Product success:

- one-command first run
- clear support matrix
- a README that feels intentional rather than vague

Community success:

- people benchmark it against `llama.cpp` and `ollama`
- people share MacBook results
- the repo becomes a credible Zig systems project rather than just an experiment

## Near-Term Priorities

- finish the repo scaffold
- implement GGUF inspection and validation
- broaden the CPU correctness path beyond `ziggy-tiny`
- wire the Metal backend
- add a minimal OpenAI-compatible server
- publish honest M3 benchmark results

## License

Apache-2.0
