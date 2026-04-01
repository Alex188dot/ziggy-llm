# ziggy-llm

Mac-first, Zig-native GGUF inference for Apple Silicon.

`ziggy-llm` is a deliberately narrow local LLM runtime:

- Apple Silicon first
- Metal first
- GGUF only
- single binary
- CLI first
- tiny OpenAI-compatible server second

The project is not trying to be Zig vLLM or a broad Ollama replacement. The goal is to build a small, understandable, high-performance inference runner that feels native to a MacBook Pro and is easy to benchmark honestly.

## Status

This repository is in the initial scaffold stage.

Today, the codebase provides:

- a working Zig build
- a CLI skeleton for the core commands
- project docs, scope, and roadmap

The actual GGUF loading, Metal backend, and inference path are not implemented yet.

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

Current scaffold:

```bash
zig build run
zig build run -- inspect -m /path/to/model.gguf
zig build run -- serve -m /path/to/model.gguf --port 8080
```

At the moment, the commands are placeholders that print intent and build configuration. They exist to establish the command shape before the engine implementation lands.

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

Initial quantization target set:

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

- Zig 0.14+ or newer

Build:

```bash
zig build
```

Run the CLI scaffold:

```bash
zig build run
```

Run a stub subcommand:

```bash
zig build run -- inspect -m /path/to/model.gguf
```

Run tests:

```bash
zig build test
```

## Repository Docs

- [PROJECT_OUTLINE.md](/Users/alessioleodori/HelloWorld/zig_/PROJECT_OUTLINE.md)
- [ROADMAP.md](/Users/alessioleodori/HelloWorld/zig_/ROADMAP.md)

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
- build a CPU correctness path
- wire the Metal backend
- add a minimal OpenAI-compatible server
- publish honest M3 benchmark results

## License

TBD
