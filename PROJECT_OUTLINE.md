# ziggy-llm

## One-line pitch

`ziggy-llm` is a Mac-first, Zig-native GGUF inference runner with first-class Apple Metal support, a single-binary CLI, and an optional tiny OpenAI-compatible HTTP server.

The goal is not to build "Zig vLLM" or "another Ollama clone". The goal is to build the fastest, cleanest, most understandable local inference runner for Apple Silicon that feels small, sharp, and easy to benchmark.

## Product thesis

Most local LLM tools are optimized for breadth:

- many model families
- many backends
- many deployment modes
- many integrations

That makes them useful, but it also makes them large, hard to understand, and harder to optimize deeply for one machine class.

`ziggy-llm` should optimize for depth instead:

- one model container format: `GGUF`
- one primary platform at the start: Apple Silicon Macs
- one primary acceleration path: Metal
- one primary use case: fast local text generation with a tiny operational footprint

Linux support should stay architecturally possible, but it is not the first-class target in v0. macOS on Apple Silicon is the first-class target.

## Why this project

This is the best balance between:

- something technically ambitious enough to matter
- something narrow enough to actually ship
- something visible enough to attract attention

A tokenizer-only repo is easier to finish, but it is much less likely to become a breakout project.

A full serving system like vLLM is too large as a first move.

A Mac-first inference runner gives us:

- a clear story
- a demo people can run quickly
- performance claims users can verify
- a good fit for Zig's strengths in memory control, binary simplicity, and explicit systems design

## Target machine

Primary development and validation machine:

- MacBook Pro with Apple M3
- 18 GB unified memory

That constraint should shape the whole project.

`ziggy-llm` should be designed first for:

- 1B to 3B instruct models as the "it should feel great" tier
- 7B quantized models as the "stretch but realistic" tier
- short setup time
- low time-to-first-token
- stable memory behavior under repeated runs

It should not assume datacenter hardware, large VRAM, or multi-GPU setups.

However, _if_ the above sizes work well on the target machine, then it is reasonable to expect that the project should be able to support larger models well and efficiently on higher-end hardware (Ultra series, M4 Max, etc..).

## First-class scope

### Must have in v0

- GGUF-only model loading
- Apple Silicon only
- Metal backend as the main acceleration path
- single binary
- CLI inference
- optional minimal HTTP server
- OpenAI-compatible core endpoints for easy testing
- `mmap` model loading where appropriate
- explicit allocator strategy
- benchmark commands and reproducible benchmark docs
- a small set of supported quantizations

### Explicit non-goals for v0

- CUDA
- ROCm
- Vulkan
- distributed inference
- continuous batching across many users
- multimodal support
- embeddings/rerank/speech/tooling sprawl
- support for every GGUF model family
- broad quantization coverage from day one
- fine-tuning or training

## Positioning

The positioning is:

> A narrow, high-performance, Mac-first Zig inference engine for GGUF models.

Not:

> a universal local AI platform

## Core design principles

### 1. Mac first, Linux later

Everything in the architecture should leave room for future Linux CPU support and possibly another backend later, but the implementation order is:

1. Apple Silicon + Metal
2. Apple Silicon CPU fallback
3. Linux CPU support
4. Only later, consider Linux GPU backends

### 2. Small surface area

The public surface should stay small:

- one binary
- one CLI
- one tiny server mode
- one model format

### 3. Restricted model support

Support a narrow list of known-good model architectures first, likely one family to begin with.

Good initial strategy:

- start with one architecture that is common and easy to validate
- support a narrow list of exact GGUF variants
- expand only after the hot path is solid

### 4. Performance claims must be reproducible

Every claim in the README should come with:

- hardware name
- model name
- quantization
- prompt shape
- generation length
- exact command

### 5. No hidden runtime behavior

The project should favor:

- explicit allocators
- clear memory ownership
- predictable startup
- minimal background magic

## Proposed v0 feature set

### CLI

Example shape:

```bash
ziggy-llm run -m /path/to/model.gguf -p "Write a haiku about compilers"
ziggy-llm chat -m /path/to/model.gguf
ziggy-llm bench -m /path/to/model.gguf
ziggy-llm inspect -m /path/to/model.gguf
```

Initial CLI capabilities:

- run one prompt
- interactive chat mode
- configurable max tokens, temperature, top-p, seed
- choose GPU layers or full Metal path if needed by the backend design
- model inspection for GGUF metadata
- benchmark mode for startup, TTFT, and tok/s

### HTTP server

Keep this deliberately small.

Example shape:

```bash
ziggy-llm serve -m /path/to/model.gguf --port 8080
```

Initial API scope:

- `/health`
- `/v1/completions`
- `/v1/chat/completions`

Optional later:

- streaming responses
- model listing

The server should exist because it makes demos, SDK integrations, and benchmarks much easier. It should not become the center of the project.

## Technical outline

### Model format

Use GGUF only.

Reasons:

- it matches the local inference ecosystem
- it keeps model ingestion simple
- it lets the project target the most common quantized local models

### Backend strategy

Primary backend:

- Metal compute on Apple Silicon

Secondary backend:

- CPU fallback on Apple Silicon

Future backend:

- Linux CPU first

The backend abstraction should be narrow and built around the actual inference hot path, not around a giant generic tensor framework.

### Quantization strategy

Do not support everything.

Pick a small set of quantizations that are common and worth optimizing:

- `Q4_K_M`
- `Q5_K_M`
- `Q6_K`
- `Q8_0`
- `F16` as a useful reference path

This set is broad enough to be useful and narrow enough to keep implementation realistic.

### Memory strategy

Core ideas:

- `mmap` the model file when it improves startup and memory behavior
- keep decode-path allocations at zero or as close to zero as possible
- use explicit allocators for setup-time structures
- separate initialization allocations from steady-state inference allocations
- pre-size buffers for token generation and KV cache where possible

This is one of the places where Zig should materially improve the codebase quality.

### Metal strategy

Metal is not a "nice to have". It is the defining feature of the first version.

That means:

- Apple Silicon performance is the primary benchmark
- kernel choices should be guided by M3 behavior, not abstract portability
- profiling on macOS should be part of the normal development loop
- CPU fallback exists for correctness and portability, not as the flagship path

### Tokenization

Tokenization does not need to be the star of the repo, but it should be integrated carefully because TTFT depends on it.

The project should:

- load tokenizer information from the model or its expected companion assets
- avoid unnecessary heap churn during prompt tokenization
- measure prompt processing separately from decode speed

### Model support strategy

For the first version, it is better to say:

- "these exact model families and quantizations are supported"

than:

- "many GGUF models may work"

Credibility is higher when the support matrix is narrow and tested.

## Suggested milestones

### Milestone 0: repo and benchmark skeleton

- project scaffold
- build system
- CLI shape
- benchmark harness
- README skeleton
- decision records for scope and non-goals

### Milestone 1: GGUF loading + inspect

- parse GGUF metadata
- print tensor info and tokenizer info
- validate supported architectures
- fail clearly on unsupported models

### Milestone 2: CPU correctness path

- minimal forward pass on CPU
- deterministic sampling
- enough correctness to compare outputs
- small model smoke tests

This is not the flagship path, but it gives us a reference implementation for validation.

### Milestone 3: Metal decode path

- Metal kernels for core inference ops
- working prompt + decode loop on Apple Silicon
- benchmarkable token generation
- correctness checks against CPU path

### Milestone 4: minimal server

- `/health`
- `/v1/completions`
- `/v1/chat/completions`
- simple JSON request/response handling

### Milestone 5: launch-quality repo

- polished README
- benchmark tables
- startup latency measurements
- comparison page vs established local runtimes
- demo recordings on the M3 MacBook Pro

## What success looks like

### Technical success

- a small, understandable codebase
- stable inference on Apple Silicon
- low startup time
- competitive TTFT on small and medium GGUF models
- clean benchmark methodology

### Product success

- users can run it in one command
- users immediately understand what it is for
- the README makes the narrow scope feel intentional, not incomplete

### Community success

- people share benchmarks
- people compare it directly against other serious local runtimes
- Zig users point to it as a serious systems project

## Repo strategy for attention

If the goal is for the repository to attract serious attention, the launch matters almost as much as the code.

The repo should eventually have:

- a very sharp README headline
- benchmark tables that are easy to trust
- a short "why this exists" section
- startup latency videos on an actual MacBook Pro M3
- explicit claims about supported models
- explicit claims about unsupported features

The project should feel opinionated, not unfinished.

## Future evolution

After the Mac-first version is solid, reasonable next steps are:

- Linux CPU support
- better streaming support
- broader model-family coverage
- a more complete benchmark suite
- limited batching

Anything beyond that should be earned by real usage, not promised upfront.

## Final project statement

`ziggy-llm` should begin as a deliberately narrow project:

- Apple Silicon first
- Metal first
- GGUF only
- single binary
- CLI first, tiny server second
- performance and clarity over breadth

If it works well on your M3 MacBook Pro and the benchmarks are honest, that is enough to build a credible first release and a repo people will pay attention to.

## Side note

Ziggy is the name of my dog 🐶 :) so I guess ziggy-llm is a good name for a project about AI and Zig.
