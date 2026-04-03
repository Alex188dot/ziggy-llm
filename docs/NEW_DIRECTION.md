# New Direction

## Summary

This project should take a moonshot direction rather than a safer incremental one.

The goal is not to become a broad local inference runtime first. The goal is to build a narrowly scoped Apple Silicon inference engine that has a real chance of producing meaningfully better decode throughput by designing the model format, execution engine, and decode strategy together.

The core thesis is to implement all three of these as one stack:

- `MoonQuant`
- speculative decoding
- fused Apple-specific kernels

These should not be treated as separate feature tracks. They reinforce each other:

- `MoonQuant` defines the weight layout and the quantization strategy that the engine is built around.
- fused Apple-specific kernels are the execution path for that layout on Metal.
- speculative decoding multiplies the benefit once the base decode path is already strong.

## Why This Direction

Trying to be good at many platforms, model families, and quantizations is the beaten path of many projects and is unlikely to produce any meaningful durable advantage for this one.

Trying to build the best Apple Silicon decode engine for a narrow GGUF slice is a meaningful bet. It gives the project a chance to win by specialization rather than by trying to be more general than existing systems.

This direction is intentionally harder, but it is coherent:

- one platform family
- one file format
- one decode-first performance target
- a small quant set
- a small model-family set

## Core Features

### 1. MoonQuant

`MoonQuant` is the project's proprietary Apple-first weight quantization and packing direction.

It should focus on:

- Apple-friendly memory layout
- bandwidth reduction on the decode hot path
- a small number of exact supported quant formats
- model-quality preservation through careful calibration and format design
- direct compatibility with fused Metal kernels instead of generic unpacking paths

The point is not to support every GGUF quant. The point is to make a few formats fast enough and clean enough that they become the preferred path for this runtime.

### 2. Speculative Decoding

Speculative decoding stays in scope, but it should be built on top of a strong base engine.

It matters because it is one of the few levers with plausible `30%+` upside on decode throughput when acceptance rates are good. It should be designed around:

- a narrow model set first
- decode-first benchmarks
- low-overhead verification in the target model path
- single-user local inference before any server-style batching work

Speculative decoding is not a substitute for a fast base runtime. It is a multiplier.

### 3. Fused Apple-Specific Kernels

The runtime should aggressively specialize for Apple Silicon and Metal instead of preserving generic execution paths in the hot decode loop.

This means:

- fused decode-first kernels
- quant-specific fast paths
- specialization for dominant shapes
- minimizing intermediate memory traffic
- minimizing launch overhead
- treating Apple Silicon as the first-class optimization target rather than as a backend port

## First Implementation Scope

The first version must stay narrow.

### Hardware

Performance tuning target:

- `M3` and newer Apple Silicon machines

Compatibility target:

- keep the architecture viable across the Apple Silicon family where practical, ideally `M1` through `M5`

The important distinction is that we should optimize for `M3+` first rather than dilute the work by trying to equalize behavior across every Apple generation.

### Model Architectures

First supported architecture:

- `llama`

Roadmap architectures after the first path is strong:

- `qwen`
- `gemma`

We should not start broad. The first win must come from a deeply optimized `llama` path.

### Quantization Scope

The first optimized quantization set should stay small and match the direction already present in the repo docs:

- `Q4_K_M`
- `Q5_K_M`
- `Q6_K`
- `Q8_0`

`F16` should remain available as a reference and validation path, but it is not the primary optimization target for this direction.

### Workload Scope

The first optimization target is:

- decode-first
- single-user
- local inference

The project should explicitly defer:

- broad prompt-processing optimization as a primary goal
- multi-user serving
- full batching systems
- broad quantization coverage
- broad model-family coverage

## Execution Order

The implementation order should be:

1. Make `MoonQuant` concrete for the first supported quant set and Apple-friendly layout strategy.
2. Build fused Metal kernels specifically for that quant and layout strategy.
3. Push the base decode path until it is clearly strong on the canonical Apple Silicon benchmark.
4. Add speculative decoding on top of that base path.

This ordering matters. A weak base runtime plus speculative decoding can produce interesting demos, but it does not create a durable engine advantage.

## Non-Goals For Now

These should stay out of the first moonshot phase:

- trying to support all GGUF variants
- trying to beat every backend on every metric
- building a broad chat-platform-style product surface first
- optimizing for generic portability over Apple-specific performance
- expanding into server orchestration before decode throughput is excellent

## Success Condition

This direction is successful if the project becomes a serious Apple Silicon decode engine with a differentiated stack:

- proprietary quantization direction
- Apple-specific fused execution
- speculative decode acceleration

That is a meaningful technical position. It is much more valuable than becoming a partially optimized general runtime with no clear edge.
