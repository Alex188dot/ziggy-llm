# Roadmap

This roadmap expands the scope defined in [PROJECT_OUTLINE.md](/Users/alessioleodori/HelloWorld/zig_/PROJECT_OUTLINE.md) into concrete milestones for the first public versions of `ziggy-llm`.

## How To Use This File

Every task in this file is a Markdown checkbox.

When a task is completed:

- replace `- [ ]` with `- [x]`
- keep the item in place instead of deleting it
- only mark it complete when the code, docs, and validation for that item are actually done

If a task turns out to be too large, split it into smaller checklist items rather than leaving a vague partially done item behind.

## v0.1.0 Foundations

- [ ] Finalize project name, binary name, and repo description
- [ ] Decide initial license and add `LICENSE`
- [ ] Create initial `build.zig.zon` if external dependencies become necessary
- [ ] Create `src/` module layout for CLI, GGUF parsing, backend, runtime, and server
- [ ] Keep the executable buildable with `zig build`
- [ ] Keep the test suite runnable with `zig build test`
- [ ] Add a `version` subcommand or `--version` output
- [ ] Add a `help` output that documents the intended command surface
- [ ] Define initial error-handling conventions for CLI and library code
- [ ] Document repository scope and non-goals in `README.md`
- [ ] Document milestone status and work sequencing in this roadmap

Definition of done for v0.1.0:

- [ ] A new contributor can clone the repo, run `zig build`, run the CLI, and understand what the project is trying to become

## v0.2.0 GGUF Inspection

- [ ] Define the exact GGUF versions and metadata fields the project will support first
- [ ] Implement GGUF header parsing
- [ ] Implement tensor metadata parsing
- [ ] Implement key/value metadata parsing
- [ ] Implement clear unsupported-model detection
- [ ] Add `inspect` output for architecture, tensor count, quantization, and tokenizer metadata
- [ ] Add validation for missing or malformed tensor metadata
- [ ] Add tests for valid GGUF fixtures
- [ ] Add tests for invalid and truncated GGUF fixtures
- [ ] Document supported and unsupported GGUF expectations in `README.md`

Definition of done for v0.2.0:

- [ ] `ziggy-llm inspect -m model.gguf` produces useful output on supported fixtures and fails clearly on unsupported ones

## v0.3.0 CPU Correctness Path

- [ ] Choose the first supported model architecture for end-to-end implementation
- [ ] Define the exact supported model family and quantization matrix for the first runtime path
- [ ] Implement tensor loading for the first supported architecture
- [ ] Implement tokenizer loading strategy for the first supported architecture
- [ ] Implement prompt tokenization on the CPU path
- [ ] Implement KV cache layout for the CPU reference path
- [ ] Implement a minimal forward pass for prompt processing
- [ ] Implement a minimal decode loop for one-token-at-a-time generation
- [ ] Implement deterministic sampling with seed control
- [ ] Add basic generation smoke tests
- [ ] Add golden-output or reference-behavior tests where realistic
- [ ] Separate initialization allocations from steady-state inference allocations
- [ ] Add instrumentation for startup time, prompt processing time, and tok/s

Definition of done for v0.3.0:

- [ ] A small supported model can run end to end on CPU with deterministic, testable behavior

## v0.4.0 Apple Silicon Runtime

- [ ] Define the first Metal backend boundary and the CPU fallback boundary
- [ ] Choose the first inference kernels to accelerate on Metal
- [ ] Implement Metal device discovery and initialization
- [ ] Implement shader compilation or shader asset loading strategy
- [ ] Implement Metal buffer management for model tensors and runtime buffers
- [ ] Implement at least one working Metal compute path for the first supported model
- [ ] Validate correctness against the CPU reference path
- [ ] Measure TTFT and tok/s on the M3 MacBook Pro target machine
- [ ] Profile hot kernels on macOS and record optimization notes
- [ ] Document Metal-specific assumptions and limitations
- [ ] Add graceful fallback behavior when Metal is unavailable or disabled

Definition of done for v0.4.0:

- [ ] The first supported model runs through the Metal path on Apple Silicon and matches CPU behavior closely enough for practical use

## v0.5.0 CLI Usability

- [ ] Finalize subcommands: `run`, `chat`, `inspect`, `bench`, `serve`
- [ ] Finalize flag naming for model path, prompt, max tokens, temperature, top-p, seed, and port
- [ ] Add interactive chat mode
- [ ] Add structured benchmark output
- [ ] Add machine-readable benchmark output format for later comparison pages
- [ ] Improve error messages for missing model paths and unsupported flags
- [ ] Add examples to `README.md`
- [ ] Add shell-usable exit codes for expected failure cases

Definition of done for v0.5.0:

- [ ] The CLI feels coherent and can support normal local evaluation workflows without reading the source code

## v0.6.0 Tiny HTTP Server

- [ ] Implement server startup and shutdown flow
- [ ] Add `/health`
- [ ] Add `/v1/completions`
- [ ] Add `/v1/chat/completions`
- [ ] Decide whether streaming is in or out for the first server release
- [ ] Implement JSON request parsing and response serialization
- [ ] Add request validation and clear error responses
- [ ] Add basic integration tests for the server endpoints
- [ ] Confirm the server can be used from standard OpenAI-compatible clients
- [ ] Document what “OpenAI-compatible” means and what it does not mean

Definition of done for v0.6.0:

- [ ] A local client can hit the tiny server and receive correct responses for the supported request shapes

## v0.7.0 Performance Discipline

- [ ] Define canonical benchmark prompts and generation lengths
- [ ] Define exact benchmark reporting format
- [ ] Add repeatable benchmark commands
- [ ] Measure startup latency on the M3 MacBook Pro
- [ ] Measure TTFT on the M3 MacBook Pro
- [ ] Measure decode tok/s on the M3 MacBook Pro
- [ ] Measure memory footprint on the M3 MacBook Pro
- [ ] Compare results against `llama.cpp`
- [ ] Compare results against `ollama` where the comparison is fair
- [ ] Publish benchmark caveats and methodology clearly

Definition of done for v0.7.0:

- [ ] Every public performance claim in the repo can be reproduced from documented commands

## v0.8.0 Launch Readiness

- [ ] Tighten README headline and positioning
- [ ] Add a clear supported-model matrix
- [ ] Add a clear unsupported-features section
- [ ] Add a quick-start path that works in a few minutes
- [ ] Add benchmark tables with exact hardware and command details
- [ ] Record startup-latency or demo videos on the M3 MacBook Pro
- [ ] Add contribution guidelines
- [ ] Add issue templates if the repo starts receiving external traffic
- [ ] Review naming and messaging so the project does not overclaim
- [ ] Tag the first public release

Definition of done for v0.8.0:

- [ ] The repository is clear, honest, demoable, and ready for public attention

## Deferred Until After The Mac-First Core Is Solid

- [ ] Linux CPU support
- [ ] Broader model-family coverage
- [ ] More quantization formats
- [ ] Stronger streaming support
- [ ] Batching or multi-request scheduling
- [ ] Broader API coverage
- [ ] More aggressive portability work

These are intentionally deferred. They should only move into active milestones after the Mac-first GGUF + Metal path is stable and benchmarked.
