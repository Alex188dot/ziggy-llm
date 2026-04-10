# ⚡️ ziggy-llm

![ziggy-llm logo](assets/ziggy-llm-logo.png)

Welcome to **ziggy-llm** — a Mac-first, Zig-native GGUF inference engine built specifically for Apple Silicon. If you're looking for an impossibly fast, understandable and deliberately narrow local AI runner, you're in the right place. 🚀

We don't try to do everything for everyone; instead, we do one thing exceptionally well. By focusing exclusively on Apple Metal and GGUF files, we deliver a single-binary, highly optimized CLI experience 💎

- Apple Silicon first
- Metal first
- GGUF only
- Single binary
- CLI first
- Tiny OpenAI-compatible server (coming soon!)

ziggy-llm is currently the fastest Zig GGUF inference engine on Apple Silicon.

## Meet MoonQuant 🌕

To squeeze every last drop of performance out of Apple hardware, we built **MoonQuant**, our proprietary quantization and layout engine. It takes standard weights and repacks them into a highly optimized, Metal-friendly memory layout designed for maximum memory bandwidth reduction during the decode phase. 🔋

As a user, your workflow doesn't change: your input is always a standard `.gguf` model file. Under the hood, ziggy-llm automatically compiles your model into our custom `.ziggy` execution format on its first run, ensuring lightning-fast loading and blazing execution speeds on all subsequent runs. ✨

## Blazing Fast Performance 🏎️

We benchmark honestly and optimize ruthlessly for single-user, local text generation on Macs. The following table compares end-to-end decode throughput on Apple Silicon (MacBook Pro M3 18GB) across ziggy-llm and llama.cpp using identical prompts and generation parameters. ZINC, another Zig GGUF inference engine (tested on M1 Max 32 GB, according to their docs) is also included for reference, although the prompt used is unknown. 📊

| Model              | GGUF   | ziggy-llm (Metal) | ZINC (Metal) | llama.cpp (Metal) |
| ------------------ | ------ | ----------------- | ------------ | ----------------- |
| **TinyLlama 1.1B** | Q4_K_M | ~133 tok/s        | —            | 151.4 tok/s       |
| **Llama 3.2 3B**   | Q4_K_M | ~40 tok/s         | —            | 53.5 tok/s        |
| **Llama 3.1 8B**   | Q4_K_M | ~18 tok/s         | ~10 tok/s    | 23.1 tok/s        |
| **Qwen3 1.7B**     | Q4_K_M | ~65 tok/s         | —            | 92.0 tok/s        |
| **Qwen3 8B**       | Q4_K_M | ~17.5 tok/s       | ~8 tok/s     | 25.0 tok/s        |

Note: ZINC's supported models are limited to the models listed in their documentation and the hardware they tested on (M1 Max 32 GB).

## Quick Start 🏁

Getting up and running takes just a few seconds. Ensure you have Zig 0.15.2 or newer installed, clone the repository, and build the release binary. 🛠️

```bash
git clone https://github.com/Alex188dot/ziggy-llm.git
cd ziggy-llm
zig build -Doptimize=ReleaseFast
```

Once built, you can immediately dive into a chat session or run a single prompt. Just point the CLI to your standard `.gguf` file and let ziggy-llm do the rest! 💬

```bash
# Start an interactive chat
./zig-out/bin/ziggy-llm chat \
  --model path/to/model.gguf \
  --backend metal \
  --temperature 0 \
  --seed 42

# Run a single prompt
./zig-out/bin/ziggy-llm run \
  --model path/to/model.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal \
  --max-tokens 128 \
  --temperature 0 \
  --seed 42
```

We also include built-in tools for benchmarking your models locally. Use the `--bench-runs` flag to automatically separate cold startup times from warm, resident-runtime measurements. ⏱️

```bash
./zig-out/bin/ziggy-llm bench \
  --model path/to/model.gguf \
  --prompt "Write one short paragraph about Zig." \
  --backend metal \
  --max-tokens 128 \
  --temperature 0 \
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

## Supported Models & Quants 🧠

Our goal is to support a deliberately narrow, highly-optimized matrix of popular models. Currently, we focus on the Qwen (2 and 3) and LLaMA (3.1, 3.2, TinyLlama) architectures. 🎯

For quantizations, we recommend our specialized MoonQuant targets: `Q4_K_M`, `Q6_K`, and `Q8_0`. We also fully support `F16` and `F32` formats as reference paths. 📉

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

## Planned HTTP API 🔌

The server should stay small.

Initial target endpoints:

- `/health`
- `/v1/completions`
- `/v1/chat/completions`

The API exists to make testing and integration easy. It should not drag the project into becoming a giant orchestration platform.

## Architecture Direction 🏗️

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

## Join the Community 🤝

ziggy-llm is open source and in active development, and we would love your help to make it even better. Check out our issue tracker for things that need immediate attention: 🏗️

- [ ] Implement OpenAI compatible server
- [ ] Add support for Qwen 3.5 (MoE and DeltaNet variants), Gemma and Mistral model families
- [ ] Make chat more robust
- [ ] Test all quants (currently tested only Q4_K_M)
- [ ] Test bigger models (of Qwen 3 and Llama families) with higher end hardware, bigger context sizes and benchmark performance

If you find this project interesting, please consider starring the repo ⭐️. It genuinely helps us grow and reach more developers in the local AI ecosystem!

## License 📜

This project is licensed under the Apache-2.0 License. Feel free to use it, modify it, and build awesome native local AI tools with it. ⚖️
