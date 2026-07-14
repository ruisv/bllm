# Changelog

All notable changes to BLLM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Reusable-prefix path** — `NativeSession.set_prefix()` / `ask()`: prefill a shared prefix
  (a document, a fixed system + tools context, few-shot exemplars) **once**, then answer many
  queries against it without re-prefilling — each `ask()` restores the prefix and is independent.
  ~3–4× per query on repeated asks over a shared context. Backed by in-memory `snapshot()` /
  `restore()` on the engines (the file-based `save_state`/`load_state` share the same core).
- **Thinking toggle** — `NativeSession.set_thinking(False)` prefills an empty `<think></think>`
  (the official template's `enable_thinking=False`) so Qwen3.5 reasoning models answer directly
  — faster, fewer tokens. Default reasons and shows the block. (The in-message `/no_think` soft
  switch is unreliable on the current checkpoints; this prefill is.)
- **`encode()` / `decode()`** on the session (the model's C++ tokenizer).
- **Qwen3.5-2B / 4B** (hybrid Gated-DeltaNet/SSM) validated on-board, 100% on-BPU int8
  (~14.5 / 6.9 tok/s), joining the 0.8B.

## [0.1.1]

### Added

- **Phi chat template** — a third chat format (`phi`, alongside `chatml` and raw `none`)
  for models that mark roles with `<|user|>` / `<|assistant|>` / `<|system|>` / `<|end|>`
  tokens instead of ChatML. `make_model_dir.py` auto-detects the markers and emits it, so
  **Phi-4-mini** now ships as a first-class multi-turn chat package (previously it only ran
  as raw completion). Existing ChatML models are unaffected.

First public release — an on-board LLM / VLM runtime for D-Robotics RDK S100 / S100P / S600,
driving compiled `.hbm` graphs natively on the BPU (hbDNN / hbUCP), no extra LLM SDK.

### Added

- **Native text runtime** — dense models (Qwen2.5-1.5B/7B, DeepSeek-R1-Distill, InternLM2,
  GLM-Edge, Phi-4-mini) and the **Qwen3.5-0.8B hybrid Gated-DeltaNet/SSM** (100% on-BPU,
  ~21 tok/s), with a built-in C++ tokenizer, ChatML templating, streaming and multi-turn.
- **Multimodal** — `NativeVlm` / `NativeVlmSession` for Qwen2.5-Omni: text + image + audio +
  video, from file paths or raw arrays, with live streaming capture.
- **`bllm.load()`** — one entry point that loads a `model.json` package **or** a bare `.hbm`
  + tokenizer dir (manifest synthesized on the fly).
- **Sampling** — greedy / temperature / top-k / top-p / min-p / typical-p, plus repeat,
  frequency and presence penalties; reproducible with a seed.
- **Production features** — perplexity, async submit, prompt-cache save/restore (bit-identical
  resume), stop strings (stops early), and BPU priority / time-slice control to share the BPU
  with a vision pipeline.
- **Packaging** — conda packages (`bllm`, `libbllm`, `tokenizers-cpp`) for linux-aarch64;
  `find_package(bllm)` for C++ consumers.

[Unreleased]: https://github.com/ruisv/bllm/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ruisv/bllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ruisv/bllm/releases/tag/v0.1.0
