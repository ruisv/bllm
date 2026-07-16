# Changelog

All notable changes to BLLM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.4]

### Added

- **`bllm-make-model-dir` ships with the package** — the model-dir tool moved into
  `bllm.make_model_dir` and installs to `$PREFIX/bin`, so `conda install bllm` is enough to
  assemble a model directory. It was previously only `scripts/make_model_dir.py` in a source
  tree, which left an installed-from-conda user with no way to turn an official release into
  something `bllm.load()` accepts. `scripts/make_model_dir.py` still works (thin wrapper), and
  `python -m bllm.make_model_dir` is equivalent.
- **[`docs/MODELS.zh.md`](docs/MODELS.zh.md) / [`docs/MODELS.en.md`](docs/MODELS.en.md)** —
  deploying an official release end-to-end: ION sizing, what the package actually contains,
  assembling dense and Omni model directories, the Omni video budget, and the failure modes
  worth knowing. Verified command-by-command on an S100P.

### Fixed

- **The ION carveout prerequisite was missing from the public docs entirely.** On-board LLMs
  need an enlarged carveout (`hb_switch_ion.sh balanced` + reboot) or the runtime segfaults on
  alloc — a first-run failure with no hint that it is a setup step, not a bug. Now in the
  README's install section, where it is a prerequisite rather than a footnote.
- **Documented the official Omni path end-to-end.** Getting from a downloaded Qwen2.5-Omni
  release to a loadable model directory was undocumented: the README's multimodal section began
  at an already-assembled directory, and the only `omni` example lived in a script docstring.
  The README (zh/en) and the new deployment docs now carry the verified command, and "making a
  model directory" is a top-level section rather than a subsection of "build from source" — a
  conda user never reached it, and it is not a build-from-source concern.
- **`embed_tokens.bin` is called out as required for Omni.** It is a separate 1.2 GB download
  in the official list, sits in `model/`, and its name does not mention Omni — so the three
  `*_Omni_3B_*.hbm` files look like a complete set. Missing it now produces an error that says
  where the file comes from, rather than just naming the flag.
- **The Omni video budget is stated instead of implied.** The README said a model directory
  "just points `visual` at the tower you want", but the official release ships only the 448px
  tower: 256 tokens per frame-pair against a 2048-slot window is ~8 s of video at 2 fps. Other
  resolutions require an offline re-export, which is outside this repo.
- **Dropped the obsolete "downgrade `tokenizer.json` to 0.19.1" advice** — that was a libxlm
  limitation. The native runtime reads `tokenizers` 0.20+ pair-format merges (with
  `ignore_merges`) as-is; verified on-board.
- **`__version__` was stuck at 0.1.2** — so the published 0.1.3 package reported the wrong
  version from `bllm.__version__`. It now tracks `CMakeLists.txt`.

## [0.1.3]

### Fixed

- **Multi-core `.hbm` binding (nash-p / S600).** An `.hbm` compiled with `core_num > 1` was
  rejected by the runtime when submitted with `HB_UCP_CORE_ANY` (*"When infer multi-core model
  task, the backend must be specified to specific cores"*). `model.json` already carried
  `bpu_cores`, but `ModelConfig` never parsed it, so the engine always submitted `CORE_ANY`.
  It is now parsed, and at load the sched core mask is bound to cores `0..(N-1)`;
  `set_bpu_priority` preserves the mask. Validated on S600: Qwen3.5-4B nash-p 4-core,
  22.6 tok/s.

## [0.1.2] — 2026-07-14

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
- **S600 multi-core BPU** — an `.hbm` compiled for N BPU cores (nash-p, N ∈ {1, 2, 4}) is
  recorded via `make_model_dir.py --bpu-cores N` (`model.json` `bpu_cores`). A large model
  saturates 4 cores (Qwen3.5-4B, 27 tok/s on S600); small models stay single-core. See
  [`docs/S600_RESULTS.md`](docs/S600_RESULTS.md).

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

[Unreleased]: https://github.com/ruisv/bllm/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/ruisv/bllm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ruisv/bllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ruisv/bllm/releases/tag/v0.1.0
