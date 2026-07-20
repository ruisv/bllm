# Changelog

All notable changes to BLLM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.5]

### Added

- **Session-aware serving layer.** `docker/server.py` is now a thin OpenAI-protocol facade
  over `docker/serving/`: an engine that owns the model on one worker thread, and a prefix
  cache of engine snapshots keyed by the token ids they cover. The OpenAI protocol is
  stateless — a client re-sends the whole history every turn — so the server used to
  re-prefill all of it; now a turn prefills only what is new. Matching is exact token-prefix
  comparison, so a wrong reuse is not possible: an edited history or different tokenization
  simply matches a shorter entry or misses and falls back to a full prefill.
  Measured on an S100P — hybrid (Qwen3.5-0.8B) 31.6 s → 1.3 s (~25x), dense
  (Qwen2.5-1.5B) TTFT 274 → 141 ms on turn 10. New knobs: `BLLM_CACHE_MAX_MB`,
  `BLLM_CACHE_TTL_S`, `BLLM_CACHE_MAX_ENTRIES`, `BLLM_MAX_QUEUE`; new `GET /metrics`.
- **Serving primitives on the session** — `snapshot()` / `restore()` (in-memory, no temp
  file), `feed_ids()` / `decode_stream()` (generate() split so a caller can snapshot exactly
  at the prefill/decode boundary), `request_cancel()` / `canceled`, plus `context_left`,
  `prefill_chunk` and `cache_align()`. `set_prefix()`/`ask()` serve ONE fixed prefix; these
  let a server hold many states and pick one per request.
- **`cache_align()` — snapshots must be chunk-aligned.** The dense engine batch-prefills runs
  of `prefill_chunk` tokens and decode-steps the remainder, and the two graphs are not
  numerically identical under int8, so a snapshot taken off a chunk boundary changes the
  reply. Measured: aligned splits reproduced the un-cached answer 5/5, unaligned 1/18 — one
  turning *"Venus is the hottest"* into *"Earth is the hottest"*. The rule lives in the
  library so callers cannot get it wrong.
- **Real token accounting.** `usage` now comes from the model's tokenizer instead of a
  4-characters-per-token estimate, and reports OpenAI's `prompt_tokens_details.cached_tokens`.

### Fixed

- **A mid-stream disconnect could corrupt the next request's answer.** The server held an
  `asyncio.Lock` around the SSE response while generation ran on a per-request thread; a
  client vanishing mid-stream closed the async generator and released the lock, leaving the
  generation thread still driving the engine, so the next request ran concurrently against
  it. Demonstrated against the old server: after a disconnect, *"reply with exactly the word:
  pineapple"* returned `"pineapple1. pineapple1.1. pineapple1. ..."`. The session is now
  owned by a single worker thread and reached only through a queue, so no HTTP-side event
  can release it early.
- **Requests queued silently and without bound** — clients just saw a timeout. Admission
  control now returns 429 past `BLLM_MAX_QUEUE` (default 8).
- **`kPrefillMinTokens` was 16, roughly 5x too high.** A prefill pass costs a padded chunk
  regardless of how few tokens it carries, so the crossover is (prefill pass)/(decode step)
  — measured 2.93 on Qwen2.5-1.5B and 3.31 on Phi-4-mini, stable because both are a single
  BPU graph invocation. It is now 4. A 15-token turn (a short follow-up like *"go on"* —
  exactly the common case the old value punished) drops from **676 ms to 135 ms**. Note this
  changes segmentation, and therefore replies, for runs of 4..15 tokens.

### Note

The Docker image installs `bllm` from the conda channel, so the server can be newer than the
runtime beneath it. When the installed build predates the snapshot primitives the server
degrades to full re-prefill and says so at startup, rather than failing — the concurrency fix
applies either way, and the cache switches itself on under a new enough runtime.
`GET /metrics` reports `supports_prefix_cache`.

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
