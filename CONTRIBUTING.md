# Contributing to BLLM

Thanks for your interest — contributions are welcome, whether a bug report, a fix, a new
model wiring, or a docs improvement. By contributing you agree that your contributions are
licensed under the project's [Apache License 2.0](LICENSE).

## The one hard constraint: build on the board

BLLM links the D-Robotics **hobot runtime** (`hbDNN` / `hbUCP` / `HBRT`), which exists **only
on RDK S100 / S100P / S600 hardware** (via the board OS, or the `hobot-dnn` conda package).
Therefore:

- The **C++ library, the Python extension, and the board tests must be built and run on an
  S100 / S100P / S600** (aarch64, Ubuntu 22.04, GCC 11, CMake ≥ 3.22).
- **Do not try to compile on a non-board host.** Editing, code review, and the pure-metadata
  unit tests (`tests/test_modelmeta.py`, `tests/test_backend_routing.py`, and the off-board
  C++ tests) work anywhere; everything that touches the runtime does not.

A typical loop is "edit on a workstation, sync to the board, build & test there."

## Project layout

```
include/bllm/, src/   the native runtime (header-heavy engines + model.json parsing)
  native_engine.h        dense KV engine over hbDNN
  native_hybrid_engine.h Gated-DeltaNet / SSM engine (Qwen3.5)
  native_vlm.h           multimodal (Qwen2.5-Omni: text + image + audio + video)
  native_llm.h           unified string-in/out session (tokenizer + ChatML + streaming)
  native_sampler.h       sampling (temp / top-k/p / min-p / typ-p / penalties)
  model_config.h         the model.json manifest
python/        nanobind bindings + the pure-Python `bllm` package
examples/      standalone C++ programs (bllm_chat_native, bllm_vlm_native, …)
tests/         pytest + off-board C++ tests
scripts/       sync / build / test / make_model_dir
cmake/         FindHobot / FindTokenizers / package config
```

## Build & test

Clone and build **on the board** (see the README's *Build from source* for the dependency
install), then run the tests there:

```bash
scripts/build.sh --python          # cmake + ninja on the board
export PYTHONPATH="$PWD/python"
python -m pytest tests/ -q -rs     # model tests skip cleanly unless BLLM_TEST_* is set
```

Off-board (any host), the metadata/routing logic can be checked without hardware:

```bash
c++ -std=c++17 -I include tests/test_choose_backend.cc -o /tmp/t && /tmp/t
PYTHONPATH=python python -m pytest tests/test_modelmeta.py tests/test_backend_routing.py -q
```

## Conventions

- Headers `.h`, implementation `.cc`; namespace `bllm`; errors via `BLLM_CHECK(...)`.
- Match the surrounding code's style, comment density, and idioms.
- A native model is a directory with a `model.json` manifest — never hand-write one; use
  `scripts/make_model_dir.py`, which resolves stop tokens from the tokenizer's own
  `generation_config.json`.

## Tests

Any behaviour change adds or updates tests:

- Pin logic with an **offline** test where possible (metadata synthesis, backend routing,
  the sampler, stop-string matching) — these run on any host.
- For anything that touches a model, add a **board end-to-end** test that **skips cleanly**
  when its model isn't present (set the `BLLM_TEST_*` env vars for the model tests).

## Commits & pull requests

- Keep commits focused; follow the [Conventional Commits](https://www.conventionalcommits.org/)
  style already in the log.
- PRs: describe what changed and how you verified it (which board, which model). Keep the diff
  scoped to one concern.

## Reporting bugs

Open an [Issue](../../issues) with: the board (S100 / S100P / S600) and OS, the model, a
minimal repro, and the actual vs expected output. Runtime errors on alloc are usually the ION
carveout — see the on-board notes.

## Scope

BLLM is the on-board **runtime** for finished `.hbm` models. Offline model conversion (how a
`.hbm` is produced) is out of scope here. Batching and the InternVL/Qwen-VL model types are
intentionally not implemented (single-core part; no shipped `.hbm`).
