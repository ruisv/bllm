<!-- English README; 中文（默认）见 README.md -->

# BLLM — On-board LLM / VLM Runtime for RDK BPU

**BLLM** (BPU LLM) is a C++17 on-board **LLM / VLM runtime** for D-Robotics
**RDK S100 / S100P / S600**. It runs compiled `.hbm` model graphs directly on the board's
BPU (hbDNN / hbUCP) — a self-built native engine with its own KV / SSM cache, sampler and
decode loop — behind a clean C++ / Python task-style API. **No extra LLM SDK required.**

It is the large-language-model sibling of [bcdl](https://github.com/ruisv/bcdl) (RDK BPU
computer-vision runtime).

> 📖 中文 README: [README.md](README.md) ·
> API docs: [English](docs/API.en.md) / [中文](docs/API.zh.md)

## Features

- **Native BPU inference** — drives `.hbm` directly on the generic hbDNN / hbUCP stack
  (same BPU driver the vision stack uses).
- **Dense + hybrid** — Qwen2.5 / DeepSeek-R1-Distill / InternLM2 / GLM-Edge / Phi-4-mini
  dense text models, plus **Qwen3.5-0.8B hybrid Gated-DeltaNet/SSM** (strict 100% on the BPU,
  ~21 tok/s).
- **Multimodal** — Qwen2.5-Omni text + image + audio + video, from file paths or raw arrays
  (zero-copy camera/microphone).
- **One-line sessions** — `bllm.load(...)` / `NativeSession`, with a built-in C++ tokenizer,
  ChatML templating, streaming, multi-turn, sampling.
- **Full sampling** — greedy / temperature / top-k / top-p / min-p / typical-p / repeat,
  frequency and presence penalties; reproducible.
- **Production features** — perplexity, async submit, prompt cache (KV+SSM state save/restore,
  bit-identical), stop strings (stops early), and BPU priority / time-slice control to share
  the BPU with a vision pipeline.

## Install

On the board (aarch64, a conda env), install from our conda channel:

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bllm
```

This pulls `bllm` (Python bindings), `libbllm` (C++ lib + headers + cmake config) and their
runtime deps (`hobot-dnn`, `tokenizers-cpp`). RDK platform libraries come with the board OS.

## Quick start

A model is a **directory** (`.hbm` + `tokenizer.json` + a `model.json` manifest); load it in
one line:

```python
import bllm

llm = bllm.load("/path/to/qwen2.5-1.5b")        # a dir, or a bare .hbm + tokenizer_dir=
llm.set_sampling(temp=0.7, top_p=0.9)           # omit for greedy
for chunk in llm.stream_chat("Introduce Beijing in one sentence"):  # streaming, multi-turn
    print(chunk, end="", flush=True)
print(llm.last_decode_tps)
```

A bare `.hbm` (OE package) works directly — the runtime synthesizes the manifest:

```python
llm = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="Qwen2.5_1.5B_config/")
print(llm.chat("Hello"))
```

C++:

```cpp
#include "bllm/native_llm.h"
bllm::NativeLlm llm("/path/to/qwen2.5-1.5b");
std::string reply = llm.chat("Hello", 400, [](const std::string& s){ std::cout << s << std::flush; });
```

REPL: `bllm_chat_native --model <dir>` (see [`examples/chat_native.cc`](examples/chat_native.cc)).

## Multimodal (Qwen2.5-Omni)

Text + image + audio + video; media as file paths **or** raw arrays (`HxWx3` uint8 frames,
16 kHz mono float PCM), so a camera/mic pipeline needs no file round-trip:

```python
vlm = bllm.load("/models/qwen2.5-omni-3b")       # model.json with arch="omni"
for chunk in vlm.stream_chat("Describe this image.", ["bus.jpg"]):
    print(chunk, end="", flush=True)

vlm.reset(); vlm.chat("What is said in this audio?", audios=["clip.wav"])
vlm.reset(); vlm.chat("Describe this video.", videos=[bllm.load_video("clip.mp4")])
```

**Live capture** — each 2-second chunk is encoded into the KV cache as it arrives, so the
question at the end answers well under the offline first-token latency:

```python
vlm.stream_begin(fps=2.0)
while capturing:
    vlm.stream_frame(frame_hwc_uint8)     # camera frame
    vlm.stream_audio(pcm_float32)         # mic, 16 kHz mono
print(vlm.stream_ask("What just happened?"))
```

The KV window **is** the context and the binding constraint (`vlm.context_left` tracks the
remaining budget; overflowing raises rather than silently evicting). A vision tower's compiled
resolution sets the video-context budget; a model directory just points `visual` at the tower
you want. See the [API docs](docs/API.en.md) for more.

## Sharing the BPU with a vision pipeline

S100/S100P have a single BPU core, so an LLM and a vision pipeline (bcdl) contend for it. A
decode step is one big graph and by default blocks a co-resident detector. Letting the detector
jump the queue needs **two** things together: a `max_time_per_fc` compile flag on the `.hbm`
(so the scheduler can switch inside the graph) plus a low LLM priority at runtime:

```python
llm.set_bpu_priority(0)      # lowest — let vision preempt the BPU queue
```

Measured (yolo26s vs Qwen2.5-1.5B decode, detector p99): with both knobs the detector p99 drops
from 41 ms to **2.95 ms** at 383 FPS, LLM 24→17 tok/s; near-free when idle.

## Supported models

Official pre-compiled **text** LLMs run through one config-free path (auto model-type,
auto chat template, per-model quirks like int8 KV handled internally):

- **Qwen2.5** 1.5B / 7B (Base + Instruct)
- **DeepSeek-R1-Distill-Qwen** 1.5B / 7B
- **InternLM2-1.8B** · **GLM-Edge** · **Phi-4-mini**
- **Qwen3.5-0.8B** (hybrid SSM, native-only)
- **Qwen2.5-Omni-3B** (multimodal)

Validated across q8/q4 × context 1024/4096 variants. Model conversion (how a `.hbm` is
produced) is an offline process, out of scope here; this repo consumes a finished `.hbm` +
tokenizer config.

## Build from source

Develop on a host, build & run on the board (the runtime lives only there):

```bash
scripts/sync.sh                    # rsync the working tree to the board (BOARD ssh alias)
scripts/board_build.sh [--python]  # cmake + ninja on the board
scripts/board_test.sh              # sync, build, set perf mode, pytest
```

One CMake target, `bllm::native`, needing only the board's generic hobot runtime (plus
`tokenizers-cpp` for string I/O):

```cmake
find_package(bllm REQUIRED)
target_link_libraries(app PRIVATE bllm::native)
```

In Python, `import bllm` works whether or not the extension is built;
`bllm.available_backends()` returns `("native",)` or `()`.

### Making a model directory

A native model is a directory with a `model.json` manifest. Generate it with
`scripts/make_model_dir.py` (don't hand-write it — it resolves stop tokens from the
tokenizer's `generation_config.json` instead of guessing):

```bash
scripts/make_model_dir.py dense ~/models/qwen2.5-1.5b \
    --hbm Qwen2.5_1.5B_Instruct_1024.hbm \
    --tokenizer Qwen2.5_1.5B_Instruct_config/tokenizer.json --cache-len 1024
```

`hybrid` (Qwen3.5 SSM) and `omni` (multimodal) pass their extra towers/embed tables as flags.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
