<!-- English README; 中文（默认）见 README.md -->

# BLLM — On-board LLM / VLM Runtime for RDK BPU

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg)](python/CMakeLists.txt)
[![Platform](https://img.shields.io/badge/platform-RDK%20S100%20%2F%20S100P%20%2F%20S600%20(aarch64)-0A7BBB.svg)](#install)
[![Version](https://img.shields.io/badge/version-0.1.5-informational.svg)](CHANGELOG.md)

[简体中文](README.md) | **English**

**BLLM** (BPU LLM) is a C++17 on-board **LLM / VLM runtime** for D-Robotics
**RDK S100 / S100P / S600**. It runs compiled `.hbm` model graphs directly on the board's
BPU (hbDNN / hbUCP) — a self-built native engine with its own KV / SSM cache, sampler and
decode loop — behind a clean C++ / Python task-style API. **No extra LLM SDK required.**

It is the large-language-model sibling of [bcdl](https://github.com/ruisv/bcdl) (RDK BPU
computer-vision runtime).

<p align="center">
  <img src="docs/demo.gif" width="760" alt="Qwen3.5-0.8B hybrid SSM streaming chat on the RDK S100P BPU"><br>
  <em>Qwen3.5-0.8B (hybrid SSM) streaming natively on the RDK S100P BPU · ~14 tok/s</em>
</p>

> 📖 API docs: [English](docs/API.en.md) · [中文](docs/API.zh.md)

## Contents

- [Features](#features) · [Install](#install) · [Quick start](#quick-start) · [Making a model directory](#making-a-model-directory)
- [Serving](#serving-openai-compatible-api) · [Multimodal](#multimodal-qwen25-omni) · [Sharing the BPU](#sharing-the-bpu-with-a-vision-pipeline)
- [Supported models](#supported-models) · [Build from source](#build-from-source)
- [Community](#community) · [Contributing](#contributing) · [Acknowledgements](#acknowledgements) · [License](#license)

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

You also need a **one-time** ION carveout enlargement plus a reboot — model weights live in ION
rather than process RSS, and the runtime **segfaults** on alloc if the carveout is too small:

```bash
sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot   # <=3B; bpu_first for 7B
```

Performance mode does not survive a reboot; set it once per boot (see
[`docs/MODELS.en.md`](docs/MODELS.en.md)).

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

## Making a model directory

An official release scatters the `.hbm` across `model/` and the tokenizer across `config/`;
`bllm-make-model-dir` assembles them into the directory above. It ships with the `bllm`
package — no need to clone this repo (equivalently `python -m bllm.make_model_dir`, or
`scripts/make_model_dir.py` in a source tree):

```bash
R=<sdk>/oellm_runtime
bllm-make-model-dir dense ~/models/qwen2.5-1.5b \
    --hbm       $R/model/Qwen2.5_1.5B_Instruct_1024.hbm \
    --tokenizer $R/config/Qwen2.5_1.5B_Instruct_config/tokenizer.json \
    --cache-len 1024
```

**Never hand-write `model.json`.** The tool resolves stop tokens from the most authoritative
source available (`generation_config.json` first, then the tokenizer's declared specials) and
prints which one it used. A guessed eos yields a model that never stops: Qwen2.5's vocabulary
holds a literal `</s>` at id 128247 that is not a stop token.

Three `arch` values: `dense` (above), `hybrid` (Qwen3.5 SSM, also needs `--embed`), and
`omni` ([multimodal](#multimodal-qwen25-omni), also needs its towers + embed table). Full
deployment steps — official package layout, ION sizing — are in
[`docs/MODELS.en.md`](docs/MODELS.en.md).

## Serving (OpenAI-compatible API)

Package the runtime as an OpenAI-compatible inference server
(`/v1/chat/completions`, streaming + non-streaming) and deploy it on the board in
a container. Full details in [`docker/README.md`](docker/README.md).

**Self-contained image (model baked in)** — `docker run` and chat, no volume:

```bash
cd docker
./bake-model.sh ~/models/qwen3.5-2b bllm-serve:qwen3.5-2b-ctx512-int8-s100p
docker run -d --network host \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu \
  bllm-serve:qwen3.5-2b-ctx512-int8-s100p
```

**Model-agnostic image (mount any model)** — via docker compose:

```bash
cd docker && ./stage-libs.sh
MODEL_DIR=~/models/qwen3.5-2b docker compose up -d --build
```

The server listens on `:8866` with CORS enabled — point any OpenAI-compatible
client at `http://<board-ip>:8866/v1`
([chatbox-lite](https://github.com/lfbear/chatbox-lite), desktop Chatbox,
Open WebUI, the OpenAI SDK, …):

```bash
curl localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```


The server **caches conversation prefixes**: the OpenAI protocol is stateless, so a client
re-sends its whole history each turn and the server prefills only what is new. Hits appear in
`usage.prompt_tokens_details.cached_tokens`, aggregates on `GET /metrics`. Matching is exact
token-prefix comparison, so an edited history simply misses and falls back to a full prefill
— it cannot return a wrong answer. Measured on-board: hybrid (Qwen3.5) 31.6 s → 1.3 s (~25x),
dense TTFT 274 → 141 ms. One generation runs at a time (single BPU graph); requests beyond
`BLLM_MAX_QUEUE` get a 429. See [`docker/README.md`](docker/README.md).

## Multimodal (Qwen2.5-Omni)

The official Omni release is **three towers plus a host embedding table**. Assemble them into
a model directory first (`bllm-make-model-dir` ships with the conda package):

```bash
R=<sdk>/oellm_runtime
bllm-make-model-dir omni ~/models/qwen2.5-omni-3b \
    --hbm         $R/model/Qwen2.5_Omni_3B_Text.hbm \
    --visual      $R/model/Qwen2.5_Omni_3B_Visual.hbm \
    --audio       $R/model/Qwen2.5_Omni_3B_Audio.hbm \
    --embed       $R/model/embed_tokens.bin \
    --mel-filters $R/config/Qwen2.5_Omni_3B_config/mel_filters_t.txt \
    --tokenizer   $R/config/Qwen2.5_Omni_3B_config/tokenizer.json \
    --cache-len 2048
```

> ⚠️ **`embed_tokens.bin` is required by Omni and easy to miss.** It is a separate 1.2 GB
> `wget` line in the official download list, it sits in `model/`, and **its name does not
> mention Omni** — so grabbing the three `*_Omni_3B_*.hbm` files looks like a complete set
> and is not. Omni keeps its embedding table on the host (the BPU graph consumes embeddings,
> not token ids), so the model cannot run without it.

Then text + image + audio + video; media as file paths **or** raw arrays (`HxWx3` uint8 frames,
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
remaining budget; overflowing raises rather than silently evicting). **The official
`Visual.hbm` is frozen at 448px = 256 tokens per frame-pair, so Omni's 2048-slot window holds
about 8 seconds of video at 2 fps** — before the prompt, and before audio (~25 tokens/s more).
Still images are unaffected: one image is a single 256-token charge. A lower-resolution tower
buys proportionally more video (`(size/14/2)²` tokens per frame-pair: 336→144, 224→64), but the
resolution is baked in at export time, so changing it means re-exporting the tower offline on
an x86 host with `leap_llm` — outside this repo, and the official release ships no alternative.

Full deployment steps are in [`docs/MODELS.en.md`](docs/MODELS.en.md); see the
[API docs](docs/API.en.md) for more.

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
- **Qwen3.5-0.8B / 2B / 4B** (hybrid SSM, native-only; strict 100% on-BPU int8)
- **Qwen2.5-Omni-3B** (multimodal)

Validated across q8/q4 × context 1024/4096 variants. Model conversion (how a `.hbm` is
produced) is an offline process, out of scope here; this repo consumes a finished `.hbm` +
tokenizer config.

> **S600 multi-core**: a large model can bind several BPU cores and decode in parallel —
> Qwen3.5-4B saturates 4 cores at **27 tok/s**, the 0.8B does **42.7 tok/s** on one core. See
> the on-board measurements in [`docs/S600_RESULTS.md`](docs/S600_RESULTS.md).

### Pre-compiled model downloads

We publish **ready-to-`bllm.load()` model packages** (each a directory: `model.hbm` +
`tokenizer.json` + `model.json`, plus `embed_tokens.bin` for hybrid models). The file names
already encode **context length / target board / quantization width**, so they read for
themselves:

**📦 [Google Drive — model downloads](https://drive.google.com/drive/folders/1tR3MtP0iriptqpeHOSzdpRMT_ckdqDQ8)**

Verify, extract, and load (Qwen3.5-4B as an example):

```bash
sha256sum -c qwen3.5-4b-ctx512-int8-s100.tar.gz.sha256   # integrity check
tar xzf qwen3.5-4b-ctx512-int8-s100.tar.gz
python -c "import bllm; s=bllm.load('qwen3.5-4b-ctx512-int8-s100'); print(s.chat('Hello'))"
```

## Build from source

Most users just `conda install bllm` (see [Install](#install)). To hack on the source, clone
and build **on the board** (the runtime lives only there):

```bash
# on an RDK S100 / S100P / S600 board
git clone https://github.com/ruisv/bllm.git && cd bllm

# build dependencies (example, from our conda channel)
conda install -c https://mirrors.ruis.ai/conda -c conda-forge \
    cmake ninja cxx-compiler nlohmann_json hobot-dnn tokenizers-cpp nanobind numpy

scripts/build.sh --python          # cmake + ninja; artifacts in build/
export PYTHONPATH="$PWD/python"     # import in-tree; or scripts/build.sh --python --install
```

One CMake target, `bllm::native`, needing only the board's generic hobot runtime (plus
`tokenizers-cpp` for string I/O):

```cmake
find_package(bllm REQUIRED)
target_link_libraries(app PRIVATE bllm::native)
```

In Python, `import bllm` works whether or not the extension is built;
`bllm.available_backends()` returns `("native",)` or `()`.

## Community

Join the **BPU tech chat group** to discuss RDK / BPU on-device deployment and this project
(shared community group with [bcdl](https://github.com/ruisv/bcdl)).

<img src="docs/assets/bllm-group-qr.jpg" alt="BPU tech chat group QR" width="240">

> WeChat group QR codes expire — if it has, please open an [Issue](../../issues) and we'll
> refresh it; you're also welcome to just discuss in [Issues](../../issues).

## Contributing

Contributions welcome — issues and pull requests alike. See [`CONTRIBUTING.md`](CONTRIBUTING.md);
in brief:

- **Develop anywhere, build & run on the board.** The hobot runtime exists only on RDK
  hardware, so the C++ library, the Python extension and the board tests must build and run
  on an S100 / S100P / S600; the pure-metadata unit tests (`tests/test_modelmeta.py` etc.) run
  on any host.
- **Conventions** — headers `.h`, impl `.cc`, namespace `bllm`, errors via `BLLM_CHECK(...)`;
  match the surrounding style.
- **Tests** — any behaviour change adds/updates tests: pin logic with the offline metadata/
  routing tests, and add a board end-to-end test that skips cleanly when its model is absent.
- **Commits** — keep them focused; follow the [Conventional Commits](https://www.conventionalcommits.org/)
  style already in the log.

## Acknowledgements

- **D-Robotics** — the RDK S100 / S100P / S600 platform and the hobot runtime (`hbDNN` /
  `hbUCP` / `HBRT`) BLLM builds on.
- **[nanobind](https://github.com/wjakob/nanobind)** — the Python binding layer.
- **[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)** +
  **[HuggingFace tokenizers](https://github.com/huggingface/tokenizers)** — C++ tokenization.
- The upstream authors of the models it runs — **Qwen** (Qwen2.5 / Qwen3.5 / Qwen2.5-Omni,
  Alibaba), **DeepSeek**, **InternLM** (Shanghai AI Lab), **GLM** (Zhipu), **Phi** (Microsoft).
  Each weight is under its own upstream license.

## License

BLLM is licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE). This covers
BLLM's own source only; the `.hbm` model weights are under their own upstream licenses.
