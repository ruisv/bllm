<!-- English README; 中文（默认）见 README.md -->

# BLLM — an on-board LLM framework for the RDK BPU

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg)](python/CMakeLists.txt)
[![Platform](https://img.shields.io/badge/platform-RDK%20S100%20%2F%20S100P%20%2F%20S600%20(aarch64)-0A7BBB.svg)](#1-install-one-minute)
[![Version](https://img.shields.io/badge/version-0.4.2-informational.svg)](CHANGELOG.md)

[简体中文](README.md) | **English**

> ### Run an on-device LLM on your board in minutes.
> One session-style C++ / Python API for chat, streaming, multimodal, structured
> output and OpenAI-compatible serving — all on the board's BPU.
> **Runs on the generic official runtime; no extra LLM SDK required.**

```python
import bllm

llm = bllm.load("~/models/qwen3.5-2b")           # one directory = one model
for chunk in llm.stream_chat("Introduce Beijing in one sentence"):
    print(chunk, end="", flush=True)
```

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bllm
```

BLLM is the large-model sibling of [**bcdl**](https://github.com/ruisv/bcdl) (RDK
BPU computer vision) — the two share one BPU and can be scheduled against each other.

<p align="center">
  <img src="docs/demo.gif" width="760" alt="Qwen3.5-2B VLM image+text chat streaming on the RDK S100P BPU, with multi-turn context memory"><br>
  <em>Qwen3.5-2B VLM (image+text, hybrid SSM + self-built vision tower, ctx4k) streaming natively on the RDK S100P BPU, bilingual — the English follow-up counts the remotes correctly without the image being re-attached</em>
</p>

**Jump to:**
[Why BLLM](#why-bllm) ·
[Embodied policies (VLA)](#embodied-policies-π05-on-the-bpu) ·
[Relation to the official runtime](#relation-to-the-official-runtime) ·
[Architecture](#architecture) ·
[Quick start](#quick-start) ·
[What you can build](#what-you-can-build) ·
[Serving](#serving-openai-compatible-api) ·
[Multimodal](#multimodal) ·
[Sharing the BPU](#sharing-the-bpu-with-a-vision-pipeline) ·
[Performance](#performance) ·
[Supported models](#supported-models) ·
[Capabilities](#capabilities) ·
[Build from source](#build-from-source) ·
[Community](#community)

## Why BLLM

The hard part of on-device LLMs is not "run inference once". It is: how to lay out
a model directory, where the stop tokens come from, how to apply the chat template,
how to manage the KV / SSM cache, how not to overflow context across turns, how to
make sampling reproducible, how to guarantee the JSON parses, how to expose an API,
and how to coexist with vision work on the same BPU. BLLM makes all of that the
default.

- **✅ One-line load** — `bllm.load(dir)` detects the model type, finds the chat
  template and resolves stop tokens from the most authoritative source available
  (a guessed eos means a model that **never stops**). A bare `.hbm` works too.
- **✅ One-line install** — prebuilt conda packages (Python 3.9–3.14); no compiler
  on the board.
- **✅ One session API** — chat / streaming / multi-turn / sampling / multimodal
  behind the same interface, with C++ and Python as peers. Swapping model does not
  mean rewriting code.
- **✅ Structured output is a guarantee, not a request** — GBNF grammar constraints
  and JSON Schema → grammar apply *while decoding*, so `json.loads()` cannot fail.
- **✅ Serving out of the box** — OpenAI-compatible HTTP (streaming + non-streaming),
  Docker images, and conversation prefix caching (measured on-board: 31.6 s → 1.3 s,
  ~25×).
- **✅ Architectures the official toolchain doesn't cover** — Qwen3.5 hybrid
  Gated-DeltaNet/SSM strictly 100% on the BPU (**BLLM-native only**), plus GLM-Edge
  and Phi-4-mini.
- **✅ Coexists with vision** — on a single BPU core, lowering the LLM's priority
  takes a co-resident detector's p99 from 41 ms down to **2.95 ms** — not "run both
  and watch them starve each other".
- **✅ Not just chat: embodied policies run on the BPU too** — π0.5 and SmolVLA,
  two vision-language-action models, load in one line and match their x86
  references on `libero_spatial` (next section).
  **The official runtime does not cover this class of model.**

## Embodied policies: π0.5 on the BPU

On-device LLMs are not only about chat. BLLM runs a whole **vision-language-action
(VLA)** model — π0.5 — on the BPU. This is a class of workload the official LLM
runtime does not cover, and it is what takes a board from "answers questions" to
"moves an arm".

<div align="center">
  <img src="docs/pi05_libero_grid.gif" width="760" alt="π0.5 replaying all ten libero_spatial tasks on an RDK S100P BPU"><br>
  <sub>libero_spatial <b>99/100</b> on-board, matching the x86 reference (99/100); <b>1016 ms per action chunk</b>, entirely on the BPU</sub>
</div>

```python
import bllm

policy = bllm.load_policy("~/models/pi05-libero-224-s100p")
actions = policy.act(scene_rgb, wrist_rgb, "pick up the black bowl")  # 10×7 chunk
```

Two camera frames plus one English instruction produce a 10×7 chunk of end-effector
deltas. All three graphs (4.57 GB) stay resident while the **process RSS is only
16 MB** (weights live in BPU/ION) and inference occupies about **1.3 of the 6 cores**,
leaving the CPU for perception and control. The model package carries its own
tokenizer, embedding table and action quantiles — openpi is not a runtime dependency.

On-board scores across three suites: `libero_spatial` **99/100** (x86 reference
99/100), `libero_goal` 97/100, `libero_object` 95/100. The spatial row was scored
**by the model package itself** — `bllm.load_policy()` driving the openpi harness
directly, through none of its transforms — i.e. the exact path you get after
installing. Replays and measurements for the other two suites are in the
[π0.5 doc](docs/PI05.md).

**📦 [ruisv/bllm-pi05-libero-224-s100p](https://huggingface.co/ruisv/bllm-pi05-libero-224-s100p)** —
download and `bllm.load_policy()` (5.3 GB: three `.hbm` files + embedding table +
tokenizer + manifest).

### SmolVLA — one API, both boards

[LeRobot](https://github.com/huggingface/lerobot)'s SmolVLA is the second VLA
here and the **first model shipped for two boards at once**. It is far smaller
than π0.5 and one chunk carries **50** actions instead of 10.

```python
policy = bllm.load_policy("~/models/smolvla-libero-512-s100p")   # or ...-s600
actions = policy.act(scene_rgb, wrist_rgb, "pick up the black bowl",
                     state=eef_state)                            # 50x7 chunk
```

| | S100P | S600 | reference (RTX 4090) |
|---|---:|---:|---:|
| libero_spatial | 68 | 71 | 73 |
| libero_object | 76 | 75 | 72 |
| libero_goal | 79 | 81 | 79 |
| libero_10 | 51 | 54 | 48 |
| **all four suites / 400** | **274** | **281** | **272** |
| per action chunk | **562 ms** | **220 ms** | 500 ms |
| graphs resident | 1.43 GB | 0.43 GB | — |

Shipped packages default to **four denoising steps** (the reference uses ten) —
the expert graph runs once per step, so this is the one latency lever that costs
no rebuild. Measured across three paired comparisons, 300 episodes:
libero_spatial 70→68 (S100P), 71→**71** (S600), libero_object 76→**76** (S100P);
**217 → 215** in total. `--steps 10` repackages with the reference schedule.

All twelve evaluations ran through the *same* harness — same env, same init
states, same seeds, same wire, with only the side computing the chunk changed.
Over 400 episodes the boards are **+2** and **+9** against the reference, and the
sign **flips between suites** — a systematic porting loss would move all four the
same way. The per-task profile is shared too: `libero_goal` task 6 is 5 on all
three, `libero_10` task 0 is 1 on all three.

`libero_10` sits at 48–54 where the other suites are 68–81, and **the reference is
the lowest of the three** — that is this checkpoint on long-horizon two-stage
tasks, not the board. Scored without the reference column, the 51 would read as
"the port collapses on long tasks".

One substantive difference from π0.5: **SmolVLA reads proprioception.** The state
is projected into a prefix token, so omitting it is a different observation
(measured at 2.0 in action space — a full gripper unit).

**📦 [ruisv/bllm-smolvla-libero-512](https://huggingface.co/ruisv/bllm-smolvla-libero-512)** —
one repo, both boards (`s100p/` and `s600/`). The whole conversion pipeline is
open in [bllm-model-zoo](https://github.com/ruisv/bllm-model-zoo/tree/main/models/smolvla-libero/convert):
the ~100 SmolVLA finetunes on the Hub share this architecture, so converting your
own is the same scripts with a different `--ckpt`.

## Relation to the official runtime

BLLM's compute comes from the same D-Robotics BPU runtime (`hbDNN` / `hbUCP`), and
it happily consumes officially released `.hbm` files. The table below compares
**abstraction level and positioning**, not quality:

| | Official OE-LLM runtime (sample-level) | BLLM |
|---|---|---|
| Shape | a standalone LLM runtime + sample programs | a reusable framework running directly on the generic hbDNN / hbUCP — **no extra LLM SDK** |
| Interface | mostly C APIs | C++17 library + Python bindings, task-style session API |
| Loading a model | organise paths and flags per the sample | `bllm.load(dir)`; `bllm-make-model-dir` writes the manifest and resolves stop tokens |
| Model coverage | the officially released `.hbm` | the official `.hbm` **plus** architectures it doesn't cover (Qwen3.5 hybrid SSM / GLM-Edge / Phi-4-mini) |
| Sessions | template, history and cache are yours to assemble | built-in C++ tokenizer, ChatML templating, multi-turn, streaming, prompt cache |
| Generation control | basic sampling | full sampling + `logit_bias` + `logprobs` + GBNF grammar constraints |
| Serving | build it yourself | OpenAI-compatible HTTP + Docker + prefix cache + `/metrics` |
| Coexisting with vision | handle it yourself | BPU priority / time-slice control alongside [bcdl](https://github.com/ruisv/bcdl) |
| Embodied policies (VLA) | not covered | π0.5 and SmolVLA load in one line and return action chunks ([above](#embodied-policies-π05-on-the-bpu)) |

In one line: **the official runtime proved the BPU can run LLMs; BLLM turns that
into a product in minutes.**

## Architecture

```
           your application / any OpenAI-compatible client
                              ↓
   ┌────────────────────────────────────────────────────┐
   │  BLLM                                               │
   │  sessions & templating · sampling & grammar          │
   │  KV/SSM cache · vision tower · prefix cache · server  │
   └────────────────────────────────────────────────────┘
                              ↓
          D-Robotics hobot runtime (official, generic)
                       hbDNN · hbUCP
                              ↓
                         BPU "Nash"
```

The inference engine is self-built — decode loop, KV / SSM cache, sampler and
tokenizer all live in BLLM — and underneath it depends only on the generic BPU
runtime the board already ships. That is the same driver the vision stack uses,
which is why the two can share a board and schedule against each other.

## Quick start

### 1. Install (one minute)

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

### 2. Get a model (one minute)

Fastest route: **download a ready-to-`bllm.load()` package** — the file names
already encode context length / target board / quantization width. Qwen3.5
image+text (VLM, S100P, ctx4k/ctx512) is published on Hugging Face with a full
acceptance record on the model card:

**📦 [ruisv/bllm-qwen3.5-0.8b](https://huggingface.co/ruisv/bllm-qwen3.5-0.8b) ·
[ruisv/bllm-qwen3.5-2b](https://huggingface.co/ruisv/bllm-qwen3.5-2b) ·
[ruisv/bllm-qwen3.5-4b](https://huggingface.co/ruisv/bllm-qwen3.5-4b)**

```bash
sha256sum -c qwen3.5-4b-ctx512-int8-s100.tar.gz.sha256   # integrity check
tar xzf qwen3.5-4b-ctx512-int8-s100.tar.gz
```

Conversion methodology, acceptance protocol, and every rejected build are in
[bllm-model-zoo](https://github.com/ruisv/bllm-model-zoo).

Already have an officially released `.hbm`? Assemble it into a model directory with
`bllm-make-model-dir` — see [Making a model directory](#making-a-model-directory).

### 3. Run (one minute)

```python
import bllm

llm = bllm.load("qwen3.5-4b-ctx512-int8-s100")  # a dir, or a bare .hbm + tokenizer_dir=
llm.set_sampling(temp=0.7, top_p=0.9)           # omit for greedy
for chunk in llm.stream_chat("Introduce Beijing in one sentence"):  # streaming, multi-turn
    print(chunk, end="", flush=True)
print(llm.last_decode_tps)
```

C++:

```cpp
#include "bllm/native_llm.h"
bllm::NativeLlm llm("/path/to/qwen2.5-1.5b");
std::string reply = llm.chat("Hello", 400, [](const std::string& s){ std::cout << s << std::flush; });
```

REPL: `bllm_chat_native --model <dir>` (see [`examples/chat_native.cc`](examples/chat_native.cc)).

> 📖 Full API docs: Python [English](docs/API.en.md) · [中文](docs/API.zh.md) | C++ [English](docs/CPP_API.en.md) · [中文](docs/CPP_API.zh.md)

## What you can build

| Capability | How it looks | Example |
|---|---|---|
| **Chat / streaming** | `llm.chat(...)` · `llm.stream_chat(...)` | [`chat_native.cc`](examples/chat_native.cc) · [`native_session.py`](examples/native_session.py) |
| **Multi-turn & context** | the session owns the history; `llm.context_left` tracks the budget | [`docs/API.en.md`](docs/API.en.md) |
| **Sampling** | `llm.set_sampling(temp=…, top_p=…, …)`, reproducible | [sampling reference](docs/API.en.md) |
| **Structured output** | `llm.set_grammar("root ::= …")` / JSON Schema | [`docs/API.en.md`](docs/API.en.md) |
| **Image + text (VLM)** | `vlm.chat("describe this", images=["a.jpg"])` | [`vlm_native.py`](examples/vlm_native.py) · [`qwen35_native.cc`](examples/qwen35_native.cc) |
| **Text / image / audio / video** | Qwen2.5-Omni, file paths or raw arrays (zero-copy camera/mic) | [`vlm_native.cc`](examples/vlm_native.cc) |
| **Text embeddings** | `NativeEmbedder` (Qwen3-VL-Embedding; C++ API, no Python binding yet) | [`embed_native.cc`](examples/embed_native.cc) |
| **Embodied policy (VLA)** | `bllm.load_policy(...).act(scene, wrist, "instruction")` → action chunk | [above](#embodied-policies-π05-on-the-bpu) · [π0.5 doc](docs/PI05.md) · [`pi05_policy.cc`](examples/pi05_policy.cc) · [`smolvla_policy.cc`](examples/smolvla_policy.cc) |
| **Serving** | OpenAI-compatible HTTP + Docker | [`docker/README.md`](docker/README.md) |
| **Sharing the BPU** | `llm.set_bpu_priority(0)` | [Sharing the BPU](#sharing-the-bpu-with-a-vision-pipeline) |

## Making a model directory

A model is a **directory** (`.hbm` + `tokenizer.json` + a `model.json` manifest). An official
release scatters the `.hbm` across `model/` and the tokenizer across `config/`;
`bllm-make-model-dir` assembles them. It ships with the `bllm` package — no need to clone this
repo (equivalently `python -m bllm.make_model_dir`, or `scripts/make_model_dir.py` in a source
tree):

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
`omni` ([multimodal](#multimodal), also needs its towers + embed table). Full
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

## Multimodal

Two routes: the official Qwen2.5-Omni (text/image/audio/video), and **Qwen3.5
image+text** with a vision tower we compiled ourselves.

### Qwen3.5 image + text

Qwen3.5 is natively an image-text-to-text model — the text-only package simply drops
the vision half at compile time. The same hybrid text `.hbm` plus a vision tower is
all it takes (the tower is converted offline; prebuilt packages are ready to use):

```bash
bllm-make-model-dir hybrid ~/models/qwen3.5-0.8b-vlm448-ctx2k \
    --hbm qwen35_0.8b_ctx2k.hbm --embed embed_fp16.bin --tokenizer tokenizer.json \
    --visual qwen35_visual_448.hbm --hbm-prefill qwen35_prefill_n32.hbm \
    --mrope-section 11 11 10 --mrope-interleaved \
    --vision-patch 16 --vision-mean 0.5 --vision-std 0.5 --cache-len 2048
```

```python
vlm = bllm.load("~/models/qwen3.5-0.8b-vlm448-ctx4096-int8-s100p")
print(vlm.chat("Describe this image in one sentence.", images=["bears.jpg"]))
# -> Two brown bears walk across a dusty, arid landscape.

vlm.set_thinking(False)                       # answer directly, skip the <think>...</think> block
m = bllm.load("~/models/qwen3.5-0.8b-vlm448-ctx4096-int8-s100p", vision=False)
print(m.chat("Tell me about yourself"))       # forces a text-only session — the vision tower's weights are never loaded
```

Measured on S100P (0.8B, with the chunked prefill graph):

| bucket | tokens/image | tower cosine vs HF fp32 | first token |
|---|---|---|---|
| 320×320 (faster, recommended) | 100 | 0.9987 | **1.23 s** |
| 448×448 | 196 | 0.99654 | **2.30 s** |

> ⚠️ Those last four settings are **mandatory and must not be copied from Omni**
> (patch 16 vs 14, mean/std 0.5 vs CLIP, rope section `11 11 10` vs `16 24 24`,
> interleaved vs contiguous frequency layout). A compiled `.hbm` does not reveal
> them, and getting one wrong yields **shape-correct garbage**. `--mrope-interleaved`
> especially: when `t==h==w` the two layouts are identical, so text-only looks
> perfect and **positions only go wrong once an image is fed**.
>
> ⚠️ Most of the first-token latency is feeding the vision rows into the text model,
> **not the tower** (the 448 tower encodes in 0.36 s). The chunked delta-rule prefill
> graph (`--hbm-prefill`) amortizes ingestion down to seconds; lower resolution is faster.
>
> Still images only for now: the video path implements Omni's TMRoPE, whereas Qwen3.5
> separates frames with timestamp tokens — passing `videos=` raises explicitly rather
> than quietly producing a mis-ordered answer.

### Qwen2.5-Omni (text / image / audio / video)

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

## Performance

Measured on the board (int8 weights; decode throughput and first-token latency):

| model | board | context | decode |
|---|---|---|---|
| Qwen3.5-0.8B (hybrid SSM) | S100P | 2048 | **22.5 tok/s** |
| Qwen3.5-0.8B (hybrid SSM) | S100P | 4096 | **19.5 tok/s** |
| Qwen3.5-0.8B (hybrid SSM, VLM) | S100P | 512 | **23.18 tok/s** |
| Qwen3.5-0.8B (hybrid SSM, VLM) | S100P | 4096 | **18.52 tok/s** |
| Qwen3.5-2B (hybrid SSM, VLM) | S100P | 512 | **15.29 tok/s** |
| Qwen3.5-2B (hybrid SSM, VLM) | S100P | 4096 | **13.20 tok/s** |
| Qwen3.5-4B (hybrid SSM, VLM) | S100P | 512 | **7.54 tok/s** |
| Qwen3.5-4B (hybrid SSM, VLM) | S100P | 4096 | **6.14 tok/s** |
| Qwen2.5-1.5B (dense) | S100P | 1024 | **~24 tok/s** |
| Qwen3.5-0.8B (hybrid SSM) | S600 (single core) | — | **42.7 tok/s** |
| Qwen3.5-4B (hybrid SSM) | S600 (4 cores) | — | **27 tok/s** |

The 2B/4B rows are measured against this release's GQA batched-matmul + mask-dedup
rewrite (4B is nearly 2x faster than the pre-rewrite 3.29 tok/s — mostly DMA saved
by dropping the old `expand_win` broadcast).

First token (including image encode, measured on S100P): Qwen3.5-0.8B VLM **1.23 s**
(320px, ctx2048) / **2.30 s** (448px, ctx2048) / **1.19 s** (320px, ctx512) /
**1.32 s** (320px, ctx4096); Qwen3.5-2B VLM **1.51 s** (ctx512) / **1.65 s**
(ctx4096); Qwen3.5-4B VLM **2.93 s** (ctx512) / **3.29 s** (ctx4096). With a
server-side prefix-cache hit, hybrid goes **31.6 s → 1.3 s** and dense TTFT
**274 → 141 ms**.

## Supported models

These run through one config-free path (auto model-type, auto chat template, per-model
quirks like int8 KV handled internally), validated across q8/q4 × context 1024/4096
variants.

**Officially released `.hbm` (D-Robotics)**:

- **Qwen2.5** 1.5B / 7B (Base + Instruct)
- **DeepSeek-R1-Distill-Qwen** 1.5B / 7B
- **InternLM2-1.8B**
- **Qwen2.5-Omni-3B** (multimodal)

**Converted by us** (architectures the official toolchain does not cover; converted
offline and validated on the board):

- **Qwen3.5-0.8B / 2B / 4B** (hybrid SSM, **BLLM-native only**; strict 100% on-BPU int8)
- **GLM-Edge** · **Phi-4-mini**

Model conversion (how a `.hbm` is produced) is an offline process, out of scope here; this
repo consumes a finished `.hbm` + tokenizer config. Prebuilt packages: see
[Quick start step 2](#2-get-a-model-one-minute).

## Capabilities

- **Native BPU inference** — drives `.hbm` directly on the generic hbDNN / hbUCP stack
  (same BPU driver the vision stack uses).
- **Dense + hybrid** — Qwen2.5 / DeepSeek-R1-Distill / InternLM2 / GLM-Edge / Phi-4-mini
  dense text models, plus **Qwen3.5 hybrid Gated-DeltaNet/SSM** (strict 100% on the BPU).
- **Multimodal** — Qwen2.5-Omni text + image + audio + video, and **Qwen3.5 image+text**
  (self-built vision tower), from file paths or raw arrays (zero-copy camera/microphone).
- **One-line sessions** — `bllm.load(...)` / `NativeSession`, with a built-in C++ tokenizer,
  ChatML templating, streaming, multi-turn, sampling.
- **Full sampling** — greedy / temperature / top-k / top-p / min-p / typical-p / repeat,
  frequency and presence penalties; reproducible. Plus `logit_bias` (force or ban tokens)
  and `logprobs` (exact full-vocab log probabilities).
- **Structured output** — GBNF grammar-constrained decoding, and JSON Schema -> grammar.
  The constraint applies *while decoding*, so `json.loads()` on the reply cannot fail —
  a guarantee, not a request made in the prompt.
- **Text embeddings** — `NativeEmbedder` (Qwen3-VL-Embedding, C++ API) for retrieval and
  similarity.
- **Chunked prefill** — hybrid models ingest prompts and images through a seq_len=N prefill
  graph, ~76× faster than token-by-token (a model without one falls back gracefully).
- **Production features** — perplexity, async submit, prompt cache (KV+SSM state save/restore,
  bit-identical), stop strings (stops early), and BPU priority / time-slice control to share
  the BPU with a vision pipeline.

## Build from source

Most users just `conda install bllm` (see [Install](#1-install-one-minute)). To hack on the
source, clone and build **on the board** (the runtime lives only there):

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

## Documentation

| Document | What it covers |
|---|---|
| [`docs/API.en.md`](docs/API.en.md) · [`docs/API.zh.md`](docs/API.zh.md) | **Python API reference** — sessions, multimodal, sampling, serving primitives. |
| [`docs/CPP_API.en.md`](docs/CPP_API.en.md) · [`docs/CPP_API.zh.md`](docs/CPP_API.zh.md) | **C++ API reference** — `NativeLlm` / `NativeVlm` / `NativeEmbedder`, grammars, the engine layer. |
| [`docs/MODELS.en.md`](docs/MODELS.en.md) · [`docs/MODELS.zh.md`](docs/MODELS.zh.md) | **Deployment guide** — official package layout, model directories, ION and performance mode. |
| [`docker/README.md`](docker/README.md) | **Serving** — the OpenAI-compatible server, images, prefix cache and metrics. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to set up, build (on the board), test, and submit changes. |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes (Keep a Changelog / SemVer). |

## Community

Join the **BPU tech chat group** to discuss RDK / BPU on-device deployment and this project
(shared community group with [bcdl](https://github.com/ruisv/bcdl)).

<img src="https://ruisv.oss-cn-beijing.aliyuncs.com/public/rdk/images/bllm-qrcode.jpg" alt="BPU tech chat group QR" width="240">

> WeChat group QR codes expire; the image above is refreshed in place at the same link. If
> scanning still fails, open an [Issue](../../issues) — you're also welcome to just discuss there.

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
