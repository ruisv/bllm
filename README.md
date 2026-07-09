# BLLM — BPU LLM Runtime for RDK S100

**BLLM** (BPU LLM) is a C++17 on-board **LLM / VLM runtime** library for
D-Robotics RDK S100 / S100P / S600. It wraps the board's OpenExplorer OE-LLM
runtime (`libxlm.so`) behind an ergonomic, task-style C++/Python API.

It is the large-language-model sibling of two vision libraries:

| Library | Target | Runtime | Scope |
|---------|--------|---------|-------|
| [bcdl](https://github.com/ruisv/bcdl) | RDK S100 BPU | `hbDNN` / `hbUCP` | computer vision (detect / seg / depth / OCR / pose …) |
| ccdl | NVIDIA | TensorRT / CUDA | computer vision |
| **bllm** | RDK S100 BPU | **`libxlm.so`** (OE-LLM) | **LLM / VLM chat** |

## Why a separate library from bcdl?

On-board LLMs do **not** run on bcdl's static-shape vision graph
(`hbDNNInferV2`). Autoregressive decoding needs a KV-cache, dynamic sequence
length, a tokenizer, and a sampling loop. D-Robotics ships this as a distinct
runtime — `libxlm.so` (OE-LLM / LeapLLM) — that holds a prefill/decode
dual-graph internally. BLLM wraps that runtime's C API (`xlm.h`). Both stacks
share the same BPU driver and `hbUCP` memory system, so CV and LLM can co-exist
on one board, but they are two different runtimes at the API layer.

## Status

Two runtimes behind one library:

- **Native runtime (self-built, no libxlm)** — drives compiled `.hbm` graphs directly
  on the generic hbDNN/hbUCP BPU stack, with our own KV/SSM cache, sampler and loop.
  It runs the official text models (Qwen2.5-1.5B/**7B**, DeepSeek-R1-Distill-Qwen-1.5B,
  InternLM2) at ~libxlm throughput, **and the Qwen3.5-0.8B hybrid (Gated-DeltaNet +
  attention) that libxlm structurally cannot run** — strict **100 % on the BPU** at
  **~21 tok/s**. Wrapped in a one-line, string-in `bllm::NativeLlm` / `bllm.NativeSession`
  with a C++ tokenizer, ChatML, streaming, multi-turn and sampling.
- **libxlm runtime** — the original `bllm::LlmSession` over the OE-LLM SDK; validated on
  board (`Qwen2.5-1.5B` ~24 tok/s), plus `Omni`/`Vlm` multimodal sessions.

See [`docs/NATIVE_RUNTIME.md`](docs/NATIVE_RUNTIME.md) (native SE-track),
[`docs/LLM_ONBOARD.md`](docs/LLM_ONBOARD.md) (bring-up + ION/perf gotchas),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and [`docs/PLAN.md`](docs/PLAN.md) (roadmap).

## Usage — native runtime (recommended)

One model **directory** (a `.hbm` + `tokenizer.json` + a `model.json` manifest), one line:

```python
import bllm

llm = bllm.NativeSession("/path/to/qwen3.5-0.8b")   # model.json → arch/tokenizer/eos/chat
llm.set_sampling(temp=0.7, top_p=0.9)               # or leave greedy
for chunk in llm.stream_chat("用一句话介绍北京"):    # streaming, multi-turn
    print(chunk, end="", flush=True)
print(llm.last_decode_tps)
```

```cpp
#include "bllm/native_llm.h"

bllm::NativeLlm llm("/path/to/qwen3.5-0.8b");
std::string reply = llm.chat("你好", 400, [](const std::string& s){ std::cout << s << std::flush; });
```

REPL: [`examples/chat_native.cc`](examples/chat_native.cc) (`bllm_chat_native --model <dir>`).

## Usage — libxlm runtime

```cpp
#include "bllm/bllm.h"

bllm::SessionOptions o;
o.model_path         = "Qwen2.5_1.5B_Instruct_1024.hbm";
o.tokenizer_dir      = "Qwen2.5_1.5B_Instruct_config/";
o.chat_template_path = "Qwen2.5_1.5B_Instruct_config/Qwen2.5_1.5B_Instruct.jinja";
// model_type auto-detected from the path; set o.model_type to be explicit.

bllm::LlmSession llm(o);
std::string reply = llm.generate("你好", [](std::string_view t) {
  std::cout << t << std::flush;          // stream tokens as they decode
});
llm.reset();                              // start a fresh conversation
```

A ready REPL is in [`examples/chat.cc`](examples/chat.cc) (built as `bllm_chat`).

### Python

```python
import bllm

llm = bllm.LlmSession(
    "Qwen2.5_1.5B_Instruct_1024.hbm",
    "Qwen2.5_1.5B_Instruct_config/",      # model_type + template auto-detected
)
for chunk in llm.stream("你好"):           # streaming generator
    print(chunk, end="", flush=True)
print(llm.last_stats.decode_tps)
llm.reset()
```

Build the module with `scripts/board_build.sh --python` (needs `nanobind` in the
env), then `export PYTHONPATH=<repo>/python`. REPL: `python examples/chat.py`.

### Multimodal (Qwen2.5-Omni)

```python
omni = bllm.OmniSession(text_hbm, visual_hbm, audio_hbm, embed_bin, config_dir)
reply = omni.generate([
    bllm.Content.image("scene.jpg"),
    bllm.Content.text("这张图里有什么？"),
])                                        # also .video(...), .audio(...)
```

C++: `bllm::OmniSession` + `bllm::Content` (`bllm/omni_session.h`); CLI:
`bllm_omni`. Handles text / image / audio / video files (offline path).

For image VLMs (InternVL / Qwen-VL) there's `bllm::VlmSession` /
`bllm.VlmSession` (`vlm.describe("img.jpg", "...")`, CLI `bllm_vlm`) — API-ready,
but SDK 1.0.0 ships no such model yet, so it's compile/guard-validated only.

### Multimodal on the native runtime (no libxlm)

The self-built engine also does **text + image + audio + video**, driving the Omni
towers on plain hbDNN. Media may be file paths **or raw arrays** — `HxWx3` uint8
frames and 16 kHz mono float PCM — so a camera/microphone pipeline needs no file
round-trip:

```python
vlm = bllm.NativeVlmSession("/models/qwen2.5-omni-3b")   # model.json, arch="omni"
for chunk in vlm.stream_chat("描述这张图片。", ["bus.jpg"]):
    print(chunk, end="", flush=True)
print(vlm.last_ttft_ms, vlm.last_decode_tps)             # ttft includes the media encode

vlm.reset(); vlm.chat("这段音频里说了什么？", audios=["clip.wav"])
vlm.reset(); vlm.chat("描述这段视频。", videos=[bllm.load_video("clip.mp4")])
```

**Live capture.** Push frames and microphone PCM as they arrive; each 2-second chunk
is encoded into the KV cache while the camera is still running, so the question at the
end answers in a fraction of the offline TTFT (4151 → 914 ms on our test clip):

```python
vlm.stream_begin(fps=2.0)
while capturing:
    vlm.stream_frame(frame_hwc_uint8)     # from the camera
    vlm.stream_audio(pcm_float32)         # from the mic, 16 kHz mono
print(vlm.stream_ask("刚才发生了什么？"))
```

The KV window **is** the context, and it is the binding constraint. A frame-pair spans
two frames, so at 2 fps video costs `vision_tokens` per second and audio costs 25:
a 2048-slot model holds **8 s of video** with the stock 448px tower, **32 s** with a
224px re-export (see below), or ~80 s of audio alone. `vlm.context_left` tracks it,
and pushing past it raises rather than silently evicting (evicting the earliest tokens
turns these full-attention models into gibberish).

C++: `bllm::NativeVlm` (`bllm/native_vlm.h`); CLI: `bllm_vlm_native --model <dir>
[--image f] [--audio f.wav] [--video f.mp4] --prompt <text>`. The audio tower emits
50 tokens per 2 s; the vision tower's cost depends on the resolution it was compiled
at. Both are spliced into the decoder's embedding stream with 3-D **mrope**/TMRoPE
positions, and a video's soundtrack interleaves with its frames in 2-second chunks.

### Choosing a vision tower

The tower's square side is baked in at export time, and it sets the whole video
context budget. We re-export it offline; the runtime reads the resolution from the
graph, so a model directory just points `visual` at the tower you want. Measured on
S100P (image TTFT, and how much video fits in 2048 slots at 2 fps):

| tower | tokens / frame-pair | video reach | image TTFT | coarse Q&A | small text in the image |
|---|---|---|---|---|---|
| 448px (official) | 256 | 8 s | 871 ms | ✅ | ✅ reads "cero emisiones" |
| **336px (re-export)** | **144** | **14 s** | **455 ms** | ✅ | ✅ still reads it |
| 224px (re-export) | 64 | **32 s** | 381 ms | ✅ | ❌ hallucinates |

These resolutions are inside the model's training distribution (the processor's
`min_pixels` is 56×56), so coarse scene understanding, counting and video description
survive all the way down. What breaks is fine detail, and the cliff is between 336 and
224 — not at 448. **336 is the default worth reaching for**: 1.8× cheaper frames, image
TTFT nearly halved, and it still reads the text on the bus. Drop to 224 only when video
reach matters more than detail. Any side that is a multiple of 28 with `side/28`
divisible by 4 works. See [`docs/NATIVE_RUNTIME.md`](docs/NATIVE_RUNTIME.md) SE5/SE8.

## Sharing the BPU with a vision pipeline

S100/S100P have a single BPU core, so an LLM and a vision pipeline (bcdl) contend for
it. An LLM decode step is one big graph, and by default a co-resident detector waits
behind the whole thing. Fixing that takes **two** things — a compile flag on the LLM's
`.hbm` and a priority gap at runtime:

```python
llm = bllm.NativeSession("/models/qwen2.5-1.5b")
llm.set_bpu_priority(0)          # lowest — the default; let vision jump the queue
```

plus `oellm_build --max_time_per_fc 1000` (µs) when compiling the LLM, which caps each
function call so the scheduler can switch inside the graph. Measured on S100P with
yolo26s running against Qwen2.5-1.5B decode (p99 latency of the detector):

| LLM `.hbm` | detector priority | detector p99 | detector FPS | LLM tok/s |
|---|---|---|---|---|
| — (detector alone) | — | 2.06 ms | 563 | — |
| stock | same as LLM | 41.33 ms | 188 | 24.22 |
| stock | above LLM | 13.49 ms | 189 | 20.66 |
| `max_time_per_fc=1000` | same as LLM | 40.98 ms | 188 | 23.87 |
| **`max_time_per_fc=1000`** | **above LLM** | **2.95 ms** | **383** | **17.11** |

Neither knob works alone. The bound is nearly free on an idle BPU (24.00 vs 24.36
tok/s); the LLM only pays when it actually yields. See
[`docs/NATIVE_RUNTIME.md`](docs/NATIVE_RUNTIME.md) SE9.

## Supported models (official pre-compiled `.hbm`)

All official **text** LLMs run through one config-free path — BLLM infers the
`model_type`, auto-discovers the chat template, and applies each model's quirks
(int8 KV-cache, template-required) itself:

- **Qwen2.5** 1.5B / 7B (Base + Instruct) — validated
- **DeepSeek-R1-Distill-Qwen** 1.5B / 7B — validated
- **InternLM2-1.8B**

in q8/q4 × ctx 1024/4096 variants. **Multimodal** samples (Qwen2.5-Omni-3B,
InternVL/Qwen-VL) are guarded pending the M5 multimodal facade. Full matrix +
per-model notes: [`docs/MODELS.md`](docs/MODELS.md).

Beyond the official set, we also publish a growing list of **pre-compiled `.hbm`
for select community models** — e.g. **Microsoft Phi-4-mini** and **GLM-Edge-1.5B**,
both running on the S100P BPU. You download the finished `.hbm` package and run it;
the offline conversion pipeline is not part of this repo. Deployment guide:
[`host_toolchain/README.md`](host_toolchain/README.md).

## Build

Develop on a host, build & run on the board (the OE-LLM SDK exists only there):

```bash
scripts/sync.sh          # rsync working tree to the board (set BOARD ssh alias)
scripts/board_build.sh   # cmake + ninja on the board
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
