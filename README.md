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

The **`bllm::LlmSession` C++ wrapper is live and on-board validated** (M1). It
loads a `.hbm` + tokenizer, streams tokens, keeps multi-turn history, and
auto-detects the model type — running `Qwen2.5-1.5B-Instruct` on the S100P BPU at
**decode ~24 tok/s** (matching the official benchmark). The underlying path was
validated in M0; see [`docs/LLM_ONBOARD.md`](docs/LLM_ONBOARD.md) for the
bring-up, gotchas (ION carveout, performance-mode registers), and raw API notes.

Design + layering: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Roadmap
(async/batch, VLM, Python bindings, packaging): [`docs/PLAN.md`](docs/PLAN.md).

## Usage

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

## Supported models (official pre-compiled `.hbm`)

All official **text** LLMs run through one config-free path — BLLM infers the
`model_type`, auto-discovers the chat template, and applies each model's quirks
(int8 KV-cache, template-required) itself:

- **Qwen2.5** 1.5B / 7B (Base + Instruct) — validated
- **DeepSeek-R1-Distill-Qwen** 1.5B / 7B — validated
- **InternLM2-1.8B**

in q8/q4 × ctx 1024/4096 variants. **Multimodal** samples (Qwen2.5-Omni-3B,
InternVL/Qwen-VL) are guarded pending the M5 multimodal facade. Full matrix +
per-model notes: [`docs/MODELS.md`](docs/MODELS.md). Custom architectures are
converted offline with the `oellm_build` toolchain.

## Build

Develop on a host, build & run on the board (the OE-LLM SDK exists only there):

```bash
scripts/sync.sh          # rsync working tree to the board (set BOARD ssh alias)
scripts/board_build.sh   # cmake + ninja on the board
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
