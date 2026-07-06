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

Early scaffold. The on-board LLM path is **validated**: the official OE-LLM SDK
runs `Qwen2.5-1.5B-Instruct` on the S100P BPU at **decode 24.5 tok/s / prefill
1969 tok/s** (matches the official benchmark). See
[`docs/LLM_ONBOARD.md`](docs/LLM_ONBOARD.md) for the full bring-up, reproduce
steps, gotchas (ION carveout, performance-mode registers), and API notes.

Next: wrap `libxlm` behind a `tasks/llm` C++ facade + Python bindings. Roadmap in
[`docs/PLAN.md`](docs/PLAN.md).

## Supported models (official pre-compiled `.hbm`)

DeepSeek-R1-Distill-Qwen 1.5B/7B · Qwen2.5 1.5B/7B (Base + Instruct) ·
InternLM2-1.8B · Qwen2.5-Omni-3B (multimodal), in q8/q4 × ctx 1024/4096 variants.
Custom architectures are converted offline with the `oellm_build` toolchain.

## Build

Develop on a host, build & run on the board (the OE-LLM SDK exists only there):

```bash
scripts/sync.sh          # rsync working tree to the board (set BOARD ssh alias)
scripts/board_build.sh   # cmake + ninja on the board
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
