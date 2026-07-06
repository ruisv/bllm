// BLLM — BPU LLM runtime for RDK S100. Public umbrella header.
//
// BLLM wraps the D-Robotics OE-LLM runtime (libxlm.so) behind a task-style
// C++ API, mirroring the bcdl / ccdl family. Autoregressive LLM decode runs on
// libxlm's prefill/decode dual-graph + KV-cache — NOT bcdl's hbDNNInferV2 path.
//
// Status: scaffold. The LlmSession wrapper lands in M1 (see docs/PLAN.md).

#ifndef BLLM_BLLM_H_
#define BLLM_BLLM_H_

#define BLLM_VERSION_MAJOR 0
#define BLLM_VERSION_MINOR 1
#define BLLM_VERSION_PATCH 0

namespace bllm {

constexpr const char* kVersion = "0.1.0";

// TODO(M1): LlmSession — RAII wrapper over xlm_handle_t.
//   - load(.hbm + tokenizer_dir + chat template + model_type)
//   - chat(prompt) -> streaming tokens via callback
//   - reset() / stop()
// See docs/LLM_ONBOARD.md for the xlm.h C API surface this wraps.

}  // namespace bllm

#endif  // BLLM_BLLM_H_
