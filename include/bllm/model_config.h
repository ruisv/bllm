// bllm::ModelConfig — the `model.json` manifest that makes a model self-describing:
// arch, backend, the .hbm + tokenizer + (hybrid) embed table, eos ids, cache_len,
// rope, and the chat template. This is the seam between the host_toolchain (which
// emits model.json at convert time) and the runtime (which loads a model in one call
// instead of guessing). See docs/PLAN.md (usability, Phase 1).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct ChatConfig {
  std::string format = "chatml";                 // "chatml" | "none" (raw completion)
  int im_start = -1, im_end = -1;                // ChatML special-token ids
  int bos = -1;                                  // BOS id prepended once (e.g. InternLM2 <s>=1); -1 = none
  std::string system = "You are a helpful assistant.";
};

struct ModelConfig {
  std::string dir;                               // model directory (paths below resolve against it)
  std::string name;
  std::string backend = "auto";                  // "native" | "libxlm" | "auto"
  std::string arch;                              // "dense" | "hybrid" (SSM) | "omni" (VLM) | "embed" (encoder)
  std::string hbm;                               // resolved absolute path to the .hbm (text tower for omni)
  std::string tokenizer;                         // resolved path to tokenizer.json
  std::string embed;                             // resolved path to the host embed table (hybrid + omni)
  std::string visual;                            // resolved path to the vision-tower .hbm (omni)
  std::string audio;                             // resolved path to the audio-tower .hbm (omni)
  std::string mel_filters;                       // resolved path to mel_filters_t.txt (omni audio)
  std::string graph = "qwen35";                  // graph name inside the .hbm (hybrid)
  std::vector<int> eos;                          // stop-token ids
  std::vector<int> mrope_section;                // omni: rope dim split across (t,h,w)
  int cache_len = 0;                             // informational
  double rope_theta = 1e7;                       // rope base
  ChatConfig chat;

  bool is_hybrid() const { return arch == "hybrid"; }
  bool is_omni() const { return arch == "omni"; }
  bool is_embed() const { return arch == "embed"; }
};

// Which runtime should drive a model.json package: "native" or "libxlm".
//
// The decision is not a preference — it is a capability boundary. libxlm's op set cannot
// express the hybrid Gated-DeltaNet/SSM arch (Qwen3.5), and the SDK gives it no multimodal
// or encoder entry point that consumes a host-owned embedding stream, so hybrid / omni /
// embed can ONLY run native. A dense model can run on either (native reached libxlm
// throughput — see NATIVE_RUNTIME.md SE6), so there the manifest's `backend` field decides,
// defaulting to native: it is one code path, stops early on stop strings, and carries the
// CV-coexistence knobs. Set `backend: "libxlm"` in model.json to pin a dense model to the
// proven runtime. `throw_on_conflict` turns an impossible request (backend "libxlm" on a
// non-dense arch) into an error instead of silently overriding it.
inline std::string chooseBackend(const ModelConfig& c, bool throw_on_conflict = true) {
  const bool dense = !c.is_hybrid() && !c.is_omni() && !c.is_embed();
  if (!dense) {
    if (throw_on_conflict && c.backend == "libxlm")
      throw std::runtime_error("[bllm] model.json requests backend=\"libxlm\" but arch=\"" +
                               c.arch + "\" only runs on the native runtime");
    return "native";
  }
  return c.backend == "libxlm" ? "libxlm" : "native";   // "native" | "auto" -> native
}

// Load + validate `<dir>/model.json`; resolves relative paths against <dir>.
ModelConfig loadModelConfig(const std::string& dir);

}  // namespace bllm
