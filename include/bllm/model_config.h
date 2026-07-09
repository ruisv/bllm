// bllm::ModelConfig — the `model.json` manifest that makes a model self-describing:
// arch, backend, the .hbm + tokenizer + (hybrid) embed table, eos ids, cache_len,
// rope, and the chat template. This is the seam between the host_toolchain (which
// emits model.json at convert time) and the runtime (which loads a model in one call
// instead of guessing). See docs/PLAN.md (usability, Phase 1).
#pragma once

#include <string>
#include <vector>

namespace bllm {

struct ChatConfig {
  std::string format = "chatml";                 // "chatml" | "none" (raw completion)
  int im_start = -1, im_end = -1;                // ChatML special-token ids
  std::string system = "You are a helpful assistant.";
};

struct ModelConfig {
  std::string dir;                               // model directory (paths below resolve against it)
  std::string name;
  std::string backend = "auto";                  // "native" | "libxlm" | "auto"
  std::string arch;                              // "hybrid" (Qwen3.5) | "omni" (VLM) | "dense" (KV models)
  std::string hbm;                               // resolved absolute path to the .hbm (text tower for omni)
  std::string tokenizer;                         // resolved path to tokenizer.json
  std::string embed;                             // resolved path to the host embed table (hybrid + omni)
  std::string visual;                            // resolved path to the vision-tower .hbm (omni)
  std::string audio;                             // resolved path to the audio-tower .hbm (omni)
  std::string graph = "qwen35";                  // graph name inside the .hbm (hybrid)
  std::vector<int> eos;                          // stop-token ids
  std::vector<int> mrope_section;                // omni: rope dim split across (t,h,w)
  int cache_len = 0;                             // informational
  double rope_theta = 1e7;                       // rope base
  ChatConfig chat;

  bool is_hybrid() const { return arch == "hybrid"; }
  bool is_omni() const { return arch == "omni"; }
};

// Load + validate `<dir>/model.json`; resolves relative paths against <dir>.
ModelConfig loadModelConfig(const std::string& dir);

}  // namespace bllm
