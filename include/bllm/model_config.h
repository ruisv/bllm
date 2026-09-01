// bllm::ModelConfig — the `model.json` manifest that makes a model self-describing:
// arch, backend, the .hbm + tokenizer + (hybrid) embed table, eos ids, cache_len,
// rope, and the chat template. This is the seam between offline conversion (which emits model.json) and the runtime (which loads a model in one call
// instead of guessing).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct ChatConfig {
  std::string format = "chatml";                 // "chatml" | "phi" | "none" (raw completion)
  int im_start = -1, im_end = -1;                // ChatML special-token ids
  int bos = -1;                                  // BOS id prepended once (e.g. InternLM2 <s>=1); -1 = none
  // Phi-style role markers (format=="phi"): each turn is `<role>content<end>`, no role
  // text and no newline — the marker itself IS the role. Generation ends on <end> (which
  // is also an eos), so a prior assistant turn is closed by re-emitting <end> next turn.
  int r_user = -1, r_assistant = -1, r_system = -1, r_end = -1;
  std::string system = "You are a helpful assistant.";
};

// The vision tower's own constants. These are per-checkpoint, not universal, and
// nothing about the compiled graph reveals them — a patch-14 preprocessor feeding a
// patch-16 graph produces correctly-SHAPED garbage. So the manifest carries them.
// Defaults are Qwen2.5-Omni / 2.5-VL (patch 14, CLIP normalization); Qwen3.5 sets
// patch 16 and mean = std = 0.5.
struct VisionCfg {
  int patch = 14, merge = 2, temporal = 2;
  float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
};

// π0.5 (openpi VLA). Three graphs rather than one, and a handful of constants
// that the .hbm cannot reveal: how many camera slots the prefix was compiled
// for, how many prompt slots, the flow-matching horizon and step count, and the
// dataset's action quantiles. `embed_prescaled` records that the shipped table
// already carries `embed_prefix`'s `sqrt(hidden)` — the runtime refuses a table
// that does not, because a prompt 45x too small is a policy that cannot read its
// own instruction and every cosine still looks perfect.
struct Pi05Cfg {
  std::string expert;                            // resolved path to the action-expert .hbm
  std::string cond_table;                        // resolved path to cond_table.f32
  int cameras = 2, prompt_len = 32;
  int horizon = 10, steps = 10;
  int vocab = 257152;
  int action_dim_real = 7;                       // dims the dataset actually uses
  bool embed_prescaled = false;
  std::vector<float> action_q01, action_q99;     // quantile unnormalisation
};

// SmolVLA (LeRobot VLA). Three graphs like π0.5's, and the same class of
// constants the .hbm cannot reveal — plus two it does not share: the state is a
// real prefix token here (π0.5's LIBERO config has none), so `state_proj` ships
// with the package; and normalisation is **MEAN_STD**, not quantiles. Using one
// where the other belongs is silent and wrong.
struct SmolVlaCfg {
  std::string expert;                            // resolved path to the action-expert .hbm
  std::string cond_table;                        // resolved path to cond_table.f32
  std::string state_proj;                        // resolved path to state_proj.f32
  int cameras = 2, prompt_len = 48;
  int horizon = 50, steps = 10;
  int vocab = 49280;
  int state_dim_real = 8;                        // dims the dataset actually uses
  int action_dim_real = 7;
  bool embed_prescaled = false;
  double rope_theta = 1e4;                       // `apply_rope`'s hardcoded default,
                                                 // NOT config.json's rope_theta
  std::vector<float> state_mean, state_std, action_mean, action_std;
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
  std::string hbm_prefill;                       // optional seq_len=N prefill graph (hybrid).
                                                 // Its absence is not an error: the engine
                                                 // falls back to one-token-at-a-time prefill.
  std::vector<int> eos;                          // stop-token ids
  std::vector<int> mrope_section;                // VLM: rope dim split across (t,h,w)
  // How that split is laid out across rope frequencies. Contiguous runs (Omni,
  // {16,24,24}) vs interleaved [THWTHW...] (Qwen3.5, {11,11,10}). Invisible on text
  // — t==h==w collapses both to plain rope — and wrong only once an image arrives.
  bool mrope_interleaved = false;
  VisionCfg vision;
  int cache_len = 0;                             // informational
  int bpu_cores = 1;                             // BPU cores the .hbm was compiled for (nash-p multi-core);
                                                 // >1 requires binding to specific cores at submit time
  double rope_theta = 1e7;                       // rope base
  double rope_theta_pi05 = 1e4;                  // PaliGemma/expert rope base
  ChatConfig chat;
  Pi05Cfg pi05;
  SmolVlaCfg smolvla;

  bool is_hybrid() const { return arch == "hybrid"; }
  bool is_omni() const { return arch == "omni"; }
  bool is_embed() const { return arch == "embed"; }
  bool is_pi05() const { return arch == "pi05"; }
  bool is_smolvla() const { return arch == "smolvla"; }
  // Multimodality is a property of the PACKAGE, not of the decoder family: "omni"
  // is a dense text tower with vision+audio, Qwen3.5 is a hybrid one with vision.
  // Anything carrying a vision tower can answer about images.
  bool has_vision() const { return !visual.empty(); }
};

// Which runtime drives a model.json package. There is only one now — the native hbDNN
// engine; the libxlm backend was removed after native proved a validated superset (see
// the design notes, tag v-libxlm-final). A leftover `backend: "libxlm"` in a manifest is
// a hard error so it's noticed, not silently reinterpreted.
inline std::string chooseBackend(const ModelConfig& c) {
  if (c.backend == "libxlm")
    throw std::runtime_error("[bllm] model.json requests backend=\"libxlm\", which was "
                             "removed — everything runs native now (recover it at tag "
                             "v-libxlm-final). Drop the backend field or set it to \"native\".");
  return "native";
}

// Load + validate `<dir>/model.json`; resolves relative paths against <dir>.
ModelConfig loadModelConfig(const std::string& dir);

}  // namespace bllm
