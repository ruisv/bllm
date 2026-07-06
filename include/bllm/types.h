// BLLM — public value types: model kinds, sampling, options, stats.
//
// These are BLLM's own types, deliberately decoupled from `xlm.h` enums so the
// public API stays stable across OE-LLM SDK versions. The .cc layer maps them
// onto the runtime's `xlm_*` enums.

#ifndef BLLM_TYPES_H_
#define BLLM_TYPES_H_

#include <cstdint>
#include <string>
#include <string_view>

namespace bllm {

// Model family. Integer values mirror `xlm_model_type` for a cheap cast, but
// callers should use the names. `Auto` asks BLLM to infer from the model /
// tokenizer path (see ModelTypeFromPath); set it explicitly if inference fails.
enum class ModelType : int {
  Auto = -1,
  InternVL = 0,
  DeepSeek = 1,
  Qwen = 2,
  Llama = 3,  // not supported by the runtime yet
  InternLM2 = 4,
  Omni = 5,
  QwenVL = 6,
  Qwen2_5 = 7,
};

// Which BPU core the decode runs on. `Any` lets the runtime pick; pin a core
// (Bpu0..3) to co-exist with a bcdl vision pipeline on the other cores.
// Values mirror `xlm_infer_backend` (Any == XLM_INFER_BACKEND_BPU_ANY).
enum class Backend : int {
  Any = 1,
  Bpu0 = 2,
  Bpu1 = 3,
  Bpu2 = 4,
  Bpu3 = 5,
};

// Scheduling priority when multiple sessions share the BPU. Mirrors
// `xlm_priority_type_t`.
enum class Priority : int { Normal = 0, High = 1, Urgent = 2 };

// Image preprocessing for the image-VLM path. Mirrors
// `xlm_img_preprocess_type`. Dynamic = the model's native dynamic-resolution
// tiling (default); None = feed as-is.
enum class ImgPreprocess : int { Dynamic = 0, None = 1 };

// Sampling knobs. Defaults reproduce the official `oellm_run` demo settings
// (validated on S100P), so a caller who touches nothing gets known-good output.
struct SamplingParams {
  int32_t top_k = 50;             // <= 0 -> vocab size
  float top_p = 1.0f;             // 1.0 = disabled
  float min_p = 0.0f;             // 0.0 = disabled
  float temp = 1.0f;              // <= 0 -> greedy
  float typ_p = 1.0f;             // typical_p, 1.0 = disabled
  int32_t min_keep = 1;
  int32_t penalty_last_n = 128;
  float penalty_repeat = 1.2f;
  float penalty_freq = 0.1f;
  float penalty_present = 0.1f;
};

// Per-generation timing. BLLM measures these with a wall clock around the
// stream, because the runtime does not surface its internal perf counters
// through the callback. ttft_ms / decode_tps / tpot_ms / decode_tokens are
// populated; prefill_tps / prefill_tokens / vit_ms are left 0 (not measurable
// from the callback — the runtime prints its own prefill number to stdout).
struct GenerationStats {
  double ttft_ms = 0.0;           // time to first token
  double tpot_ms = 0.0;           // time per output (decode) token
  double prefill_tps = 0.0;       // not measured (see above)
  double decode_tps = 0.0;        // decode tokens/s (wall-clock estimate)
  int64_t prefill_tokens = 0;     // not measured
  int64_t decode_tokens = 0;      // number of streamed chunks
  double vit_ms = 0.0;            // not measured
};

// Everything needed to construct an LlmSession. Only model_path + tokenizer_dir
// are strictly required; the rest have working defaults or are auto-derived.
struct SessionOptions {
  std::string model_path;         // the compiled .hbm
  std::string tokenizer_dir;      // tokenizer.json (+ config) directory
  std::string config_path;        // model config file (required for InternVL)

  ModelType model_type = ModelType::Auto;
  int32_t context_size = 0;       // 0 -> runtime default (matches the .hbm)

  // Chat template: pass either the .jinja file path (loaded once at construction)
  // or its raw content. Required for Instruct/DeepSeek models; omit for Base.
  // If both are empty and auto_discover_template is true, BLLM looks for a single
  // *.jinja inside tokenizer_dir (matches the official config layout) — so for
  // Instruct models you can just point at the model + config dir and go, while
  // Base/InternLM2 dirs (no .jinja) correctly stay template-free.
  std::string chat_template_path;
  std::string chat_template;
  bool auto_discover_template = true;
  std::string system_prompt;      // applied to every turn if set

  // Force int8 KV-cache. Left as-is for most models; BLLM auto-enables it for
  // InternLM2 (the runtime requires it) unless you override here.
  bool k_cache_int8 = false;

  SamplingParams sampling;
  Backend backend = Backend::Any;
  Priority priority = Priority::Normal;
};

// Best-effort ModelType inference from a model/tokenizer path (case-insensitive
// substring match). Returns ModelType::Auto if nothing matches.
ModelType ModelTypeFromPath(std::string_view path);

// Parse a friendly name ("qwen2.5", "deepseek", "internlm2", ...) -> ModelType.
// Returns ModelType::Auto on no match.
ModelType ModelTypeFromString(std::string_view name);

// Human-readable name, e.g. for logging / errors.
const char* ToString(ModelType t);

}  // namespace bllm

#endif  // BLLM_TYPES_H_
