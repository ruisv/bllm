#include "bllm/llm_session.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <utility>

#include "xlm.h"  // OE-LLM runtime C API (board SDK only; located by CMake)

namespace bllm {
namespace {

// ── enum mapping: BLLM public types -> xlm.h runtime enums ──────────────────
xlm_model_type ToXlm(ModelType t) {
  return static_cast<xlm_model_type>(static_cast<int>(t));
}
xlm_infer_backend ToXlm(Backend b) {
  return static_cast<xlm_infer_backend>(static_cast<int>(b));
}
xlm_priority_type_t ToXlm(Priority p) {
  return static_cast<xlm_priority_type_t>(static_cast<int>(p));
}

common_params_sampling_t ToXlm(const SamplingParams& s) {
  common_params_sampling_t o;
  std::memset(&o, 0, sizeof(o));
  o.top_k = s.top_k;
  o.top_p = s.top_p;
  o.min_p = s.min_p;
  o.temp = s.temp;
  o.typ_p = s.typ_p;
  o.min_keep = s.min_keep;
  o.penalty_last_n = s.penalty_last_n;
  o.penalty_repeat = s.penalty_repeat;
  o.penalty_freq = s.penalty_freq;
  o.penalty_present = s.penalty_present;
  return o;
}

// Per-model quirks the official demos / docs hardcode — encoded once here.
bool NeedsInt8KCache(ModelType t) { return t == ModelType::InternLM2; }
bool RequiresChatTemplate(ModelType t) { return t == ModelType::DeepSeek; }

// Multimodal families take image/audio/video through the xlm_omni*/multimodal
// input path, not the text prompt path this class drives. Guarded until M5.
bool IsMultimodal(ModelType t) {
  return t == ModelType::Omni || t == ModelType::InternVL ||
         t == ModelType::QwenVL;
}

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  BLLM_CHECK(f.good(), "cannot open chat template: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// The official config dirs hold exactly one *.jinja for chat models (Base /
// InternLM2 dirs hold none). Return that single template path, or "" if there
// isn't exactly one — callers treat "" as "run template-free (Base)".
std::string FindSingleJinja(const std::string& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return "";
  std::string found;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (ec) return "";
    if (e.is_regular_file(ec) && e.path().extension() == ".jinja") {
      if (!found.empty()) return "";  // ambiguous -> don't guess
      found = e.path().string();
    }
  }
  return found;
}

using Clock = std::chrono::steady_clock;

// Routed per-generate() through xlm's userdata. The C callback (registered once
// at init) casts userdata back to this and accumulates the stream.
//
// The runtime does NOT expose its perf counters via the callback
// (result->performance is always zero here — it prints them internally), so we
// measure timing ourselves with a wall clock: TTFT to the first chunk, decode
// rate over the rest of the stream.
struct CallContext {
  const LlmSession::TokenCallback* on_token = nullptr;
  std::string accum;
  bool error = false;

  Clock::time_point t_start;   // set in generate() before xlm_infer
  Clock::time_point t_first;   // first non-empty chunk
  Clock::time_point t_end;     // XLM_STATE_END
  int32_t chunk_count = 0;     // ~= decoded tokens (one chunk per token here)
  bool have_first = false;
};

// The single C callback for every session. `userdata` selects the live call.
void Trampoline(xlm_result_t* result, xlm_state_t state, void* userdata) {
  auto* ctx = static_cast<CallContext*>(userdata);
  if (!ctx) return;
  switch (state) {
    case XLM_STATE_START:
    case XLM_STATE_RUNNING:
      if (result && result->text && result->text[0] != '\0') {
        if (!ctx->have_first) {
          ctx->have_first = true;
          ctx->t_first = Clock::now();
        }
        ++ctx->chunk_count;
        ctx->accum += result->text;
        if (ctx->on_token && *ctx->on_token) (*ctx->on_token)(result->text);
      }
      break;
    case XLM_STATE_END:
      ctx->t_end = Clock::now();
      break;
    case XLM_STATE_ERROR:
      ctx->error = true;
      ctx->t_end = Clock::now();
      break;
  }
}

// Derive wall-clock stats from a finished CallContext.
GenerationStats ComputeStats(const CallContext& ctx) {
  GenerationStats s;
  if (!ctx.have_first) return s;
  auto ms = [](Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  s.ttft_ms = ms(ctx.t_first - ctx.t_start);
  s.decode_tokens = ctx.chunk_count;
  const double decode_ms = ms(ctx.t_end - ctx.t_first);
  if (ctx.chunk_count > 1 && decode_ms > 0.0) {
    // First token is attributed to prefill (TTFT); the rest to decode.
    s.decode_tps = (ctx.chunk_count - 1) * 1000.0 / decode_ms;
    s.tpot_ms = decode_ms / (ctx.chunk_count - 1);
  }
  return s;
}

}  // namespace

// ── pimpl ───────────────────────────────────────────────────────────────────
struct LlmSession::Impl {
  SessionOptions opts;          // resolved (model_type/int8/template filled in)
  xlm_handle_t handle = nullptr;
  std::mutex mu;                // serialize infer on a single handle
  bool next_new_chat = true;    // first turn (and post-reset) starts fresh
  GenerationStats last_stats;

  // Backing storage kept alive for the handle's lifetime (xlm keeps pointers).
  std::string model_path, tokenizer_dir, config_path, chat_template, system_prompt;
};

LlmSession::LlmSession(SessionOptions options)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.opts = std::move(options);

  BLLM_CHECK(!s.opts.model_path.empty(), "SessionOptions.model_path is required");
  BLLM_CHECK(!s.opts.tokenizer_dir.empty(),
             "SessionOptions.tokenizer_dir is required");

  // Resolve model type (Auto -> infer from path).
  if (s.opts.model_type == ModelType::Auto) {
    s.opts.model_type = ModelTypeFromPath(s.opts.model_path);
    if (s.opts.model_type == ModelType::Auto)
      s.opts.model_type = ModelTypeFromPath(s.opts.tokenizer_dir);
    BLLM_CHECK(s.opts.model_type != ModelType::Auto,
               "could not infer model_type from paths; set it explicitly");
  }
  BLLM_CHECK(s.opts.model_type != ModelType::Llama,
             "llama model_type is not supported by the runtime yet");
  BLLM_CHECK(!IsMultimodal(s.opts.model_type),
             std::string("model_type '") + ToString(s.opts.model_type) +
                 "' is multimodal (image/audio/video); LlmSession is text-only. "
                 "The multimodal API lands in M5 — see docs/PLAN.md");

  // Per-model default: InternLM2 requires int8 KV-cache.
  if (NeedsInt8KCache(s.opts.model_type)) s.opts.k_cache_int8 = true;

  // Resolve chat template: explicit content > explicit path > auto-discovered
  // single *.jinja in tokenizer_dir (Base/InternLM2 dirs have none -> stays
  // template-free). Then enforce where the runtime requires one.
  if (s.opts.chat_template.empty() && s.opts.chat_template_path.empty() &&
      s.opts.auto_discover_template) {
    s.opts.chat_template_path = FindSingleJinja(s.opts.tokenizer_dir);
  }
  if (s.opts.chat_template.empty() && !s.opts.chat_template_path.empty())
    s.opts.chat_template = ReadFile(s.opts.chat_template_path);
  if (RequiresChatTemplate(s.opts.model_type))
    BLLM_CHECK(!s.opts.chat_template.empty(),
               "this model_type requires a chat template "
               "(none found in tokenizer_dir; set chat_template_path)");

  // Stash backing strings, then point xlm params at them.
  s.model_path = s.opts.model_path;
  s.tokenizer_dir = s.opts.tokenizer_dir;
  s.config_path = s.opts.config_path;
  s.chat_template = s.opts.chat_template;
  s.system_prompt = s.opts.system_prompt;

  xlm_common_params_t p = xlm_create_default_param();
  p.model_path = s.model_path.c_str();
  p.token_config_path = s.tokenizer_dir.c_str();
  if (!s.config_path.empty()) p.config_path = s.config_path.c_str();
  p.model_type = ToXlm(s.opts.model_type);
  if (s.opts.context_size > 0) p.context_size = s.opts.context_size;
  p.k_cache_int8 = s.opts.k_cache_int8;
  p.sampling = ToXlm(s.opts.sampling);

  int ret = xlm_init(&p, &Trampoline, &s.handle);
  BLLM_CHECK_XLM(ret, "xlm_init failed (check .hbm/tokenizer paths and ION "
                      "carveout — see docs/LLM_ONBOARD.md)");
  BLLM_CHECK(s.handle != nullptr, "xlm_init returned a null handle");
}

LlmSession::~LlmSession() {
  if (impl_ && impl_->handle) {
    xlm_destroy(&impl_->handle);
    impl_->handle = nullptr;
  }
}

LlmSession::LlmSession(LlmSession&&) noexcept = default;
LlmSession& LlmSession::operator=(LlmSession&&) noexcept = default;

std::string LlmSession::generate(std::string_view prompt) {
  return generate(prompt, nullptr);
}

std::string LlmSession::generate(std::string_view prompt,
                                 const TokenCallback& on_token) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);

  CallContext ctx;
  ctx.on_token = on_token ? &on_token : nullptr;
  ctx.t_start = Clock::now();

  const std::string prompt_str(prompt);

  xlm_lm_request_t req;
  std::memset(&req, 0, sizeof(req));
  req.request_id = 0;
  req.type = XLM_INPUT_PROMPT;
  req.new_chat = s.next_new_chat;
  req.prompt = prompt_str.c_str();
  if (!s.system_prompt.empty()) req.system_prompt = s.system_prompt.c_str();
  if (!s.chat_template.empty()) req.chat_template = s.chat_template.c_str();
  req.infer_backend = ToXlm(s.opts.backend);
  req.priority.type = ToXlm(s.opts.priority);
  req.priority.priority = 0;

  xlm_input_t input;
  std::memset(&input, 0, sizeof(input));
  input.request_num = 1;
  input.requests = &req;

  int ret = xlm_infer(s.handle, &input, &ctx);
  BLLM_CHECK_XLM(ret, "xlm_infer failed");
  BLLM_CHECK(!ctx.error, "runtime reported XLM_STATE_ERROR during generation");

  s.next_new_chat = false;  // subsequent turns continue the conversation
  s.last_stats = ComputeStats(ctx);
  return std::move(ctx.accum);
}

void LlmSession::reset() { impl_->next_new_chat = true; }

const GenerationStats& LlmSession::last_stats() const {
  return impl_->last_stats;
}

const SessionOptions& LlmSession::options() const { return impl_->opts; }

}  // namespace bllm
