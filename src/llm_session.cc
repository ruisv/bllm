#include "bllm/llm_session.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "bllm/package.h"

#include "xlm_internal.h"  // shared: xlm.h + enum maps + Trampoline + stats

namespace bllm {
namespace {

using detail::CallContext;
using detail::Clock;
using detail::ComputeStats;
using detail::ReadFile;
using detail::ToXlm;
using detail::Trampoline;

// Per-model quirks the official demos / docs hardcode — encoded once here.
bool NeedsInt8KCache(ModelType t) { return t == ModelType::InternLM2; }
bool RequiresChatTemplate(ModelType t) { return t == ModelType::DeepSeek; }

// Multimodal families take image/audio/video through their own facades, not the
// text prompt path this class drives.
bool IsMultimodal(ModelType t) {
  return t == ModelType::Omni || t == ModelType::InternVL ||
         t == ModelType::QwenVL;
}
const char* MultimodalFacade(ModelType t) {
  return t == ModelType::Omni ? "bllm::OmniSession (bllm/omni_session.h)"
                              : "bllm::VlmSession (bllm/vlm_session.h)";
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

}  // namespace

// ── pimpl ───────────────────────────────────────────────────────────────────
struct LlmSession::Impl {
  SessionOptions opts;          // resolved (model_type/int8/template filled in)
  xlm_handle_t handle = nullptr;
  std::mutex mu;                // serialize infer on a single handle
  bool next_new_chat = true;    // first turn (and post-reset) starts fresh
  GenerationStats last_stats;
  std::vector<std::string> stop; // stop strings (trimmed post-hoc; libxlm can't cancel)

  // Backing storage kept alive for the handle's lifetime (xlm keeps pointers).
  std::string model_path, tokenizer_dir, config_path, chat_template, system_prompt;
};

LlmSession::LlmSession(SessionOptions options)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.opts = std::move(options);

  BLLM_CHECK(!s.opts.model_path.empty(), "SessionOptions.model_path is required");

  // Accept a package (.tar.gz / directory) as model_path — extract on first use
  // and fill in the tokenizer dir from it if the caller didn't set one.
  {
    ModelPackage pkg = ResolvePackage(s.opts.model_path);
    s.opts.model_path = pkg.hbm_path;
    if (s.opts.tokenizer_dir.empty() && !pkg.tokenizer_dir.empty())
      s.opts.tokenizer_dir = pkg.tokenizer_dir;
  }
  BLLM_CHECK(!s.opts.tokenizer_dir.empty(),
             "tokenizer_dir is required (or pass a package with a config/ dir)");

  // Resolve model type (Auto -> infer from path, else from the config.json arch
  // so a downloaded package "just works" without --model-type).
  if (s.opts.model_type == ModelType::Auto) {
    s.opts.model_type = ModelTypeFromPath(s.opts.model_path);
    if (s.opts.model_type == ModelType::Auto)
      s.opts.model_type = ModelTypeFromPath(s.opts.tokenizer_dir);
    if (s.opts.model_type == ModelType::Auto) {
      // Read <tokenizer_dir>/config.json; our converted packages are Qwen2-layout.
      std::ifstream cf(s.opts.tokenizer_dir + "/config.json");
      if (cf) {
        const std::string cj((std::istreambuf_iterator<char>(cf)),
                             std::istreambuf_iterator<char>());
        if (cj.find("Qwen2ForCausalLM") != std::string::npos ||
            cj.find("\"qwen2\"") != std::string::npos)
          s.opts.model_type = ModelType::Qwen2_5;
      }
    }
    BLLM_CHECK(s.opts.model_type != ModelType::Auto,
               "could not infer model_type; set it explicitly (e.g. qwen2.5)");
  }
  BLLM_CHECK(s.opts.model_type != ModelType::Llama,
             "llama model_type is not supported by the runtime yet");
  BLLM_CHECK(!IsMultimodal(s.opts.model_type),
             std::string("model_type '") + ToString(s.opts.model_type) +
                 "' is multimodal (image/audio/video); LlmSession is text-only. "
                 "Use " + MultimodalFacade(s.opts.model_type) + " instead.");

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
  ctx.stop = StopMatcher(s.stop);
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
  ctx.finish_stream();      // flush a held-back tail / trim the reply at a stop string

  s.next_new_chat = false;  // subsequent turns continue the conversation
  s.last_stats = ComputeStats(ctx);
  return std::move(ctx.accum);
}

void LlmSession::reset() { impl_->next_new_chat = true; }

void LlmSession::set_stop(std::vector<std::string> stop) {
  impl_->stop = std::move(stop);
}

const GenerationStats& LlmSession::last_stats() const {
  return impl_->last_stats;
}

const SessionOptions& LlmSession::options() const { return impl_->opts; }

}  // namespace bllm
