#include "bllm/omni_session.h"

#include <unistd.h>  // getpid

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <utility>

#include "xlm_internal.h"  // shared: xlm.h + enum maps + Trampoline + stats

namespace bllm {
namespace {

using detail::CallContext;
using detail::ComputeStats;
using detail::ToXlm;
using detail::Trampoline;

// Escape a string for embedding as a JSON string value.
std::string EscapeJson(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

// Build the runtime's `conversation` JSON from a system prompt + content parts.
std::string BuildConversationJson(const std::string& system_prompt,
                                  const std::vector<Content>& parts) {
  std::ostringstream j;
  j << "{\"conversation\":[";
  if (!system_prompt.empty()) {
    j << "{\"role\":\"system\",\"content\":[{\"type\":\"text\",\"text\":\""
      << EscapeJson(system_prompt) << "\"}]},";
  }
  j << "{\"role\":\"user\",\"content\":[";
  for (size_t i = 0; i < parts.size(); ++i) {
    const Content& c = parts[i];
    if (i) j << ',';
    switch (c.kind) {
      case Content::Kind::Text:
        j << "{\"type\":\"text\",\"text\":\"" << EscapeJson(c.value) << "\"}";
        break;
      case Content::Kind::Image:
        j << "{\"type\":\"image\",\"image\":\"" << EscapeJson(c.value)
          << "\",\"resized_width\":" << (c.width > 0 ? c.width : 448)
          << ",\"resized_height\":" << (c.height > 0 ? c.height : 448) << "}";
        break;
      case Content::Kind::Audio:
        j << "{\"type\":\"audio\",\"audio\":\"" << EscapeJson(c.value) << "\"}";
        break;
      case Content::Kind::Video:
        j << "{\"type\":\"video\",\"video\":\"" << EscapeJson(c.value)
          << "\",\"resized_width\":" << (c.width > 0 ? c.width : 448)
          << ",\"resized_height\":" << (c.height > 0 ? c.height : 448) << "}";
        break;
    }
  }
  j << "]}]}";
  return j.str();
}

// The runtime's `prompt_json` field is a PATH to a JSON file, not inline JSON —
// so we materialize the conversation to a unique temp file per turn.
std::filesystem::path WriteTempJson(const std::string& json) {
  namespace fs = std::filesystem;
  static std::atomic<uint64_t> seq{0};
  fs::path p = fs::temp_directory_path() /
               ("bllm_omni_" + std::to_string(::getpid()) + "_" +
                std::to_string(seq.fetch_add(1)) + ".json");
  std::ofstream o(p, std::ios::binary | std::ios::trunc);
  BLLM_CHECK(o.good(), "cannot write temp prompt json: " + p.string());
  o << json;
  o.close();
  BLLM_CHECK(o.good(), "failed writing temp prompt json: " + p.string());
  return p;
}

}  // namespace

struct OmniSession::Impl {
  OmniOptions opts;
  xlm_handle_t handle = nullptr;
  std::mutex mu;
  bool next_new_chat = true;
  GenerationStats last_stats;

  // Backing storage kept alive for the handle's lifetime.
  std::string text_path, visual_path, audio_path, embed_path, tokenizer_dir,
      system_prompt;
};

OmniSession::OmniSession(OmniOptions options)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.opts = std::move(options);

  BLLM_CHECK(!s.opts.text_model_path.empty(), "text_model_path is required");
  BLLM_CHECK(!s.opts.visual_model_path.empty(), "visual_model_path is required");
  BLLM_CHECK(!s.opts.audio_model_path.empty(), "audio_model_path is required");
  BLLM_CHECK(!s.opts.embed_tokens_path.empty(), "embed_tokens_path is required");
  BLLM_CHECK(!s.opts.tokenizer_dir.empty(), "tokenizer_dir is required");

  s.text_path = s.opts.text_model_path;
  s.visual_path = s.opts.visual_model_path;
  s.audio_path = s.opts.audio_model_path;
  s.embed_path = s.opts.embed_tokens_path;
  s.tokenizer_dir = s.opts.tokenizer_dir;
  s.system_prompt = s.opts.system_prompt;

  xlm_common_params_t p = xlm_create_default_param();
  p.omni_text_model_path = s.text_path.c_str();
  p.omni_visual_model_path = s.visual_path.c_str();
  p.omni_audio_model_path = s.audio_path.c_str();
  p.embed_tokens = s.embed_path.c_str();
  p.token_config_path = s.tokenizer_dir.c_str();
  p.model_type = ToXlm(s.opts.model_type);
  p.omni_online_mode = false;  // offline (file-based) path
  if (s.opts.context_size > 0) p.context_size = s.opts.context_size;
  p.sampling = ToXlm(s.opts.sampling);

  int ret = xlm_init(&p, &Trampoline, &s.handle);
  BLLM_CHECK_XLM(ret, "xlm_init failed for Omni (check the 4 model paths + "
                      "tokenizer, and ION sizing — see docs/LLM_ONBOARD.md)");
  BLLM_CHECK(s.handle != nullptr, "xlm_init returned a null handle");
}

OmniSession::~OmniSession() {
  if (impl_ && impl_->handle) {
    xlm_destroy(&impl_->handle);
    impl_->handle = nullptr;
  }
}

OmniSession::OmniSession(OmniSession&&) noexcept = default;
OmniSession& OmniSession::operator=(OmniSession&&) noexcept = default;

std::string OmniSession::generate(const std::vector<Content>& content) {
  return generate(content, nullptr);
}

std::string OmniSession::generate(const std::vector<Content>& content,
                                  const TokenCallback& on_token) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  BLLM_CHECK(!content.empty(), "generate() needs at least one Content part");

  CallContext ctx;
  detail::TokenSink sink;
  if (on_token) sink = [&on_token](std::string_view c) { on_token(c); };
  ctx.on_token = on_token ? &sink : nullptr;
  ctx.t_start = detail::Clock::now();

  const std::string json = BuildConversationJson(s.system_prompt, content);
  const std::filesystem::path json_path = WriteTempJson(json);
  const std::string json_path_str = json_path.string();

  xlm_lm_request_t req;
  std::memset(&req, 0, sizeof(req));
  req.request_id = 0;
  req.type = XLM_INPUT_PROMPT;
  req.new_chat = s.next_new_chat;
  req.prompt_json = json_path_str.c_str();  // a PATH, not inline JSON
  req.infer_backend = ToXlm(s.opts.backend);
  req.priority.type = ToXlm(s.opts.priority);
  req.priority.priority = 0;

  xlm_input_t input;
  std::memset(&input, 0, sizeof(input));
  input.request_num = 1;
  input.requests = &req;

  int ret = xlm_omni(s.handle, &input, &ctx);
  std::error_code rm_ec;
  std::filesystem::remove(json_path, rm_ec);  // always clean up the temp file
  BLLM_CHECK_XLM(ret, "xlm_omni failed");
  BLLM_CHECK(!ctx.error, "runtime reported XLM_STATE_ERROR during generation");

  s.next_new_chat = false;
  s.last_stats = ComputeStats(ctx);
  return std::move(ctx.accum);
}

std::string OmniSession::ask(std::string_view text) {
  return generate({Content::Text(std::string(text))});
}
std::string OmniSession::ask(std::string_view text,
                             const TokenCallback& on_token) {
  return generate({Content::Text(std::string(text))}, on_token);
}

void OmniSession::reset() { impl_->next_new_chat = true; }
const GenerationStats& OmniSession::last_stats() const {
  return impl_->last_stats;
}
const OmniOptions& OmniSession::options() const { return impl_->opts; }

}  // namespace bllm
