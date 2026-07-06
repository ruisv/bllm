#include "bllm/vlm_session.h"

#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "xlm_internal.h"  // shared: xlm.h + enum maps + Trampoline + stats

namespace bllm {
namespace {

using detail::CallContext;
using detail::ComputeStats;
using detail::ToXlm;
using detail::Trampoline;

xlm_img_preprocess_type ToXlm(ImgPreprocess p) {
  return static_cast<xlm_img_preprocess_type>(static_cast<int>(p));
}

bool IsImageVlm(ModelType t) {
  return t == ModelType::InternVL || t == ModelType::QwenVL;
}
bool RequiresConfigPath(ModelType t) { return t == ModelType::InternVL; }

}  // namespace

struct VlmSession::Impl {
  VlmOptions opts;
  xlm_handle_t handle = nullptr;
  std::mutex mu;
  bool next_new_chat = true;
  GenerationStats last_stats;

  std::string model_path, tokenizer_dir, config_path, system_prompt;
};

VlmSession::VlmSession(VlmOptions options) : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.opts = std::move(options);

  BLLM_CHECK(!s.opts.model_path.empty(), "VlmOptions.model_path is required");
  BLLM_CHECK(!s.opts.tokenizer_dir.empty(),
             "VlmOptions.tokenizer_dir is required");

  if (s.opts.model_type == ModelType::Auto) {
    s.opts.model_type = ModelTypeFromPath(s.opts.model_path);
    if (s.opts.model_type == ModelType::Auto)
      s.opts.model_type = ModelTypeFromPath(s.opts.tokenizer_dir);
  }
  BLLM_CHECK(IsImageVlm(s.opts.model_type),
             std::string("VlmSession handles image VLMs (InternVL / Qwen-VL); "
                         "model_type '") +
                 ToString(s.opts.model_type) +
                 "' is not one. Use LlmSession (text) or OmniSession (omni).");
  if (RequiresConfigPath(s.opts.model_type))
    BLLM_CHECK(!s.opts.config_path.empty(),
               "InternVL requires config_path (the model config file)");

  s.model_path = s.opts.model_path;
  s.tokenizer_dir = s.opts.tokenizer_dir;
  s.config_path = s.opts.config_path;
  s.system_prompt = s.opts.system_prompt;

  xlm_common_params_t p = xlm_create_default_param();
  p.model_path = s.model_path.c_str();
  p.token_config_path = s.tokenizer_dir.c_str();
  if (!s.config_path.empty()) p.config_path = s.config_path.c_str();
  p.model_type = ToXlm(s.opts.model_type);
  if (s.opts.context_size > 0) p.context_size = s.opts.context_size;
  p.sampling = ToXlm(s.opts.sampling);

  int ret = xlm_init(&p, &Trampoline, &s.handle);
  BLLM_CHECK_XLM(ret, "xlm_init failed for VLM (check .hbm/tokenizer/config "
                      "paths and ION sizing — see docs/LLM_ONBOARD.md)");
  BLLM_CHECK(s.handle != nullptr, "xlm_init returned a null handle");
}

VlmSession::~VlmSession() {
  if (impl_ && impl_->handle) {
    xlm_destroy(&impl_->handle);
    impl_->handle = nullptr;
  }
}

VlmSession::VlmSession(VlmSession&&) noexcept = default;
VlmSession& VlmSession::operator=(VlmSession&&) noexcept = default;

std::string VlmSession::generate(const std::vector<VlmImage>& images,
                                 std::string_view prompt) {
  return generate(images, prompt, nullptr);
}

std::string VlmSession::generate(const std::vector<VlmImage>& images,
                                 std::string_view prompt,
                                 const TokenCallback& on_token) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  BLLM_CHECK(!images.empty(), "generate() needs at least one image");

  CallContext ctx;
  detail::TokenSink sink;
  if (on_token) sink = [&on_token](std::string_view c) { on_token(c); };
  ctx.on_token = on_token ? &sink : nullptr;
  ctx.t_start = detail::Clock::now();

  const std::string prompt_str(prompt);

  // Marshal the images into the C array (kept alive across the infer call).
  std::vector<xlm_input_image_t> ximgs(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    std::memset(&ximgs[i], 0, sizeof(xlm_input_image_t));
    const VlmImage& in = images[i];
    if (in.data) {
      ximgs[i].image_data = in.data;
      ximgs[i].image_width = in.width;
      ximgs[i].image_height = in.height;
    } else {
      BLLM_CHECK(!in.path.empty(), "VlmImage has neither path nor data");
      ximgs[i].image_path = in.path.c_str();
    }
    ximgs[i].image_preprocess = ToXlm(in.preprocess);
  }

  xlm_lm_request_t req;
  std::memset(&req, 0, sizeof(req));
  req.request_id = 0;
  req.type = XLM_INPUT_MULTI_MODAL;
  req.new_chat = s.next_new_chat;
  req.multi_modal_requset.prompt = prompt_str.c_str();
  req.multi_modal_requset.image_num = static_cast<int32_t>(ximgs.size());
  req.multi_modal_requset.images = ximgs.data();
  if (!s.system_prompt.empty()) req.system_prompt = s.system_prompt.c_str();
  req.infer_backend = ToXlm(s.opts.backend);
  req.priority.type = ToXlm(s.opts.priority);
  req.priority.priority = 0;

  xlm_input_t input;
  std::memset(&input, 0, sizeof(input));
  input.request_num = 1;
  input.requests = &req;

  int ret = xlm_infer(s.handle, &input, &ctx);
  BLLM_CHECK_XLM(ret, "xlm_infer failed (VLM)");
  BLLM_CHECK(!ctx.error, "runtime reported XLM_STATE_ERROR during generation");

  s.next_new_chat = false;
  s.last_stats = ComputeStats(ctx);
  return std::move(ctx.accum);
}

std::string VlmSession::describe(const std::string& image_path,
                                 std::string_view prompt) {
  return generate({VlmImage::File(image_path)}, prompt);
}
std::string VlmSession::describe(const std::string& image_path,
                                 std::string_view prompt,
                                 const TokenCallback& on_token) {
  return generate({VlmImage::File(image_path)}, prompt, on_token);
}

void VlmSession::reset() { impl_->next_new_chat = true; }
const GenerationStats& VlmSession::last_stats() const {
  return impl_->last_stats;
}
const VlmOptions& VlmSession::options() const { return impl_->opts; }

}  // namespace bllm
