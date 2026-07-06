// BLLM — VlmSession: image + text vision-language chat (InternVL / Qwen-VL).
//
// These models take one or more images plus a text prompt through the runtime's
// XLM_INPUT_MULTI_MODAL path (xlm_infer), distinct from Qwen2.5-Omni's separate
// xlm_omni path (see OmniSession). Output is streamed text — same RAII +
// streaming ergonomics as LlmSession.
//
// NOTE: the OE-LLM S100 SDK 1.0.0 ships NO pre-compiled InternVL/Qwen-VL `.hbm`
// (only the demo code + this API are ready). This facade is validated by
// compilation; end-to-end board validation awaits a model (an official later
// release, or one converted offline with oellm_build). See docs/MODELS.md.
//
// The `xlm.h` handle is hidden behind a pimpl so this header needs no OE-LLM SDK.

#ifndef BLLM_VLM_SESSION_H_
#define BLLM_VLM_SESSION_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "bllm/common.h"
#include "bllm/types.h"

namespace bllm {

// One image input: a file path, or raw pixels + dimensions.
struct VlmImage {
  std::string path;                 // image file (with data == nullptr)
  const uint8_t* data = nullptr;    // raw pixels (alternative to path)
  int32_t width = 0;                // required when data != nullptr
  int32_t height = 0;
  ImgPreprocess preprocess = ImgPreprocess::Dynamic;

  static VlmImage File(std::string path) {
    VlmImage i;
    i.path = std::move(path);
    return i;
  }
  static VlmImage Pixels(const uint8_t* data, int32_t w, int32_t h) {
    VlmImage i;
    i.data = data;
    i.width = w;
    i.height = h;
    return i;
  }
};

struct VlmOptions {
  std::string model_path;      // the compiled VLM .hbm
  std::string tokenizer_dir;   // tokenizer config directory
  std::string config_path;     // model config file (required for InternVL)

  // InternVL (0) or QwenVL (6). Auto -> inferred from model_path.
  ModelType model_type = ModelType::Auto;
  int32_t context_size = 0;
  std::string system_prompt;

  SamplingParams sampling;
  Backend backend = Backend::Any;
  Priority priority = Priority::Normal;
};

class VlmSession {
 public:
  using TokenCallback = std::function<void(std::string_view chunk)>;

  explicit VlmSession(VlmOptions options);
  ~VlmSession();

  VlmSession(VlmSession&&) noexcept;
  VlmSession& operator=(VlmSession&&) noexcept;
  VlmSession(const VlmSession&) = delete;
  VlmSession& operator=(const VlmSession&) = delete;

  // Ask about one or more images. Blocks until end-of-stream; returns the reply.
  std::string generate(const std::vector<VlmImage>& images,
                       std::string_view prompt);
  std::string generate(const std::vector<VlmImage>& images,
                       std::string_view prompt, const TokenCallback& on_token);

  // Convenience: a single image by path.
  std::string describe(const std::string& image_path, std::string_view prompt);
  std::string describe(const std::string& image_path, std::string_view prompt,
                       const TokenCallback& on_token);

  void reset();
  const GenerationStats& last_stats() const;
  const VlmOptions& options() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bllm

#endif  // BLLM_VLM_SESSION_H_
