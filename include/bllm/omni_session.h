// BLLM — OmniSession: multimodal (text + image + audio + video) chat.
//
// Qwen2.5-Omni runs as three sub-models (text / visual / audio `.hbm`) plus an
// embed_tokens weight, loaded together with model_type = Omni. Input is a turn
// of content parts (text and/or media file paths); output is streamed text —
// same RAII + streaming ergonomics as LlmSession.
//
// This wraps the runtime's OFFLINE omni path (media supplied as files, via
// xlm_omni). The ONLINE path (raw nv12/PCM frame feeding) is not wrapped yet.
//
// The `xlm.h` handle is hidden behind a pimpl so this header needs no OE-LLM SDK.

#ifndef BLLM_OMNI_SESSION_H_
#define BLLM_OMNI_SESSION_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "bllm/common.h"
#include "bllm/types.h"

namespace bllm {

// One item in a user turn: literal text, or a path to an image / audio / video
// file. Build with the factories; the runtime infers modality from the kind.
struct Content {
  enum class Kind { Text, Image, Audio, Video };

  Kind kind = Kind::Text;
  std::string value;      // text, or media file path
  int width = 0;          // video resize target (0 -> default 448)
  int height = 0;

  static Content Text(std::string text) {
    return {Kind::Text, std::move(text), 0, 0};
  }
  static Content Image(std::string path, int w = 448, int h = 448) {
    return {Kind::Image, std::move(path), w, h};
  }
  static Content Audio(std::string path) {
    return {Kind::Audio, std::move(path), 0, 0};
  }
  static Content Video(std::string path, int w = 448, int h = 448) {
    return {Kind::Video, std::move(path), w, h};
  }
};

struct OmniOptions {
  std::string text_model_path;     // Qwen2.5_Omni_3B_Text.hbm
  std::string visual_model_path;   // Qwen2.5_Omni_3B_Visual.hbm
  std::string audio_model_path;    // Qwen2.5_Omni_3B_Audio.hbm
  std::string embed_tokens_path;   // embed_tokens.bin
  std::string tokenizer_dir;       // Qwen2.5_Omni_3B_config/

  ModelType model_type = ModelType::Omni;
  int32_t context_size = 0;        // 0 -> runtime default (Omni: 2048)

  // Default system prompt matches the official Omni demo; override as needed.
  std::string system_prompt =
      "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
      "capable of perceiving auditory and visual inputs, as well as generating "
      "text and speech.";

  SamplingParams sampling;
  Backend backend = Backend::Any;
  Priority priority = Priority::Normal;
};

class OmniSession {
 public:
  using TokenCallback = std::function<void(std::string_view chunk)>;

  // Loads the three sub-models + embed_tokens + tokenizer. Throws bllm::Error on
  // failure. Note: Omni is the heaviest model — ensure the ION carveout is sized
  // for it (see docs/LLM_ONBOARD.md).
  explicit OmniSession(OmniOptions options);
  ~OmniSession();

  OmniSession(OmniSession&&) noexcept;
  OmniSession& operator=(OmniSession&&) noexcept;
  OmniSession(const OmniSession&) = delete;
  OmniSession& operator=(const OmniSession&) = delete;

  // Generate a reply from a multimodal turn, blocking until end-of-stream.
  std::string generate(const std::vector<Content>& content);
  std::string generate(const std::vector<Content>& content,
                       const TokenCallback& on_token);

  // Convenience: a text-only turn.
  std::string ask(std::string_view text);
  std::string ask(std::string_view text, const TokenCallback& on_token);

  void reset();
  const GenerationStats& last_stats() const;
  const OmniOptions& options() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bllm

#endif  // BLLM_OMNI_SESSION_H_
