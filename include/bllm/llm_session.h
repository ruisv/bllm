// BLLM — LlmSession: the one deep module callers touch.
//
// RAII wrapper over an OE-LLM `xlm_handle_t`: construct with a model + tokenizer,
// call generate(), destruct to release the BPU. Multi-turn history is kept by
// the runtime; this class just drives the `new_chat` flag (reset() starts over).
//
// Thread-safety: one generate() runs at a time per session (serialized with an
// internal mutex). Use separate sessions — or pin different Backend cores — for
// concurrency.
//
// The `xlm.h` handle is hidden behind a pimpl so this header needs no OE-LLM SDK.

#ifndef BLLM_LLM_SESSION_H_
#define BLLM_LLM_SESSION_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "bllm/common.h"
#include "bllm/types.h"

namespace bllm {

class LlmSession {
 public:
  // Called once per streamed chunk with the newly decoded text (UTF-8). The
  // full reply is also returned by generate(), so a callback is optional.
  using TokenCallback = std::function<void(std::string_view chunk)>;

  // Loads the model + tokenizer and initializes the BPU runtime. Throws
  // bllm::Error on any failure (bad path, alloc failure, unsupported type).
  explicit LlmSession(SessionOptions options);
  ~LlmSession();

  LlmSession(LlmSession&&) noexcept;
  LlmSession& operator=(LlmSession&&) noexcept;
  LlmSession(const LlmSession&) = delete;
  LlmSession& operator=(const LlmSession&) = delete;

  // Generate a reply, blocking until the runtime signals end-of-stream. The
  // conversation continues from prior turns unless reset() was called (or this
  // is the first turn). Returns the full decoded reply.
  std::string generate(std::string_view prompt);

  // Streaming form: on_token fires for each chunk as it decodes; the full reply
  // is still returned.
  std::string generate(std::string_view prompt, const TokenCallback& on_token);

  // Drop the runtime-side KV-cache / history so the next generate() starts a
  // fresh conversation.
  void reset();

  // Timing of the most recent generate() (TTFT, decode tok/s, ...).
  const GenerationStats& last_stats() const;

  // The resolved options this session was built with (model_type after Auto
  // inference, k_cache_int8 after per-model defaults, etc.).
  const SessionOptions& options() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bllm

#endif  // BLLM_LLM_SESSION_H_
