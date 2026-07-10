// bllm::NativeLlm — the unified, string-in/string-out session over the self-built
// native engines (no libxlm). Loads a model directory (model.json), builds the right
// engine (hybrid Gated-DeltaNet for Qwen3.5, or dense KV for Qwen2.5/DeepSeek/…),
// holds a C++ tokenizer, and exposes generate()/chat()/reset() on STRINGS with
// streaming. This is the "one object, one line" surface (roadmap Phase 1, usability).
//
//   bllm::NativeLlm llm("/path/to/model_dir");
//   std::string reply = llm.chat("你好", 256, [](const std::string& s){ std::cout << s; });
#pragma once

#include "bllm/model_config.h"
#include "bllm/native_engine.h"
#include "bllm/native_hybrid_engine.h"
#include "bllm/stop_match.h"
#include "bllm/tokenizer.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

class NativeLlm {
 public:
  using OnText = std::function<void(const std::string&)>;

  explicit NativeLlm(const std::string& model_dir) : NativeLlm(loadModelConfig(model_dir)) {}
  explicit NativeLlm(ModelConfig cfg)
      : cfg_(std::move(cfg)), tk_(Tokenizer::fromFile(cfg_.tokenizer)) {
    if (cfg_.is_omni())
      throw std::runtime_error("[bllm] '" + cfg_.name + "' is arch=\"omni\" (multimodal); "
                               "load it with bllm::NativeVlm / bllm.NativeVlmSession");
    if (cfg_.is_embed())
      throw std::runtime_error("[bllm] '" + cfg_.name + "' is arch=\"embed\" (an encoder, not a "
                               "generator); load it with bllm::NativeEmbedder");
    if (cfg_.is_hybrid())
      hybrid_ = std::make_unique<NativeHybridEngine>(cfg_.hbm, cfg_.embed, cfg_.graph, cfg_.rope_theta);
    else
      dense_ = std::make_unique<NativeEngine>(cfg_.hbm);
  }

  const ModelConfig& config() const { return cfg_; }
  const Tokenizer& tokenizer() const { return tk_; }

  // Sampling controls (default greedy). Applied to both dense and hybrid backends.
  void set_sampling(float temp, float top_p, int top_k, float rep_pen, uint64_t seed) {
    sampling_.temp = temp; sampling_.top_p = top_p; sampling_.top_k = top_k;
    sampling_.rep_pen = rep_pen; sampling_.seed = seed;
  }

  // Stop strings: end a turn as soon as any of them appears in the decoded text, and
  // return the reply WITHOUT the stop string. Complements eos-token stopping (a stop
  // string can span tokens, and need not be a token). Empty clears them. Persists across
  // turns; a per-call `stop` argument overrides for that call only.
  void set_stop(std::vector<std::string> stops) { stop_ = std::move(stops); }
  // BPU queue priority (0..255). Lowest (the default) lets a co-resident vision
  // pipeline jump the queue. Note: it orders the queue, it does not preempt a graph
  // that is already running — see native_detail::BpuSched.
  void set_bpu_priority(int priority) {
    native_detail::BpuSched s;
    s.priority = priority;
    if (hybrid_) hybrid_->set_sched(s); else dense_->set_sched(s);
  }

  // Tokens that still fit in this conversation's KV window.
  int context_left() const { return hybrid_ ? hybrid_->context_left() : dense_->context_left(); }

  double last_decode_tps() const {
    return hybrid_ ? hybrid_->last_stats().decode_tps : dense_->last_stats().decode_tps;
  }

  // Fresh conversation / context.
  void reset() {
    if (hybrid_) hybrid_->reset(); else dense_->reset();
    first_turn_ = true;
  }

  // Raw completion: continue `prompt`. Streams decoded text to on_text (delta) if set.
  std::string generate(const std::string& prompt, int max_new = 256, const OnText& on_text = {},
                       const std::vector<std::string>& stop = {}) {
    feedIds(tk_.encode(prompt));
    return streamDecode(max_new, on_text, stop);
  }

  // ChatML multi-turn: context persists across calls (feed only the new turn).
  std::string chat(const std::string& msg, int max_new = 256, const OnText& on_text = {},
                   const std::vector<std::string>& stop = {}) {
    if (cfg_.chat.format != "chatml" || cfg_.chat.im_start < 0)
      return generate(msg, max_new, on_text, stop);   // no template → raw
    const int S = cfg_.chat.im_start, E = cfg_.chat.im_end;
    std::vector<int> ids;
    auto add = [&](const std::vector<int>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    if (first_turn_) { add({S}); add(tk_.encode("system\n" + cfg_.chat.system)); add({E}); }
    else             { add({E}); }               // close the previous assistant turn
    add({S}); add(tk_.encode("user\n" + msg)); add({E});
    add({S}); add(tk_.encode("assistant\n"));
    first_turn_ = false;
    feedIds(ids);
    return streamDecode(max_new, on_text, stop);
  }

 private:
  // Ingest prompt/turn tokens into the engine (leaving the last logits for sampling).
  void feedIds(const std::vector<int>& ids) {
    if (hybrid_) hybrid_->feed(ids);
    else dense_->feed(ids);
  }

  // Generate up to max_new, streaming decoded deltas; returns the full reply text.
  // A stop string ends the turn and is trimmed from both the stream and the return value;
  // its own prefix is held back from the stream until it completes or is ruled out.
  std::string streamDecode(int max_new, const OnText& on_text,
                           const std::vector<std::string>& stop) {
    const StopMatcher matcher(stop.empty() ? stop_ : stop);
    std::vector<int> gen;
    size_t emitted = 0;      // bytes already streamed to on_text
    size_t cut = std::string::npos;
    std::string full;
    auto on_token = [&](int id) -> bool {
      gen.push_back(id);
      full = tk_.decode(gen);
      const auto sc = matcher.scan(full);
      const size_t show = sc.cut != std::string::npos ? sc.cut : sc.safe;
      if (on_text && show > emitted) on_text(full.substr(emitted, show - emitted));
      emitted = std::max(emitted, show);
      cut = sc.cut;
      return cut != std::string::npos;           // stop string hit → break the loop
    };
    NativeSamplingParams sp = sampling_;
    sp.max_new = max_new;
    if (!cfg_.eos.empty()) sp.eos = cfg_.eos;
    if (hybrid_) hybrid_->generate(sp, on_token);
    else dense_->generate(sp, on_token);
    if (cut != std::string::npos) return full.substr(0, cut);
    // No stop string: flush any suffix held back mid-stream (a stop-prefix that never
    // completed) so the stream ends up with the whole reply.
    if (on_text && full.size() > emitted) on_text(full.substr(emitted));
    return full.empty() ? tk_.decode(gen) : full;
  }

  ModelConfig cfg_;
  Tokenizer tk_;
  std::unique_ptr<NativeHybridEngine> hybrid_;
  std::unique_ptr<NativeEngine> dense_;
  NativeSamplingParams sampling_;                 // default: greedy
  std::vector<std::string> stop_;                 // persistent stop strings (default: none)
  bool first_turn_ = true;
};

}  // namespace bllm
