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
  std::string generate(const std::string& prompt, int max_new = 256, const OnText& on_text = {}) {
    feedIds(tk_.encode(prompt));
    return streamDecode(max_new, on_text);
  }

  // ChatML multi-turn: context persists across calls (feed only the new turn).
  std::string chat(const std::string& msg, int max_new = 256, const OnText& on_text = {}) {
    if (cfg_.chat.format != "chatml" || cfg_.chat.im_start < 0)
      return generate(msg, max_new, on_text);   // no template → raw
    const int S = cfg_.chat.im_start, E = cfg_.chat.im_end;
    std::vector<int> ids;
    auto add = [&](const std::vector<int>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    if (first_turn_) { add({S}); add(tk_.encode("system\n" + cfg_.chat.system)); add({E}); }
    else             { add({E}); }               // close the previous assistant turn
    add({S}); add(tk_.encode("user\n" + msg)); add({E});
    add({S}); add(tk_.encode("assistant\n"));
    first_turn_ = false;
    feedIds(ids);
    return streamDecode(max_new, on_text);
  }

 private:
  // Ingest prompt/turn tokens into the engine (leaving the last logits for sampling).
  void feedIds(const std::vector<int>& ids) {
    if (hybrid_) hybrid_->feed(ids);
    else dense_->feed(ids);
  }

  // Generate up to max_new, streaming decoded deltas; returns the full reply text.
  std::string streamDecode(int max_new, const OnText& on_text) {
    std::vector<int> gen;
    std::string emitted;
    auto on_token = [&](int id) {
      gen.push_back(id);
      std::string full = tk_.decode(gen);
      if (on_text && full.size() > emitted.size()) on_text(full.substr(emitted.size()));
      emitted = std::move(full);
    };
    NativeSamplingParams sp = sampling_;
    sp.max_new = max_new;
    if (!cfg_.eos.empty()) sp.eos = cfg_.eos;
    if (hybrid_) hybrid_->generate(sp, on_token);
    else dense_->generate(sp, on_token);
    return tk_.decode(gen);
  }

  ModelConfig cfg_;
  Tokenizer tk_;
  std::unique_ptr<NativeHybridEngine> hybrid_;
  std::unique_ptr<NativeEngine> dense_;
  NativeSamplingParams sampling_;                 // default: greedy
  bool first_turn_ = true;
};

}  // namespace bllm
