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

#include <atomic>
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
    // A multi-core .hbm (nash-p / S600) MUST be submitted to specific BPU cores — the
    // runtime rejects HB_UCP_CORE_ANY for multi-core tasks. Bind cores 0..(bpu_cores-1)
    // = mask (1<<N)-1 (HB_UCP_BPU_CORE_k == 1<<k). Single-core keeps the CORE_ANY default.
    if (cfg_.bpu_cores > 1) {
      native_detail::BpuSched s;
      s.core_mask = (static_cast<uint64_t>(1) << cfg_.bpu_cores) - 1;
      if (hybrid_) hybrid_->set_sched(s); else dense_->set_sched(s);
    }
  }

  const ModelConfig& config() const { return cfg_; }
  const Tokenizer& tokenizer() const { return tk_; }

  // Sampling controls (default greedy). Full parity with libxlm's knobs; applied to both
  // the dense and hybrid backends. min_p/typ_p disabled at 0/1, penalties at 1/0/0.
  void set_sampling(float temp, float top_p, int top_k, float rep_pen, uint64_t seed,
                    float min_p = 0.0f, float typ_p = 1.0f, int min_keep = 1,
                    int penalty_last_n = 64, float penalty_freq = 0.0f,
                    float penalty_present = 0.0f) {
    sampling_.temp = temp; sampling_.top_p = top_p; sampling_.top_k = top_k;
    sampling_.rep_pen = rep_pen; sampling_.seed = seed;
    sampling_.min_p = min_p; sampling_.typ_p = typ_p; sampling_.min_keep = min_keep;
    sampling_.penalty_last_n = penalty_last_n; sampling_.penalty_freq = penalty_freq;
    sampling_.penalty_present = penalty_present;
  }

  // Stop strings: end a turn as soon as any of them appears in the decoded text, and
  // return the reply WITHOUT the stop string. Complements eos-token stopping (a stop
  // string can span tokens, and need not be a token). Empty clears them. Persists across
  // turns; a per-call `stop` argument overrides for that call only.
  void set_stop(std::vector<std::string> stops) { stop_ = std::move(stops); }

  // Thinking control (Qwen3.5 reasoning models). `enabled=false` prefills a closed empty
  // `<think></think>` after the assistant marker (== the official template's
  // enable_thinking=False) so the model answers DIRECTLY, no reasoning — faster, fewer
  // tokens. When enabled (the default), the model reasons and the `<think>...</think>` block
  // IS shown — if you asked for thinking, you get to see it. Note: the in-message `/no_think`
  // soft switch is NOT reliable on this checkpoint; this prefill is.
  void set_thinking(bool enabled) { thinking_ = enabled; }
  bool thinking() const { return thinking_; }
  // BPU queue priority (0..255). Lowest (the default) lets a co-resident vision
  // pipeline jump the queue. Note: it orders the queue, it does not preempt a graph
  // that is already running — see native_detail::BpuSched.
  void set_bpu_priority(int priority) {
    // Preserve core_mask (set at load for multi-core .hbm) — only change priority.
    native_detail::BpuSched s = hybrid_ ? hybrid_->sched() : dense_->sched();
    s.priority = priority;
    if (hybrid_) hybrid_->set_sched(s); else dense_->set_sched(s);
  }

  // Tokens that still fit in this conversation's KV window.
  int context_left() const { return hybrid_ ? hybrid_->context_left() : dense_->context_left(); }

  // Perplexity of `text` under the model (teacher-forced, raw text — no chat template).
  // Resets the conversation. This is libxlm's xlm_ppl on the native path.
  double perplexity(const std::string& text) {
    const std::vector<int> ids = tk_.encode(text);
    first_turn_ = true;
    return hybrid_ ? hybrid_->perplexity(ids) : dense_->perplexity(ids);
  }

  // Save/restore the conversation's KV (+ SSM) state — libxlm's path_prompt_cache. Evaluate
  // a shared prefix once, save, and reload later to skip re-prefill; generate() resumes
  // bit-identically. A state is bound to this model; loading one from another is rejected.
  void save_state(const std::string& path) {
    if (hybrid_) hybrid_->save_state(path); else dense_->save_state(path);
  }
  void load_state(const std::string& path) {
    if (hybrid_) hybrid_->load_state(path); else dense_->load_state(path);
    first_turn_ = false;                 // a restored context is mid-conversation
  }

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
    std::vector<int> ids;
    auto add = [&](const std::vector<int>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    // Phi-style template: `<|system|>sys<|end|><|user|>msg<|end|><|assistant|>`. No role
    // text, no newline — the marker token is the role. Generation stops on <|end|> (an eos),
    // which is therefore NOT in the cache, so a prior assistant turn is closed by prepending
    // <|end|> again on the next turn (mirrors the ChatML close below).
    if (cfg_.chat.format == "phi" && cfg_.chat.r_user >= 0) {
      const int U = cfg_.chat.r_user, A = cfg_.chat.r_assistant;
      const int SY = cfg_.chat.r_system, PE = cfg_.chat.r_end;
      if (first_turn_) {
        if (cfg_.chat.bos >= 0) add({cfg_.chat.bos});
        if (SY >= 0) { add({SY}); add(tk_.encode(cfg_.chat.system)); add({PE}); }
      } else            { add({PE}); }              // close the previous assistant turn
      add({U}); add(tk_.encode(msg)); add({PE});
      add({A});
      first_turn_ = false;
      feedIds(ids);
      return streamDecode(max_new, on_text, stop);
    }
    if (cfg_.chat.format != "chatml" || cfg_.chat.im_start < 0)
      return generate(msg, max_new, on_text, stop);   // no template → raw
    const int S = cfg_.chat.im_start, E = cfg_.chat.im_end;
    // Canonical ChatML: every role block is `<|im_start|>role\n{content}<|im_end|>\n`. The
    // newline after <|im_end|> matters — omitting it leaves the model with a malformed
    // boundary (InternLM2 echoes the prompt without it; Qwen is more forgiving but this is
    // its real template too). Some models (InternLM2) also prepend a BOS before the first
    // block; ChatML models (Qwen) do not — cfg_.chat.bos carries it, -1 otherwise.
    const std::vector<int> NL = tk_.encode("\n");
    if (first_turn_) {
      if (cfg_.chat.bos >= 0) add({cfg_.chat.bos});
      add({S}); add(tk_.encode("system\n" + cfg_.chat.system)); add({E}); add(NL);
    } else            { add({E}); add(NL); }     // close the previous assistant turn
    add({S}); add(tk_.encode("user\n" + msg)); add({E}); add(NL);
    add({S}); add(tk_.encode("assistant\n"));
    if (!thinking_) add(tk_.encode("<think>\n\n</think>\n\n"));   // enable_thinking=False: answer directly
    first_turn_ = false;
    feedIds(ids);
    return streamDecode(max_new, on_text, stop);
  }

  // ---- Reusable-prefix path (prompt cache × fixed prefix) -----------------------------
  // For workloads that ask MANY independent questions against ONE shared context (RAG over a
  // document, a fixed system prompt + tool defs, few-shot exemplars): prefill the shared
  // prefix ONCE with set_prefix(), then ask() each query — each restores the snapshot instead
  // of re-prefilling the (possibly long) prefix. Unlike chat(), ask() does NOT accumulate:
  // every call starts from the same prefix. The one-time prefix prefill still runs here at the
  // current speed; chunked prefill (roadmap) accelerates that single ingest.

  // Prefill the system block + optional shared context and snapshot it as the reusable base.
  void set_prefix(const std::string& shared_context = "") {
    reset();
    std::vector<int> ids;
    buildPrefix(ids, shared_context);
    if (ids.empty())
      throw std::runtime_error("[bllm] set_prefix needs a system prompt or shared context "
                               "(the model's chat.format is \"none\" and shared_context is empty)");
    feedIds(ids);
    prefix_ = engineSnapshot();      // requires valid logits — feedIds ran >=1 step
    prefixSet_ = true;
  }

  // Answer `query` against the fixed prefix, independently (does not accumulate across asks).
  std::string ask(const std::string& query, int max_new = 256, const OnText& on_text = {},
                  const std::vector<std::string>& stop = {}) {
    if (!prefixSet_) throw std::runtime_error("[bllm] ask() needs set_prefix() first");
    engineRestore(prefix_);          // back to the prefix; discard any previous ask's tail
    std::vector<int> ids;
    buildUserTurn(ids, query);
    feedIds(ids);
    return streamDecode(max_new, on_text, stop);
  }

  bool has_prefix() const { return prefixSet_; }
  void clear_prefix() { prefix_.clear(); prefixSet_ = false; }

  // ---- Serving primitives (session pool × prefix cache) --------------------------------
  // set_prefix()/ask() serve ONE fixed prefix. A server juggling many conversations needs
  // to hold MANY states and pick one per request, so it drives the snapshot itself:
  //
  //   ids = encode(prompt); if (cached prefix of ids) restore(blob); else reset();
  //   feed_ids(delta); blob = snapshot();        // state == the whole prompt, cache it
  //   decode_stream(...);
  //
  // The blob is the same payload as save_state(), in memory. Cheap relative to a prefill,
  // but NOT free — restore() clears the whole cache window before copying rows back.
  std::string snapshot() const { return engineSnapshot(); }
  void restore(const std::string& blob) {
    engineRestore(blob);
    first_turn_ = false;               // a restored context is mid-conversation
  }

  // Prefill chunk width, or 0 when this engine has no chunked prefill (the hybrid engine
  // ingests token by token). THIS MATTERS FOR CACHING: the dense engine ingests a run of
  // tokens as batch-prefill chunks of this width and decode-steps the remainder, and the
  // two graphs are not numerically identical under int8. Splitting a prompt at an
  // arbitrary point therefore changes which tokens went through which graph — and so can
  // change the reply. Snapshot only at multiples of this width and the segmentation is the
  // same as a straight-through prefill, making reuse exact. 0 means any split is exact.
  int prefill_chunk() const { return hybrid_ ? 0 : dense_->chunk(); }

  // generate() split in two, so a caller can snapshot exactly at the prefill/decode
  // boundary. feed_ids() takes ids rather than text: the server has already tokenized
  // (it matches its cache on token ids), and re-encoding a text delta could tokenize
  // differently at the seam than the ids the cache was keyed on.
  void feed_ids(const std::vector<int>& ids) {
    feedIds(ids);
    first_turn_ = false;
  }
  std::string decode_stream(int max_new = 256, const OnText& on_text = {},
                            const std::vector<std::string>& stop = {}) {
    return streamDecode(max_new, on_text, stop);
  }

  // Ask the in-flight generation to stop at the next token boundary. Thread-safe — call it
  // from another thread while chat()/generate()/decode_stream() runs (a server cancelling a
  // disconnected client). It breaks the decode loop the same way a stop string does, so the
  // context stays consistent: nothing is generated past the cancel point. The flag is
  // one-shot and is cleared when the next generation starts, so a cancel that lands between
  // turns is dropped rather than killing the following turn.
  void request_cancel() { cancelReq_.store(true, std::memory_order_relaxed); }
  // Whether the LAST generation ended because of request_cancel() (vs eos/stop/max_new).
  bool canceled() const { return canceled_; }

 private:
  std::string engineSnapshot() const { return hybrid_ ? hybrid_->snapshot() : dense_->snapshot(); }
  void engineRestore(const std::string& blob) {
    if (hybrid_) hybrid_->restore(blob); else dense_->restore(blob);
  }

  // Build the reusable prefix: BOS (if any) + a system block carrying cfg system prompt and,
  // appended, the shared context. Mirrors chat()'s first-turn opener for each format.
  void buildPrefix(std::vector<int>& ids, const std::string& shared) {
    auto add = [&](const std::vector<int>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    std::string sys = cfg_.chat.system;
    if (!shared.empty()) sys += (sys.empty() ? "" : "\n\n") + shared;
    if (cfg_.chat.format == "phi" && cfg_.chat.r_user >= 0) {
      if (cfg_.chat.bos >= 0) add({cfg_.chat.bos});
      if (cfg_.chat.r_system >= 0 && !sys.empty()) { add({cfg_.chat.r_system}); add(tk_.encode(sys)); add({cfg_.chat.r_end}); }
      return;
    }
    if (cfg_.chat.format == "chatml" && cfg_.chat.im_start >= 0) {
      if (cfg_.chat.bos >= 0) add({cfg_.chat.bos});
      if (!sys.empty()) { add({cfg_.chat.im_start}); add(tk_.encode("system\n" + sys)); add({cfg_.chat.im_end}); add(tk_.encode("\n")); }
      return;
    }
    if (!shared.empty()) add(tk_.encode(shared));    // no template: shared text is the raw prefix
  }

  // Build one user turn (query) opening the assistant, per format. Same wire form chat() uses.
  void buildUserTurn(std::vector<int>& ids, const std::string& msg) {
    auto add = [&](const std::vector<int>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    if (cfg_.chat.format == "phi" && cfg_.chat.r_user >= 0) {
      add({cfg_.chat.r_user}); add(tk_.encode(msg)); add({cfg_.chat.r_end}); add({cfg_.chat.r_assistant});
      return;
    }
    if (cfg_.chat.format == "chatml" && cfg_.chat.im_start >= 0) {
      add({cfg_.chat.im_start}); add(tk_.encode("user\n" + msg)); add({cfg_.chat.im_end}); add(tk_.encode("\n"));
      add({cfg_.chat.im_start}); add(tk_.encode("assistant\n"));
      if (!thinking_) add(tk_.encode("<think>\n\n</think>\n\n"));
      return;
    }
    add(tk_.encode(msg));                            // no template: raw completion
  }

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
    cancelReq_.store(false, std::memory_order_relaxed);   // drop a cancel that landed between turns
    canceled_ = false;
    auto on_token = [&](int id) -> bool {
      gen.push_back(id);
      full = tk_.decode(gen);
      const auto sc = matcher.scan(full);
      const size_t show = sc.cut != std::string::npos ? sc.cut : sc.safe;
      if (on_text && show > emitted) on_text(full.substr(emitted, show - emitted));
      emitted = std::max(emitted, show);
      cut = sc.cut;
      if (cut != std::string::npos) return true;          // stop string hit → break the loop
      // Checked after streaming this token, so the client keeps whatever already reached it.
      canceled_ = cancelReq_.exchange(false, std::memory_order_relaxed);
      return canceled_;
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
  bool thinking_ = true;                          // Qwen3.5 reasoning: prefill empty <think> when false
  std::string prefix_;                            // in-memory snapshot of the reusable prefix
  bool prefixSet_ = false;
  std::atomic<bool> cancelReq_{false};            // set from another thread by request_cancel()
  bool canceled_ = false;                         // did the last generation end on a cancel
};

}  // namespace bllm
