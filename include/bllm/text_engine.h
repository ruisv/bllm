// bllm::TextEngine — the decoder half of a multimodal session, behind the small
// surface the VLM path actually needs.
//
// Why this exists: NativeVlm was welded to NativeEngine (dense KV, Qwen2.5-Omni).
// Qwen3.5 is a HYBRID (Gated-DeltaNet + attention) model driven by
// NativeHybridEngine, and it is also natively multimodal — so the same session
// logic (splice vision rows into the embedding stream, assign 3-D mrope positions,
// decode) has to run over either engine.
//
// The surface is deliberately tiny and deliberately NOT "everything both engines
// can do": prompt-cache snapshots, perplexity and chunked prefill stay on the
// concrete classes, because they differ in kind rather than in detail. What is here
// is exactly the multimodal seam — feed rows at positions, decode, and turn a token
// id back into a row.
#pragma once

#include "bllm/native_embed.h"
#include "bllm/native_engine.h"
#include "bllm/native_hybrid_engine.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bllm {

class TextEngine {
 public:
  virtual ~TextEngine() = default;

  virtual int hidden() const = 0;
  virtual int vocab() const = 0;
  virtual int position() const = 0;
  virtual int context_left() const = 0;
  // The next PLAIN-rope position. Not position() + 1 once an image has been fed:
  // a 14x14 grid occupies 196 slots but advances rope by only 14.
  virtual int next_pos() const = 0;

  virtual void reset() = 0;
  virtual void mark_turn_start() = 0;
  virtual void set_sched(const native_detail::BpuSched& s) = 0;

  // token id -> hidden row, from whichever embedding table this engine owns.
  virtual void embed(int id, float* dst) const = 0;
  // Splice n already-embedded rows in at explicit 3-D positions. `extras` carries the
  // per-token rows a prefill graph adds on top (Qwen3-VL DeepStack), if it takes any.
  virtual void feed_embeds(const float* rows, int n, const Pos3* pos,
                           const PrefillExtras* extras) = 0;
  // How many such extra maps the prefill graph takes (0: none).
  virtual int prefill_extras() const { return 0; }
  // A graph that takes scaled-int16 embeddings: their scale and the extra maps' (else
  // false). Lets a quantised vision tower requantise straight to them.
  virtual bool input_scales(float& embed, std::vector<float>& extras) const { return false; }
  virtual std::vector<int> generate(const NativeSamplingParams& p,
                                    const std::function<bool(int)>& on_token) = 0;

  virtual double decode_tps() const = 0;
  virtual double ttft_ms() const = 0;
};

// ── dense KV dual-graph: engine + a separately mmap'd table ─────────────────────
// Two contracts: Qwen2.5-Omni's (host rope from cos/sin, mrope sections ours to set) and
// the official LLM 2.0 one (int32 [3,1,T] positions, rope inside the graph — so the
// rope arguments are unused, and the positions must have all three rows).
class DenseTextEngine final : public TextEngine {
 public:
  DenseTextEngine(const std::string& hbm, const std::string& embed_path,
                  double rope_theta, const std::vector<int>& mrope_section,
                  bool mrope_interleaved,
                  const char* prefill = "text_model_prefill",
                  const char* decode = "text_model_decode")
      : e_(std::make_unique<NativeEngine>(hbm, prefill, decode)) {
    if (e_->input_mode() == NativeEngine::InputMode::EmbedPos) {
      if (e_->pos_rows() != 3)
        throw std::runtime_error("[bllm] a VLM text tower needs 3-row (mrope) position_ids, "
                                 "this one has " + std::to_string(e_->pos_rows()));
    } else if (!e_->embed_input()) {
      throw std::runtime_error("[bllm] a VLM text tower must take an embedding input");
    } else {
      e_->set_rope(rope_theta, mrope_section, mrope_interleaved);
    }
    tbl_ = std::make_unique<EmbedTable>(embed_path, e_->vocab(), e_->hidden());
    e_->set_embedder([this](int id, float* dst) { tbl_->lookup(id, dst); });
  }

  int hidden() const override { return e_->hidden(); }
  int vocab() const override { return e_->vocab(); }
  int position() const override { return e_->position(); }
  int context_left() const override { return e_->context_left(); }
  int next_pos() const override { return e_->next_pos(); }
  void reset() override { e_->reset(); }
  void mark_turn_start() override { e_->mark_turn_start(); }
  void set_sched(const native_detail::BpuSched& s) override { e_->set_sched(s); }
  void embed(int id, float* dst) const override { tbl_->lookup(id, dst); }
  void feed_embeds(const float* rows, int n, const Pos3* pos,
                   const PrefillExtras* extras) override {
    e_->feed_embeds(rows, n, pos, extras);
  }
  int prefill_extras() const override { return e_->prefill_extras(); }
  bool input_scales(float& embed, std::vector<float>& extras) const override {
    if (e_->input_mode() != NativeEngine::InputMode::EmbedPos) return false;
    embed = e_->prefill_embed_scale();
    extras = e_->prefill_extra_scales();
    return true;
  }
  std::vector<int> generate(const NativeSamplingParams& p,
                            const std::function<bool(int)>& on_token) override {
    return e_->generate(p, on_token);
  }
  double decode_tps() const override { return e_->last_stats().decode_tps; }
  double ttft_ms() const override { return e_->last_stats().ttft_ms; }

  NativeEngine& raw() { return *e_; }

 private:
  std::unique_ptr<NativeEngine> e_;
  std::unique_ptr<EmbedTable> tbl_;
};

// ── hybrid GDN+attention single graph (Qwen3.5) ────────────────────────────────
// This engine already holds its own fp16 embedding table (it needs one to decode),
// so embed() forwards to that rather than mapping a second ~500 MB copy.
class HybridTextEngine final : public TextEngine {
 public:
  HybridTextEngine(const std::string& hbm, const std::string& embed_path,
                   const std::string& graph, double rope_theta,
                   const std::vector<int>& mrope_section, bool mrope_interleaved,
                   const std::string& hbm_prefill = "")
      : e_(std::make_unique<NativeHybridEngine>(hbm, embed_path, graph, rope_theta,
                                                hbm_prefill)) {
    e_->set_mrope(mrope_section, mrope_interleaved);
  }

  int hidden() const override { return e_->hidden(); }
  int vocab() const override { return e_->vocab(); }
  int position() const override { return e_->position(); }
  int context_left() const override { return e_->context_left(); }
  int next_pos() const override { return e_->next_pos(); }
  void reset() override { e_->reset(); }
  void mark_turn_start() override { e_->mark_turn_start(); }
  void set_sched(const native_detail::BpuSched& s) override { e_->set_sched(s); }
  void embed(int id, float* dst) const override { e_->embed_row(id, dst); }
  void feed_embeds(const float* rows, int n, const Pos3* pos,
                   const PrefillExtras* extras) override {
    if (extras) throw std::runtime_error("[bllm] the hybrid engine takes no DeepStack rows");
    e_->feed_embeds(rows, n, pos);
  }
  std::vector<int> generate(const NativeSamplingParams& p,
                            const std::function<bool(int)>& on_token) override {
    return e_->generate(p, on_token);
  }
  double decode_tps() const override { return e_->last_stats().decode_tps; }
  double ttft_ms() const override { return e_->last_stats().ttft_ms; }

  NativeHybridEngine& raw() { return *e_; }

 private:
  std::unique_ptr<NativeHybridEngine> e_;
};

}  // namespace bllm
