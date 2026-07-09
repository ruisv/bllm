// bllm::NativeHybridEngine — the self-built autoregressive engine for HYBRID
// (Gated-DeltaNet + attention) models like Qwen3.5, driving a single-graph
// `.hbm` directly on hbDNN/hbUCP, WITHOUT libxlm. Unlike NativeEngine (KV-only
// dual-graph), this drives ONE seq_len==1 decode graph whose I/O is:
//   inputs : x[1,H] (host-embedded), cos[ROT], sin[ROT], mask[nH,CL], then N
//            mixed per-layer caches (GDN: state+conv; attn: K+V) as graph tensors
//   outputs: logits[1,vocab], then the N updated caches (the graph rolls/overwrites
//            internally, so cache update is a uniform output->input hand-off)
//
// Perf design (to match/beat libxlm):
//   * Zero-copy cache ping-pong: every cache is double-buffered; each step binds
//     one buffer set as inputs and the other as outputs, then swaps. No per-token
//     memcpy of the (large) mixed caches.
//   * Caches are BPU-write then BPU-read only — the CPU never touches them, so
//     they need NO cache flush. Only the small CPU-written inputs (x/cos/sin/mask)
//     are cleaned and the logits output invalidated.
//   * Embedding lookup is host-side from an fp16 table (aarch64 native __fp16).
// See docs/NATIVE_RUNTIME.md (SE4).
#pragma once

#include "bllm/native_engine.h"   // native_detail::{Mem, check, elemSize, alignUp, kAlign}
#include "bllm/native_sampler.h"  // bllm::Sampler (temperature/top-k/top-p/rep-pen)

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct HybridStats { double decode_tps = 0, prefill_tps = 0; int ntok = 0; };

class NativeHybridEngine {
  using Mem = native_detail::Mem;

 public:
  // hbm: the hybrid model; embed_fp16: raw fp16 embedding table [vocab*hidden];
  // model_name: the graph name inside the .hbm (default "qwen35").
  NativeHybridEngine(const std::string& hbm, const std::string& embed_fp16,
                     const std::string& model_name = "qwen35", double rope_theta = 1e7)
      : theta_(rope_theta) {
    const char* files[] = {hbm.c_str()};
    BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&packed_, files, 1));
    BLLM_NATIVE_CK(hbDNNGetModelHandle(&h_, packed_, model_name.c_str()));
    int ic = 0, oc = 0;
    BLLM_NATIVE_CK(hbDNNGetInputCount(&ic, h_));
    BLLM_NATIVE_CK(hbDNNGetOutputCount(&oc, h_));
    if (ic < 5) throw std::runtime_error("[hybrid] unexpected input count");
    in_.resize(ic); out_.resize(oc);
    nCache_ = ic - kFixedIn;                 // caches after x,cos,sin,mask
    if (oc != 1 + nCache_) throw std::runtime_error("[hybrid] output count != 1 + nCache");

    // --- inputs ---
    for (int i = 0; i < ic; ++i) {
      auto& p = in_[i].properties;
      BLLM_NATIVE_CK(hbDNNGetInputTensorProperties(&p, h_, i));
      fixStride(p);
      inBytes_.push_back(tensorBytes(p));
    }
    H_   = dim(in_[0].properties, in_[0].properties.validShape.numDimensions - 1);
    ROT_ = dim(in_[1].properties, in_[1].properties.validShape.numDimensions - 1);
    { const auto& ms = in_[3].properties.validShape;
      nHead_ = ms.dimensionSize[0]; CL_ = ms.dimensionSize[1]; }
    // fixed CPU-written inputs get one buffer each
    for (int i = 0; i < kFixedIn; ++i) { fixedMem_.emplace_back(); fixedMem_[i].alloc(inBytes_[i]); in_[i].sysMem = fixedMem_[i].m; }
    // caches: two buffers each (ping-pong), zero-initialised
    bufA_.resize(nCache_); bufB_.resize(nCache_);
    for (int c = 0; c < nCache_; ++c) {
      uint64_t b = inBytes_[kFixedIn + c];
      bufA_[c].alloc(b); bufB_[c].alloc(b);
      std::memset(bufA_[c].p(), 0, b); bufA_[c].clean();
      std::memset(bufB_[c].p(), 0, b); bufB_[c].clean();
    }

    // --- outputs ---
    for (int i = 0; i < oc; ++i) {
      BLLM_NATIVE_CK(hbDNNGetOutputTensorProperties(&out_[i].properties, h_, i));
    }
    logitMem_.alloc(out_[0].properties.alignedByteSize);
    out_[0].sysMem = logitMem_.m;
    const auto& ls = out_[0].properties.validShape;
    vocab_ = ls.dimensionSize[ls.numDimensions - 1];
    logitType_ = out_[0].properties.tensorType;
    logitScale_ = out_[0].properties.scale.scaleData && out_[0].properties.scale.scaleLen > 0
                      ? out_[0].properties.scale.scaleData[0] : 1.0f;

    // rope inv-freq: inv[i] = theta^(-2i/ROT)
    inv_.resize(ROT_ / 2);
    for (int i = 0; i < ROT_ / 2; ++i) inv_[i] = std::pow(theta_, -(double)(2 * i) / ROT_);

    // embedding table (fp16)
    std::ifstream f(embed_fp16, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("[hybrid] cannot open embed file: " + embed_fp16);
    size_t sz = f.tellg(); f.seekg(0);
    emb_.resize(sz / 2);
    f.read(reinterpret_cast<char*>(emb_.data()), sz);
    if (emb_.size() < (size_t)vocab_ * H_) throw std::runtime_error("[hybrid] embed table too small");

    reset();
  }
  ~NativeHybridEngine() { if (packed_) hbDNNRelease(packed_); }
  NativeHybridEngine(const NativeHybridEngine&) = delete;
  NativeHybridEngine& operator=(const NativeHybridEngine&) = delete;

  void set_sched(const native_detail::BpuSched& s) { sched_ = s; }
  const native_detail::BpuSched& sched() const { return sched_; }

  int hidden() const { return H_; }
  int vocab() const { return vocab_; }
  int n_cache() const { return nCache_; }
  int cache_len() const { return CL_; }
  int position() const { return P_; }
  // Tokens that still fit. The attention layers hold a cache_len window (the GDN
  // layers are O(1) in length), and rolling past it evicts the earliest tokens —
  // which turns a full-attention model into gibberish. Refuse rather than degrade.
  int context_left() const { return CL_ - P_; }
  const HybridStats& last_stats() const { return stats_; }

  void reset() {
    for (int c = 0; c < nCache_; ++c) {
      std::memset(bufA_[c].p(), 0, inBytes_[kFixedIn + c]); bufA_[c].clean();
      std::memset(bufB_[c].p(), 0, inBytes_[kFixedIn + c]); bufB_[c].clean();
    }
    P_ = 0; phase_ = false; curLogits_ = nullptr;
  }

  // ingest prompt tokens (updates caches; keeps the last logits for sampling).
  void feed(const std::vector<int>& ids) {
    if ((int)ids.size() > context_left())
      throw std::runtime_error("[hybrid] context overflow: " + std::to_string(ids.size()) +
                               " tokens do not fit in the remaining " + std::to_string(context_left()) +
                               " of " + std::to_string(CL_) +
                               " cache slots. reset(), shorten the turn, or use a longer-context build.");
    for (int id : ids) step(id);
  }

  // greedy-generate up to max_new; stop on any eos; call on_token(id) per token.
  std::vector<int> generate(int max_new, const std::vector<int>& eos,
                            const std::function<void(int)>& on_token = {}) {
    std::vector<int> gen;
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (int i = 0; i < max_new && curLogits_valid_ && P_ < CL_; ++i) {
      int tok = argmax();
      bool stop = false; for (int e : eos) if (e == tok) { stop = true; break; }
      if (stop) break;
      gen.push_back(tok);
      if (on_token) on_token(tok);
      step(tok);
    }
    double dt = std::chrono::duration<double>(clk::now() - t0).count();
    stats_.ntok = (int)gen.size();
    stats_.decode_tps = dt > 0 ? gen.size() / dt : 0.0;
    return gen;
  }

  // sample up to p.max_new (temperature/top-k/top-p/rep-pen, seeded); stop on p.eos.
  std::vector<int> generate(const NativeSamplingParams& p,
                            const std::function<void(int)>& on_token = {}) {
    Sampler s(p);
    std::unordered_set<int> eos(p.eos.begin(), p.eos.end());
    auto logit = [this](int i) -> float {
      return logitType_ == HB_DNN_TENSOR_TYPE_F32
                 ? reinterpret_cast<const float*>(curLogits_)[i]
                 : reinterpret_cast<const int16_t*>(curLogits_)[i] * logitScale_;
    };
    std::vector<int> gen;
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (int i = 0; i < p.max_new && curLogits_valid_ && P_ < CL_; ++i) {
      int tok = s.pick(logit, vocab_);
      if (eos.count(tok)) break;
      gen.push_back(tok);
      if (on_token) on_token(tok);
      s.record(tok);
      step(tok);
    }
    double dt = std::chrono::duration<double>(clk::now() - t0).count();
    stats_.ntok = (int)gen.size();
    stats_.decode_tps = dt > 0 ? gen.size() / dt : 0.0;
    return gen;
  }

  // run one decode step for `token`; leaves logits available for argmax()/generate().
  void step(int token) {
    // x = embed[token]  (fp16 -> f32)
    float* x = reinterpret_cast<float*>(fixedMem_[0].p());
    const __fp16* row = emb_.data() + (size_t)token * H_;
    for (int i = 0; i < H_; ++i) x[i] = (float)row[i];
    fixedMem_[0].clean();
    // cos/sin for position P_
    float* cs = reinterpret_cast<float*>(fixedMem_[1].p());
    float* sn = reinterpret_cast<float*>(fixedMem_[2].p());
    const int half = ROT_ / 2;
    for (int i = 0; i < half; ++i) {
      double ang = (double)P_ * inv_[i];
      float c = (float)std::cos(ang), s = (float)std::sin(ang);
      cs[i] = cs[i + half] = c; sn[i] = sn[i + half] = s;
    }
    fixedMem_[1].clean(); fixedMem_[2].clean();
    // causal mask [nHead, CL]: valid = last min(P+1,CL) slots
    float* mask = reinterpret_cast<float*>(fixedMem_[3].p());
    int valid = P_ + 1; if (valid > CL_) valid = CL_;
    const int lo = CL_ - valid;
    for (int j = 0; j < CL_; ++j) { float m = (j >= lo) ? 0.0f : -1e9f; for (int hh = 0; hh < nHead_; ++hh) mask[hh * CL_ + j] = m; }
    fixedMem_[3].clean();
    // bind ping-pong caches: cur -> inputs, next -> outputs
    std::vector<Mem>& cur = phase_ ? bufB_ : bufA_;
    std::vector<Mem>& nxt = phase_ ? bufA_ : bufB_;
    for (int c = 0; c < nCache_; ++c) { in_[kFixedIn + c].sysMem = cur[c].m; out_[1 + c].sysMem = nxt[c].m; }
    // infer
    hbUCPTaskHandle_t task = nullptr;
    BLLM_NATIVE_CK(hbDNNInferV2(&task, out_.data(), in_.data(), h_));
    native_detail::submitAndWait(task, sched_);
    hbUCPReleaseTask(task);
    logitMem_.inval();                          // logits are CPU-read
    curLogits_ = logitMem_.p(); curLogits_valid_ = true;
    phase_ = !phase_;
    ++P_;
  }

  int argmax() const {
    if (!curLogits_valid_) return -1;
    if (logitType_ == HB_DNN_TENSOR_TYPE_F32) {
      const float* lg = reinterpret_cast<const float*>(curLogits_);
      int best = 0; float bv = lg[0];
      for (int i = 1; i < vocab_; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
      return best;
    } else {  // S16 logits
      const int16_t* lg = reinterpret_cast<const int16_t*>(curLogits_);
      return native_detail::argmaxS16(lg, vocab_);
    }
  }

 private:
  static constexpr int kFixedIn = 4;            // x, cos, sin, mask
  static int dim(const hbDNNTensorProperties& p, int idx) { return p.validShape.dimensionSize[idx]; }
  static void fixStride(hbDNNTensorProperties& p) {
    const int nd = p.validShape.numDimensions;
    if (nd > 0 && p.stride[nd - 1] == -1) p.stride[nd - 1] = (int64_t)native_detail::elemSize(p.tensorType);
    for (int d = nd - 2; d >= 0; --d)
      if (p.stride[d] == -1)
        p.stride[d] = native_detail::alignUp(p.stride[d + 1] * p.validShape.dimensionSize[d + 1], native_detail::kAlign);
  }
  static uint64_t tensorBytes(const hbDNNTensorProperties& p) {
    const int nd = p.validShape.numDimensions;
    return nd > 0 ? (uint64_t)(p.stride[0] * p.validShape.dimensionSize[0]) : 0;
  }

  hbDNNPackedHandle_t packed_ = nullptr;
  hbDNNHandle_t h_ = nullptr;
  std::vector<hbDNNTensor> in_, out_;
  std::vector<uint64_t> inBytes_;
  std::vector<Mem> fixedMem_, bufA_, bufB_;
  Mem logitMem_;
  std::vector<__fp16> emb_;
  std::vector<double> inv_;
  native_detail::BpuSched sched_;
  int H_ = 0, ROT_ = 0, nHead_ = 0, CL_ = 0, nCache_ = 0, vocab_ = 0, logitType_ = 0;
  float logitScale_ = 1.0f;
  double theta_ = 1e7;
  int P_ = 0;
  bool phase_ = false, curLogits_valid_ = false;
  void* curLogits_ = nullptr;
  HybridStats stats_;
};

}  // namespace bllm
