// bllm::NativeEngine — the self-built autoregressive LLM engine that drives a
// compiled `.hbm` (prefill + decode dual-graph) directly on the generic hbDNN /
// hbUCP BPU runtime, WITHOUT libxlm. Header-only so the CLI (examples/selfengine)
// and the Python binding (python/_bllm_native) share one implementation.
//
// Stateful: the KV cache lives in the decode graph's input buffers and persists
// across feed()/generate(), giving multi-turn chat + chunked prefill. Every model
// shape is introspected from the graph (model-agnostic).
//
// Two graph contracts are supported, distinguished by how many non-cache inputs
// the decode graph has (the cache count follows from the output count):
//
//   TokenInput (3): [token_id, position, mask]     — Qwen2.5 / DeepSeek / Qwen3 / …
//                   ids in, the graph embeds and ropes internally.
//   EmbedInput (4): [embedding, mask, cos, sin]    — Qwen2.5-Omni text tower.
//                   the HOST embeds (EmbedTable) and computes rope. That seam is
//                   what makes VLM possible: vision/audio rows are spliced into
//                   the embedding stream, and mrope 3-D positions are ours to pick.
//
// See docs/NATIVE_RUNTIME.md.
#pragma once

#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bllm {
namespace native_detail {

inline void check(int32_t e, const char* what) {
  if (e != 0) throw std::runtime_error(std::string("[native] hbDNN error: ") + what);
}
#define BLLM_NATIVE_CK(expr) ::bllm::native_detail::check((expr), #expr)

constexpr int64_t kAlign = 32;
inline int64_t alignUp(int64_t v, int64_t a) { return (v + (a - 1)) & ~(a - 1); }

inline size_t elemSize(int t) {
  switch (t) {
    case HB_DNN_TENSOR_TYPE_S8: case HB_DNN_TENSOR_TYPE_U8: case HB_DNN_TENSOR_TYPE_BOOL8: return 1;
    case HB_DNN_TENSOR_TYPE_S16: case HB_DNN_TENSOR_TYPE_U16: case HB_DNN_TENSOR_TYPE_F16: return 2;
    case HB_DNN_TENSOR_TYPE_S32: case HB_DNN_TENSOR_TYPE_U32: case HB_DNN_TENSOR_TYPE_F32: return 4;
    default: return 0;
  }
}

inline int16_t quantS16(float v, float inv_scale) {
  const float q = std::rint(v * inv_scale);
  if (q <= -32768.0f) return -32768;
  if (q >= 32767.0f) return 32767;
  return (int16_t)q;
}

struct Mem {
  hbUCPSysMem m{};
  void alloc(uint64_t sz) { BLLM_NATIVE_CK(hbUCPMallocCached(&m, sz, 0)); }
  void clean() const { if (m.virAddr) BLLM_NATIVE_CK(hbUCPMemFlush(&m, HB_SYS_MEM_CACHE_CLEAN)); }
  void inval() const { if (m.virAddr) BLLM_NATIVE_CK(hbUCPMemFlush(&m, HB_SYS_MEM_CACHE_INVALIDATE)); }
  void* p() const { return m.virAddr; }
  ~Mem() { if (m.virAddr) hbUCPFree(&m); }
  Mem() = default;
  Mem(const Mem&) = delete;
  Mem& operator=(const Mem&) = delete;
  Mem(Mem&& o) noexcept : m(o.m) { o.m = hbUCPSysMem{}; }
  Mem& operator=(Mem&& o) noexcept {
    if (this != &o) { if (m.virAddr) hbUCPFree(&m); m = o.m; o.m = hbUCPSysMem{}; }
    return *this;
  }
};

struct Graph {
  hbDNNHandle_t h = nullptr;
  std::vector<hbDNNTensor> in, out;
  std::vector<Mem> inMem, outMem;
  void init(hbDNNPackedHandle_t packed, const char* name) {
    BLLM_NATIVE_CK(hbDNNGetModelHandle(&h, packed, name));
    int ic = 0, oc = 0;
    BLLM_NATIVE_CK(hbDNNGetInputCount(&ic, h));
    BLLM_NATIVE_CK(hbDNNGetOutputCount(&oc, h));
    in.resize(ic); out.resize(oc); inMem.resize(ic); outMem.resize(oc);
    for (int i = 0; i < ic; ++i) {
      auto& p = in[i].properties;
      BLLM_NATIVE_CK(hbDNNGetInputTensorProperties(&p, h, i));
      const int nd = p.validShape.numDimensions;
      if (nd > 0 && p.stride[nd - 1] == -1) p.stride[nd - 1] = (int64_t)elemSize(p.tensorType);
      for (int d = nd - 2; d >= 0; --d)
        if (p.stride[d] == -1)
          p.stride[d] = alignUp(p.stride[d + 1] * p.validShape.dimensionSize[d + 1], kAlign);
      const uint64_t bytes = nd > 0 ? (uint64_t)(p.stride[0] * p.validShape.dimensionSize[0]) : 0;
      inMem[i].alloc(bytes);
      in[i].sysMem = inMem[i].m;
    }
    for (int i = 0; i < oc; ++i) {
      BLLM_NATIVE_CK(hbDNNGetOutputTensorProperties(&out[i].properties, h, i));
      outMem[i].alloc((uint64_t)out[i].properties.alignedByteSize);
      out[i].sysMem = outMem[i].m;
    }
  }
  int inCount() const { return (int)in.size(); }
  int outCount() const { return (int)out.size(); }
  const hbDNNTensorShape& inShape(int i) const { return in[i].properties.validShape; }
  const hbDNNTensorShape& outShape(int i) const { return out[i].properties.validShape; }
  int inType(int i) const { return in[i].properties.tensorType; }
  static float scaleOf(const hbDNNTensorProperties& p) {
    const auto& s = p.scale;
    return (s.scaleData && s.scaleLen > 0) ? s.scaleData[0] : 1.0f;
  }
  float inScale(int i) const { return scaleOf(in[i].properties); }
  float outScale(int i) const { return scaleOf(out[i].properties); }
  void inClean(int i) { inMem[i].clean(); }
  void* inPtr(int i) { return inMem[i].p(); }
  void* outPtr(int i) { return outMem[i].p(); }
  void infer() {
    hbUCPTaskHandle_t task = nullptr;
    BLLM_NATIVE_CK(hbDNNInferV2(&task, out.data(), in.data(), h));
    hbUCPSchedParam sched;
    HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    sched.priority = HB_UCP_PRIORITY_LOWEST;
    BLLM_NATIVE_CK(hbUCPSubmitTask(task, &sched));
    BLLM_NATIVE_CK(hbUCPWaitTaskDone(task, 0));
    hbUCPReleaseTask(task);
    for (auto& m : outMem) m.inval();
  }
};

inline int argmaxS16(const int16_t* row, int n) {
  int best = 0; int16_t bv = row[0];
  for (int i = 1; i < n; ++i) if (row[i] > bv) { bv = row[i]; best = i; }
  return best;
}

}  // namespace native_detail

// Sampling knobs (temp<=0 => greedy).
struct NativeSamplingParams {
  float temp = 0.0f, top_p = 1.0f, rep_pen = 1.0f;
  int top_k = 0, max_new = 200;
  uint64_t seed = 1234;
  std::vector<int> eos = {151645, 151643};
};

struct NativeStats { double ttft_ms = 0, decode_tps = 0; int ntok = 0; };

// An mrope position: (temporal, height, width). Plain-rope tokens set all three
// equal; image tokens vary h/w across the patch grid.
struct Pos3 {
  int32_t t = 0, h = 0, w = 0;
  static Pos3 scalar(int32_t p) { return {p, p, p}; }
  int32_t max() const { return std::max(t, std::max(h, w)); }
};

class NativeEngine {
 public:
  enum class InputMode { Token, Embed };

  explicit NativeEngine(const std::string& hbm,
                        const char* prefill_name = "prefill",
                        const char* decode_name = "decode") {
    const char* files[] = {hbm.c_str()};
    BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&packed_, files, 1));
    prefill_.init(packed_, prefill_name);
    decode_.init(packed_, decode_name);

    // outputs are [logits] + K[L] + V[L]; inputs are [fixed…] + K[L] + V[L].
    nLayers_ = (decode_.outCount() - 1) / 2;
    const int nFixed = decode_.inCount() - 2 * nLayers_;
    if (nFixed == 3) mode_ = InputMode::Token;
    else if (nFixed == 4) mode_ = InputMode::Embed;
    else throw std::runtime_error("[native] unsupported graph: " + std::to_string(nFixed) +
                                  " non-cache inputs (expected 3 or 4)");
    base_ = nFixed;
    vInBase_ = base_ + nLayers_;
    vOutBase_ = 1 + nLayers_;

    // KV cache is [.., cache_len, n_kv, head_dim] — dense graphs emit it 3-D,
    // the Omni text graph 4-D (leading batch), so index from the back.
    const auto& ks = decode_.inShape(base_);
    const int knd = ks.numDimensions;
    cacheLen_ = ks.dimensionSize[knd - 3]; nKV_ = ks.dimensionSize[knd - 2]; headDim_ = ks.dimensionSize[knd - 1];
    rowK_ = (size_t)nKV_ * headDim_ * native_detail::elemSize(decode_.inType(base_));
    rowV_ = (size_t)nKV_ * headDim_ * native_detail::elemSize(decode_.inType(vInBase_));

    const auto& ls = decode_.outShape(0);
    vocab_ = ls.dimensionSize[ls.numDimensions - 1];
    logitStride_ = native_detail::alignUp((int64_t)vocab_ * 2, native_detail::kAlign) / 2;
    logitScale_ = decode_.outScale(0);

    const auto& ps = prefill_.inShape(0);
    if (mode_ == InputMode::Token) {
      chunk_ = ps.dimensionSize[ps.numDimensions - 1];
    } else {
      chunk_ = ps.dimensionSize[ps.numDimensions - 2];
      hidden_ = ps.dimensionSize[ps.numDimensions - 1];
      rot_ = decode_.inShape(3).dimensionSize[decode_.inShape(3).numDimensions - 1];
      embInvScale_ = 1.0f / decode_.inScale(0);
      ropeInvScale_ = 1.0f / decode_.inScale(2);
      rowBuf_.resize(hidden_);
      set_rope(1e6, {16, 24, 24});  // Qwen2.5-Omni defaults; override via set_rope()
    }
    reset();
  }
  ~NativeEngine() { if (packed_) hbDNNRelease(packed_); }
  NativeEngine(const NativeEngine&) = delete;
  NativeEngine& operator=(const NativeEngine&) = delete;

  InputMode input_mode() const { return mode_; }
  bool embed_input() const { return mode_ == InputMode::Embed; }
  int n_layers() const { return nLayers_; }
  int n_kv() const { return nKV_; }
  int head_dim() const { return headDim_; }
  int cache_len() const { return cacheLen_; }
  int vocab() const { return vocab_; }
  int chunk() const { return chunk_; }
  int hidden() const { return hidden_; }
  int position() const { return P_; }
  // Next plain-rope position (one past the largest mrope coordinate seen).
  int next_pos() const { return maxPos_ + 1; }
  const NativeStats& last_stats() const { return stats_; }

  // Embed-mode only: how the engine turns a sampled token id back into a hidden
  // row (normally EmbedTable::lookup). Required before generate().
  void set_embedder(std::function<void(int, float*)> fn) { embedder_ = std::move(fn); }

  // Embed-mode only: rope base + mrope section split (sums to rot/2; empty = plain).
  void set_rope(double theta, std::vector<int> mrope_section) {
    inv_.resize(rot_ / 2);
    for (int i = 0; i < rot_ / 2; ++i) inv_[i] = std::pow(theta, -(double)(2 * i) / rot_);
    sec_.assign(rot_ / 2, 0);
    if (!mrope_section.empty()) {
      int d = 0;
      for (int s = 0; s < (int)mrope_section.size() && d < rot_ / 2; ++s)
        for (int k = 0; k < mrope_section[s] && d < rot_ / 2; ++k) sec_[d++] = s;
      if (d != rot_ / 2)
        throw std::runtime_error("[native] mrope_section does not cover rot/2");
    }
  }

  void reset() {
    for (int l = 0; l < nLayers_; ++l) {
      std::memset(kBuf(l), 0, cacheLen_ * rowK_); decode_.inClean(base_ + l);
      std::memset(vBuf(l), 0, cacheLen_ * rowV_); decode_.inClean(vInBase_ + l);
    }
    P_ = 0; maxPos_ = -1; curLogits_ = nullptr;
  }

  // Start the clock for the next reply's TTFT. feed() does this automatically;
  // call it earlier when work precedes the feed (e.g. a VLM's vision encode).
  void mark_turn_start() { tTurn_ = std::chrono::steady_clock::now(); }

  // ingest tokens (prompt or a new conversation turn) into the cache.
  void feed(const std::vector<int>& ids) {
    mark_turn_start();
    if (mode_ == InputMode::Embed) {
      if (!embedder_) throw std::runtime_error("[native] embed-mode feed() needs set_embedder()");
      std::vector<float> rows((size_t)ids.size() * hidden_);
      for (size_t i = 0; i < ids.size(); ++i) embedder_((int)ids[i], rows.data() + i * hidden_);
      feedEmbedsNoMark(rows.data(), (int)ids.size(), nullptr);
      return;
    }
    int i = 0, n = (int)ids.size();
    while (i < n) {
      const int left = n - i;
      if (!prefillWorthwhile(left)) break;
      const int cl = std::min(left, chunk_);
      prefillTokens(ids.data() + i, cl);
      i += cl;
    }
    for (; i < n; ++i) decodeStepToken(ids[i]);
  }

  // Embed-mode: ingest `n` hidden rows (n*hidden floats). `pos` may be null, in
  // which case positions continue sequentially from the current maximum. Does NOT
  // reset the TTFT clock — a VLM marks the turn before encoding its media.
  void feed_embeds(const float* rows, int n, const Pos3* pos) {
    feedEmbedsNoMark(rows, n, pos);
  }

  // sample up to p.max_new tokens; stop on eos; call on_token(id) per token.
  std::vector<int> generate(const NativeSamplingParams& p,
                            const std::function<void(int)>& on_token = {}) {
    Sampler s;
    s.temp = p.temp; s.topp = p.top_p; s.rep_pen = p.rep_pen; s.topk = p.top_k;
    s.rng.seed(p.seed);
    std::unordered_set<int> eos(p.eos.begin(), p.eos.end());
    std::vector<int> gen;
    using clk = std::chrono::steady_clock;
    // TTFT is measured from the start of the turn (prefill, and any vision encode
    // the caller marked), not from the first sample — the logits already exist here.
    clk::time_point tFirst;
    for (int step = 0; step < p.max_new && curLogits_; ++step) {
      int tok = s.pick(curLogits_, logitScale_, vocab_);
      if (eos.count(tok)) break;
      if (gen.empty()) { tFirst = clk::now(); stats_.ttft_ms = std::chrono::duration<double>(tFirst - tTurn_).count() * 1000.0; }
      gen.push_back(tok);
      if (on_token) on_token(tok);
      s.record(tok);
      decodeStepToken(tok);
    }
    stats_.ntok = (int)gen.size();
    const double tail = gen.empty() ? 0.0 : std::chrono::duration<double>(clk::now() - tFirst).count();
    stats_.decode_tps = (stats_.ntok > 1 && tail > 0) ? (stats_.ntok - 1) / tail : 0.0;
    if (gen.empty()) stats_.ttft_ms = 0.0;
    return gen;
  }

 private:
  struct Sampler {
    float temp = 0, topp = 1, rep_pen = 1; int topk = 0;
    std::mt19937 rng{1234u};
    std::unordered_set<int> seen;
    std::vector<int> idxbuf;
    void record(int id) { if (rep_pen != 1.0f) seen.insert(id); }
    bool stochastic() const { return temp > 0 || rep_pen != 1 || topk > 0 || topp < 1; }
    int pick(const int16_t* lg, float scale, int n) {
      using native_detail::argmaxS16;
      if (!stochastic()) return argmaxS16(lg, n);
      const int k = topk > 0 ? std::min(topk, n) : std::min(n, 512);
      if ((int)idxbuf.size() != n) { idxbuf.resize(n); for (int i = 0; i < n; ++i) idxbuf[i] = i; }
      std::nth_element(idxbuf.begin(), idxbuf.begin() + k, idxbuf.end(),
                       [&](int a, int b) { return lg[a] > lg[b]; });
      std::vector<std::pair<float, int>> cand(k);
      for (int i = 0; i < k; ++i) {
        int id = idxbuf[i]; float v = lg[id] * scale;
        if (rep_pen != 1.0f && seen.count(id)) v = v > 0 ? v / rep_pen : v * rep_pen;
        cand[i] = {v, id};
      }
      std::sort(cand.begin(), cand.end(),
                [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; });
      const float T = temp > 0 ? temp : 1.0f; const float mx = cand[0].first;
      std::vector<double> pr(k); double sum = 0;
      for (int i = 0; i < k; ++i) { pr[i] = std::exp((cand[i].first - mx) / T); sum += pr[i]; }
      double cum = 0; int keep = k;
      for (int i = 0; i < k; ++i) { cum += pr[i] / sum; if (cum >= topp) { keep = i + 1; break; } }
      double ks = 0; for (int i = 0; i < keep; ++i) ks += pr[i];
      std::uniform_real_distribution<double> U(0.0, ks);
      double r = U(rng), acc = 0;
      for (int i = 0; i < keep; ++i) { acc += pr[i]; if (r <= acc) return cand[i].second; }
      return cand[keep - 1].second;
    }
  };

  uint8_t* kBuf(int l) { return (uint8_t*)decode_.inPtr(base_ + l); }
  uint8_t* vBuf(int l) { return (uint8_t*)decode_.inPtr(vInBase_ + l); }

  // roll each layer's KV window left by one and append the step's new row.
  void rollCaches() {
    for (int l = 0; l < nLayers_; ++l) {
      uint8_t* k = kBuf(l);
      std::memmove(k, k + rowK_, (cacheLen_ - 1) * rowK_);
      std::memcpy(k + (cacheLen_ - 1) * rowK_, decode_.outPtr(1 + l), rowK_);
      decode_.inClean(base_ + l);
      uint8_t* v = vBuf(l);
      std::memmove(v, v + rowV_, (cacheLen_ - 1) * rowV_);
      std::memcpy(v + (cacheLen_ - 1) * rowV_, decode_.outPtr(vOutBase_ + l), rowV_);
      decode_.inClean(vInBase_ + l);
    }
    ++P_;
  }

  // A prefill pass costs a whole static chunk of compute, so it only beats
  // decode-stepping once enough tokens are left to amortize it.
  static constexpr int kPrefillMinTokens = 16;
  bool prefillWorthwhile(int left) const {
    return left >= kPrefillMinTokens && P_ + chunk_ <= cacheLen_;
  }

  // Like decode (which shifts its window left by one and appends the new row), the
  // prefill graph shifts its input window left by a whole chunk and writes the chunk's
  // KV into the freed tail. So the cached rows go in with the SAME right-aligned
  // layout the decode cache uses; the graph relocates them to [base-P, base).
  void loadPrefillCaches() {
    for (int l = 0; l < nLayers_; ++l) {
      uint8_t* pk = (uint8_t*)prefill_.inPtr(base_ + l);
      std::memset(pk, 0, cacheLen_ * rowK_);
      if (P_ > 0) std::memcpy(pk + (size_t)(cacheLen_ - P_) * rowK_, kBuf(l) + (size_t)(cacheLen_ - P_) * rowK_, (size_t)P_ * rowK_);
      prefill_.inClean(base_ + l);
      uint8_t* pv = (uint8_t*)prefill_.inPtr(vInBase_ + l);
      std::memset(pv, 0, cacheLen_ * rowV_);
      if (P_ > 0) std::memcpy(pv + (size_t)(cacheLen_ - P_) * rowV_, vBuf(l) + (size_t)(cacheLen_ - P_) * rowV_, (size_t)P_ * rowV_);
      prefill_.inClean(vInBase_ + l);
    }
  }

  // Append the chunk's `cl` new KV rows to the right-aligned decode window.
  void appendPrefill(int cl) {
    curLogits_ = (int16_t*)prefill_.outPtr(0) + (size_t)(cl - 1) * logitStride_;
    for (int l = 0; l < nLayers_; ++l) {
      uint8_t* k = kBuf(l);
      if (P_ > 0) std::memmove(k + (size_t)(cacheLen_ - P_ - cl) * rowK_, k + (size_t)(cacheLen_ - P_) * rowK_, (size_t)P_ * rowK_);
      std::memcpy(k + (size_t)(cacheLen_ - cl) * rowK_, prefill_.outPtr(1 + l), (size_t)cl * rowK_);
      decode_.inClean(base_ + l);
      uint8_t* v = vBuf(l);
      if (P_ > 0) std::memmove(v + (size_t)(cacheLen_ - P_ - cl) * rowV_, v + (size_t)(cacheLen_ - P_) * rowV_, (size_t)P_ * rowV_);
      std::memcpy(v + (size_t)(cacheLen_ - cl) * rowV_, prefill_.outPtr(vOutBase_ + l), (size_t)cl * rowV_);
      decode_.inClean(vInBase_ + l);
    }
    P_ += cl;
  }

  // ── token-input graphs ─────────────────────────────────────────────────────
  void decodeStepToken(int token) {
    if (mode_ == InputMode::Embed) {
      if (!embedder_) throw std::runtime_error("[native] embed-mode decode needs set_embedder()");
      embedder_(token, rowBuf_.data());
      decodeStepEmbed(rowBuf_.data(), Pos3::scalar(maxPos_ + 1));
      return;
    }
    *(int32_t*)decode_.inPtr(0) = token; decode_.inClean(0);
    *(int32_t*)decode_.inPtr(1) = P_;    decode_.inClean(1);
    writeDecodeMask((int16_t*)decode_.inPtr(2));
    decode_.inClean(2);
    decode_.infer();
    curLogits_ = (int16_t*)decode_.outPtr(0);
    rollCaches();
  }

  void prefillTokens(const int* ids, int cl) {
    auto* tk = (int32_t*)prefill_.inPtr(0);
    std::memset(tk, 0, chunk_ * sizeof(int32_t));
    for (int i = 0; i < cl; ++i) tk[i] = ids[i];
    prefill_.inClean(0);
    auto* pos = (int32_t*)prefill_.inPtr(1);
    for (int i = 0; i < chunk_; ++i) pos[i] = P_ + i;
    prefill_.inClean(1);
    writePrefillMask((int16_t*)prefill_.inPtr(2));
    prefill_.inClean(2);
    loadPrefillCaches();
    prefill_.infer();
    const int p0 = P_;
    appendPrefill(cl);
    maxPos_ = p0 + cl - 1;
  }

  // ── embed-input graphs (host embedding + host rope) ────────────────────────
  void writeRope(int16_t* cs, int16_t* sn, Pos3 p) const {
    const int half = rot_ / 2;
    const int32_t pv[3] = {p.t, p.h, p.w};
    for (int i = 0; i < half; ++i) {
      const double ang = (double)pv[sec_[i]] * inv_[i];
      const int16_t c = native_detail::quantS16((float)std::cos(ang), ropeInvScale_);
      const int16_t s = native_detail::quantS16((float)std::sin(ang), ropeInvScale_);
      cs[i] = cs[i + half] = c;
      sn[i] = sn[i + half] = s;
    }
  }
  void quantRow(const float* src, int16_t* dst) const {
    for (int i = 0; i < hidden_; ++i) dst[i] = native_detail::quantS16(src[i], embInvScale_);
  }

  void decodeStepEmbed(const float* row, Pos3 pos) {
    quantRow(row, (int16_t*)decode_.inPtr(0)); decode_.inClean(0);
    writeDecodeMask((int16_t*)decode_.inPtr(1)); decode_.inClean(1);
    writeRope((int16_t*)decode_.inPtr(2), (int16_t*)decode_.inPtr(3), pos);
    decode_.inClean(2); decode_.inClean(3);
    decode_.infer();
    curLogits_ = (int16_t*)decode_.outPtr(0);
    rollCaches();
    maxPos_ = std::max(maxPos_, pos.max());
  }

  void prefillEmbeds(const float* rows, const Pos3* pos, int cl) {
    auto* x = (int16_t*)prefill_.inPtr(0);
    std::memset(x, 0, (size_t)chunk_ * hidden_ * sizeof(int16_t));
    for (int i = 0; i < cl; ++i) quantRow(rows + (size_t)i * hidden_, x + (size_t)i * hidden_);
    prefill_.inClean(0);
    writePrefillMask((int16_t*)prefill_.inPtr(1));
    prefill_.inClean(1);
    auto* cs = (int16_t*)prefill_.inPtr(2);
    auto* sn = (int16_t*)prefill_.inPtr(3);
    std::memset(cs, 0, (size_t)chunk_ * rot_ * sizeof(int16_t));
    std::memset(sn, 0, (size_t)chunk_ * rot_ * sizeof(int16_t));
    for (int i = 0; i < cl; ++i) writeRope(cs + (size_t)i * rot_, sn + (size_t)i * rot_, pos[i]);
    prefill_.inClean(2); prefill_.inClean(3);
    loadPrefillCaches();
    prefill_.infer();
    appendPrefill(cl);
    for (int i = 0; i < cl; ++i) maxPos_ = std::max(maxPos_, pos[i].max());
  }

  void feedEmbedsNoMark(const float* rows, int n, const Pos3* pos) {
    if (mode_ != InputMode::Embed) throw std::runtime_error("[native] feed_embeds() needs an embed-input graph");
    std::vector<Pos3> seq;
    if (!pos) {
      seq.resize(n);
      for (int i = 0; i < n; ++i) seq[i] = Pos3::scalar(maxPos_ + 1 + i);
      pos = seq.data();
    }
    int i = 0;
    while (i < n) {
      const int left = n - i;
      if (!prefillWorthwhile(left)) break;
      const int cl = std::min(left, chunk_);
      prefillEmbeds(rows + (size_t)i * hidden_, pos + i, cl);
      i += cl;
    }
    for (; i < n; ++i) decodeStepEmbed(rows + (size_t)i * hidden_, pos[i]);
  }

  // ── masks (identical for both contracts) ──────────────────────────────────
  // decode: the graph shifts its input window left and appends the new row, so
  // the valid slots are the last (P+1) of the cacheLen-wide window.
  void writeDecodeMask(int16_t* mask) const {
    int lo = cacheLen_ - 1 - P_; if (lo < 0) lo = 0;
    for (int j = 0; j < cacheLen_; ++j) mask[j] = (j >= lo) ? 0 : -32768;
  }
  // prefill: token i lands at window slot base+i, and attends everything from the
  // start of the already-cached rows (base-P) up to itself.
  void writePrefillMask(int16_t* m) const {
    const int base = cacheLen_ - chunk_;
    int lo = base - P_; if (lo < 0) lo = 0;
    for (int i = 0; i < chunk_; ++i)
      for (int j = 0; j < cacheLen_; ++j)
        m[(size_t)i * cacheLen_ + j] = (j >= lo && j <= base + i) ? 0 : -32768;
  }

  hbDNNPackedHandle_t packed_ = nullptr;
  native_detail::Graph prefill_, decode_;
  InputMode mode_ = InputMode::Token;
  int nLayers_ = 0, nKV_ = 0, headDim_ = 0, cacheLen_ = 0, chunk_ = 0, vocab_ = 0;
  int hidden_ = 0, rot_ = 0;
  size_t rowK_ = 0, rowV_ = 0;
  int base_ = 3, vInBase_ = 0, vOutBase_ = 0, logitStride_ = 0;
  float logitScale_ = 1.0f, embInvScale_ = 1.0f, ropeInvScale_ = 1.0f;
  int P_ = 0, maxPos_ = -1;
  const int16_t* curLogits_ = nullptr;
  std::chrono::steady_clock::time_point tTurn_ = std::chrono::steady_clock::now();
  NativeStats stats_;
  std::function<void(int, float*)> embedder_;
  std::vector<double> inv_;
  std::vector<int> sec_;
  std::vector<float> rowBuf_;
};

}  // namespace bllm
