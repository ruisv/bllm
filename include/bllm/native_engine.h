// bllm::NativeEngine — the self-built autoregressive LLM engine that drives a
// compiled `.hbm` (prefill + decode dual-graph) directly on the generic hbDNN /
// hbUCP BPU runtime, WITHOUT libxlm. Header-only so the CLI (examples/selfengine)
// and the Python binding (python/_bllm_native) share one implementation.
//
// Stateful: the KV cache lives in the decode graph's input buffers and persists
// across feed()/generate(), giving multi-turn chat + chunked prefill. Every model
// shape is introspected from the graph (model-agnostic). Speaks token ids in/out;
// tokenization lives above this class.
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
  const hbDNNTensorShape& inShape(int i) const { return in[i].properties.validShape; }
  const hbDNNTensorShape& outShape(int i) const { return out[i].properties.validShape; }
  int inType(int i) const { return in[i].properties.tensorType; }
  float outScale(int i) const {
    const auto& s = out[i].properties.scale;
    return (s.scaleData && s.scaleLen > 0) ? s.scaleData[0] : 1.0f;
  }
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

class NativeEngine {
 public:
  explicit NativeEngine(const std::string& hbm) {
    const char* files[] = {hbm.c_str()};
    BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&packed_, files, 1));
    prefill_.init(packed_, "prefill");
    decode_.init(packed_, "decode");
    const int nCache = decode_.inCount() - 3;
    nLayers_ = nCache / 2;
    vInBase_ = 3 + nLayers_;
    vOutBase_ = 1 + nLayers_;
    const auto& ks = decode_.inShape(3);
    cacheLen_ = ks.dimensionSize[0]; nKV_ = ks.dimensionSize[1]; headDim_ = ks.dimensionSize[2];
    rowK_ = (size_t)nKV_ * headDim_ * native_detail::elemSize(decode_.inType(3));
    rowV_ = (size_t)nKV_ * headDim_ * native_detail::elemSize(decode_.inType(vInBase_));
    const auto& ls = decode_.outShape(0);
    vocab_ = ls.dimensionSize[ls.numDimensions - 1];
    logitStride_ = native_detail::alignUp((int64_t)vocab_ * 2, native_detail::kAlign) / 2;
    logitScale_ = decode_.outScale(0);
    const auto& ps = prefill_.inShape(0);
    chunk_ = ps.dimensionSize[ps.numDimensions - 1];
    reset();
  }
  ~NativeEngine() { if (packed_) hbDNNRelease(packed_); }
  NativeEngine(const NativeEngine&) = delete;
  NativeEngine& operator=(const NativeEngine&) = delete;

  int n_layers() const { return nLayers_; }
  int n_kv() const { return nKV_; }
  int head_dim() const { return headDim_; }
  int cache_len() const { return cacheLen_; }
  int vocab() const { return vocab_; }
  int chunk() const { return chunk_; }
  int position() const { return P_; }
  const NativeStats& last_stats() const { return stats_; }

  void reset() {
    for (int l = 0; l < nLayers_; ++l) {
      std::memset(kBuf(l), 0, cacheLen_ * rowK_); decode_.inClean(3 + l);
      std::memset(vBuf(l), 0, cacheLen_ * rowV_); decode_.inClean(vInBase_ + l);
    }
    P_ = 0; curLogits_ = nullptr;
  }

  // ingest tokens (prompt or a new conversation turn) into the cache.
  void feed(const std::vector<int>& ids) {
    size_t i = 0;
    if (P_ == 0 && !ids.empty()) {
      int cl = std::min((int)ids.size(), chunk_);
      prefillFirst(ids, cl);
      i = cl;
    }
    for (; i < ids.size(); ++i) decodeStep(ids[i]);
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
    auto t0 = clk::now(); double t_first = -1;
    for (int step = 0; step < p.max_new && curLogits_; ++step) {
      int tok = s.pick(curLogits_, logitScale_, vocab_);
      if (eos.count(tok)) break;
      if (t_first < 0) t_first = std::chrono::duration<double>(clk::now() - t0).count();
      gen.push_back(tok);
      if (on_token) on_token(tok);
      s.record(tok);
      decodeStep(tok);
    }
    double t_last = std::chrono::duration<double>(clk::now() - t0).count();
    stats_.ntok = (int)gen.size();
    stats_.ttft_ms = (t_first < 0 ? 0 : t_first) * 1000.0;
    stats_.decode_tps = (stats_.ntok > 1 && t_last > t_first) ? (stats_.ntok - 1) / (t_last - t_first) : 0.0;
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

  uint8_t* kBuf(int l) { return (uint8_t*)decode_.inPtr(3 + l); }
  uint8_t* vBuf(int l) { return (uint8_t*)decode_.inPtr(vInBase_ + l); }

  void decodeStep(int token) {
    *(int32_t*)decode_.inPtr(0) = token; decode_.inClean(0);
    *(int32_t*)decode_.inPtr(1) = P_;    decode_.inClean(1);
    auto* mask = (int16_t*)decode_.inPtr(2);
    int lo = cacheLen_ - 1 - P_; if (lo < 0) lo = 0;
    for (int j = 0; j < cacheLen_; ++j) mask[j] = (j >= lo) ? 0 : -32768;
    decode_.inClean(2);
    decode_.infer();
    curLogits_ = (int16_t*)decode_.outPtr(0);
    for (int l = 0; l < nLayers_; ++l) {
      uint8_t* k = kBuf(l);
      std::memmove(k, k + rowK_, (cacheLen_ - 1) * rowK_);
      std::memcpy(k + (cacheLen_ - 1) * rowK_, decode_.outPtr(1 + l), rowK_);
      decode_.inClean(3 + l);
      uint8_t* v = vBuf(l);
      std::memmove(v, v + rowV_, (cacheLen_ - 1) * rowV_);
      std::memcpy(v + (cacheLen_ - 1) * rowV_, decode_.outPtr(vOutBase_ + l), rowV_);
      decode_.inClean(vInBase_ + l);
    }
    ++P_;
  }

  void prefillFirst(const std::vector<int>& ids, int cl) {
    auto* tk = (int32_t*)prefill_.inPtr(0);
    std::memset(tk, 0, chunk_ * sizeof(int32_t));
    for (int i = 0; i < cl; ++i) tk[i] = ids[i];
    prefill_.inClean(0);
    auto* pos = (int32_t*)prefill_.inPtr(1);
    for (int i = 0; i < chunk_; ++i) pos[i] = i;
    prefill_.inClean(1);
    const int base = cacheLen_ - chunk_;
    auto* m = (int16_t*)prefill_.inPtr(2);
    for (int i = 0; i < chunk_; ++i)
      for (int j = 0; j < cacheLen_; ++j)
        m[i * cacheLen_ + j] = (j >= base && j <= base + i) ? 0 : -32768;
    prefill_.inClean(2);
    for (int l = 0; l < nLayers_; ++l) {
      std::memset(prefill_.inPtr(3 + l), 0, cacheLen_ * rowK_); prefill_.inClean(3 + l);
      std::memset(prefill_.inPtr(vInBase_ + l), 0, cacheLen_ * rowV_); prefill_.inClean(vInBase_ + l);
    }
    prefill_.infer();
    curLogits_ = (int16_t*)prefill_.outPtr(0) + (size_t)(cl - 1) * logitStride_;
    for (int l = 0; l < nLayers_; ++l) {
      std::memcpy(kBuf(l) + (size_t)(cacheLen_ - cl) * rowK_, prefill_.outPtr(1 + l), (size_t)cl * rowK_);
      decode_.inClean(3 + l);
      std::memcpy(vBuf(l) + (size_t)(cacheLen_ - cl) * rowV_, prefill_.outPtr(vOutBase_ + l), (size_t)cl * rowV_);
      decode_.inClean(vInBase_ + l);
    }
    P_ = cl;
  }

  hbDNNPackedHandle_t packed_ = nullptr;
  native_detail::Graph prefill_, decode_;
  int nLayers_ = 0, nKV_ = 0, headDim_ = 0, cacheLen_ = 0, chunk_ = 0, vocab_ = 0;
  size_t rowK_ = 0, rowV_ = 0;
  int vInBase_ = 0, vOutBase_ = 0, logitStride_ = 0;
  float logitScale_ = 1.0f;
  int P_ = 0;
  const int16_t* curLogits_ = nullptr;
  NativeStats stats_;
};

}  // namespace bllm
