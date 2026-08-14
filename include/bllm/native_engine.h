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
//
#pragma once

#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bllm/native_sampler.h"   // bllm::NativeSamplingParams + bllm::Sampler

namespace bllm {
namespace native_detail {

inline void check(int32_t e, const char* what) {
  if (e != 0) throw std::runtime_error(std::string("[native] hbDNN error: ") + what);
}
#define BLLM_NATIVE_CK(expr) ::bllm::native_detail::check((expr), #expr)

constexpr int64_t kAlign = 32;
inline int64_t alignUp(int64_t v, int64_t a) { return (v + (a - 1)) & ~(a - 1); }

// Graphs from the OpenExplorer pi0 pipeline carry fp16 at every boundary. The
// board is aarch64, where `__fp16` is a storage-and-arithmetic type the compiler
// converts in one instruction; nothing else in this header needs a software
// half, so there is no fallback path to keep correct.
#if defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
using half_t = __fp16;
#else
#error "native_engine: fp16 tensor I/O needs __fp16 (aarch64); this header is board-only"
#endif

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
  // A view into this allocation at `off` bytes. hbDNN only needs {phyAddr, virAddr},
  // so a sub-view of a registered allocation binds like any other tensor buffer.
  hbUCPSysMem view(uint64_t off) const { return {m.phyAddr + off, (uint8_t*)m.virAddr + off, m.memSize - off}; }
  void flush(uint64_t off, uint64_t len, int flag) const {
    const hbUCPSysMem s{m.phyAddr + off, (uint8_t*)m.virAddr + off, len};
    BLLM_NATIVE_CK(hbUCPMemFlush(&s, flag));
  }
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

// How BPU tasks are submitted. Shared by every graph a session owns (text tower,
// vision/audio towers), because they contend for the same accelerator.
//
// `priority` (0..255) is RELATIVE, and it only bites where the running graph offers
// preemption points. Sharing a board with a vision pipeline needs BOTH halves:
//   * the LLM `.hbm` compiled with `--max_time_per_fc` (µs), which caps each function
//     call so the scheduler can switch inside a big graph, and
//   * the LLM submitting BELOW the vision pipeline (the default here is LOWEST).
// Measured on S100P with yolo26s alongside Qwen2.5-1.5B decode — CV p99 latency:
//   unbounded fc, equal priority   41.3 ms      (CV waits out a whole decode step)
//   unbounded fc, CV at 255        13.5 ms
//   1 ms fc,      equal priority   41.0 ms
//   1 ms fc,      CV at 255         2.95 ms     (idle CV is 2.06 ms)
// The bound is nearly free on an idle BPU (24.00 vs 24.36 tok/s); the LLM only pays
// when it actually yields (23.9 -> 17.1 tok/s under a CV saturating the core).
//
// `core_mask` picks BPU cores (HB_UCP_BPU_CORE_0, …). S100/S100P expose exactly ONE
// BPU core, so it is a no-op there; it matters on multi-core parts (nash-p / J6P).
struct BpuSched {
  int priority = HB_UCP_PRIORITY_LOWEST;   // yield the queue to latency-critical work
  uint64_t core_mask = HB_UCP_CORE_ANY;
  uint32_t device_id = 0;
};

inline void submitAndWait(hbUCPTaskHandle_t task, const BpuSched& s) {
  hbUCPSchedParam sched;
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
  sched.priority = s.priority;
  sched.backend = s.core_mask;
  sched.deviceId = s.device_id;
  check(hbUCPSubmitTask(task, &sched), "hbUCPSubmitTask");
  check(hbUCPWaitTaskDone(task, 0), "hbUCPWaitTaskDone");
}

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
  int outType(int i) const { return out[i].properties.tensorType; }
  // Bytes to advance index `dim` by one. Dtype- and padding-agnostic, unlike a hand
  // computed stride.
  int64_t outStride(int i, int dim) const { return out[i].properties.stride[dim]; }
  int64_t inStride(int i, int dim) const { return in[i].properties.stride[dim]; }
  // Rows are padded up to kAlign, so a dense [rows, cols] buffer is only safe to
  // memcpy when cols*sizeof(float) already lands on that boundary. It usually
  // does (Qwen2.5-Omni's 1176-float patch row is 4704 bytes), which is exactly
  // why writing dense and hoping goes unnoticed until a tower whose row is not —
  // InternViT's 588 floats are 2352 bytes, 16 short of the next multiple of 32,
  // and every row after the first lands askew with no error anywhere.
  // Byte pitch between consecutive dense rows of `cols` elements, for a tensor of
  // any rank. Callers flatten trailing dimensions freely — a [512, 8, 128] cache
  // is naturally written as 512 rows of 1024, and a [1, N, W] mask as N rows of
  // W — so the padded axis is whichever one has exactly `cols` elements below it,
  // not a fixed dimension index.
  static int64_t rowPitch(const hbDNNTensorShape& s, const int64_t* stride, int64_t cols) {
    int64_t below = 1;
    for (int d = s.numDimensions - 1; d >= 0; --d) {
      if (below == cols) return stride[d];
      below *= s.dimensionSize[d];
    }
    return cols * (int64_t)sizeof(float);   // the whole tensor is one row
  }

  void writeRowsF32(int i, const float* src, int64_t rows, int64_t cols) {
    const int64_t pitch = rowPitch(inShape(i), in[i].properties.stride, cols);
    auto* dst = (uint8_t*)inPtr(i);
    if (pitch == cols * (int64_t)sizeof(float)) {
      std::memcpy(dst, src, (size_t)rows * cols * sizeof(float));
    } else {
      for (int64_t r = 0; r < rows; ++r)
        std::memcpy(dst + r * pitch, src + r * cols, (size_t)cols * sizeof(float));
    }
    inClean(i);
  }
  // `writeRowsF32` is a raw copy and assumes the graph takes float32, which our
  // own graphs do. Graphs built by the OpenExplorer pi0 pipeline are **fp16 in
  // and out**, with int64 position ids, so feeding them needs conversion. These
  // two speak f32 to the caller and whatever the tensor declares to the board.
  void writeRowsF32As(int i, const float* src, int64_t rows, int64_t cols) {
    const int64_t pitch = rowPitch(inShape(i), in[i].properties.stride, cols);
    auto* dst = (uint8_t*)inPtr(i);
    const int type = inType(i);
    for (int64_t r = 0; r < rows; ++r) {
      const float* s = src + r * cols;
      uint8_t* d = dst + r * pitch;
      switch (type) {
        case HB_DNN_TENSOR_TYPE_F32:
          std::memcpy(d, s, (size_t)cols * sizeof(float));
          break;
        case HB_DNN_TENSOR_TYPE_F16: {
          auto* p = (native_detail::half_t*)d;
          for (int64_t k = 0; k < cols; ++k) p[k] = (native_detail::half_t)s[k];
          break;
        }
        case HB_DNN_TENSOR_TYPE_S64: {
          auto* p = (int64_t*)d;
          for (int64_t k = 0; k < cols; ++k) p[k] = (int64_t)std::llround(s[k]);
          break;
        }
        case HB_DNN_TENSOR_TYPE_S32: {
          auto* p = (int32_t*)d;
          for (int64_t k = 0; k < cols; ++k) p[k] = (int32_t)std::lround(s[k]);
          break;
        }
        default:
          throw std::runtime_error("[native] writeRowsF32As: unsupported input type " +
                                   std::to_string(type));
      }
    }
    inClean(i);
  }

  void readRowsF32As(int i, float* dst, int64_t rows, int64_t cols) const {
    const int64_t pitch = rowPitch(outShape(i), out[i].properties.stride, cols);
    const auto* base = (const uint8_t*)out[i].sysMem.virAddr;
    const int type = out[i].properties.tensorType;
    const float scale = scaleOf(out[i].properties);
    for (int64_t r = 0; r < rows; ++r) {
      const uint8_t* s = base + r * pitch;
      float* d = dst + r * cols;
      switch (type) {
        case HB_DNN_TENSOR_TYPE_F32:
          std::memcpy(d, s, (size_t)cols * sizeof(float));
          break;
        case HB_DNN_TENSOR_TYPE_F16: {
          const auto* p = (const native_detail::half_t*)s;
          for (int64_t k = 0; k < cols; ++k) d[k] = (float)p[k];
          break;
        }
        case HB_DNN_TENSOR_TYPE_S16: {
          const auto* p = (const int16_t*)s;
          for (int64_t k = 0; k < cols; ++k) d[k] = p[k] * scale;
          break;
        }
        case HB_DNN_TENSOR_TYPE_S8: {
          const auto* p = (const int8_t*)s;
          for (int64_t k = 0; k < cols; ++k) d[k] = p[k] * scale;
          break;
        }
        default:
          throw std::runtime_error("[native] readRowsF32As: unsupported output type " +
                                   std::to_string(type));
      }
    }
  }

  int64_t outRowPitch(int i, int64_t cols) const {
    return rowPitch(outShape(i), out[i].properties.stride, cols);
  }
  static float scaleOf(const hbDNNTensorProperties& p) {
    const auto& s = p.scale;
    return (s.scaleData && s.scaleLen > 0) ? s.scaleData[0] : 1.0f;
  }
  float inScale(int i) const { return scaleOf(in[i].properties); }
  float outScale(int i) const { return scaleOf(out[i].properties); }
  void inClean(int i) { inMem[i].clean(); }
  void* inPtr(int i) { return inMem[i].p(); }
  void* outPtr(int i) { return outMem[i].p(); }
  BpuSched sched;
  void infer() {
    hbUCPTaskHandle_t task = nullptr;
    BLLM_NATIVE_CK(hbDNNInferV2(&task, out.data(), in.data(), h));
    submitAndWait(task, sched);
    hbUCPReleaseTask(task);
    for (auto& m : outMem) m.inval();
  }
};

inline int argmaxS16(const int16_t* row, int n) {
  int best = 0; int16_t bv = row[0];
  for (int i = 1; i < n; ++i) if (row[i] > bv) { bv = row[i]; best = i; }
  return best;
}

// Which of the three mrope axes (0=t, 1=h, 2=w) each rope frequency reads, for
// `half` = rot/2 frequencies.
//
// Two families, and they are NOT interchangeable:
//   * CONTIGUOUS (Qwen2.5-Omni, section {16,24,24}) — a run of t, then h, then w.
//   * INTERLEAVED (Qwen3.5, section {11,11,10}) — [THWTHWTHW...TT]. Upstream builds
//     it by starting from all-t and overwriting strided slices: h takes
//     range(1, sec[1]*3, 3) and w takes range(2, sec[2]*3, 3), so anything past
//     those strided runs stays t. Reproduced exactly, not approximated by i%3 —
//     that shortcut happens to agree for {11,11,10} but not in general.
// For text tokens t==h==w, so both collapse to plain 1-D rope; the layout only
// starts to matter once an image supplies differing h/w.
inline std::vector<uint8_t> mropeAxisMap(int half, const std::vector<int>& section,
                                         bool interleaved) {
  std::vector<uint8_t> sec((size_t)half, 0);
  if (section.empty()) return sec;
  if (interleaved) {
    for (int d = 1; d <= 2 && d < (int)section.size(); ++d)
      for (int i = d; i < section[d] * 3 && i < half; i += 3) sec[i] = (uint8_t)d;
    return sec;
  }
  int d = 0;
  for (int s = 0; s < (int)section.size() && d < half; ++s)
    for (int k = 0; k < section[s] && d < half; ++k) sec[d++] = (uint8_t)s;
  if (d != half) throw std::runtime_error("[native] mrope_section does not cover rot/2");
  return sec;
}

}  // namespace native_detail

// NativeSamplingParams now lives in native_sampler.h (shared with bllm::Sampler).

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
    // Fail with a HINT, not a cryptic "Model not exists: prefill". The most common
    // mistake is pointing this DENSE engine at a HYBRID (Qwen3.5 Gated-DeltaNet/SSM)
    // .hbm, whose graph is named "qwen35" (no prefill/decode) and which also needs a
    // host embed table — so the dense prefill/decode lookup below would abort deep in
    // the DNN layer with no guidance. Check the graph names first and say what to use.
    {
      const char** names = nullptr;
      int32_t count = 0;
      hbDNNGetModelNameList(&names, &count, packed_);
      bool hasPrefill = false, hasDecode = false;
      std::string avail;
      for (int32_t i = 0; i < count && names; ++i) {
        if (!names[i]) continue;
        if (!avail.empty()) avail += ", ";
        avail += names[i];
        if (std::string(prefill_name) == names[i]) hasPrefill = true;
        if (std::string(decode_name) == names[i]) hasDecode = true;
      }
      if (!hasPrefill || !hasDecode)
        throw std::runtime_error(
            "[native] this .hbm has no dense '" + std::string(prefill_name) + "'/'" +
            std::string(decode_name) + "' graphs (found: [" + avail + "]).\n"
            "  NativeEngine / bllm_selfengine drives DENSE models only.\n"
            "  For a hybrid Qwen3.5 (Gated-DeltaNet/SSM) model use the hybrid path:\n"
            "    bllm_qwen35 --hbm <model>.hbm --embed <embed_tokens>.bin --ids \"..\"\n"
            "  or, simplest, bllm.load(<model_dir>) which auto-routes by arch.");
    }
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

    // Zero-copy KV: hold each layer's cache in a ring longer than the window, so a
    // decode step just slides the window one row and lets the BPU write the new row
    // into the freed slot. No per-token memmove, and no flush at all — the caches are
    // BPU-write then BPU-read. Compaction happens once every `slack_` tokens.
    slack_ = std::max(512, chunk_ + 2);
    ringRows_ = cacheLen_ + slack_;
    kRing_.resize(nLayers_); vRing_.resize(nLayers_);
    for (int l = 0; l < nLayers_; ++l) {
      kRing_[l].alloc((uint64_t)(ringRows_ + 1) * rowK_);   // +1 row of slop for tensor padding
      vRing_[l].alloc((uint64_t)(ringRows_ + 1) * rowV_);
      decode_.inMem[base_ + l] = Mem{};                     // free the graph's own buffers
      decode_.inMem[vInBase_ + l] = Mem{};
      decode_.outMem[1 + l] = Mem{};
      decode_.outMem[vOutBase_ + l] = Mem{};
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
  // Tokens that still fit. The KV window IS the context: these graphs are compiled
  // for full attention over cache_len slots, and rolling past it — evicting the
  // earliest tokens, which act as attention sinks — collapses the model into
  // gibberish. Measured on both the prefill and the decode-step path, byte for byte.
  // So the engine refuses to overflow rather than degrade silently.
  int context_left() const { return cacheLen_ - P_; }
  const NativeStats& last_stats() const { return stats_; }

  // Per-token logprobs for the last generate(), one entry per generated token, in order.
  // Empty unless that generate() ran with params.logprobs >= 0.
  const std::vector<TokenLogprobs>& last_logprobs() const { return logprobs_; }

  // How this engine's BPU tasks are queued. Default: lowest priority, any core, so a
  // co-resident vision pipeline wins the queue. See native_detail::BpuSched — priority
  // orders the queue but does not preempt a running graph.
  void set_sched(const native_detail::BpuSched& s) { prefill_.sched = s; decode_.sched = s; }
  const native_detail::BpuSched& sched() const { return decode_.sched; }

  // Embed-mode only: how the engine turns a sampled token id back into a hidden
  // row (normally EmbedTable::lookup). Required before generate().
  void set_embedder(std::function<void(int, float*)> fn) { embedder_ = std::move(fn); }

  // Embed-mode only: rope base + mrope section split (sums to rot/2; empty = plain).
  // `interleaved` picks Qwen3.5's [THWTHW...] layout over Omni's contiguous runs —
  // see native_detail::mropeAxisMap.
  void set_rope(double theta, std::vector<int> mrope_section, bool interleaved = false) {
    inv_.resize(rot_ / 2);
    for (int i = 0; i < rot_ / 2; ++i) inv_[i] = std::pow(theta, -(double)(2 * i) / rot_);
    sec_ = native_detail::mropeAxisMap(rot_ / 2, mrope_section, interleaved);
  }

  void reset() {
    W_ = 0;
    for (int l = 0; l < nLayers_; ++l) {
      std::memset(kRing_[l].p(), 0, (size_t)cacheLen_ * rowK_);
      kRing_[l].flush(0, (uint64_t)cacheLen_ * rowK_, HB_SYS_MEM_CACHE_CLEAN);
      std::memset(vRing_[l].p(), 0, (size_t)cacheLen_ * rowV_);
      vRing_[l].flush(0, (uint64_t)cacheLen_ * rowV_, HB_SYS_MEM_CACHE_CLEAN);
    }
    P_ = 0; maxPos_ = -1; curLogits_ = nullptr;
  }

  // Save/restore the conversation's KV state to a file — libxlm's path_prompt_cache. A long
  // shared prefix can be fed once, saved, and reloaded to skip re-prefill. The exact last
  // logits are saved too, so generate() resumes bit-identically. Bound to this model's
  // shape (rejected on load if it differs) and to a context within the window (P_ <=
  // cache_len). Not for the sliding-window regime.
  void save_state(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[native] cannot write state: " + path);
    writeState(f);
    if (!f) throw std::runtime_error("[native] failed writing state: " + path);
  }
  void load_state(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[native] cannot read state: " + path);
    readState(f);
  }
  // In-memory snapshot of the current context — same payload as save_state, as a byte blob.
  // Backs the cheap prefix-reuse path (snapshot a prefilled prefix once, restore before each
  // query) without a temp file.
  std::string snapshot() const {
    std::ostringstream os(std::ios::binary);
    const_cast<NativeEngine*>(this)->writeState(os);
    return os.str();
  }
  void restore(const std::string& blob) {
    std::istringstream is(blob, std::ios::binary);
    readState(is);
  }

 private:
  void writeState(std::ostream& f) {
    if (!curLogits_) throw std::runtime_error("[native] no state to save — feed a prompt first");
    if (mode_ != InputMode::Token)
      throw std::runtime_error("[native] save_state is only for the token path");
    if (P_ > cacheLen_) throw std::runtime_error("[native] cannot snapshot a context past cache_len");
    const int32_t hdr[] = {kStateMagic, nLayers_, (int32_t)rowK_, (int32_t)rowV_, vocab_,
                           cacheLen_, P_, maxPos_};
    f.write((const char*)hdr, sizeof(hdr));
    const int D = filled();
    for (int l = 0; l < nLayers_; ++l) {
      kRing_[l].flush((uint64_t)(W_ + cacheLen_ - D) * rowK_, (uint64_t)D * rowK_, HB_SYS_MEM_CACHE_INVALIDATE);
      f.write((const char*)kRow(l, W_ + cacheLen_ - D), (size_t)D * rowK_);
      vRing_[l].flush((uint64_t)(W_ + cacheLen_ - D) * rowV_, (uint64_t)D * rowV_, HB_SYS_MEM_CACHE_INVALIDATE);
      f.write((const char*)vRow(l, W_ + cacheLen_ - D), (size_t)D * rowV_);
    }
    f.write((const char*)curLogits_, (size_t)vocab_ * sizeof(int16_t));
  }
  void readState(std::istream& f) {
    int32_t hdr[8];
    f.read((char*)hdr, sizeof(hdr));
    if (!f || hdr[0] != kStateMagic || hdr[1] != nLayers_ || hdr[2] != (int32_t)rowK_ ||
        hdr[3] != (int32_t)rowV_ || hdr[4] != vocab_ || hdr[5] != cacheLen_)
      throw std::runtime_error("[native] state does not match this model");
    const int savedP = hdr[6], savedMax = hdr[7], D = savedP;   // P_ <= cache_len at save
    reset();
    for (int l = 0; l < nLayers_; ++l) {
      f.read((char*)kRow(l, cacheLen_ - D), (size_t)D * rowK_);
      kRing_[l].flush((uint64_t)(cacheLen_ - D) * rowK_, (uint64_t)D * rowK_, HB_SYS_MEM_CACHE_CLEAN);
      f.read((char*)vRow(l, cacheLen_ - D), (size_t)D * rowV_);
      vRing_[l].flush((uint64_t)(cacheLen_ - D) * rowV_, (uint64_t)D * rowV_, HB_SYS_MEM_CACHE_CLEAN);
    }
    savedLogits_.resize(vocab_);
    f.read((char*)savedLogits_.data(), (size_t)vocab_ * sizeof(int16_t));
    if (!f) throw std::runtime_error("[native] truncated / corrupt state");
    P_ = savedP; maxPos_ = savedMax; W_ = 0;
    curLogits_ = savedLogits_.data();      // generate() reads this until the first decode step
  }

 public:

  // Start the clock for the next reply's TTFT. feed() does this automatically;
  // call it earlier when work precedes the feed (e.g. a VLM's vision encode).
  void mark_turn_start() { tTurn_ = std::chrono::steady_clock::now(); }

  // ingest tokens (prompt or a new conversation turn) into the cache.
  void feed(const std::vector<int>& ids) {
    mark_turn_start();
    checkRoom((int)ids.size());
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
    checkRoom(n);
    feedEmbedsNoMark(rows, n, pos);
  }

  // Teacher-forced perplexity of `tokens` on a FRESH context (this reset()s the session):
  // exp( mean over i of -log p(token_i | token_<i) ). libxlm's xlm_ppl equivalent.
  // Per-position log-probs need the decode graph one token at a time (prefill only yields
  // the last position's logits), so this costs ~N decode steps.
  double perplexity(const std::vector<int>& tokens) {
    const int N = (int)tokens.size();
    if (N < 2) throw std::runtime_error("[native] perplexity needs >= 2 tokens");
    checkRoom(N);
    reset();
    double lp = 0.0;
    int count = 0;
    decodeStepToken(tokens[0]);                      // curLogits_ now predicts token 1
    for (int i = 1; i < N && curLogits_ && P_ < cacheLen_; ++i) {
      lp += logProbOfNext(tokens[i]);
      ++count;
      if (i + 1 < N) decodeStepToken(tokens[i]);
    }
    return count > 0 ? std::exp(-lp / count) : 0.0;
  }

  // sample up to p.max_new tokens; stop on eos; call on_token(id) per token. on_token may
  // return true to stop after that token (e.g. a session-level stop string matched) —
  // this breaks before the next decode step, so nothing is generated past the stop.
  std::vector<int> generate(const NativeSamplingParams& p,
                            const std::function<bool(int)>& on_token = {}) {
    Sampler s(p);
    // Reads the CURRENT logits each call — decodeStepToken advances curLogits_ per step.
    auto logit = [this](int i) { return curLogits_[i] * logitScale_; };
    std::unordered_set<int> eos(p.eos.begin(), p.eos.end());
    std::vector<int> gen;
    logprobs_.clear();
    using clk = std::chrono::steady_clock;
    // TTFT is measured from the start of the turn (prefill, and any vision encode
    // the caller marked), not from the first sample — the logits already exist here.
    clk::time_point tFirst;
    // Stop at the window rather than evict — see context_left().
    for (int step = 0; step < p.max_new && curLogits_ && P_ < cacheLen_; ++step) {
      int tok = s.pick(logit, vocab_);
      if (tok < 0) break;                 // a constraint (grammar) has nothing left to allow
      if (eos.count(tok)) break;
      if (gen.empty()) { tFirst = clk::now(); stats_.ttft_ms = std::chrono::duration<double>(tFirst - tTurn_).count() * 1000.0; }
      gen.push_back(tok);
      // Report against the logits this token came out of, before the step advances them.
      if (p.logprobs >= 0) logprobs_.push_back(computeLogprobs(logit, vocab_, tok, p.logprobs));
      const bool stop = on_token && on_token(tok);
      if (stop) break;                          // skip the next decode step we won't use
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
  // log p(tok) under a softmax over the FULL vocab of the current logits (exact
  // normaliser: logit[tok] - logsumexp(all logits)). Both in the dequantised domain.
  double logProbOfNext(int tok) const {
    double m = curLogits_[0] * (double)logitScale_;
    for (int v = 1; v < vocab_; ++v) {
      const double x = curLogits_[v] * (double)logitScale_;
      if (x > m) m = x;
    }
    double se = 0.0;
    for (int v = 0; v < vocab_; ++v) se += std::exp(curLogits_[v] * (double)logitScale_ - m);
    return curLogits_[tok] * (double)logitScale_ - (m + std::log(se));
  }

  using Mem = native_detail::Mem;

  void checkRoom(int n) const {
    if (n > context_left())
      throw std::runtime_error("[native] context overflow: " + std::to_string(n) +
                               " tokens do not fit in the remaining " + std::to_string(context_left()) +
                               " of " + std::to_string(cacheLen_) +
                               " cache slots. reset(), shorten the turn, or use a longer-context build.");
  }

  uint8_t* kRow(int l, int row) { return (uint8_t*)kRing_[l].p() + (size_t)row * rowK_; }
  uint8_t* vRow(int l, int row) { return (uint8_t*)vRing_[l].p() + (size_t)row * rowV_; }
  int filled() const { return std::min(P_, cacheLen_); }   // valid rows inside the window

  // Bind the decode graph at the current window: caches in at row W_, and the step's
  // new KV out straight into row W_+cacheLen_ — which is the last row of the window
  // one slide later. The CPU never touches them, so nothing needs flushing.
  void bindDecodeCaches() {
    for (int l = 0; l < nLayers_; ++l) {
      decode_.in[base_ + l].sysMem = kRing_[l].view((uint64_t)W_ * rowK_);
      decode_.in[vInBase_ + l].sysMem = vRing_[l].view((uint64_t)W_ * rowV_);
      decode_.out[1 + l].sysMem = kRing_[l].view((uint64_t)(W_ + cacheLen_) * rowK_);
      decode_.out[vOutBase_ + l].sysMem = vRing_[l].view((uint64_t)(W_ + cacheLen_) * rowV_);
    }
  }

  // Slide the ring back to the start once it runs out of room (every `slack_` tokens).
  void ensureRingRoom(int rows) {
    if (W_ + cacheLen_ + rows <= ringRows_) return;
    const int D = filled();
    for (int l = 0; l < nLayers_; ++l) {
      kRing_[l].flush((uint64_t)(W_ + cacheLen_ - D) * rowK_, (uint64_t)D * rowK_, HB_SYS_MEM_CACHE_INVALIDATE);
      std::memmove(kRow(l, cacheLen_ - D), kRow(l, W_ + cacheLen_ - D), (size_t)D * rowK_);
      kRing_[l].flush((uint64_t)(cacheLen_ - D) * rowK_, (uint64_t)D * rowK_, HB_SYS_MEM_CACHE_CLEAN);
      vRing_[l].flush((uint64_t)(W_ + cacheLen_ - D) * rowV_, (uint64_t)D * rowV_, HB_SYS_MEM_CACHE_INVALIDATE);
      std::memmove(vRow(l, cacheLen_ - D), vRow(l, W_ + cacheLen_ - D), (size_t)D * rowV_);
      vRing_[l].flush((uint64_t)(cacheLen_ - D) * rowV_, (uint64_t)D * rowV_, HB_SYS_MEM_CACHE_CLEAN);
    }
    W_ = 0;
  }

  // A prefill pass costs a whole static chunk of compute, so it only beats
  // decode-stepping once enough tokens are left to amortize it. The crossover is
  // (prefill pass) / (decode step), which is stable across models because both are a
  // single BPU graph invocation — measured on an S100P:
  //
  //   qwen2.5-1.5b   prefill 135.2 ms   decode 46.1 ms   -> 2.93 tokens
  //   phi-4-mini     prefill 351.6 ms   decode 106.2 ms  -> 3.31 tokens
  //
  // 4 rather than 3, so the widest measured crossover (3.31) still comes out ahead.
  // This was 16, which made a 15-token turn cost 676 ms instead of 134 ms — short
  // follow-ups ("go on", "why?") are exactly the common case it punished.
  //
  // NOTE: changing this changes SEGMENTATION, and prefill and decode are not
  // numerically identical under int8, so it changes replies for runs of 4..15 tokens.
  // It does NOT affect the prefix cache's alignment rule: feed()'s chunk boundaries
  // depend only on how many tokens remain, so they stay at multiples of chunk_.
  static constexpr int kPrefillMinTokens = 4;
  bool prefillWorthwhile(int left) const {
    return left >= kPrefillMinTokens && P_ + chunk_ <= cacheLen_;
  }

  // Like decode (which shifts its window left by one and appends the new row), the
  // prefill graph shifts its input window left by a whole chunk and writes the chunk's
  // KV into the freed tail. So the cached rows go in with the SAME right-aligned
  // layout the decode window uses; the graph relocates them to [base-P, base).
  // These rows were written by the BPU, so invalidate before the CPU reads them.
  void loadPrefillCaches() {
    for (int l = 0; l < nLayers_; ++l) {
      uint8_t* pk = (uint8_t*)prefill_.inPtr(base_ + l);
      std::memset(pk, 0, cacheLen_ * rowK_);
      uint8_t* pv = (uint8_t*)prefill_.inPtr(vInBase_ + l);
      std::memset(pv, 0, cacheLen_ * rowV_);
      if (P_ > 0) {
        kRing_[l].flush((uint64_t)(W_ + cacheLen_ - P_) * rowK_, (uint64_t)P_ * rowK_, HB_SYS_MEM_CACHE_INVALIDATE);
        std::memcpy(pk + (size_t)(cacheLen_ - P_) * rowK_, kRow(l, W_ + cacheLen_ - P_), (size_t)P_ * rowK_);
        vRing_[l].flush((uint64_t)(W_ + cacheLen_ - P_) * rowV_, (uint64_t)P_ * rowV_, HB_SYS_MEM_CACHE_INVALIDATE);
        std::memcpy(pv + (size_t)(cacheLen_ - P_) * rowV_, vRow(l, W_ + cacheLen_ - P_), (size_t)P_ * rowV_);
      }
      prefill_.inClean(base_ + l);
      prefill_.inClean(vInBase_ + l);
    }
  }

  // Append the chunk's `cl` new KV rows just past the window, then slide by cl — the
  // cached rows keep their addresses, so there is nothing to move.
  void appendPrefill(int cl) {
    curLogits_ = (int16_t*)prefill_.outPtr(0) + (size_t)(cl - 1) * logitStride_;
    for (int l = 0; l < nLayers_; ++l) {
      std::memcpy(kRow(l, W_ + cacheLen_), prefill_.outPtr(1 + l), (size_t)cl * rowK_);
      kRing_[l].flush((uint64_t)(W_ + cacheLen_) * rowK_, (uint64_t)cl * rowK_, HB_SYS_MEM_CACHE_CLEAN);
      std::memcpy(vRow(l, W_ + cacheLen_), prefill_.outPtr(vOutBase_ + l), (size_t)cl * rowV_);
      vRing_[l].flush((uint64_t)(W_ + cacheLen_) * rowV_, (uint64_t)cl * rowV_, HB_SYS_MEM_CACHE_CLEAN);
    }
    W_ += cl;
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
    ensureRingRoom(1);
    bindDecodeCaches();
    decode_.infer();
    curLogits_ = (int16_t*)decode_.outPtr(0);
    ++W_; ++P_;
  }

  void prefillTokens(const int* ids, int cl) {
    ensureRingRoom(cl);
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
    ensureRingRoom(1);
    bindDecodeCaches();
    decode_.infer();
    curLogits_ = (int16_t*)decode_.outPtr(0);
    ++W_; ++P_;
    maxPos_ = std::max(maxPos_, pos.max());
  }

  void prefillEmbeds(const float* rows, const Pos3* pos, int cl) {
    ensureRingRoom(cl);
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
  std::vector<Mem> kRing_, vRing_;               // per layer, ringRows_+1 rows each
  int W_ = 0, ringRows_ = 0, slack_ = 0;         // window starts at row W_
  InputMode mode_ = InputMode::Token;
  int nLayers_ = 0, nKV_ = 0, headDim_ = 0, cacheLen_ = 0, chunk_ = 0, vocab_ = 0;
  int hidden_ = 0, rot_ = 0;
  size_t rowK_ = 0, rowV_ = 0;
  int base_ = 3, vInBase_ = 0, vOutBase_ = 0, logitStride_ = 0;
  float logitScale_ = 1.0f, embInvScale_ = 1.0f, ropeInvScale_ = 1.0f;
  int P_ = 0, maxPos_ = -1;
  const int16_t* curLogits_ = nullptr;
  static constexpr int32_t kStateMagic = 0x424C4B31;   // "BLK1" — prompt-cache file tag
  std::vector<int16_t> savedLogits_;                   // holds the restored last logits
  std::chrono::steady_clock::time_point tTurn_ = std::chrono::steady_clock::now();
  NativeStats stats_;
  std::vector<TokenLogprobs> logprobs_;                // last generate()'s, when asked for
  std::function<void(int, float*)> embedder_;
  std::vector<double> inv_;
  std::vector<uint8_t> sec_;
  std::vector<float> rowBuf_;
};

}  // namespace bllm
