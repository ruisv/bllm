// bllm::native_detail — the hbDNN / hbUCP layer everything else in this library
// sits on: a packed-`.hbm` handle, a cached BPU allocation, and one compiled
// graph with its bound input/output buffers.
//
// This is the ONE piece of BLLM that is compiled rather than header-only. It
// earns that: it is a thin, stable wrapper over the vendor runtime with no
// templates and nothing worth inlining across a call boundary (every function
// that is not a field accessor either talks to the driver or copies whole rows),
// while the model classes above it — NativeEngine, VisionTower, BeingHStack —
// stay in headers so the CLI, the Python module and bllm-serve share one
// implementation with no ABI to keep.
//
// Split out of native_engine.h so that a class needing only a graph (the vision
// and audio towers, pi0.5, Being-H) does not drag in the whole autoregressive
// engine and the sampler behind it.
#pragma once

#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {
namespace native_detail {

void check(int32_t e, const char* what);
#define BLLM_NATIVE_CK(expr) ::bllm::native_detail::check((expr), #expr)

constexpr int64_t kAlign = 32;
inline int64_t alignUp(int64_t v, int64_t a) { return (v + (a - 1)) & ~(a - 1); }

// Graphs from the OpenExplorer pi0 pipeline carry fp16 at every boundary. The
// board is aarch64, where `__fp16` is a storage-and-arithmetic type the compiler
// converts in one instruction; nothing else here needs a software half, so there
// is no fallback path to keep correct.
#if defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
using half_t = __fp16;
#else
#error "native_graph: fp16 tensor I/O needs __fp16 (aarch64); this header is board-only"
#endif

size_t elemSize(int t);

// Inline on purpose: this one runs per element of a quantized row.
inline int16_t quantS16(float v, float inv_scale) {
  const float q = std::rint(v * inv_scale);
  if (q <= -32768.0f) return -32768;
  if (q >= 32767.0f) return 32767;
  return (int16_t)q;
}

// Owns an hbDNN packed handle for the lifetime of whoever holds it.
//
// Every model class here loads its .hbm in the constructor and then keeps
// validating — graph shapes, tensor types, a KV window that can `bad_alloc`. A
// throw from any of those means the object was never constructed, so its own
// destructor never runs and a raw handle leaks the whole (GB-scale) mapping on
// every failed load. Declared ahead of the Graph members, it is also released
// *after* their buffers rather than before, which is the correct order.
class PackedHbm {
 public:
  PackedHbm() = default;
  explicit PackedHbm(const std::vector<std::string>& files) { load(files); }
  ~PackedHbm();
  PackedHbm(const PackedHbm&) = delete;
  PackedHbm& operator=(const PackedHbm&) = delete;

  void load(const std::vector<std::string>& files);
  hbDNNPackedHandle_t get() const { return h_; }
  operator hbDNNPackedHandle_t() const { return h_; }
  explicit operator bool() const { return h_ != nullptr; }

 private:
  hbDNNPackedHandle_t h_ = nullptr;
};

struct Mem {
  hbUCPSysMem m{};
  void alloc(uint64_t sz);
  void clean() const;
  void inval() const;
  void* p() const { return m.virAddr; }
  // A view into this allocation at `off` bytes. hbDNN only needs {phyAddr, virAddr},
  // so a sub-view of a registered allocation binds like any other tensor buffer.
  hbUCPSysMem view(uint64_t off) const { return {m.phyAddr + off, (uint8_t*)m.virAddr + off, m.memSize - off}; }
  void flush(uint64_t off, uint64_t len, int flag) const;
  ~Mem();
  Mem() = default;
  Mem(const Mem&) = delete;
  Mem& operator=(const Mem&) = delete;
  Mem(Mem&& o) noexcept : m(o.m) { o.m = hbUCPSysMem{}; }
  Mem& operator=(Mem&& o) noexcept;
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
// BPU core, so it is a no-op there; it matters on multi-core parts (nash-p / J6P),
// where `Graph::init` sets it from the model's own compiled core count — a
// multi-core hbm is *rejected* under HB_UCP_CORE_ANY rather than falling back.
struct BpuSched {
  int priority = HB_UCP_PRIORITY_LOWEST;   // yield the queue to latency-critical work
  uint64_t core_mask = HB_UCP_CORE_ANY;
  uint32_t device_id = 0;
};

void submitAndWait(hbUCPTaskHandle_t task, const BpuSched& s);

struct Graph {
  hbDNNHandle_t h = nullptr;
  std::vector<hbDNNTensor> in, out;
  std::vector<Mem> inMem, outMem;
  BpuSched sched;

  void init(hbDNNPackedHandle_t packed, const char* name);

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
  static int64_t rowPitch(const hbDNNTensorShape& s, const int64_t* stride, int64_t cols);
  int64_t outRowPitch(int i, int64_t cols) const;

  void writeRowsF32(int i, const float* src, int64_t rows, int64_t cols);
  // `writeRowsF32` is a raw copy and assumes the graph takes float32, which our
  // own graphs do. Graphs built by the OpenExplorer pi0 pipeline are **fp16 in
  // and out**, with int64 position ids, so feeding them needs conversion. These
  // two speak f32 to the caller and whatever the tensor declares to the board.
  void writeRowsF32As(int i, const float* src, int64_t rows, int64_t cols);
  void readRowsF32As(int i, float* dst, int64_t rows, int64_t cols) const;

  static float scaleOf(const hbDNNTensorProperties& p) {
    const auto& s = p.scale;
    return (s.scaleData && s.scaleLen > 0) ? s.scaleData[0] : 1.0f;
  }
  float inScale(int i) const { return scaleOf(in[i].properties); }
  float outScale(int i) const { return scaleOf(out[i].properties); }
  void inClean(int i) { inMem[i].clean(); }
  void* inPtr(int i) { return inMem[i].p(); }
  void* outPtr(int i) { return outMem[i].p(); }

  void infer();
};

// Inline on purpose: one pass over a 250k-row logit vector, called per token.
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
std::vector<uint8_t> mropeAxisMap(int half, const std::vector<int>& section, bool interleaved);

}  // namespace native_detail

// An mrope position: (temporal, height, width). Plain-rope tokens set all three
// equal; image tokens vary h/w across the patch grid. Lives beside
// `mropeAxisMap` because the two only make sense together.
struct Pos3 {
  int32_t t = 0, h = 0, w = 0;
  static Pos3 scalar(int32_t p) { return {p, p, p}; }
  int32_t max() const { return std::max(t, std::max(h, w)); }
};

}  // namespace bllm
