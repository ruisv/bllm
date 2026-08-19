// Out-of-line half of include/bllm/native_graph.h — see that header for why this
// layer, and only this layer, is compiled.
#include "bllm/native_graph.h"

#include <cstring>

namespace bllm {
namespace native_detail {

void check(int32_t e, const char* what) {
  if (e != 0) throw std::runtime_error(std::string("[native] hbDNN error: ") + what);
}

size_t elemSize(int t) {
  switch (t) {
    case HB_DNN_TENSOR_TYPE_S8: case HB_DNN_TENSOR_TYPE_U8: case HB_DNN_TENSOR_TYPE_BOOL8: return 1;
    case HB_DNN_TENSOR_TYPE_S16: case HB_DNN_TENSOR_TYPE_U16: case HB_DNN_TENSOR_TYPE_F16: return 2;
    case HB_DNN_TENSOR_TYPE_S32: case HB_DNN_TENSOR_TYPE_U32: case HB_DNN_TENSOR_TYPE_F32: return 4;
    default: return 0;
  }
}

// ---- PackedHbm -------------------------------------------------------------

PackedHbm::~PackedHbm() { if (h_) hbDNNRelease(h_); }

void PackedHbm::load(const std::vector<std::string>& files) {
  std::vector<const char*> c;
  c.reserve(files.size());
  for (const auto& s : files) c.push_back(s.c_str());
  BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&h_, c.data(), (int32_t)c.size()));
}

// ---- Mem -------------------------------------------------------------------

void Mem::alloc(uint64_t sz) { BLLM_NATIVE_CK(hbUCPMallocCached(&m, sz, 0)); }
void Mem::clean() const { if (m.virAddr) BLLM_NATIVE_CK(hbUCPMemFlush(&m, HB_SYS_MEM_CACHE_CLEAN)); }
void Mem::inval() const { if (m.virAddr) BLLM_NATIVE_CK(hbUCPMemFlush(&m, HB_SYS_MEM_CACHE_INVALIDATE)); }

void Mem::flush(uint64_t off, uint64_t len, int flag) const {
  const hbUCPSysMem s{m.phyAddr + off, (uint8_t*)m.virAddr + off, len};
  BLLM_NATIVE_CK(hbUCPMemFlush(&s, flag));
}

Mem::~Mem() { if (m.virAddr) hbUCPFree(&m); }

Mem& Mem::operator=(Mem&& o) noexcept {
  if (this != &o) { if (m.virAddr) hbUCPFree(&m); m = o.m; o.m = hbUCPSysMem{}; }
  return *this;
}

// ---- scheduling ------------------------------------------------------------

void submitAndWait(hbUCPTaskHandle_t task, const BpuSched& s) {
  hbUCPSchedParam sched;
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
  sched.priority = s.priority;
  sched.backend = s.core_mask;
  sched.deviceId = s.device_id;
  check(hbUCPSubmitTask(task, &sched), "hbUCPSubmitTask");
  check(hbUCPWaitTaskDone(task, 0), "hbUCPWaitTaskDone");
}

// ---- Graph -----------------------------------------------------------------

namespace {
// The runtime reports a stride of -1 for "dense, compute it yourself". Left raw,
// a reader that trusts it gets a negative pitch and walks backwards off the
// front of the buffer.
void resolveStrides(hbDNNTensorProperties& p) {
  const int nd = p.validShape.numDimensions;
  if (nd > 0 && p.stride[nd - 1] == -1) p.stride[nd - 1] = (int64_t)elemSize(p.tensorType);
  for (int d = nd - 2; d >= 0; --d)
    if (p.stride[d] == -1)
      p.stride[d] = alignUp(p.stride[d + 1] * p.validShape.dimensionSize[d + 1], kAlign);
}
}  // namespace

void Graph::init(hbDNNPackedHandle_t packed, const char* name) {
  BLLM_NATIVE_CK(hbDNNGetModelHandle(&h, packed, name));
  int ic = 0, oc = 0;
  BLLM_NATIVE_CK(hbDNNGetInputCount(&ic, h));
  BLLM_NATIVE_CK(hbDNNGetOutputCount(&oc, h));
  in.resize(ic); out.resize(oc); inMem.resize(ic); outMem.resize(oc);
  for (int i = 0; i < ic; ++i) {
    auto& p = in[i].properties;
    BLLM_NATIVE_CK(hbDNNGetInputTensorProperties(&p, h, i));
    resolveStrides(p);
    const int nd = p.validShape.numDimensions;
    const uint64_t bytes = nd > 0 ? (uint64_t)(p.stride[0] * p.validShape.dimensionSize[0]) : 0;
    inMem[i].alloc(bytes);
    in[i].sysMem = inMem[i].m;
  }
  for (int i = 0; i < oc; ++i) {
    auto& p = out[i].properties;
    BLLM_NATIVE_CK(hbDNNGetOutputTensorProperties(&p, h, i));
    resolveStrides(p);
    outMem[i].alloc((uint64_t)p.alignedByteSize);
    out[i].sysMem = outMem[i].m;
  }
}

int64_t Graph::rowPitch(const hbDNNTensorShape& s, const int64_t* stride, int64_t cols) {
  int64_t below = 1;
  for (int d = s.numDimensions - 1; d >= 0; --d) {
    if (below == cols) return stride[d];
    below *= s.dimensionSize[d];
  }
  return cols * (int64_t)sizeof(float);   // the whole tensor is one row
}

int64_t Graph::outRowPitch(int i, int64_t cols) const {
  return rowPitch(outShape(i), out[i].properties.stride, cols);
}

void Graph::writeRowsF32(int i, const float* src, int64_t rows, int64_t cols) {
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

void Graph::writeRowsF32As(int i, const float* src, int64_t rows, int64_t cols) {
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
        auto* p = (half_t*)d;
        for (int64_t k = 0; k < cols; ++k) p[k] = (half_t)s[k];
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

void Graph::readRowsF32As(int i, float* dst, int64_t rows, int64_t cols) const {
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
        const auto* p = (const half_t*)s;
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

void Graph::infer() {
  hbUCPTaskHandle_t task = nullptr;
  BLLM_NATIVE_CK(hbDNNInferV2(&task, out.data(), in.data(), h));
  submitAndWait(task, sched);
  hbUCPReleaseTask(task);
  for (auto& m : outMem) m.inval();
}

// ---- mrope -----------------------------------------------------------------

std::vector<uint8_t> mropeAxisMap(int half, const std::vector<int>& section, bool interleaved) {
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
}  // namespace bllm
