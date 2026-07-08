// bllm_selfengine — a self-built autoregressive LLM engine that drives a
// compiled `.hbm` (prefill + decode dual-graph) DIRECTLY on the generic
// hbDNN / hbUCP BPU runtime — the SAME layer bcdl/vision uses — with our own
// KV-cache management and sampling. It NEVER loads libxlm.
//
// SE1: stateful engine (persistent KV cache -> multi-turn, chunked prefill,
//      sampling, stop logic).
// SE2: MODEL-AGNOSTIC. Every shape (layer count, kv heads, head_dim, cache_len,
//      chunk, vocab, per-tensor dtype/scale, K/V index ranges) is discovered by
//      introspecting the decode/prefill graphs via the hbDNN API — nothing is
//      hardcoded. So one binary runs any dense `.hbm` we ship (Qwen2.5, DeepSeek,
//      InternLM2, Phi-4, GLM-Edge, Qwen3, …) and, because we own the cache,
//      head_dim != 128 too (which libxlm cannot do). Per-model bits not in the
//      graph (eos ids) come from --eos (the launcher reads config.json).
//
// Graph contract (stable across the leap_llm recipe family):
//   in : [0]=tokens S32 [1]=pos S32 [2]=mask S16(.,cache_len)
//        [3 .. 3+L-1]=K, [3+L .. 3+2L-1]=V   (each (cache_len, n_kv, head_dim))
//   out: [0]=logits S16(.,.,vocab) [1..L]=new K [1+L..2L]=new V
// K/V quant scales match prefill-out vs decode-in, so quantized bytes move
// directly. mask uses INT16 -32768 == -inf; logits S16 x scale (greedy is exact
// on raw int16; sampling dequantizes the top-k).
//
// Build:  g++ -O2 -std=c++17 -I/usr/include/hobot examples/selfengine.cc \
//              -L/usr/hobot/lib -ldnn -lhbucp -lhbrt4 -o bllm_selfengine
// One-shot:  ./bllm_selfengine --hbm m.hbm --ids "785,..." --max-new 24
// Server  :  ./bllm_selfengine --hbm m.hbm --serve [--temp .7 --top-p .9 --eos "151645,151643"]

#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#define CK(expr)                                                            \
  do {                                                                      \
    int32_t _e = (expr);                                                    \
    if (_e != 0) {                                                          \
      std::fprintf(stderr, "[selfengine] error %d at %s:%d: %s\n", _e,      \
                   __FILE__, __LINE__, #expr);                              \
      std::exit(2);                                                         \
    }                                                                       \
  } while (0)

namespace {

constexpr int64_t kAlign = 32;   // S100/S100P BPU width-stride alignment

int64_t alignUp(int64_t v, int64_t a) { return (v + (a - 1)) & ~(a - 1); }

size_t elemSize(int t) {
  switch (t) {
    case HB_DNN_TENSOR_TYPE_S8:
    case HB_DNN_TENSOR_TYPE_U8:
    case HB_DNN_TENSOR_TYPE_BOOL8:
      return 1;
    case HB_DNN_TENSOR_TYPE_S16:
    case HB_DNN_TENSOR_TYPE_U16:
    case HB_DNN_TENSOR_TYPE_F16:
      return 2;
    case HB_DNN_TENSOR_TYPE_S32:
    case HB_DNN_TENSOR_TYPE_U32:
    case HB_DNN_TENSOR_TYPE_F32:
      return 4;
    default:
      return 0;
  }
}

// RAII over an hbUCP cached device buffer (shared CPU/BPU memory).
struct Mem {
  hbUCPSysMem m{};
  void alloc(uint64_t sz) { CK(hbUCPMallocCached(&m, sz, 0)); }
  void clean() const {
    if (m.virAddr) CK(hbUCPMemFlush(&m, HB_SYS_MEM_CACHE_CLEAN));
  }
  void inval() const {
    if (m.virAddr) CK(hbUCPMemFlush(&m, HB_SYS_MEM_CACHE_INVALIDATE));
  }
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

// One compiled graph (prefill or decode) with its I/O tensors resident.
struct Graph {
  hbDNNHandle_t h = nullptr;
  std::vector<hbDNNTensor> in, out;
  std::vector<Mem> inMem, outMem;

  void init(hbDNNPackedHandle_t packed, const char* name) {
    CK(hbDNNGetModelHandle(&h, packed, name));
    int ic = 0, oc = 0;
    CK(hbDNNGetInputCount(&ic, h));
    CK(hbDNNGetOutputCount(&oc, h));
    in.resize(ic); out.resize(oc); inMem.resize(ic); outMem.resize(oc);
    for (int i = 0; i < ic; ++i) {
      auto& p = in[i].properties;
      CK(hbDNNGetInputTensorProperties(&p, h, i));
      const int nd = p.validShape.numDimensions;
      if (nd > 0 && p.stride[nd - 1] == -1)
        p.stride[nd - 1] = static_cast<int64_t>(elemSize(p.tensorType));
      for (int d = nd - 2; d >= 0; --d)
        if (p.stride[d] == -1)
          p.stride[d] = alignUp(p.stride[d + 1] * p.validShape.dimensionSize[d + 1], kAlign);
      const uint64_t bytes =
          nd > 0 ? static_cast<uint64_t>(p.stride[0] * p.validShape.dimensionSize[0]) : 0;
      inMem[i].alloc(bytes);
      in[i].sysMem = inMem[i].m;
    }
    for (int i = 0; i < oc; ++i) {
      CK(hbDNNGetOutputTensorProperties(&out[i].properties, h, i));
      outMem[i].alloc(static_cast<uint64_t>(out[i].properties.alignedByteSize));
      out[i].sysMem = outMem[i].m;
    }
  }
  int inCount() const { return static_cast<int>(in.size()); }
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
    CK(hbDNNInferV2(&task, out.data(), in.data(), h));
    hbUCPSchedParam sched;
    HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    sched.priority = HB_UCP_PRIORITY_LOWEST;
    CK(hbUCPSubmitTask(task, &sched));
    CK(hbUCPWaitTaskDone(task, 0));
    hbUCPReleaseTask(task);
    for (auto& m : outMem) m.inval();
  }
};

int argmaxS16(const int16_t* row, int n) {
  int best = 0;
  int16_t bv = row[0];
  for (int i = 1; i < n; ++i)
    if (row[i] > bv) { bv = row[i]; best = i; }
  return best;
}

std::vector<int> parseIds(const std::string& s) {
  std::vector<int> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) v.push_back(std::atoi(s.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return v;
}

// ---------------------------------------------------------------- sampling ---
struct Sampler {
  float temp = 0.0f, topp = 1.0f, rep_pen = 1.0f;
  int topk = 0;
  std::mt19937 rng{1234u};
  std::unordered_set<int> seen;
  std::vector<int> idxbuf;

  void reset() { seen.clear(); }
  void record(int id) { if (rep_pen != 1.0f) seen.insert(id); }
  bool stochastic() const { return temp > 0.0f || rep_pen != 1.0f || topk > 0 || topp < 1.0f; }

  int pick(const int16_t* logits, float scale, int n) {
    if (!stochastic()) return argmaxS16(logits, n);
    const int k = topk > 0 ? std::min(topk, n) : std::min(n, 512);
    if (static_cast<int>(idxbuf.size()) != n) {
      idxbuf.resize(n);
      for (int i = 0; i < n; ++i) idxbuf[i] = i;
    }
    std::nth_element(idxbuf.begin(), idxbuf.begin() + k, idxbuf.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    std::vector<std::pair<float, int>> cand(k);
    for (int i = 0; i < k; ++i) {
      int id = idxbuf[i];
      float v = logits[id] * scale;
      if (rep_pen != 1.0f && seen.count(id)) v = v > 0 ? v / rep_pen : v * rep_pen;
      cand[i] = {v, id};
    }
    std::sort(cand.begin(), cand.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
              });
    const float T = temp > 0.0f ? temp : 1.0f;
    const float mx = cand[0].first;
    std::vector<double> pr(k);
    double sum = 0;
    for (int i = 0; i < k; ++i) { pr[i] = std::exp((cand[i].first - mx) / T); sum += pr[i]; }
    double cum = 0; int keep = k;
    for (int i = 0; i < k; ++i) { cum += pr[i] / sum; if (cum >= topp) { keep = i + 1; break; } }
    double ks = 0;
    for (int i = 0; i < keep; ++i) ks += pr[i];
    std::uniform_real_distribution<double> U(0.0, ks);
    double r = U(rng), acc = 0;
    for (int i = 0; i < keep; ++i) { acc += pr[i]; if (r <= acc) return cand[i].second; }
    return cand[keep - 1].second;
  }
};

// --------------------------------------------------------------- LLM engine --
struct LlmEngine {
  Graph prefill, decode;
  // introspected model shape
  int nLayers = 0, nKV = 0, headDim = 0, cacheLen = 0, chunk = 0, vocab = 0;
  size_t rowK = 0, rowV = 0;   // per-position K/V bytes (dtype-aware)
  int vInBase = 0;             // decode input index of first V cache (K starts at 3)
  int vOutBase = 0;            // decode output index of first V cache (K starts at 1)
  int logitStride = 0;
  float logitScale = 1.0f;
  std::unordered_set<int> eos{151645, 151643};

  int P = 0;
  const int16_t* curLogits = nullptr;

  uint8_t* kBuf(int l) { return static_cast<uint8_t*>(decode.inPtr(3 + l)); }
  uint8_t* vBuf(int l) { return static_cast<uint8_t*>(decode.inPtr(vInBase + l)); }

  void init(hbDNNPackedHandle_t packed) {
    prefill.init(packed, "prefill");
    decode.init(packed, "decode");
    // in: 0 tokens, 1 pos, 2 mask, 3.. caches (K then V)
    const int nCache = decode.inCount() - 3;
    nLayers = nCache / 2;
    vInBase = 3 + nLayers;
    vOutBase = 1 + nLayers;
    const auto& kShape = decode.inShape(3);        // (cache_len, n_kv, head_dim)
    cacheLen = kShape.dimensionSize[0];
    nKV = kShape.dimensionSize[1];
    headDim = kShape.dimensionSize[2];
    rowK = static_cast<size_t>(nKV) * headDim * elemSize(decode.inType(3));
    rowV = static_cast<size_t>(nKV) * headDim * elemSize(decode.inType(vInBase));
    const auto& logitsShape = decode.outShape(0);  // (1,1,vocab)
    vocab = logitsShape.dimensionSize[logitsShape.numDimensions - 1];
    logitStride = alignUp(static_cast<int64_t>(vocab) * 2, kAlign) / 2;
    logitScale = decode.outScale(0);
    chunk = prefill.inShape(0).dimensionSize[prefill.inShape(0).numDimensions - 1];
    std::fprintf(stderr,
        "[selfengine] introspected: layers=%d kv=%d head_dim=%d cache_len=%d "
        "chunk=%d vocab=%d K=%zuB V=%zuB scale=%.6g\n",
        nLayers, nKV, headDim, cacheLen, chunk, vocab, rowK, rowV, logitScale);
    reset();
  }

  void reset() {
    for (int l = 0; l < nLayers; ++l) {
      std::memset(kBuf(l), 0, cacheLen * rowK); decode.inClean(3 + l);
      std::memset(vBuf(l), 0, cacheLen * rowV); decode.inClean(vInBase + l);
    }
    P = 0; curLogits = nullptr;
  }

  void decodeStep(int token) {
    *static_cast<int32_t*>(decode.inPtr(0)) = token; decode.inClean(0);
    *static_cast<int32_t*>(decode.inPtr(1)) = P;     decode.inClean(1);
    auto* mask = static_cast<int16_t*>(decode.inPtr(2));
    int lo = cacheLen - 1 - P; if (lo < 0) lo = 0;
    for (int j = 0; j < cacheLen; ++j) mask[j] = (j >= lo) ? 0 : -32768;
    decode.inClean(2);
    decode.infer();
    curLogits = static_cast<int16_t*>(decode.outPtr(0));
    for (int l = 0; l < nLayers; ++l) {
      uint8_t* k = kBuf(l);
      std::memmove(k, k + rowK, (cacheLen - 1) * rowK);
      std::memcpy(k + (cacheLen - 1) * rowK, decode.outPtr(1 + l), rowK);
      decode.inClean(3 + l);
      uint8_t* v = vBuf(l);
      std::memmove(v, v + rowV, (cacheLen - 1) * rowV);
      std::memcpy(v + (cacheLen - 1) * rowV, decode.outPtr(vOutBase + l), rowV);
      decode.inClean(vInBase + l);
    }
    ++P;
  }

  void prefillFirst(const std::vector<int>& ids, int cl) {
    auto* tk = static_cast<int32_t*>(prefill.inPtr(0));
    std::memset(tk, 0, chunk * sizeof(int32_t));
    for (int i = 0; i < cl; ++i) tk[i] = ids[i];
    prefill.inClean(0);
    auto* pos = static_cast<int32_t*>(prefill.inPtr(1));
    for (int i = 0; i < chunk; ++i) pos[i] = i;
    prefill.inClean(1);
    const int base = cacheLen - chunk;
    auto* m = static_cast<int16_t*>(prefill.inPtr(2));
    for (int i = 0; i < chunk; ++i)
      for (int j = 0; j < cacheLen; ++j)
        m[i * cacheLen + j] = (j >= base && j <= base + i) ? 0 : -32768;
    prefill.inClean(2);
    for (int l = 0; l < nLayers; ++l) {
      std::memset(prefill.inPtr(3 + l), 0, cacheLen * rowK); prefill.inClean(3 + l);
      std::memset(prefill.inPtr(vInBase + l), 0, cacheLen * rowV); prefill.inClean(vInBase + l);
    }
    prefill.infer();
    curLogits = static_cast<int16_t*>(prefill.outPtr(0)) + static_cast<size_t>(cl - 1) * logitStride;
    for (int l = 0; l < nLayers; ++l) {
      std::memcpy(kBuf(l) + static_cast<size_t>(cacheLen - cl) * rowK,
                  prefill.outPtr(1 + l), static_cast<size_t>(cl) * rowK);
      decode.inClean(3 + l);
      std::memcpy(vBuf(l) + static_cast<size_t>(cacheLen - cl) * rowV,
                  prefill.outPtr(vOutBase + l), static_cast<size_t>(cl) * rowV);
      decode.inClean(vInBase + l);
    }
    P = cl;
  }

  void feed(const std::vector<int>& ids) {
    size_t i = 0;
    if (P == 0 && !ids.empty()) {
      int cl = std::min(static_cast<int>(ids.size()), chunk);
      prefillFirst(ids, cl);
      i = cl;
    }
    for (; i < ids.size(); ++i) decodeStep(ids[i]);
  }

  template <class Emit>
  void generate(int max_new, Sampler& s, Emit emit) {
    for (int step = 0; step < max_new && curLogits; ++step) {
      int tok = s.pick(curLogits, logitScale, vocab);
      if (eos.count(tok)) break;
      emit(tok);
      s.record(tok);
      decodeStep(tok);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string hbm, ids_str, eos_str;
  int max_new = 24;
  bool serve = false;
  Sampler sampler;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() { return std::string(argv[++i]); };
    if (a == "--hbm") hbm = val();
    else if (a == "--ids") ids_str = val();
    else if (a == "--max-new") max_new = std::atoi(val().c_str());
    else if (a == "--serve") serve = true;
    else if (a == "--temp") sampler.temp = std::atof(val().c_str());
    else if (a == "--top-k") sampler.topk = std::atoi(val().c_str());
    else if (a == "--top-p") sampler.topp = std::atof(val().c_str());
    else if (a == "--rep-pen") sampler.rep_pen = std::atof(val().c_str());
    else if (a == "--seed") sampler.rng.seed(std::strtoul(val().c_str(), nullptr, 10));
    else if (a == "--eos") eos_str = val();
  }
  if (hbm.empty() || (!serve && ids_str.empty())) {
    std::fprintf(stderr, "usage: %s --hbm <model.hbm> [--ids \"id,..\" | --serve]"
                 " [--max-new N] [--temp T --top-k K --top-p P --rep-pen R --seed S]"
                 " [--eos \"id,id\"]\n", argv[0]);
    return 1;
  }

  hbDNNPackedHandle_t packed = nullptr;
  const char* files[] = {hbm.c_str()};
  CK(hbDNNInitializeFromFiles(&packed, files, 1));
  LlmEngine eng;
  eng.init(packed);
  if (!eos_str.empty()) {
    eng.eos.clear();
    for (int id : parseIds(eos_str)) eng.eos.insert(id);
  }
  std::fprintf(stderr, "[selfengine] loaded via hbDNN (libxlm never touched)\n");

  if (serve) {
    if (max_new < 2) max_new = 512;
    std::fprintf(stderr, "[selfengine] serve ready (max-new=%d temp=%.2f top-k=%d top-p=%.2f)\n",
                 max_new, sampler.temp, sampler.topk, sampler.topp);
    std::fflush(stderr);
    std::printf("READY\n"); std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "R" || line == "r") {
        eng.reset(); sampler.reset();
        std::printf("OK\n"); std::fflush(stdout); continue;
      }
      std::vector<int> ids = parseIds(line);
      if (ids.empty()) { std::printf("END\n"); std::fflush(stdout); continue; }
      eng.feed(ids);
      eng.generate(max_new, sampler, [](int t) { std::printf("TOK %d\n", t); std::fflush(stdout); });
      std::printf("END\n"); std::fflush(stdout);
    }
  } else {
    std::vector<int> prompt = parseIds(ids_str);
    if (prompt.empty()) { std::fprintf(stderr, "[selfengine] empty prompt\n"); return 1; }
    eng.feed(prompt);
    std::vector<int> gen;
    eng.generate(max_new, sampler, [&](int t) { gen.push_back(t); });
    std::printf("GEN:");
    for (int id : gen) std::printf(" %d", id);
    std::printf("\n");
  }
  hbDNNRelease(packed);
  return 0;
}
