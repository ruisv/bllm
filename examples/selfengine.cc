// bllm_selfengine — a self-built autoregressive LLM engine that drives a
// compiled `.hbm` (prefill + decode dual-graph) DIRECTLY on the generic
// hbDNN / hbUCP BPU runtime — the SAME layer bcdl/vision uses — with our own
// KV-cache management and sampling. It NEVER loads libxlm.
//
// SE1 (core hardening): a STATEFUL engine. The KV cache persists across calls,
// so we get real multi-turn chat (each turn only ingests the new tokens, never
// re-prefills history) and prompts longer than one chunk (first 256 via the
// prefill graph, the rest appended through the decode graph). Sampling supports
// greedy / temperature / top-k / top-p / repetition-penalty; generation stops on
// eos, and the KV window slides once the context passes cache_len.
//
// Contract (Qwen-family, cache_len 1024, chunk 256, 28 layers, GQA 8 kv heads,
// head_dim 128), reverse-engineered from the leap_llm recipe + hrt_model_exec:
//   prefill inputs : [0]=tokens S32(1,256) [1]=pos S32(256) [2]=mask S16(256,1024)
//                    [3..30]=K S16(1024,8,128) [31..58]=V S8(1024,8,128)
//   prefill outputs: [0]=logits S16(1,256,V) [1..28]=K S16(256,8,128) [29..56]=V S8(256,8,128)
//   decode  inputs : [0]=tok S32(1,1) [1]=pos S32(1) [2]=mask S16(1,1024) [3..]=K/V windows
//   decode  outputs: [0]=logits S16(1,1,V) [1..28]=newK S16(1,8,128) [29..56]=newV S8(1,8,128)
// K/V-cache quant scales match between prefill-out and decode-in, so quantized
// bytes move directly. mask uses INT16 -32768 == -inf. logits are S16 with a
// positive scale (read from tensor properties) — greedy is exact on int16;
// sampling dequantizes the top-k candidates.
//
// Build (on board):
//   g++ -O2 -std=c++17 -I/usr/include/hobot examples/selfengine.cc \
//       -L/usr/hobot/lib -ldnn -lhbucp -lhbrt4 -o bllm_selfengine
// One-shot:  ./bllm_selfengine --hbm m.hbm --ids "785,6722,315,9625,374" --max-new 24
// Server  :  ./bllm_selfengine --hbm m.hbm --serve [--temp 0.7 --top-p 0.9 ...]
//   protocol (stdin lines): "R" resets context; otherwise a comma-id line is
//   ingested then generated, streaming "TOK <id>" per token, ending with "END".

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

constexpr int kChunk = 256;      // prefill sequence length
constexpr int kCacheLen = 1024;  // KV window length
constexpr int kNKV = 8;          // key/value heads (GQA)
constexpr int kHeadDim = 128;
constexpr int kNLayers = 28;
constexpr int kVocab = 151936;
constexpr int64_t kAlign = 32;   // S100/S100P BPU width-stride alignment
constexpr int kEosImEnd = 151645;
constexpr int kEosEot = 151643;

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
  ~Mem() {
    if (m.virAddr) hbUCPFree(&m);
  }
  Mem() = default;
  Mem(const Mem&) = delete;
  Mem& operator=(const Mem&) = delete;
  Mem(Mem&& o) noexcept : m(o.m) { o.m = hbUCPSysMem{}; }
  Mem& operator=(Mem&& o) noexcept {
    if (this != &o) {
      if (m.virAddr) hbUCPFree(&m);
      m = o.m;
      o.m = hbUCPSysMem{};
    }
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
    in.resize(ic);
    out.resize(oc);
    inMem.resize(ic);
    outMem.resize(oc);
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

  void setIn(int i, const void* data, size_t bytes) {
    std::memcpy(inMem[i].p(), data, bytes);
    inMem[i].clean();
  }
  void* inPtr(int i) { return inMem[i].p(); }
  void inClean(int i) { inMem[i].clean(); }
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

const size_t kRowK = static_cast<size_t>(kNKV) * kHeadDim * 2;  // S16 row bytes
const size_t kRowV = static_cast<size_t>(kNKV) * kHeadDim * 1;  // S8  row bytes

// ---------------------------------------------------------------- sampling ---
struct Sampler {
  float temp = 0.0f;    // <= 0 => greedy
  float topp = 1.0f;
  float rep_pen = 1.0f;
  int topk = 0;         // 0 => cap at 512 candidates
  std::mt19937 rng{1234u};
  std::unordered_set<int> seen;   // for repetition penalty (per conversation)
  std::vector<int> idxbuf;

  void reset() { seen.clear(); }
  void record(int id) { if (rep_pen != 1.0f) seen.insert(id); }
  bool stochastic() const {
    return temp > 0.0f || rep_pen != 1.0f || topk > 0 || topp < 1.0f;
  }

  int pick(const int16_t* logits, float scale, int n) {
    if (!stochastic()) return argmaxS16(logits, n);
    const int k = topk > 0 ? std::min(topk, n) : std::min(n, 512);
    if (static_cast<int>(idxbuf.size()) != n) {
      idxbuf.resize(n);
      for (int i = 0; i < n; ++i) idxbuf[i] = i;
    }
    // top-k by raw int16 (scale > 0 preserves order)
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
    // nucleus (top-p)
    double cum = 0;
    int keep = k;
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
// Stateful: the KV cache lives IN the decode graph's input buffers (in[3..58])
// and persists across feed()/generate() calls, so multi-turn chat keeps context
// in the cache with no host-side copy. Each step shifts the window and appends
// the new key/value in place (one copy, not two). libxlm never touched.
struct LlmEngine {
  Graph prefill, decode;
  int P = 0;
  int logitStride = 0;       // int16 elems per logits row
  float logitScale = 1.0f;
  const int16_t* curLogits = nullptr;  // prediction for the NEXT token

  uint8_t* kBuf(int l) { return static_cast<uint8_t*>(decode.inPtr(3 + l)); }
  uint8_t* vBuf(int l) { return static_cast<uint8_t*>(decode.inPtr(31 + l)); }

  void init(hbDNNPackedHandle_t packed) {
    prefill.init(packed, "prefill");
    decode.init(packed, "decode");
    logitStride = alignUp(static_cast<int64_t>(kVocab) * 2, kAlign) / 2;
    const auto& sc = decode.out[0].properties.scale;
    logitScale = (sc.scaleData && sc.scaleLen > 0) ? sc.scaleData[0] : 1.0f;
    reset();
  }

  void reset() {
    for (int l = 0; l < kNLayers; ++l) {
      std::memset(kBuf(l), 0, kCacheLen * kRowK); decode.inClean(3 + l);
      std::memset(vBuf(l), 0, kCacheLen * kRowV); decode.inClean(31 + l);
    }
    P = 0;
    curLogits = nullptr;
  }

  // decode one token at position P; append its k/v into the persistent window;
  // advance; set curLogits.
  void decodeStep(int token) {
    *static_cast<int32_t*>(decode.inPtr(0)) = token; decode.inClean(0);
    *static_cast<int32_t*>(decode.inPtr(1)) = P;     decode.inClean(1);
    auto* mask = static_cast<int16_t*>(decode.inPtr(2));
    int lo = kCacheLen - 1 - P; if (lo < 0) lo = 0;  // slides once P >= cache_len
    for (int j = 0; j < kCacheLen; ++j) mask[j] = (j >= lo) ? 0 : -32768;
    decode.inClean(2);
    // K/V input buffers already hold the current window (from prefill / prev step)
    decode.infer();
    curLogits = static_cast<int16_t*>(decode.outPtr(0));
    // roll each window in place: drop oldest row, append this step's key/value
    for (int l = 0; l < kNLayers; ++l) {
      uint8_t* k = kBuf(l);
      std::memmove(k, k + kRowK, (kCacheLen - 1) * kRowK);
      std::memcpy(k + (kCacheLen - 1) * kRowK, decode.outPtr(1 + l), kRowK);
      decode.inClean(3 + l);
      uint8_t* v = vBuf(l);
      std::memmove(v, v + kRowV, (kCacheLen - 1) * kRowV);
      std::memcpy(v + (kCacheLen - 1) * kRowV, decode.outPtr(29 + l), kRowV);
      decode.inClean(31 + l);
    }
    ++P;
  }

  // prefill the first `cl` tokens (positions 0..cl-1) from a fresh context, then
  // seed the decode window (right-aligned) directly into the decode input bufs.
  void prefillFirst(const std::vector<int>& ids, int cl) {
    auto* tk = static_cast<int32_t*>(prefill.inPtr(0));
    std::memset(tk, 0, kChunk * sizeof(int32_t));
    for (int i = 0; i < cl; ++i) tk[i] = ids[i];
    prefill.inClean(0);
    auto* pos = static_cast<int32_t*>(prefill.inPtr(1));
    for (int i = 0; i < kChunk; ++i) pos[i] = i;
    prefill.inClean(1);
    const int base = kCacheLen - kChunk;  // current chunk lands at window cols base..1023
    auto* m = static_cast<int16_t*>(prefill.inPtr(2));
    for (int i = 0; i < kChunk; ++i)
      for (int j = 0; j < kCacheLen; ++j)
        m[i * kCacheLen + j] = (j >= base && j <= base + i) ? 0 : -32768;
    prefill.inClean(2);
    for (int t = 3; t <= 58; ++t) {
      std::memset(prefill.inPtr(t), 0, t <= 30 ? kCacheLen * kRowK : kCacheLen * kRowV);
      prefill.inClean(t);
    }
    prefill.infer();
    curLogits = static_cast<int16_t*>(prefill.outPtr(0)) + static_cast<size_t>(cl - 1) * logitStride;
    for (int l = 0; l < kNLayers; ++l) {
      std::memcpy(kBuf(l) + static_cast<size_t>(kCacheLen - cl) * kRowK,
                  prefill.outPtr(1 + l), static_cast<size_t>(cl) * kRowK);
      decode.inClean(3 + l);
      std::memcpy(vBuf(l) + static_cast<size_t>(kCacheLen - cl) * kRowV,
                  prefill.outPtr(29 + l), static_cast<size_t>(cl) * kRowV);
      decode.inClean(31 + l);
    }
    P = cl;
  }

  // ingest tokens into the cache (a prompt, or a new conversation turn).
  void feed(const std::vector<int>& ids) {
    size_t i = 0;
    if (P == 0 && !ids.empty()) {
      int cl = std::min(static_cast<int>(ids.size()), kChunk);
      prefillFirst(ids, cl);
      i = cl;
    }
    for (; i < ids.size(); ++i) decodeStep(ids[i]);
  }

  // sample up to max_new tokens; stop on eos. Emit(id) is called per token.
  template <class Emit>
  void generate(int max_new, Sampler& s, Emit emit) {
    for (int step = 0; step < max_new && curLogits; ++step) {
      int tok = s.pick(curLogits, logitScale, kVocab);
      if (tok == kEosImEnd || tok == kEosEot) break;
      emit(tok);
      s.record(tok);
      decodeStep(tok);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string hbm, ids_str;
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
  }
  if (hbm.empty() || (!serve && ids_str.empty())) {
    std::fprintf(stderr, "usage: %s --hbm <model.hbm> [--ids \"id,..\" | --serve]"
                 " [--max-new N] [--temp T --top-k K --top-p P --rep-pen R --seed S]\n", argv[0]);
    return 1;
  }

  // ---- load the .hbm ONCE via the generic hbDNN runtime (NOT libxlm) --------
  hbDNNPackedHandle_t packed = nullptr;
  const char* files[] = {hbm.c_str()};
  CK(hbDNNInitializeFromFiles(&packed, files, 1));
  LlmEngine eng;
  eng.init(packed);
  std::fprintf(stderr, "[selfengine] loaded prefill+decode via hbDNN (libxlm never touched)\n");

  if (serve) {
    if (max_new < 2) max_new = 512;
    std::fprintf(stderr, "[selfengine] serve ready (max-new=%d, temp=%.2f top-k=%d top-p=%.2f)\n",
                 max_new, sampler.temp, sampler.topk, sampler.topp);
    std::fflush(stderr);
    std::printf("READY\n"); std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "R" || line == "r") {         // reset conversation context
        eng.reset(); sampler.reset();
        std::printf("OK\n"); std::fflush(stdout); continue;
      }
      std::vector<int> ids = parseIds(line);
      if (ids.empty()) { std::printf("END\n"); std::fflush(stdout); continue; }
      eng.feed(ids);   // multi-turn: only the NEW tokens; cache keeps history
      eng.generate(max_new, sampler, [](int t) {
        std::printf("TOK %d\n", t); std::fflush(stdout);
      });
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
