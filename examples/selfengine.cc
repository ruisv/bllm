// bllm_selfengine — a self-built autoregressive LLM engine that drives a
// compiled `.hbm` (prefill + decode dual-graph) DIRECTLY on the generic
// hbDNN / hbUCP BPU runtime — the SAME layer bcdl/vision uses — with our own
// KV-cache management and sampling. It NEVER loads libxlm.
//
// The point: prove that the on-board LLM runtime (libxlm) is only an
// orchestration layer. The graphs are ordinary static BPU models; the KV cache
// is ordinary graph I/O whose shape/semantics WE define. Owning the decode loop
// is what makes SSM-state caches / head_dim!=128 possible later — none of which
// libxlm allows.
//
// Contract reverse-engineered from the leap_llm recipe + hrt_model_exec
// model_info (Qwen-family, cache_len 1024, chunk 256, 28 layers, GQA 8 kv heads,
// head_dim 128):
//   prefill inputs : [0]=tokens S32(1,256) [1]=pos S32(256) [2]=mask S16(256,1024)
//                    [3..30]=K S16(1024,8,128) [31..58]=V S8(1024,8,128)
//   prefill outputs: [0]=logits S16(1,256,V) [1..28]=K S16(256,8,128) [29..56]=V S8(256,8,128)
//   decode  inputs : [0]=tok S32(1,1) [1]=pos S32(1) [2]=mask S16(1,1024) [3..]=K/V windows
//   decode  outputs: [0]=logits S16(1,1,V) [1..28]=newK S16(1,8,128) [29..56]=newV S8(1,8,128)
// K/V-cache quant scales are identical between prefill-out and decode-in, so the
// quantized bytes move directly (no de/requant). mask uses INT16 -32768 == -inf.
// logits are S16 with a positive scale, so argmax is exact on the raw int16.
//
// Build (on board, no cmake needed):
//   g++ -O2 -std=c++17 -I/usr/include/hobot examples/selfengine.cc \
//       -L/usr/hobot/lib -ldnn -lhbucp -lhbrt4 -o bllm_selfengine
// Run:
//   ./bllm_selfengine --hbm qwen3-1.7b.hbm --ids "785,6722,315,9625,374" --max-new 24

#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
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
      // Input strides are dynamic (-1): resolve innermost->outermost, BPU-aligned.
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

// One independent generation: prefill the prompt, then decode greedily. Our own
// KV cache; libxlm untouched. If stream, emit "TOK <id>" per token (flushed).
std::vector<int> generate(Graph& prefill, Graph& decode,
                          const std::vector<int>& prompt, int max_new, bool stream) {
  const int L = static_cast<int>(prompt.size());
  const int base = kCacheLen - kChunk;  // 768: current chunk lands at window cols base..1023

  // ---- PREFILL ----
  auto* tok = static_cast<int32_t*>(prefill.inPtr(0));
  std::memset(tok, 0, kChunk * sizeof(int32_t));
  for (int i = 0; i < L; ++i) tok[i] = prompt[i];
  prefill.inClean(0);
  auto* pos = static_cast<int32_t*>(prefill.inPtr(1));
  for (int i = 0; i < kChunk; ++i) pos[i] = i;
  prefill.inClean(1);
  auto* pmask = static_cast<int16_t*>(prefill.inPtr(2));  // causal over the current chunk
  for (int i = 0; i < kChunk; ++i)
    for (int j = 0; j < kCacheLen; ++j)
      pmask[i * kCacheLen + j] = (j >= base && j <= base + i) ? 0 : -32768;
  prefill.inClean(2);
  for (int t = 3; t <= 58; ++t) {  // zero KV history
    std::memset(prefill.inPtr(t), 0, t <= 30 ? kCacheLen * kRowK : kCacheLen * kRowV);
    prefill.inClean(t);
  }
  prefill.infer();

  std::vector<int> gen;
  auto* plogits = static_cast<int16_t*>(prefill.outPtr(0));
  const int rowStride = alignUp(static_cast<int64_t>(kVocab) * 2, kAlign) / 2;  // int16 elems/row
  int first = argmaxS16(plogits + static_cast<size_t>(L - 1) * rowStride, kVocab);
  gen.push_back(first);
  if (stream && first != 151645 && first != 151643) {
    std::printf("TOK %d\n", first);
    std::fflush(stdout);
  }

  // ---- seed decode KV windows from prefill outputs (right-aligned) ----
  std::vector<std::vector<uint8_t>> Kwin(kNLayers, std::vector<uint8_t>(kCacheLen * kRowK, 0));
  std::vector<std::vector<uint8_t>> Vwin(kNLayers, std::vector<uint8_t>(kCacheLen * kRowV, 0));
  for (int l = 0; l < kNLayers; ++l) {
    std::memcpy(Kwin[l].data() + static_cast<size_t>(kCacheLen - L) * kRowK,
                prefill.outPtr(1 + l), static_cast<size_t>(L) * kRowK);
    std::memcpy(Vwin[l].data() + static_cast<size_t>(kCacheLen - L) * kRowV,
                prefill.outPtr(29 + l), static_cast<size_t>(L) * kRowV);
  }

  // ---- DECODE loop ----
  int P = L;
  for (int step = 0; step < max_new - 1; ++step) {
    int cur = gen.back();
    if (cur == 151645 || cur == 151643) break;  // Qwen eos
    *static_cast<int32_t*>(decode.inPtr(0)) = cur;  decode.inClean(0);
    *static_cast<int32_t*>(decode.inPtr(1)) = P;    decode.inClean(1);
    auto* mask = static_cast<int16_t*>(decode.inPtr(2));
    int lo = kCacheLen - 1 - P; if (lo < 0) lo = 0;
    for (int j = 0; j < kCacheLen; ++j) mask[j] = (j >= lo) ? 0 : -32768;
    decode.inClean(2);
    for (int l = 0; l < kNLayers; ++l) {
      decode.setIn(3 + l, Kwin[l].data(), Kwin[l].size());
      decode.setIn(31 + l, Vwin[l].data(), Vwin[l].size());
    }
    decode.infer();
    int nxt = argmaxS16(static_cast<int16_t*>(decode.outPtr(0)), kVocab);
    gen.push_back(nxt);
    if (stream && nxt != 151645 && nxt != 151643) {
      std::printf("TOK %d\n", nxt);
      std::fflush(stdout);
    }
    for (int l = 0; l < kNLayers; ++l) {
      std::memmove(Kwin[l].data(), Kwin[l].data() + kRowK, (kCacheLen - 1) * kRowK);
      std::memcpy(Kwin[l].data() + (kCacheLen - 1) * kRowK, decode.outPtr(1 + l), kRowK);
      std::memmove(Vwin[l].data(), Vwin[l].data() + kRowV, (kCacheLen - 1) * kRowV);
      std::memcpy(Vwin[l].data() + (kCacheLen - 1) * kRowV, decode.outPtr(29 + l), kRowV);
    }
    ++P;
  }
  return gen;
}

}  // namespace

int main(int argc, char** argv) {
  std::string hbm, ids_str;
  int max_new = 24;
  bool serve = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() { return std::string(argv[++i]); };
    if (a == "--hbm") hbm = val();
    else if (a == "--ids") ids_str = val();
    else if (a == "--max-new") max_new = std::atoi(val().c_str());
    else if (a == "--serve") serve = true;
  }
  if (hbm.empty() || (!serve && ids_str.empty())) {
    std::fprintf(stderr, "usage: %s --hbm <model.hbm> [--ids \"id,..\" | --serve] [--max-new N]\n", argv[0]);
    return 1;
  }

  // ---- load the .hbm ONCE via the generic hbDNN runtime (NOT libxlm) --------
  hbDNNPackedHandle_t packed = nullptr;
  const char* files[] = {hbm.c_str()};
  CK(hbDNNInitializeFromFiles(&packed, files, 1));
  Graph prefill, decode;
  prefill.init(packed, "prefill");
  decode.init(packed, "decode");
  std::fprintf(stderr, "[selfengine] loaded prefill+decode via hbDNN (libxlm never touched)\n");

  if (serve) {
    // Persistent server: read one prompt (comma ids) per stdin line, stream
    // tokens, terminate each turn with "END". Model stays resident -> fast.
    if (max_new < 2) max_new = 200;
    std::fprintf(stderr, "[selfengine] serve ready (max-new=%d)\n", max_new);
    std::fflush(stderr);
    std::string line;
    std::printf("READY\n"); std::fflush(stdout);
    while (std::getline(std::cin, line)) {
      if (line.empty()) { std::printf("END\n"); std::fflush(stdout); continue; }
      std::vector<int> prompt = parseIds(line);
      if (prompt.empty() || static_cast<int>(prompt.size()) > kChunk) {
        std::printf("END\n"); std::fflush(stdout); continue;
      }
      generate(prefill, decode, prompt, max_new, /*stream=*/true);
      std::printf("END\n"); std::fflush(stdout);
    }
  } else {
    std::vector<int> prompt = parseIds(ids_str);
    if (prompt.empty() || static_cast<int>(prompt.size()) > kChunk) {
      std::fprintf(stderr, "[selfengine] prompt length must be 1..%d\n", kChunk);
      return 1;
    }
    std::vector<int> gen = generate(prefill, decode, prompt, max_new, /*stream=*/false);
    std::printf("GEN:");
    for (int id : gen) std::printf(" %d", id);
    std::printf("\n");
  }
  hbDNNRelease(packed);
  return 0;
}
