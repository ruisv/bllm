// bllm_graph_probe — run ANY compiled hbDNN graph from f32 files and dump its
// outputs, so a host-side reference can be diffed against the board directly.
//
// The vision probe knows a tower's shape conventions and refuses anything else,
// which is right for a tower and useless for bisecting. This one knows nothing:
// it reports every input/output's shape and dtype, fills inputs from the files
// you name (in order; a missing file leaves that input zeroed), and writes each
// output as f32. That covers a partial-stage vision graph, and it covers the
// LLM graphs whose 60 inputs are mostly zero-initialized caches.
//
//   ./bllm_graph_probe --model g.hbm --graph beingh_vit_blk \
//        --in patches.f32 --out-prefix /tmp/stage --repeat 5
#include "bllm/native_engine.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

const char* argOf(int argc, char** argv, const char* flag, const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}

// Every --in in order; "-" means "leave this input at zero".
std::vector<std::string> argsOf(int argc, char** argv, const char* flag) {
  std::vector<std::string> v;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) v.emplace_back(argv[i + 1]);
  return v;
}

size_t elems(const hbDNNTensorShape& s) {
  size_t n = 1;
  for (int i = 0; i < s.numDimensions; ++i) n *= (size_t)s.dimensionSize[i];
  return n;
}

std::string shapeStr(const hbDNNTensorShape& s) {
  std::string r = "[";
  for (int i = 0; i < s.numDimensions; ++i)
    r += (i ? "," : "") + std::to_string(s.dimensionSize[i]);
  return r + "]";
}

}  // namespace

int main(int argc, char** argv) {
  const char* model = argOf(argc, argv, "--model");
  const char* graph = argOf(argc, argv, "--graph");
  if (!model || !graph) {
    std::fprintf(stderr,
                 "usage: %s --model <x.hbm> --graph <name> [--in f32]... "
                 "[--out-prefix p] [--repeat N]\n"
                 "       --in may repeat, in input order; \"-\" zero-fills one\n",
                 argv[0]);
    return 2;
  }
  const auto ins = argsOf(argc, argv, "--in");
  const char* prefix = argOf(argc, argv, "--out-prefix");
  const int repeat = std::atoi(argOf(argc, argv, "--repeat", "1"));

  hbDNNPackedHandle_t packed = nullptr;
  const char* files[] = {model};
  BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&packed, files, 1));
  bllm::native_detail::Graph g;
  g.init(packed, graph);

  std::printf("[graph] %s: %d inputs, %d outputs\n", graph, g.inCount(), g.outCount());
  for (int i = 0; i < g.inCount(); ++i)
    std::printf("  in[%2d]  %-18s type=%d\n", i, shapeStr(g.inShape(i)).c_str(), g.inType(i));
  for (int i = 0; i < g.outCount(); ++i)
    std::printf("  out[%2d] %-18s type=%d scale=%g\n", i, shapeStr(g.outShape(i)).c_str(),
                g.outType(i), g.outScale(i));

  for (int i = 0; i < g.inCount(); ++i) {
    const auto& s = g.inShape(i);
    const size_t want = elems(s);
    const int64_t cols = s.numDimensions ? s.dimensionSize[s.numDimensions - 1] : 1;
    // Rows may be padded up to kAlign, so zero the whole padded buffer and write
    // through writeRowsF32 rather than assuming the layout is dense.
    std::memset(g.inPtr(i), 0, (size_t)(g.inStride(i, 0) * s.dimensionSize[0]));
    if (i >= (int)ins.size() || ins[i] == "-") continue;
    std::ifstream f(ins[i], std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", ins[i].c_str()); return 1; }
    const size_t have = (size_t)f.tellg() / sizeof(float);
    if (have != want) {
      std::fprintf(stderr, "[graph] %s holds %zu floats, in[%d] wants %zu\n",
                   ins[i].c_str(), have, i, want);
      return 1;
    }
    f.seekg(0);
    std::vector<float> buf(want);
    f.read((char*)buf.data(), want * sizeof(float));
    // Typed: pi0-pipeline graphs are fp16 in and int64 for position ids.
    g.writeRowsF32As(i, buf.data(), (int64_t)(want / (size_t)cols), cols);
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) g.infer();
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
      repeat;
  std::printf("[graph] infer %.2f ms\n", ms);

  for (int i = 0; i < g.outCount(); ++i) {
    const auto& os = g.outShape(i);
    const size_t n = elems(os);
    const int nd = os.numDimensions;
    const int64_t cols = nd ? os.dimensionSize[nd - 1] : 1;
    const int64_t rows = cols ? (int64_t)(n / (size_t)cols) : 0;
    // The last dim is contiguous and the axis above it is padded up to kAlign;
    // [16,257,257] pads, because 257 floats is 1028 bytes.
    const int64_t pitch = g.outRowPitch(i, cols);
    std::vector<float> v(n);
    (void)pitch;
    try {
      g.readRowsF32As(i, v.data(), rows, cols);
    } catch (const std::exception& e) {
      std::printf("  out[%d]: %s, skipped\n", i, e.what());
      continue;
    }
    double ss = 0;
    for (float x : v) ss += (double)x * x;
    std::printf("  out[%2d] rms=%.6f  head: %.4f %.4f %.4f %.4f\n", i,
                std::sqrt(ss / (double)n), v[0], v[1], v[2], v[3]);
    if (prefix) {
      const std::string path = std::string(prefix) + "_" + std::to_string(i) + ".f32";
      std::ofstream o(path, std::ios::binary);
      o.write((const char*)v.data(), n * sizeof(float));
    }
  }
  hbDNNRelease(packed);
  return 0;
}
