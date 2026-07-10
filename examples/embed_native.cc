// bllm_embed_native — text embeddings on the BPU (Qwen3-VL-Embedding), no libxlm.
//
//   ./bllm_embed_native --model <dir> --text "巴黎是法国的首都。" [--dim 256]
//   ./bllm_embed_native --model <dir> --verify ref.json --ref emb_ref.npy
//
// --verify replays the token ids the HF reference used and scores the result two ways:
// per-text cosine, and whether the pairwise-similarity RANKING survives. A vector can
// sit at cosine 0.99 and still reorder a retrieval result, so the ranking is the test
// that matters.
#include "bllm/native_embedder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char* argOf(int argc, char** argv, const char* flag, const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}

// Minimal reader for the reference dump: a JSON with "token_ids": [[...], ...].
std::vector<std::vector<int>> readTokenIds(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss; ss << f.rdbuf();
  const std::string s = ss.str();
  size_t p = s.find("\"token_ids\"");
  if (p == std::string::npos) throw std::runtime_error("no token_ids in " + path);
  std::vector<std::vector<int>> out;
  p = s.find('[', p);                       // outer list
  size_t depth = 0;
  for (size_t i = p; i < s.size(); ++i) {
    if (s[i] == '[') {
      if (++depth == 2) {
        std::vector<int> row;
        size_t j = i + 1, e = s.find(']', j);
        std::stringstream rs(s.substr(j, e - j));
        for (std::string t; std::getline(rs, t, ',');) row.push_back(std::stoi(t));
        out.push_back(std::move(row));
        i = e; --depth;
      }
    } else if (s[i] == ']' && depth-- == 1) break;
  }
  return out;
}

// float32 .npy, C-order, shape (n, d).
std::vector<std::vector<float>> readNpy(const std::string& path, int d) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6]; f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not a .npy");
  unsigned char major; f.read((char*)&major, 1); f.seekg(1, std::ios::cur);
  uint32_t hlen = 0;
  if (major == 1) { uint16_t h; f.read((char*)&h, 2); hlen = h; }
  else            { f.read((char*)&hlen, 4); }
  std::string hdr(hlen, '\0'); f.read(&hdr[0], hlen);
  if (hdr.find("'<f4'") == std::string::npos && hdr.find("\"<f4\"") == std::string::npos)
    throw std::runtime_error("reference .npy must be float32");
  std::vector<std::vector<float>> rows;
  while (f) {
    std::vector<float> r(d);
    f.read((char*)r.data(), (std::streamsize)d * 4);
    if (!f) break;
    rows.push_back(std::move(r));
  }
  return rows;
}

double dot(const std::vector<float>& a, const std::vector<float>& b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  const char* model = argOf(argc, argv, "--model");
  if (!model) {
    std::fprintf(stderr,
                 "usage: %s --model <dir> --text <text> [--dim N]\n"
                 "       %s --model <dir> --verify <ref.json> --ref <emb_ref.npy>\n",
                 argv[0], argv[0]);
    return 2;
  }
  try {
    bllm::NativeEmbedder emb(model);
    std::fprintf(stderr, "[%s] dim=%d  max_tokens=%d\n", emb.config().name.c_str(),
                 emb.dimension(), emb.max_tokens());

    if (const char* refj = argOf(argc, argv, "--verify")) {
      const char* refn = argOf(argc, argv, "--ref");
      if (!refn) { std::fprintf(stderr, "--verify needs --ref <emb_ref.npy>\n"); return 2; }
      const auto ids = readTokenIds(refj);
      const auto ref = readNpy(refn, emb.dimension());
      if (ids.size() != ref.size()) throw std::runtime_error("ref.json / .npy length mismatch");

      std::vector<std::vector<float>> got;
      for (const auto& row : ids) got.push_back(emb.embed_ids(row));

      double worst = 1.0;
      std::printf("per-text cosine vs HF:");
      for (size_t i = 0; i < got.size(); ++i) {
        const double c = dot(got[i], ref[i]);
        worst = std::min(worst, c);
        std::printf(" %.6f", c);
      }
      std::printf("\n  min %.6f   (HF's own bf16-vs-fp32 gap is 0.9998)\n", worst);

      // Ranking: for each query, does the BPU order the others the same way?
      const size_t n = got.size();
      bool order_ok = true; double maxd = 0;
      for (size_t i = 0; i < n; ++i) {
        std::vector<size_t> a(n), b(n);
        std::iota(a.begin(), a.end(), 0); std::iota(b.begin(), b.end(), 0);
        std::stable_sort(a.begin(), a.end(), [&](size_t x, size_t y) { return dot(ref[i], ref[x]) > dot(ref[i], ref[y]); });
        std::stable_sort(b.begin(), b.end(), [&](size_t x, size_t y) { return dot(got[i], got[x]) > dot(got[i], got[y]); });
        if (a != b) order_ok = false;
        for (size_t j = 0; j < n; ++j) maxd = std::max(maxd, std::fabs(dot(ref[i], ref[j]) - dot(got[i], got[j])));
      }
      std::printf("  pairwise-similarity max|diff| %.6f   ranking preserved: %s\n",
                  maxd, order_ok ? "yes" : "NO");
      const bool pass = worst > 0.99 && order_ok;
      std::printf("\n%s\n", pass ? "PASS" : "FAIL");
      return pass ? 0 : 1;
    }

    const std::string text = argOf(argc, argv, "--text", "The capital of France is Paris.");
    const int dim = std::atoi(argOf(argc, argv, "--dim", "0"));
    const auto v = emb.embed(text, dim);
    std::printf("[%zu tokens -> %zu-d]\n", emb.build_ids(text).size(), v.size());
    for (size_t i = 0; i < std::min<size_t>(v.size(), 8); ++i) std::printf("%+.5f ", v[i]);
    std::printf("...\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
