// bllm_embed_native — text embeddings on the BPU (Qwen3-VL-Embedding), no libxlm.
//
//   ./bllm_embed_native --model <dir> --text "巴黎是法国的首都。" [--dim 256]
//   ./bllm_embed_native --model <dir> --verify ref.json --ref emb_ref.npy
//
// --verify replays the token ids the HF reference used and reports per-text cosine plus
// whether the pairwise-similarity ranking survives. It is a SMOKE TEST, not a ship gate:
// the five reference sentences are near-duplicates whose pairwise similarities are nearly
// tied, so their ranking flips on noise. Measured on SciFact (1000 docs, 300 queries), the
// preserve_precision build reorders that 5-way ranking and still retrieves as well as fp32
// — nDCG@10 0.8437 vs 0.8469 — while the build with int16 RMSNorm variance collapses to
// 0.3358. What separates them is the cosine, not the ranking. The real gate is
#include "bllm/native_embedder.h"

#include <algorithm>
#include <chrono>
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

// float32 .npy, C-order, shape (n, d). Minimal writer — enough for the eval pipeline.
void writeNpy(const std::string& path, const std::vector<std::vector<float>>& rows) {
  if (rows.empty()) throw std::runtime_error("nothing to write");
  std::ostringstream hdr;
  hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows.size() << ", "
      << rows[0].size() << "), }";
  std::string h = hdr.str();
  size_t pre = 10 + h.size() + 1;                 // magic+ver+len, header, trailing '\n'
  h.append((16 - pre % 16) % 16, ' ');
  h.push_back('\n');
  const uint16_t hlen = (uint16_t)h.size();
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write("\x93NUMPY\x01\x00", 8);
  f.write((const char*)&hlen, 2);
  f.write(h.data(), h.size());
  for (const auto& r : rows) f.write((const char*)r.data(), r.size() * sizeof(float));
}

int main(int argc, char** argv) {
  const char* model = argOf(argc, argv, "--model");
  if (!model) {
    std::fprintf(stderr,
                 "usage: %s --model <dir> --text <text> [--dim N]\n"
                 "       %s --model <dir> --verify <ref.json> --ref <emb_ref.npy>\n"
                 "       %s --model <dir> --dump <eval.json> --out <emb.npy>\n",
                 argv[0], argv[0], argv[0]);
    return 2;
  }
  try {
    bllm::NativeEmbedder emb(model);
    std::fprintf(stderr, "[%s] dim=%d  max_tokens=%d\n", emb.config().name.c_str(),
                 emb.dimension(), emb.max_tokens());

    // Batch mode: replay a make_scifact_eval.py id set and dump the vectors, so a real
    // retrieval eval can be scored off-board against the fp32 reference.
    if (const char* dumpj = argOf(argc, argv, "--dump")) {
      const char* outn = argOf(argc, argv, "--out");
      if (!outn) { std::fprintf(stderr, "--dump needs --out <emb.npy>\n"); return 2; }
      const auto ids = readTokenIds(dumpj);
      std::vector<std::vector<float>> got;
      got.reserve(ids.size());
      const auto t0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i < ids.size(); ++i) {
        got.push_back(emb.embed_ids(ids[i]));
        if ((i + 1) % 200 == 0) {
          const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          std::fprintf(stderr, "  %zu/%zu  %.0fs\n", i + 1, ids.size(), el);
        }
      }
      writeNpy(outn, got);
      const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "wrote %zu x %zu -> %s  (%.1f ms/text)\n", got.size(),
                   got[0].size(), outn, el * 1000.0 / got.size());
      return 0;
    }

    if (const char* refj = argOf(argc, argv, "--verify")) {
      const char* refn = argOf(argc, argv, "--ref");
      if (!refn) { std::fprintf(stderr, "--verify needs --ref <emb_ref.npy>\n"); return 2; }
      const auto ids = readTokenIds(refj);
      const auto ref = readNpy(refn, emb.dimension());
      if (ids.size() != ref.size()) throw std::runtime_error("ref.json / .npy length mismatch");

      std::vector<std::vector<float>> got;
      for (const auto& row : ids) got.push_back(emb.embed_ids(row));

      double worst = 1.0, mean = 0.0;
      std::printf("per-text cosine vs HF:");
      for (size_t i = 0; i < got.size(); ++i) {
        const double c = dot(got[i], ref[i]);
        worst = std::min(worst, c);
        mean += c / got.size();
        std::printf(" %.6f", c);
      }
      std::printf("\n  min %.6f  mean %.6f\n", worst, mean);

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
      std::printf("  pairwise-similarity max|diff| %.6f   ranking preserved: %s"
                  "  (informational — these 5 sentences are near-ties)\n",
                  maxd, order_ok ? "yes" : "no");
      // Threshold from measurement, not taste: on SciFact the mean-cosine 0.9825 build
      // retrieves at fp32 quality and the mean-cosine 0.5665 build is destroyed. Nothing
      // observed lands between. Run eval_retrieval.py before shipping.
      const bool intact = mean > 0.95;
      std::printf("\n%s\n", intact ? "ENCODER INTACT (smoke test; gate on eval_retrieval.py)"
                                   : "ENCODER BROKEN");
      return intact ? 0 : 1;
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
