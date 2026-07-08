// bllm_selfengine — CLI over bllm::NativeEngine (include/bllm/native_engine.h):
// drive a compiled `.hbm` directly on the generic hbDNN/hbUCP BPU runtime, with
// our own KV cache and sampling, WITHOUT libxlm. The engine itself lives in the
// header so this CLI and the Python binding (python/_bllm_native) share one
// implementation. See docs/NATIVE_RUNTIME.md.
//
// Build (project):  ninja -C build bllm_selfengine   (gated on FindHobot)
// One-shot:  ./bllm_selfengine --hbm m.hbm --ids "785,6722,315,9625,374" --max-new 24
// Server  :  ./bllm_selfengine --hbm m.hbm --serve [--temp .7 --top-p .9 --eos "151645,151643"]
//   protocol (stdin lines): "R" resets context; "M:id,id,.." caps this request at
//   M tokens; a bare "id,id,.." uses the default. Each request streams "TOK <id>"
//   then "STATS ttft_ms=.. decode_tps=.. ntok=.." then "END".

#include "bllm/native_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
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
}  // namespace

int main(int argc, char** argv) {
  std::string hbm, ids_str, eos_str;
  bool serve = false;
  bllm::NativeSamplingParams sp;
  sp.max_new = 24;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() { return std::string(argv[++i]); };
    if (a == "--hbm") hbm = val();
    else if (a == "--ids") ids_str = val();
    else if (a == "--max-new") sp.max_new = std::atoi(val().c_str());
    else if (a == "--serve") serve = true;
    else if (a == "--temp") sp.temp = std::atof(val().c_str());
    else if (a == "--top-k") sp.top_k = std::atoi(val().c_str());
    else if (a == "--top-p") sp.top_p = std::atof(val().c_str());
    else if (a == "--rep-pen") sp.rep_pen = std::atof(val().c_str());
    else if (a == "--seed") sp.seed = std::strtoul(val().c_str(), nullptr, 10);
    else if (a == "--eos") eos_str = val();
  }
  if (hbm.empty() || (!serve && ids_str.empty())) {
    std::fprintf(stderr, "usage: %s --hbm <model.hbm> [--ids \"id,..\" | --serve]"
                 " [--max-new N] [--temp T --top-k K --top-p P --rep-pen R --seed S]"
                 " [--eos \"id,id\"]\n", argv[0]);
    return 1;
  }

  bllm::NativeEngine eng(hbm);
  if (!eos_str.empty()) sp.eos = parseIds(eos_str);
  std::fprintf(stderr, "[selfengine] loaded via hbDNN (libxlm never touched): "
               "layers=%d kv=%d head_dim=%d cache_len=%d vocab=%d\n",
               eng.n_layers(), eng.n_kv(), eng.head_dim(), eng.cache_len(), eng.vocab());

  if (serve) {
    if (sp.max_new < 2) sp.max_new = 512;
    std::fprintf(stderr, "[selfengine] serve ready\n"); std::fflush(stderr);
    std::printf("READY\n"); std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "R" || line == "r") {
        eng.reset();
        std::printf("OK\n"); std::fflush(stdout); continue;
      }
      bllm::NativeSamplingParams req = sp;
      size_t colon = line.find(':');
      if (colon != std::string::npos && line.find_first_not_of("0123456789") == colon) {
        req.max_new = std::atoi(line.substr(0, colon).c_str());
        line = line.substr(colon + 1);
      }
      std::vector<int> ids = parseIds(line);
      if (ids.empty()) { std::printf("END\n"); std::fflush(stdout); continue; }
      eng.feed(ids);
      eng.generate(req, [](int t) { std::printf("TOK %d\n", t); std::fflush(stdout); });
      const auto& s = eng.last_stats();
      std::printf("STATS ttft_ms=%.1f decode_tps=%.2f ntok=%d\n",
                  s.ttft_ms, s.decode_tps, s.ntok);
      std::printf("END\n"); std::fflush(stdout);
    }
  } else {
    std::vector<int> prompt = parseIds(ids_str);
    eng.feed(prompt);
    std::vector<int> gen = eng.generate(sp);
    std::printf("GEN:");
    for (int id : gen) std::printf(" %d", id);
    std::printf("\n");
  }
  return 0;
}
