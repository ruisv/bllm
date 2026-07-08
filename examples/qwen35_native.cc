// bllm_qwen35 — CLI over bllm::NativeHybridEngine: drive the full Qwen3.5-0.8B
// hybrid (Gated-DeltaNet + attention) model on hbDNN/hbUCP, in-process, WITHOUT
// libxlm. Reports decode tok/s (the perf number) and prints generated token ids
// (decode them with the tokenizer for coherence).
//
// Build: ninja -C build bllm_qwen35   (gated on FindHobot)
// Run  : ./bllm_qwen35 --hbm qwen35_0.8b_fast.hbm --embed embed_fp16.bin \
//          --ids "760,6511,314,9338,369" --max-new 20 [--eos 248044]
#include "bllm/native_hybrid_engine.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
std::vector<int> parseIds(const std::string& s) {
  std::vector<int> v; size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i); if (j == std::string::npos) j = s.size();
    if (j > i) v.push_back(std::atoi(s.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  std::string hbm, embed, ids_str, eos_str = "248044", name = "qwen35";
  int max_new = 20;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() { return std::string(argv[++i]); };
    if (a == "--hbm") hbm = val();
    else if (a == "--embed") embed = val();
    else if (a == "--ids") ids_str = val();
    else if (a == "--max-new") max_new = std::atoi(val().c_str());
    else if (a == "--eos") eos_str = val();
    else if (a == "--name") name = val();
  }
  if (hbm.empty() || embed.empty() || ids_str.empty()) {
    std::fprintf(stderr, "usage: %s --hbm m.hbm --embed embed_fp16.bin --ids \"id,..\""
                 " [--max-new N] [--eos \"id,..\"]\n", argv[0]);
    return 1;
  }
  std::vector<int> ids = parseIds(ids_str), eos = parseIds(eos_str);

  bllm::NativeHybridEngine eng(hbm, embed, name);
  std::fprintf(stderr, "[loaded] hidden=%d vocab=%d caches=%d cache_len=%d\n",
               eng.hidden(), eng.vocab(), eng.n_cache(), eng.cache_len());

  // prefill (all but last token), timing separately
  using clk = std::chrono::steady_clock;
  auto tp = clk::now();
  for (size_t i = 0; i + 1 < ids.size(); ++i) eng.step(ids[i]);
  double prefill_s = std::chrono::duration<double>(clk::now() - tp).count();
  eng.step(ids.back());   // establishes first logits

  std::printf("ids:");
  auto t0 = clk::now();
  std::vector<int> gen = eng.generate(max_new, eos, [](int id) { std::printf(" %d", id); std::fflush(stdout); });
  double dt = std::chrono::duration<double>(clk::now() - t0).count();
  std::printf("\n");
  std::fprintf(stderr, "[prefill %zu tok in %.3fs] [decode %zu tok in %.3fs = %.2f tok/s]\n",
               ids.size(), prefill_s, gen.size(), dt, gen.size() / (dt > 0 ? dt : 1));
  return 0;
}
