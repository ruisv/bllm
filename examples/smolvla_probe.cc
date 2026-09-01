// End-to-end SmolVLA on the board: patches in, action chunk out, scored against
// the golden the host reference produced for the same inputs.
//
//   ./bllm_smolvla_probe --vision vision.hbm --prefix prefix.hbm --expert expert.hbm \
//        --patches cams.f32 --prompt prompt.f32 --state state.f32 \
//        --cos cos.f32 --sin sin.f32 --bias bias.f32 \
//        --acos acos.f32 --asin asin.f32 --abias abias.f32 \
//        --cond cond_table.f32 --noise noise.f32 --ref ref_chunk.f32
//
// Every stage has already been scored on its own; this asks the only question
// left, which is whether the chain holds when the errors compose — and what it
// costs with all three graphs resident in one process rather than reloaded.
#include "bllm/native_smolvla.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* argOf(int argc, char** argv, const char* flag, const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}

std::vector<float> readF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const size_t n = (size_t)f.tellg() / sizeof(float);
  std::vector<float> v(n);
  f.seekg(0);
  f.read((char*)v.data(), n * sizeof(float));
  return v;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
  }
  return num / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bllm::SmolVlaShape shape;
    shape.cameras = std::atoi(argOf(argc, argv, "--cameras", "2"));
    shape.prompt_len = std::atoi(argOf(argc, argv, "--prompt-len", "48"));
    shape.horizon = std::atoi(argOf(argc, argv, "--horizon", "50"));
    shape.steps = std::atoi(argOf(argc, argv, "--steps", "10"));
    const int repeat = std::atoi(argOf(argc, argv, "--repeat", "3"));

    bllm::SmolVlaStack stack(argOf(argc, argv, "--vision"), argOf(argc, argv, "--prefix"),
                             argOf(argc, argv, "--expert"), shape);
    std::printf("[smolvla] prefix %d = %d x %d image + %d language + 1 state; "
                "horizon %d, %d steps\n",
                shape.prefix(), shape.cameras, shape.vision_tokens, shape.prompt_len,
                shape.horizon, shape.steps);

    const auto patches = readF32(argOf(argc, argv, "--patches"));
    const auto prompt = readF32(argOf(argc, argv, "--prompt"));
    const auto state = readF32(argOf(argc, argv, "--state"));
    const auto cos_ = readF32(argOf(argc, argv, "--cos"));
    const auto sin_ = readF32(argOf(argc, argv, "--sin"));
    const auto bias = readF32(argOf(argc, argv, "--bias"));
    const auto acos_ = readF32(argOf(argc, argv, "--acos"));
    const auto asin_ = readF32(argOf(argc, argv, "--asin"));
    const auto abias = readF32(argOf(argc, argv, "--abias"));
    const auto cond = readF32(argOf(argc, argv, "--cond"));
    const auto noise = readF32(argOf(argc, argv, "--noise"));
    std::vector<float> actions((size_t)shape.horizon * shape.action_dim);

    double msObs = 0, msLoop = 0;
    for (int r = 0; r < repeat; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      stack.setObservation(patches.data(), prompt.data(), state.data(), cos_.data(),
                           sin_.data(), bias.data());
      auto t1 = std::chrono::steady_clock::now();
      stack.sampleActions(noise.data(), cond.data(), acos_.data(), asin_.data(),
                          abias.data(), actions.data());
      auto t2 = std::chrono::steady_clock::now();
      msObs += std::chrono::duration<double, std::milli>(t1 - t0).count();
      msLoop += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    std::printf("[smolvla] observation %.1f ms   flow loop %.1f ms   chunk %.1f ms\n",
                msObs / repeat, msLoop / repeat, (msObs + msLoop) / repeat);

    if (const char* rp = argOf(argc, argv, "--ref")) {
      const auto ref = readF32(rp);
      if (ref.size() != actions.size())
        throw std::runtime_error("ref has " + std::to_string(ref.size()) + " floats, chunk has " +
                                 std::to_string(actions.size()));
      double maxd = 0, na = 0, nb = 0;
      for (size_t i = 0; i < ref.size(); ++i) {
        maxd = std::max(maxd, (double)std::abs(ref[i] - actions[i]));
        na += (double)actions[i] * actions[i];
        nb += (double)ref[i] * ref[i];
      }
      std::printf("[smolvla] chunk vs golden: cosine %.7f   gain %.6f   max|d| %.5f\n",
                  cosine(ref, actions), std::sqrt(na / nb), maxd);
    }
    if (const char* op = argOf(argc, argv, "--out")) {
      std::ofstream o(op, std::ios::binary);
      o.write((const char*)actions.data(), actions.size() * sizeof(float));
      std::printf("[smolvla] wrote %s\n", op);
    }
    std::printf("[smolvla] head: %.4f %.4f %.4f %.4f\n", actions[0], actions[1],
                actions[2], actions[3]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
