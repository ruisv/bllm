// End-to-end π0.5 on the board: patches in, action chunk out, scored against the
// golden the host reference produced for the same inputs.
//
//   ./bllm_pi05_probe --vit siglip.hbm --prefill paligemma.hbm --expert expert.hbm \
//        --patches cam.f32 --prompt prompt.f32 --cond cond_table.f32 \
//        --noise noise.f32 --ref ref_actions.f32
//
// Every stage has already been scored on its own; this asks the only question
// left, which is whether the chain holds when the errors compose.
#include "bllm/native_pi05.h"

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
    bllm::Pi05Shape shape;
    shape.cameras = std::atoi(argOf(argc, argv, "--cameras", "2"));
    shape.prompt_len = std::atoi(argOf(argc, argv, "--prompt-len", "20"));
    shape.horizon = std::atoi(argOf(argc, argv, "--horizon", "10"));
    shape.steps = std::atoi(argOf(argc, argv, "--steps", "10"));
    const int repeat = std::atoi(argOf(argc, argv, "--repeat", "3"));

    bllm::Pi05Stack stack(argOf(argc, argv, "--vit"), argOf(argc, argv, "--prefill"),
                          argOf(argc, argv, "--expert"), shape);
    std::printf("[pi05] prefix %d = %d cameras x %d + %d prompt; horizon %d, %d steps\n",
                shape.prefix(), shape.cameras, shape.vision_tokens, shape.prompt_len,
                shape.horizon, shape.steps);

    const auto patches = readF32(argOf(argc, argv, "--patches"));
    const auto prompt = readF32(argOf(argc, argv, "--prompt"));
    const auto cond = readF32(argOf(argc, argv, "--cond"));
    const auto noise = readF32(argOf(argc, argv, "--noise"));
    // The action rows' RoPE depends on the instruction length, so the table is
    // an input now; `make_pi05_case.py` writes it beside the other blobs.
    const auto ropeCos = readF32(argOf(argc, argv, "--rope-cos"));
    const auto ropeSin = readF32(argOf(argc, argv, "--rope-sin"));
    std::vector<float> bias((size_t)shape.prefix(), 0.0f);
    if (const char* bp = argOf(argc, argv, "--bias")) bias = readF32(bp);
    std::vector<float> actions((size_t)shape.horizon * shape.action_dim);

    double msObs = 0, msLoop = 0;
    for (int r = 0; r < repeat; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      stack.setObservation(patches.data(), prompt.data(), bias.data());
      auto t1 = std::chrono::steady_clock::now();
      stack.sampleActions(noise.data(), cond.data(), ropeCos.data(), ropeSin.data(),
                          actions.data());
      auto t2 = std::chrono::steady_clock::now();
      msObs += std::chrono::duration<double, std::milli>(t1 - t0).count();
      msLoop += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    std::printf("[pi05] observation %.1f ms   flow loop %.1f ms   chunk %.1f ms\n",
                msObs / repeat, msLoop / repeat, (msObs + msLoop) / repeat);

    if (const char* rp = argOf(argc, argv, "--ref")) {
      const auto ref = readF32(rp);
      if (ref.size() != actions.size()) {
        std::fprintf(stderr, "ref has %zu floats, chunk has %zu\n", ref.size(),
                     actions.size());
        return 1;
      }
      double maxd = 0;
      for (size_t i = 0; i < ref.size(); ++i)
        maxd = std::max(maxd, (double)std::abs(ref[i] - actions[i]));
      std::printf("[pi05] chunk vs golden: cosine %.7f   max|d| %.5f\n",
                  cosine(ref, actions), maxd);
    }
    if (const char* op = argOf(argc, argv, "--out")) {
      std::ofstream o(op, std::ios::binary);
      o.write((const char*)actions.data(), actions.size() * sizeof(float));
      std::printf("[pi05] wrote %s\n", op);
    }
    std::printf("[pi05] head: %.4f %.4f %.4f %.4f\n", actions[0], actions[1],
                actions[2], actions[3]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
