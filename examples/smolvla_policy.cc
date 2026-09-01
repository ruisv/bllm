// SmolVLA from a model package: two images, a state vector, a sentence, actions out.
//
// This is what "supported" should mean for a VLA — no lerobot, no checkpoint, no
// host-side transform code. Everything comes from the package that
// `export_smolvla_package.py` + `bllm-make-model-dir smolvla` produced.
//
// Unlike π0.5's LIBERO config, SmolVLA genuinely reads proprioception: the state
// becomes a prefix token, so `--state` is not decorative.
//
//   bllm_smolvla_policy --model ~/models/smolvla-libero-224-s100p \
//       --image scene.jpg --wrist wrist.jpg \
//       --prompt "pick up the black bowl and place it on the plate"
//
// `--noise f32` pins the flow-matching latent so two runs are comparable; SmolVLA
// is a flow model integrated from a Gaussian, and without pinning it, two runs
// of the *same* policy differ by more than a port defect does.
#include "bllm/image_io.h"
#include "bllm/native_smolvla_policy.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const char* argOf(int argc, char** argv, const char* key,
                         const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return dflt;
}

int main(int argc, char** argv) {
  try {
    const char* dir = argOf(argc, argv, "--model");
    const char* prompt = argOf(argc, argv, "--prompt");
    const char* imgPath = argOf(argc, argv, "--image");
    const char* wristPath = argOf(argc, argv, "--wrist");
    if (!dir || !prompt || !imgPath) {
      std::fprintf(stderr,
                   "usage: %s --model DIR --image RGB --wrist RGB --prompt TEXT\n"
                   "       [--state state.f32] [--noise noise.f32] [--repeat N]\n"
                   "       [--out actions.f32]\n",
                   argv[0]);
      return 2;
    }
    const int repeat = std::atoi(argOf(argc, argv, "--repeat", "1"));

    auto cfg = bllm::loadModelConfig(dir);
    if (!cfg.is_smolvla())
      throw std::runtime_error("[smolvla] " + std::string(dir) + " is arch=" + cfg.arch);
    bllm::SmolVlaPolicy policy(cfg);

    auto img = bllm::loadImageRGB(imgPath);
    bllm::OwnedImage wrist;
    if (wristPath) wrist = bllm::loadImageRGB(wristPath);

    auto readF32 = [](const char* path) {
      std::ifstream f(path, std::ios::binary | std::ios::ate);
      if (!f) throw std::runtime_error("[smolvla] cannot open " + std::string(path));
      const std::streamsize n = f.tellg();
      f.seekg(0);
      std::vector<float> v((size_t)n / 4);
      f.read((char*)v.data(), n);
      return v;
    };
    std::vector<float> state;
    if (const char* sp = argOf(argc, argv, "--state")) state = readF32(sp);

    std::vector<float> noise;
    if (const char* np = argOf(argc, argv, "--noise")) {
      std::ifstream f(np, std::ios::binary | std::ios::ate);
      if (!f) throw std::runtime_error("[smolvla] cannot open " + std::string(np));
      const std::streamsize n = f.tellg();
      f.seekg(0);
      noise.resize((size_t)n / 4);
      f.read((char*)noise.data(), n);
    }

    bllm::SmolVlaObservation obs;
    obs.image = img.rgb.data();
    obs.image_h = img.height;
    obs.image_w = img.width;
    if (wristPath) {
      obs.wrist_image = wrist.rgb.data();
      obs.wrist_h = wrist.height;
      obs.wrist_w = wrist.width;
    }
    obs.state = state.empty() ? nullptr : state.data();
    obs.prompt = prompt;

    std::vector<float> a;
    double ms = 0;
    for (int r = 0; r < repeat; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      a = policy.act(obs, noise.empty() ? nullptr : noise.data());
      ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
    }

    const int A = policy.actionDim(), H = policy.horizon();
    std::printf("[smolvla] %s\n[smolvla] %d x %d actions in %.0f ms/chunk\n",
                cfg.name.empty() ? dir : cfg.name.c_str(), H, A, ms / repeat);
    for (int t = 0; t < H; ++t) {
      std::printf("  t%-2d ", t);
      for (int j = 0; j < A; ++j) std::printf("% .4f ", a[(size_t)t * A + j]);
      std::printf("\n");
    }
    if (const char* op = argOf(argc, argv, "--out")) {
      std::ofstream o(op, std::ios::binary);
      o.write((const char*)a.data(), a.size() * sizeof(float));
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
