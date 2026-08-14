// π0.5 from a model package: two images, a sentence, actions out.
//
// This is what "supported" should mean for a VLA — no openpi, no checkpoint, no
// host-side transform code. Everything comes from the package that
// `export_pi05_package.py` + `bllm-make-model-dir pi05` produced.
//
//   bllm_pi05_policy --model ~/models/pi05-libero-224-s100p \
//       --image scene.jpg --wrist wrist.jpg \
//       --prompt "pick up the black bowl and place it on the plate"
//
// `--noise f32` pins the flow-matching latent so two runs are comparable; π0.5
// is a flow model integrated from a Gaussian, and without pinning it, two runs
// of the *same* policy differ by more than a port defect does.
#include "bllm/image_io.h"
#include "bllm/native_pi05_policy.h"

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
                   "       [--noise noise.f32] [--repeat N] [--out actions.f32]\n",
                   argv[0]);
      return 2;
    }
    const int repeat = std::atoi(argOf(argc, argv, "--repeat", "1"));

    auto cfg = bllm::loadModelConfig(dir);
    if (!cfg.is_pi05())
      throw std::runtime_error("[pi05] " + std::string(dir) + " is arch=" + cfg.arch);
    bllm::Pi05Policy policy(cfg);

    auto img = bllm::loadImageRGB(imgPath);
    bllm::OwnedImage wrist;
    if (wristPath) wrist = bllm::loadImageRGB(wristPath);

    std::vector<float> noise;
    if (const char* np = argOf(argc, argv, "--noise")) {
      std::ifstream f(np, std::ios::binary | std::ios::ate);
      if (!f) throw std::runtime_error("[pi05] cannot open " + std::string(np));
      const std::streamsize n = f.tellg();
      f.seekg(0);
      noise.resize((size_t)n / 4);
      f.read((char*)noise.data(), n);
    }

    bllm::Pi05Observation obs;
    obs.image = img.rgb.data();
    obs.image_h = img.height;
    obs.image_w = img.width;
    if (wristPath) {
      obs.wrist_image = wrist.rgb.data();
      obs.wrist_h = wrist.height;
      obs.wrist_w = wrist.width;
    }
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
    std::printf("[pi05] %s\n[pi05] %d x %d actions in %.0f ms/chunk\n",
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
