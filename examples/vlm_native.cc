// bllm_vlm_native — image+text chat on the self-built native runtime (no libxlm).
//
//   ./bllm_vlm_native --model /models/qwen2.5-omni-3b --image cat.jpg \
//                     --prompt "描述这张图片" [--max-new 128] [--temp 0.7]
//
// With no --image it is a plain text chat, which is the cheapest way to check that
// the host embedding + mrope + mask plumbing matches the graph's expectations.
#include "bllm/image_io.h"
#include "bllm/native_vlm.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* argOf(int argc, char** argv, const char* flag, const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model = argOf(argc, argv, "--model");
  if (!model) {
    std::fprintf(stderr,
                 "usage: %s --model <dir> [--image <file>]... [--prompt <text>]\n"
                 "          [--max-new N] [--temp T] [--top-p P] [--top-k K] [--seed S]\n",
                 argv[0]);
    return 2;
  }
  const std::string prompt = argOf(argc, argv, "--prompt", "描述一下这张图片。");
  const int max_new = std::atoi(argOf(argc, argv, "--max-new", "128"));
  const float temp = (float)std::atof(argOf(argc, argv, "--temp", "0"));
  const float top_p = (float)std::atof(argOf(argc, argv, "--top-p", "1"));
  const int top_k = std::atoi(argOf(argc, argv, "--top-k", "0"));
  const uint64_t seed = (uint64_t)std::atoll(argOf(argc, argv, "--seed", "1234"));

  std::vector<std::string> image_paths;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--image") == 0) image_paths.emplace_back(argv[i + 1]);

  try {
    bllm::NativeVlm vlm(model);
    vlm.set_sampling(temp, top_p, top_k, 1.0f, seed);

    std::vector<bllm::OwnedImage> owned;
    std::vector<bllm::ImageRGB> images;
    owned.reserve(image_paths.size());
    for (const std::string& p : image_paths) {
      owned.push_back(bllm::loadImageRGB(p));
      std::fprintf(stderr, "[image] %s  %dx%d -> %d vision tokens\n", p.c_str(),
                   owned.back().width, owned.back().height, vlm.vision_tokens());
    }
    for (const auto& o : owned) images.push_back({o.rgb.data(), o.width, o.height});

    std::cout << "> " << prompt << "\n";
    vlm.chat(prompt, images, max_new, [](const std::string& s) {
      std::cout << s << std::flush;
    });
    std::cout << "\n";
    std::fprintf(stderr, "[stats] ttft %.0f ms, decode %.1f tok/s\n",
                 vlm.last_ttft_ms(), vlm.last_decode_tps());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
