// bllm_vlm — image + text VLM demo (InternVL / Qwen-VL). Feed one or more
// images and a prompt, stream the reply.
//
// NOTE: the OE-LLM S100 SDK 1.0.0 ships no pre-compiled InternVL/Qwen-VL model;
// point --hbm at one once available (official later release or oellm_build).
//
//   ./bllm_vlm --hbm <vlm.hbm> --tokenizer <dir> --config <config.json> \
//              --image a.jpg --image b.jpg --prompt "对比这两张图"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "bllm/bllm.h"

namespace {

void PrintUsage() {
  std::cout <<
      "Usage: bllm_vlm --hbm <vlm.hbm> --tokenizer <dir> [options]\n"
      "  --hbm <path>         compiled VLM .hbm (required)\n"
      "  --tokenizer <dir>    tokenizer config directory (required)\n"
      "  --config <path>      model config file (required for InternVL)\n"
      "  --model-type <name>  internvl|qwen-vl (default: auto)\n"
      "  --image <path>       image file (repeatable)\n"
      "  --prompt <text>      the question\n"
      "  --system <text>      system prompt\n";
}

const char* ArgVal(int argc, char** argv, int& i) {
  if (i + 1 >= argc) { std::cerr << "missing value for " << argv[i] << "\n"; std::exit(1); }
  return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) { PrintUsage(); return 0; }

  bllm::VlmOptions opts;
  std::vector<bllm::VlmImage> images;
  std::string prompt = "描述这张图片。";

  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* f) { return !std::strcmp(argv[i], f); };
    if (is("-h") || is("--help")) { PrintUsage(); return 0; }
    else if (is("--hbm")) opts.model_path = ArgVal(argc, argv, i);
    else if (is("--tokenizer")) opts.tokenizer_dir = ArgVal(argc, argv, i);
    else if (is("--config")) opts.config_path = ArgVal(argc, argv, i);
    else if (is("--model-type")) opts.model_type = bllm::ModelTypeFromString(ArgVal(argc, argv, i));
    else if (is("--system")) opts.system_prompt = ArgVal(argc, argv, i);
    else if (is("--image")) images.push_back(bllm::VlmImage::File(ArgVal(argc, argv, i)));
    else if (is("--prompt")) prompt = ArgVal(argc, argv, i);
    else { std::cerr << "unknown argument: " << argv[i] << "\n"; PrintUsage(); return 1; }
  }

  if (images.empty()) { std::cerr << "no --image given\n"; return 1; }

  try {
    bllm::VlmSession vlm(opts);
    std::cout << "[Assistant] >>> " << std::flush;
    vlm.generate(images, prompt, [](std::string_view chunk) {
      std::cout << chunk << std::flush;
    });
    std::cout << "\n";
    const auto& s = vlm.last_stats();
    std::cout << "  (ttft " << s.ttft_ms << " ms, decode " << s.decode_tps
              << " tok/s, " << s.decode_tokens << " tokens)\n";
  } catch (const bllm::Error& e) {
    std::cerr << "\nbllm error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
