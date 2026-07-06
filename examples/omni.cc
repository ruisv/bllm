// bllm_omni — offline multimodal demo for Qwen2.5-Omni. Feed any of
// text/image/audio/video and stream the text reply.
//
//   export LD_LIBRARY_PATH=<sdk>/oellm_runtime/lib:$LD_LIBRARY_PATH
//   ./bllm_omni --text  Qwen2.5_Omni_3B_Text.hbm \
//               --visual Qwen2.5_Omni_3B_Visual.hbm \
//               --audio Qwen2.5_Omni_3B_Audio.hbm \
//               --embed embed_tokens.bin \
//               --tokenizer Qwen2.5_Omni_3B_config/ \
//               --video draw_guitar.mp4 --prompt "描述这段视频"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "bllm/bllm.h"

namespace {

void PrintUsage() {
  std::cout <<
      "Usage: bllm_omni --text <t.hbm> --visual <v.hbm> --audio <a.hbm>\n"
      "                 --embed <embed_tokens.bin> --tokenizer <dir> [input...]\n"
      "  model (all required):\n"
      "    --text/--visual/--audio <hbm>   the three Omni sub-models\n"
      "    --embed <bin>                   embed_tokens.bin\n"
      "    --tokenizer <dir>               Qwen2.5_Omni_3B_config/\n"
      "  input (mix any; at least one):\n"
      "    --prompt <text>    --image <path>    --audio-in <path>\n"
      "    --video <path>     --system <text>\n";
}

const char* ArgVal(int argc, char** argv, int& i) {
  if (i + 1 >= argc) { std::cerr << "missing value for " << argv[i] << "\n"; std::exit(1); }
  return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) { PrintUsage(); return 0; }

  bllm::OmniOptions opts;
  std::vector<bllm::Content> parts;

  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* f) { return !std::strcmp(argv[i], f); };
    if (is("-h") || is("--help")) { PrintUsage(); return 0; }
    else if (is("--text")) opts.text_model_path = ArgVal(argc, argv, i);
    else if (is("--visual")) opts.visual_model_path = ArgVal(argc, argv, i);
    else if (is("--audio")) opts.audio_model_path = ArgVal(argc, argv, i);
    else if (is("--embed")) opts.embed_tokens_path = ArgVal(argc, argv, i);
    else if (is("--tokenizer")) opts.tokenizer_dir = ArgVal(argc, argv, i);
    else if (is("--system")) opts.system_prompt = ArgVal(argc, argv, i);
    else if (is("--prompt")) parts.push_back(bllm::Content::Text(ArgVal(argc, argv, i)));
    else if (is("--image")) parts.push_back(bllm::Content::Image(ArgVal(argc, argv, i)));
    else if (is("--audio-in")) parts.push_back(bllm::Content::Audio(ArgVal(argc, argv, i)));
    else if (is("--video")) parts.push_back(bllm::Content::Video(ArgVal(argc, argv, i)));
    else { std::cerr << "unknown argument: " << argv[i] << "\n"; PrintUsage(); return 1; }
  }

  if (parts.empty()) {
    std::cerr << "no input given — add --prompt/--image/--audio-in/--video\n";
    return 1;
  }

  try {
    bllm::OmniSession omni(opts);
    std::cout << "[Assistant] >>> " << std::flush;
    omni.generate(parts, [](std::string_view chunk) {
      std::cout << chunk << std::flush;
    });
    std::cout << "\n";
    const auto& s = omni.last_stats();
    std::cout << "  (ttft " << s.ttft_ms << " ms, decode " << s.decode_tps
              << " tok/s, " << s.decode_tokens << " tokens)\n";
  } catch (const bllm::Error& e) {
    std::cerr << "\nbllm error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
