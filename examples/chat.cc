// bllm_chat — minimal multi-turn REPL, the bllm-API equivalent of the SDK's
// oellm_run demo. Streams tokens to stdout; type `reset` for a fresh chat,
// `exit` to quit.
//
//   export LD_LIBRARY_PATH=<sdk>/oellm_runtime/lib:$LD_LIBRARY_PATH
//   ./bllm_chat --hbm      Qwen2.5_1.5B_Instruct_1024.hbm \
//               --tokenizer Qwen2.5_1.5B_Instruct_config/ \
//               --template  Qwen2.5_1.5B_Instruct_config/Qwen2.5_1.5B_Instruct.jinja

#include <cstring>
#include <iostream>
#include <string>

#include "bllm/bllm.h"

namespace {

void PrintUsage() {
  std::cout <<
      "Usage: bllm_chat --hbm <model.hbm> --tokenizer <dir> [options]\n"
      "  --hbm <path>         compiled .hbm model (required)\n"
      "  --tokenizer <dir>    tokenizer config directory (required)\n"
      "  --template <path>    .jinja chat template (auto-found in tokenizer dir\n"
      "                       if present; omit for Base/InternLM2 models)\n"
      "  --config <path>      model config (required for InternVL)\n"
      "  --model-type <name>  qwen2.5|deepseek|internlm2|... (default: auto)\n"
      "  --system <text>      system prompt\n"
      "  --core <0-3>         pin a BPU core (default: any)\n"
      "  --stats              print timing after each reply\n";
}

const char* ArgVal(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    std::cerr << "missing value for " << argv[i] << "\n";
    std::exit(1);
  }
  return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    PrintUsage();
    return 0;
  }

  bllm::SessionOptions opts;
  bool show_stats = false;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      PrintUsage();
      return 0;
    } else if (!std::strcmp(argv[i], "--hbm")) {
      opts.model_path = ArgVal(argc, argv, i);
    } else if (!std::strcmp(argv[i], "--tokenizer")) {
      opts.tokenizer_dir = ArgVal(argc, argv, i);
    } else if (!std::strcmp(argv[i], "--template")) {
      opts.chat_template_path = ArgVal(argc, argv, i);
    } else if (!std::strcmp(argv[i], "--config")) {
      opts.config_path = ArgVal(argc, argv, i);
    } else if (!std::strcmp(argv[i], "--model-type")) {
      opts.model_type = bllm::ModelTypeFromString(ArgVal(argc, argv, i));
    } else if (!std::strcmp(argv[i], "--system")) {
      opts.system_prompt = ArgVal(argc, argv, i);
    } else if (!std::strcmp(argv[i], "--core")) {
      int c = std::atoi(ArgVal(argc, argv, i));
      opts.backend = static_cast<bllm::Backend>(
          static_cast<int>(bllm::Backend::Bpu0) + c);
    } else if (!std::strcmp(argv[i], "--stats")) {
      show_stats = true;
    } else {
      std::cerr << "unknown argument: " << argv[i] << "\n";
      PrintUsage();
      return 1;
    }
  }

  try {
    bllm::LlmSession llm(opts);
    std::cout << "bllm_chat ready (model_type=" << bllm::ToString(llm.options().model_type)
              << "). Type 'reset' to clear history, 'exit' to quit.\n\n";

    std::string line;
    std::cout << "[User] <<< " << std::flush;
    while (std::getline(std::cin, line)) {
      if (line == "exit") break;
      if (line == "reset") {
        llm.reset();
        std::cout << "[system] history cleared\n[User] <<< " << std::flush;
        continue;
      }
      if (line.empty()) {
        std::cout << "[User] <<< " << std::flush;
        continue;
      }

      std::cout << "[Assistant] >>> " << std::flush;
      llm.generate(line, [](std::string_view chunk) {
        std::cout << chunk << std::flush;
      });
      std::cout << "\n";
      if (show_stats) {
        const auto& s = llm.last_stats();
        std::cout << "  (ttft " << s.ttft_ms << " ms, decode " << s.decode_tps
                  << " tok/s, " << s.decode_tokens << " tokens)\n";
      }
      std::cout << "[User] <<< " << std::flush;
    }
  } catch (const bllm::Error& e) {
    std::cerr << "\nbllm error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
