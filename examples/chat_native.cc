// bllm_chat_native — interactive chat over bllm::NativeLlm, entirely in C++ (native
// engine + C++ tokenizer + ChatML, no libxlm, no Python). One model directory in,
// streaming chat out.
//   ./bllm_chat_native --model <model_dir> [--max-new 400] [--raw]
//   commands: /reset  /exit
#include "bllm/native_llm.h"

#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string dir;
  int max_new = 400, top_k = 0;
  float temp = 0.0f, top_p = 1.0f, rep_pen = 1.0f;
  uint64_t seed = 1234;
  bool raw = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model") dir = argv[++i];
    else if (a == "--max-new") max_new = std::atoi(argv[++i]);
    else if (a == "--temp") temp = std::atof(argv[++i]);
    else if (a == "--top-p") top_p = std::atof(argv[++i]);
    else if (a == "--top-k") top_k = std::atoi(argv[++i]);
    else if (a == "--rep-pen") rep_pen = std::atof(argv[++i]);
    else if (a == "--seed") seed = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--raw") raw = true;
  }
  if (dir.empty()) {
    std::fprintf(stderr, "usage: %s --model <dir> [--max-new N] [--raw]"
                 " [--temp T --top-p P --top-k K --rep-pen R --seed S]\n", argv[0]);
    return 1;
  }

  bllm::NativeLlm llm(dir);
  llm.set_sampling(temp, top_p, top_k, rep_pen, seed);
  const auto& c = llm.config();
  std::fprintf(stderr, "[loaded] %s  arch=%s  backend=native  vocab=%d  (%s)\n",
               c.name.c_str(), c.arch.c_str(), llm.tokenizer().vocab_size(),
               raw ? "raw completion" : "chat");
  std::printf("commands: /reset  /exit\n\n");

  auto stream = [](const std::string& s) { std::printf("%s", s.c_str()); std::fflush(stdout); };
  std::string line;
  while (true) {
    std::printf("\033[1mYou:\033[0m ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (line == "/exit" || line == "/quit") break;
    if (line == "/reset") { llm.reset(); std::printf("[context reset]\n\n"); continue; }
    if (line.empty()) continue;
    std::printf("\033[1mBot:\033[0m ");
    if (raw) llm.generate(line, max_new, stream);
    else     llm.chat(line, max_new, stream);
    std::fprintf(stderr, "\n\033[2m[%.2f tok/s]\033[0m\n\n", llm.last_decode_tps());
  }
  return 0;
}
