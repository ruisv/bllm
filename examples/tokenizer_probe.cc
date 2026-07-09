// bllm_tokenizer_probe — sanity-check the pure-C++ tokenizer (bllm::Tokenizer over
// HuggingFace `tokenizers`). Load a tokenizer.json, encode/decode a few strings.
// This is the seam that lets the native engine speak strings, not token ids.
//   ./bllm_tokenizer_probe qwen35_tok/tokenizer.json
#include "bllm/tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <tokenizer.json> [text ...]\n", argv[0]);
    return 1;
  }
  auto tk = bllm::Tokenizer::fromFile(argv[1]);
  std::printf("vocab_size=%d\n", tk.vocab_size());
  std::vector<std::string> texts;
  for (int i = 2; i < argc; ++i) texts.push_back(argv[i]);
  if (texts.empty()) texts = {"The capital of France is", "你好，世界"};
  for (const auto& s : texts) {
    auto ids = tk.encode(s);
    std::printf("encode(%s) =", s.c_str());
    for (int id : ids) std::printf(" %d", id);
    std::printf("\n  decode -> '%s'\n", tk.decode(ids).c_str());
  }
  return 0;
}
