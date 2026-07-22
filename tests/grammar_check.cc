// A tiny CLI over bllm::Grammar, so the Python-side JSON-Schema -> GBNF converter can be
// tested against the REAL matcher instead of against its own idea of the output.
//
//   grammar_check <grammar-file> <text>
//     exit 0  text matches and the grammar is satisfied
//     exit 1  text is a valid prefix but incomplete
//     exit 2  text is rejected
//     exit 3  the grammar itself does not parse (message on stderr)
//
// Header-only and hobot-free — tests/test_json_grammar.py compiles it on demand with
// whatever c++ is around.
#include "bllm/native_grammar.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: grammar_check <grammar-file> <text>\n");
    return 3;
  }
  std::ifstream f(argv[1]);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 3; }
  std::stringstream ss;
  ss << f.rdbuf();
  try {
    bllm::Grammar g(ss.str());
    const std::string text = argv[2];
    // Feed it byte by byte: a token boundary can fall anywhere, including mid-character,
    // so this also exercises the partial-UTF-8 path on every input.
    for (size_t i = 0; i < text.size(); ++i) {
      const std::string piece(1, text[i]);
      if (!g.accepts(piece)) return 2;
      g.accept(piece);
    }
    return g.can_end() ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 3;
  }
}
