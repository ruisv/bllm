// bllm::Tokenizer — a thin C++ wrapper over HuggingFace `tokenizers` (via
// mlc-ai/tokenizers-cpp, which links the official Rust `tokenizers` crate). This is
// what lets the native engine speak STRINGS, not just token ids — the key step to
// making BLLM a real library rather than a set of scripts (usability, roadmap
// Phase 1). Loads a standard `tokenizer.json`; encode/decode match HF exactly.
#pragma once

#include <tokenizers_cpp.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

class Tokenizer {
 public:
  // Load from a HuggingFace `tokenizer.json` file.
  static Tokenizer fromFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[bllm] cannot open tokenizer.json: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return Tokenizer(ss.str());
  }
  // Load from an in-memory tokenizer.json blob.
  static Tokenizer fromBlob(const std::string& json) { return Tokenizer(json); }

  std::vector<int> encode(const std::string& text) const {
    auto v = tk_->Encode(text);
    return std::vector<int>(v.begin(), v.end());
  }
  std::string decode(const std::vector<int>& ids) const {
    return tk_->Decode(std::vector<int32_t>(ids.begin(), ids.end()));
  }
  int vocab_size() const { return static_cast<int>(tk_->GetVocabSize()); }
  int token_to_id(const std::string& tok) const { return tk_->TokenToId(tok); }
  std::string id_to_token(int id) const { return tk_->IdToToken(id); }

 private:
  explicit Tokenizer(const std::string& json)
      : tk_(tokenizers::Tokenizer::FromBlobJSON(json)) {
    if (!tk_) throw std::runtime_error("[bllm] failed to parse tokenizer.json");
  }
  std::shared_ptr<tokenizers::Tokenizer> tk_;
};

}  // namespace bllm
