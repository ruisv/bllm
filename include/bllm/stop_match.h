// bllm::StopMatcher — stop-string detection for a streaming generator.
//
// A stop string is a string-space concept, but the engine generates in token space and
// a stop string can span several tokens (and a token can decode to several bytes). So the
// only correct place to look is the accumulated decoded text, and the only correct thing
// to stream is text that CANNOT still turn into a stop string. This owns both:
//
//   * cut  — the byte offset of the earliest COMPLETE stop string in the text so far, or
//            npos. When set, generation must stop and the reply is text[0, cut).
//   * safe — how many leading bytes are safe to emit now: everything except a trailing
//            run that is a proper prefix of some stop string and might complete next token.
//
// Holding back that prefix is what stops a half-formed stop string ("\n\nUse") from leaking
// to the stream before it either completes ("\n\nUser:") or is ruled out ("\n\nUsage").
#pragma once

#include <string>
#include <vector>

namespace bllm {

class StopMatcher {
 public:
  StopMatcher() = default;
  explicit StopMatcher(std::vector<std::string> stops) {
    for (auto& s : stops)
      if (!s.empty()) stops_.push_back(std::move(s));
  }

  bool empty() const { return stops_.empty(); }

  struct Scan {
    size_t cut = std::string::npos;  // start of the earliest complete stop, or npos
    size_t safe = 0;                 // leading bytes safe to emit right now
  };

  // Scan the whole accumulated text. O(text * stops) per call; fine for chat-length
  // replies, and the caller feeds a monotonically growing string.
  Scan scan(const std::string& text) const {
    Scan r;
    r.safe = text.size();
    if (stops_.empty()) return r;

    for (const auto& s : stops_) {
      const size_t p = text.find(s);
      if (p != std::string::npos && p < r.cut) r.cut = p;
    }
    if (r.cut != std::string::npos) {   // a complete stop wins: emit up to it, then halt
      r.safe = r.cut;
      return r;
    }

    // No complete stop. Hold back the longest suffix of `text` that is a proper prefix of
    // some stop string — it may complete on the next token.
    size_t hold = 0;
    for (const auto& s : stops_) {
      const size_t kmax = std::min(s.size() - 1, text.size());
      for (size_t k = kmax; k > hold; --k)
        if (text.compare(text.size() - k, k, s, 0, k) == 0) { hold = k; break; }
    }
    r.safe = text.size() - hold;
    return r;
  }

 private:
  std::vector<std::string> stops_;
};

}  // namespace bllm
