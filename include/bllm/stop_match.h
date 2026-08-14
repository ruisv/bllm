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
//
// `utf8SafeEnd` guards the OTHER way a byte offset can be wrong: splitting a character.
#pragma once

#include <string>
#include <vector>

namespace bllm {

// Largest offset <= `end` that does not cut a UTF-8 sequence in half.
//
// A streaming delta is a BYTE slice of the accumulated text, but the offsets that produce
// it (StopMatcher's `safe`, or simply "everything decoded so far") know nothing about
// character boundaries. Byte-level BPE routinely emits a token carrying the first byte or
// two of a multi-byte character — every CJK character is three bytes and is frequently
// split — so a slice ending there hands the client half a character, and the NEXT slice
// begins with a continuation byte. A consumer that decodes each delta on its own then
// fails on the second one, at position 0, with "invalid start byte". Holding the partial
// tail back until the rest of its bytes arrive costs one token of latency and makes every
// delta independently decodable.
inline size_t utf8SafeEnd(const std::string& s, size_t end) {
  if (end > s.size()) end = s.size();
  size_t i = end;
  while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) --i;  // continuations
  if (i == 0) return end;                       // no lead byte in range: leave it alone
  const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
  size_t need = 1;
  if ((lead & 0x80) == 0x00) need = 1;
  else if ((lead & 0xE0) == 0xC0) need = 2;
  else if ((lead & 0xF0) == 0xE0) need = 3;
  else if ((lead & 0xF8) == 0xF0) need = 4;
  else return end;                              // malformed lead: not ours to repair
  // The character starts at i-1 and needs `need` bytes. Emit it only once they are all here.
  size_t out = (i - 1) + need <= end ? end : i - 1;

  // Also hold back a trailing U+FFFD. A byte-level BPE decoder does not hand back
  // raw fragments — decoding a prefix that ends mid-character yields REPLACEMENT
  // CHARACTER instead, which is perfectly well-formed UTF-8 and would sail past the
  // check above. It is three bytes where the real character is often four:
  //
  //   tokens [9008]           -> "\uFFFD"   (3 bytes)   <- emitted, offset now 3
  //   tokens [9008, 236]      -> "\uFFFD"   (3 bytes)
  //   tokens [9008, 236, 231] -> "🎉"        (4 bytes)   <- slice from 3 is its LAST byte
  //
  // so emitting the placeholder desynchronises every later offset and the next delta
  // starts on a continuation byte. Holding it costs one token and is self-correcting:
  // whatever it becomes is emitted whole, and a genuine U+FFFD still reaches the client
  // on the end-of-stream flush.
  while (out >= 3 && static_cast<unsigned char>(s[out - 3]) == 0xEF &&
         static_cast<unsigned char>(s[out - 2]) == 0xBF &&
         static_cast<unsigned char>(s[out - 1]) == 0xBD)
    out -= 3;
  return out;
}

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
