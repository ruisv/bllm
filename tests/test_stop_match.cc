// Unit test for bllm::StopMatcher — pure string logic, no hobot deps, so it builds and
// runs anywhere (including the Mac dev host). Covers the two things that are easy to get
// wrong: a stop string that spans several streamed chunks, and holding its prefix back
// from the stream until it either completes or is ruled out.
//
//   c++ -std=c++17 -I include tests/test_stop_match.cc -o /tmp/t && /tmp/t
#include "bllm/stop_match.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
  } while (0)

// Drive a matcher over the full text one byte at a time (a stand-in for token-by-token
// growth), reproducing the session's emit/holdback/stop loop. Returns the streamed text
// and the final reply, so a test can assert BOTH the stream and the return value.
struct Run {
  std::string streamed, reply;
  bool stopped;
  std::vector<std::string> deltas;   // each chunk as the client would receive it
};

static Run drive(const bllm::StopMatcher& m, const std::string& text) {
  Run r{"", "", false, {}};
  size_t emitted = 0;
  std::string full;
  auto emit = [&](std::string chunk) {
    r.streamed += chunk;
    r.deltas.push_back(std::move(chunk));
  };
  for (size_t n = 1; n <= text.size(); ++n) {
    full.assign(text, 0, n);
    const auto sc = m.scan(full);
    // Mirrors the sessions exactly (native_llm.h / native_vlm.h), including the
    // character-boundary trim — a test that sliced differently would pass while
    // the shipped code splits characters.
    const size_t show =
        sc.cut != std::string::npos ? sc.cut : bllm::utf8SafeEnd(full, sc.safe);
    if (show > emitted) { emit(full.substr(emitted, show - emitted)); emitted = show; }
    if (sc.cut != std::string::npos) { r.reply = full.substr(0, sc.cut); r.stopped = true; return r; }
  }
  if (full.size() > emitted) emit(full.substr(emitted));   // flush held-back tail
  r.reply = full;
  return r;
}

// Is `s` well-formed UTF-8 on its own? This is the property a streaming client
// relies on when it decodes each delta independently; the customer-reported
// failure was exactly a delta that began with a continuation byte.
static bool validUtf8(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t need;
    if ((c & 0x80) == 0x00) need = 0;
    else if ((c & 0xE0) == 0xC0) need = 1;
    else if ((c & 0xF0) == 0xE0) need = 2;
    else if ((c & 0xF8) == 0xF0) need = 3;
    else return false;                       // continuation byte or malformed lead
    if (i + need >= s.size() + 0 && i + need > s.size() - 1) return false;
    for (size_t k = 1; k <= need; ++k)
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    i += need + 1;
  }
  return true;
}

int main() {
  // No stop strings: everything streams, nothing is cut.
  {
    bllm::StopMatcher m;
    CHECK(m.empty());
    auto r = drive(m, "hello world");
    CHECK(r.streamed == "hello world" && r.reply == "hello world" && !r.stopped);
  }

  // Stop string mid-text: reply and stream are trimmed at it, nothing past it leaks.
  {
    bllm::StopMatcher m({"\n\nUser:"});
    auto r = drive(m, "The answer is 4.\n\nUser: next question");
    CHECK(r.stopped);
    CHECK(r.reply == "The answer is 4.");
    CHECK(r.streamed == "The answer is 4.");   // the stop string itself never streamed
  }

  // A prefix of the stop string that never completes must still end up streamed in full.
  {
    bllm::StopMatcher m({"\n\nUser:"});
    auto r = drive(m, "cost is $5\n\nUsage climbs");   // "\n\nUs" is a live prefix, then ruled out
    CHECK(!r.stopped);
    CHECK(r.reply == "cost is $5\n\nUsage climbs");
    CHECK(r.streamed == "cost is $5\n\nUsage climbs");
  }

  // Earliest of several stops wins; the one that appears first in the text, not the list.
  {
    bllm::StopMatcher m({"END", "STOP"});
    auto r = drive(m, "aaa STOP bbb END ccc");
    CHECK(r.stopped && r.reply == "aaa " && r.streamed == "aaa ");
  }

  // Stop at position 0 yields an empty reply and streams nothing.
  {
    bllm::StopMatcher m({"##"});
    auto r = drive(m, "## heading");
    CHECK(r.stopped && r.reply.empty() && r.streamed.empty());
  }

  // Single-character stop.
  {
    bllm::StopMatcher m({"\n"});
    auto r = drive(m, "one line\nsecond");
    CHECK(r.stopped && r.reply == "one line" && r.streamed == "one line");
  }

  // A stop string equal to the whole reply → empty result, cleanly stopped.
  {
    bllm::StopMatcher m({"quit"});
    auto r = drive(m, "quit");
    CHECK(r.stopped && r.reply.empty());
  }

  // Empty stop strings are ignored (they must not match everywhere).
  {
    bllm::StopMatcher m({"", "X"});
    CHECK(!m.empty());
    auto r = drive(m, "abcXdef");
    CHECK(r.stopped && r.reply == "abc");
  }

  // --- UTF-8 boundaries -----------------------------------------------------
  // Reported from the field: a Qwen3.5-VL stream died with
  //   'utf-8' codec can't decode byte 0x8a in position 0: invalid start byte
  // Byte-level BPE splits multi-byte characters across tokens, so a delta sliced
  // at an arbitrary byte offset can end mid-character; the NEXT delta then starts
  // with a continuation byte and fails on its own.
  {
    // Every delta must be independently decodable, and the stream must still be
    // the whole text. Chinese is the common case (3 bytes per character).
    const std::string cn = "任务，随时告诉我，我会尽力协助！";
    for (const std::vector<std::string>& stops :
         std::vector<std::vector<std::string>>{{}, {"<|im_end|>"}, {"\n\nUser:"}}) {
      bllm::StopMatcher m(stops);
      auto r = drive(m, cn);
      CHECK(r.streamed == cn);
      for (const auto& d : r.deltas) CHECK(validUtf8(d));
    }
  }
  {
    // Mixed widths in one string: ASCII, 2-byte, 3-byte, and a 4-byte emoji.
    const std::string mixed = "ok é 中 \xF0\x9F\x98\x80 done";
    bllm::StopMatcher m;
    auto r = drive(m, mixed);
    CHECK(r.streamed == mixed);
    for (const auto& d : r.deltas) CHECK(validUtf8(d));
  }
  {
    // utf8SafeEnd itself: it may only ever move an offset back, never forward,
    // and a complete character must not be held back.
    const std::string cn = "中文";                       // 3 + 3 bytes
    CHECK(bllm::utf8SafeEnd(cn, 0) == 0);
    CHECK(bllm::utf8SafeEnd(cn, 1) == 0);                // mid-character → back to 0
    CHECK(bllm::utf8SafeEnd(cn, 2) == 0);
    CHECK(bllm::utf8SafeEnd(cn, 3) == 3);                // exactly one character
    CHECK(bllm::utf8SafeEnd(cn, 4) == 3);
    CHECK(bllm::utf8SafeEnd(cn, 6) == 6);                // both characters
    CHECK(bllm::utf8SafeEnd("abc", 2) == 2);             // ASCII untouched
    CHECK(bllm::utf8SafeEnd("", 0) == 0);
    // A stray continuation byte is not ours to repair — pass it through rather
    // than looping or truncating the caller's data.
    CHECK(bllm::utf8SafeEnd(std::string("\x8a", 1), 1) == 1);
  }

  // --- the field failure, reproduced -----------------------------------------
  // A customer's Qwen3.5-VL stream died with
  //   'utf-8' codec can't decode byte 0x8a in position 0: invalid start byte
  // 0x8a is the last byte of U+1F38A. Byte-level BPE spreads a 4-byte emoji over
  // several tokens, and decoding a partial prefix yields U+FFFD — three bytes where
  // the finished character is four. Emitting the placeholder leaves the stream offset
  // three bytes into a four-byte character, so the next delta begins on a continuation
  // byte. This drives the real decode sequence rather than a growing byte string,
  // because that shrink-and-regrow is the whole bug.
  {
    const std::vector<std::string> steps = {
        "\xEF\xBF\xBD",                  // tokens[0..0]  -> U+FFFD
        "\xEF\xBF\xBD",                  // tokens[0..1]  -> U+FFFD (still incomplete)
        "\xF0\x9F\x8E\x8A",             // tokens[0..2]  -> 🎊, four bytes
        "\xF0\x9F\x8E\x8A ok",          // more text after it
    };
    bllm::StopMatcher m;
    size_t emitted = 0;
    std::vector<std::string> deltas;
    for (const auto& full : steps) {
      const auto sc = m.scan(full);
      const size_t show =
          sc.cut != std::string::npos ? sc.cut : bllm::utf8SafeEnd(full, sc.safe);
      if (show > emitted) { deltas.push_back(full.substr(emitted, show - emitted)); emitted = show; }
    }
    const std::string& last = steps.back();
    if (last.size() > emitted) deltas.push_back(last.substr(emitted));   // flush
    std::string joined;
    for (const auto& d : deltas) { CHECK(validUtf8(d)); joined += d; }
    CHECK(joined == last);            // nothing lost, nothing duplicated
  }

  if (failures == 0) std::printf("all stop-match tests passed\n");
  return failures == 0 ? 0 : 1;
}
