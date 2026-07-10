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
struct Run { std::string streamed, reply; bool stopped; };

static Run drive(const bllm::StopMatcher& m, const std::string& text) {
  Run r{"", "", false};
  size_t emitted = 0;
  std::string full;
  for (size_t n = 1; n <= text.size(); ++n) {
    full.assign(text, 0, n);
    const auto sc = m.scan(full);
    const size_t show = sc.cut != std::string::npos ? sc.cut : sc.safe;
    if (show > emitted) { r.streamed += full.substr(emitted, show - emitted); emitted = show; }
    if (sc.cut != std::string::npos) { r.reply = full.substr(0, sc.cut); r.stopped = true; return r; }
  }
  if (full.size() > emitted) r.streamed += full.substr(emitted);   // flush held-back tail
  r.reply = full;
  return r;
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

  if (failures == 0) std::printf("all stop-match tests passed\n");
  return failures == 0 ? 0 : 1;
}
