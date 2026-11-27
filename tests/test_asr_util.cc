// Unit test for bllm::asr_detail — pure logic, no hobot deps, so it builds and runs
// anywhere (including the Mac dev host).
//
//   c++ -std=c++17 -I include tests/test_asr_util.cc -o /tmp/t && /tmp/t
#include "bllm/asr_util.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
  } while (0)

using namespace bllm::asr_detail;

static void token_count() {
  // Measured on the official package and its tensor dumps: an 800-frame window is 104
  // tokens, an 11.41 s clip (1141 frames) 104 + 45, a 3.86 s one 50.
  CHECK(tokensFor(800) == 104);
  CHECK(tokensFor(341) == 45);
  CHECK(tokensFor(386) == 50);
  CHECK(tokensFor(100) == 13);
  CHECK(tokensFor(1) == 1);
  CHECK(tokensFor(0) == 0);
}

static void split() {
  // A clip that fits comes back whole.
  std::vector<float> a(1000, 0.5f);
  auto p = splitAtQuiet(a.data(), a.size(), 2000, 100, 10);
  CHECK(p.size() == 1 && p[0].first == 0 && p[0].second == 1000);

  // Loud everywhere except a silent stretch near the cut: the boundary lands in it.
  std::vector<float> b(3000, 0.5f);
  for (size_t i = 900; i < 940; ++i) b[i] = 0.0f;
  p = splitAtQuiet(b.data(), b.size(), 1000, 200, 10);
  CHECK(p.size() >= 2);
  CHECK(p[0].first == 0 && p[0].second >= 900 && p[0].second < 940);
  // Contiguous, no overlap, covers everything.
  size_t at = 0;
  for (const auto& q : p) { CHECK(q.first == at); CHECK(q.second > q.first); at = q.second; }
  CHECK(at == b.size());
  for (const auto& q : p) CHECK(q.second - q.first <= 1000 + 200);
}

static void repetitions() {
  CHECK(fixRepetitions("hello") == "hello");
  CHECK(fixRepetitions(std::string(25, 'a')) == "a");                 // run > 20 -> one
  // The pattern pass collapses >= 20 repeats (the character pass needs > 20), as upstream.
  CHECK(fixRepetitions(std::string(20, 'a')) == "a");
  CHECK(fixRepetitions(std::string(19, 'a')) == std::string(19, 'a'));
  std::string loop;
  for (int i = 0; i < 25; ++i) loop += "好的";
  CHECK(fixRepetitions("嗯" + loop + "。") == "嗯好的。");               // code points, not bytes
  std::string few;
  for (int i = 0; i < 5; ++i) few += "对";
  CHECK(fixRepetitions(few) == few);
}

static void language() {
  CHECK(normalizeLanguage("chinese") == "Chinese");
  CHECK(normalizeLanguage(" ENGLISH ") == "English");
  CHECK(normalizeLanguage("zh") == "Chinese");
  CHECK(normalizeLanguage("en") == "English");
  CHECK(normalizeLanguage("") == "");
}

static void joining() {
  CHECK(needsSpace("to go.", "She turned"));
  CHECK(!needsSpace("变化的。", "动植物"));
  CHECK(!needsSpace("hello ", "world"));
  CHECK(!needsSpace("", "x"));
  CHECK(trim("  a b \n") == "a b");
  CHECK(toUtf8(toU32("中文 abc")) == "中文 abc");
}

int main() {
  token_count();
  split();
  repetitions();
  language();
  joining();
  if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
  std::printf("test_asr_util: all passed\n");
  return 0;
}
