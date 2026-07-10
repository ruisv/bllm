// Unit test for bllm::Sampler — the token sampler at parity with libxlm's knobs. Pure
// logic over a logit functor, no hobot deps, so it builds and runs on the Mac.
//
//   c++ -std=c++17 -I include tests/test_native_sampler.cc -o /tmp/t && /tmp/t
#include "bllm/native_sampler.h"

#include <cstdio>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
  } while (0)

static int pick(bllm::NativeSamplingParams p, const std::vector<float>& lg) {
  bllm::Sampler s(p);
  return s.pick([&](int i) { return lg[i]; }, (int)lg.size());
}

int main() {
  // A clear argmax; greedy (all knobs disabled) always returns it.
  std::vector<float> lg = {0.1f, 5.0f, 0.2f, 0.3f, -1.0f};
  {
    bllm::NativeSamplingParams p;                 // defaults => greedy
    CHECK(pick(p, lg) == 1);
    // temp<=0 stays greedy even with other knobs set.
    p.temp = 0.0f; p.top_p = 0.9f; p.min_p = 0.1f;
    CHECK(pick(p, lg) == 1);
  }

  // Determinism: same seed, same temperature → identical draws.
  {
    bllm::NativeSamplingParams a; a.temp = 1.0f; a.seed = 42;
    bllm::NativeSamplingParams b = a;
    bllm::Sampler sa(a), sb(b);
    for (int i = 0; i < 50; ++i)
      CHECK(sa.pick([&](int j) { return lg[j]; }, 5) ==
            sb.pick([&](int j) { return lg[j]; }, 5));
  }

  // min_p at 1.0 keeps only tokens tied with the max prob → argmax only.
  {
    bllm::NativeSamplingParams p; p.temp = 1.0f; p.min_p = 1.0f; p.seed = 7;
    for (int i = 0; i < 30; ++i) CHECK(pick(p, lg) == 1);
  }

  // top_k = 1 collapses to argmax regardless of temperature.
  {
    bllm::NativeSamplingParams p; p.temp = 2.0f; p.top_k = 1; p.seed = 9;
    for (int i = 0; i < 30; ++i) CHECK(pick(p, lg) == 1);
  }

  // Frequency/presence penalties push a repeated token below a rival and flip the argmax.
  // Two near-tied leaders; penalise the leader for having occurred, the other should win.
  {
    std::vector<float> two = {3.00f, 2.90f, -2.0f};  // token 0 slightly ahead of token 1
    bllm::NativeSamplingParams p;                    // greedy otherwise
    p.penalty_last_n = 8;
    p.penalty_present = 1.0f;                         // subtract 1.0 from any token seen
    bllm::Sampler s(p);
    // No history yet: token 0 wins.
    CHECK(s.pick([&](int i) { return two[i]; }, 3) == 0);
    s.record(0);                                     // token 0 now carries a presence penalty
    // 3.00 - 1.0 = 2.00 < 2.90 → token 1 wins.
    CHECK(s.pick([&](int i) { return two[i]; }, 3) == 1);
  }

  // Repetition penalty (>1) also demotes a seen token.
  {
    std::vector<float> two = {4.0f, 3.0f, -2.0f};
    bllm::NativeSamplingParams p; p.penalty_last_n = 8; p.rep_pen = 2.0f;
    bllm::Sampler s(p);
    CHECK(s.pick([&](int i) { return two[i]; }, 3) == 0);
    s.record(0);                                     // 4.0 / 2.0 = 2.0 < 3.0 → token 1
    CHECK(s.pick([&](int i) { return two[i]; }, 3) == 1);
  }

  // The penalty window forgets: after penalty_last_n newer tokens, the old one recovers.
  {
    std::vector<float> two = {4.0f, 3.0f, -2.0f};
    bllm::NativeSamplingParams p; p.penalty_last_n = 2; p.rep_pen = 2.0f;
    bllm::Sampler s(p);
    s.record(0);                                     // window: [0]
    CHECK(s.pick([&](int i) { return two[i]; }, 3) == 1);   // 0 penalised
    s.record(2); s.record(2);                        // window slides past the 0
    CHECK(s.pick([&](int i) { return two[i]; }, 3) == 0);   // 0 no longer in window
  }

  if (failures == 0) std::printf("all native-sampler tests passed\n");
  return failures == 0 ? 0 : 1;
}
