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

  // logit_bias forces a token in and bans another, greedily.
  {
    bllm::NativeSamplingParams p;                    // greedy
    p.logit_bias[3] = 100.0f;                        // OpenAI's "always pick this"
    CHECK(pick(p, lg) == 3);
    p.logit_bias.clear();
    p.logit_bias[1] = -100.0f;                       // ban the argmax
    CHECK(pick(p, lg) == 3);                         // next-best of {0.1, 0.2, 0.3}
  }

  // The bias must survive candidate truncation. With a vocab far larger than the 512-logit
  // cap, boost a token buried in the tail: the top-k cut is taken on RAW logits and cannot
  // see it, so this fails unless biased ids are unioned into the candidate set.
  {
    std::vector<float> big(4000, 0.0f);
    for (int i = 0; i < 600; ++i) big[i] = 10.0f + (float)i / 1000.0f;   // the top-512 live here
    const int buried = 3999;
    big[buried] = -5.0f;
    bllm::NativeSamplingParams p;
    p.logit_bias[buried] = 100.0f;
    CHECK(pick(p, big) == buried);
    // And with temperature on, the sampled token is still always the boosted one.
    p.temp = 1.0f; p.seed = 5;
    for (int i = 0; i < 20; ++i) CHECK(pick(p, big) == buried);
  }

  // Banning the top few tokens hands it to the next one down, not to a duplicate or a
  // token outside the candidate set.
  {
    bllm::NativeSamplingParams p; p.temp = 1.0f; p.seed = 11; p.top_k = 2;
    p.logit_bias[1] = -100.0f;                       // ban 5.0 → {0.3, 0.2} survive top_k=2
    for (int i = 0; i < 30; ++i) { const int t = pick(p, lg); CHECK(t == 3 || t == 2); }
  }

  // A bias on an out-of-range id is ignored rather than sampled or crashing.
  {
    bllm::NativeSamplingParams p;
    p.logit_bias[99999] = 100.0f;
    p.logit_bias[-1] = 100.0f;
    CHECK(pick(p, lg) == 1);
  }

  // computeLogprobs: an exact full-vocab log-softmax.
  {
    // Uniform logits over 4 tokens => every logprob is log(1/4).
    std::vector<float> flat(4, 2.5f);
    auto lp = bllm::computeLogprobs([&](int i) { return flat[i]; }, 4, 2, 4);
    CHECK(std::fabs(lp.logprob - std::log(0.25)) < 1e-5);
    CHECK(lp.id == 2);
    CHECK(lp.top.size() == 4);
    double mass = 0.0;
    for (const auto& t : lp.top) mass += std::exp(t.second);
    CHECK(std::fabs(mass - 1.0) < 1e-5);              // the whole distribution sums to 1

    // Skewed: the alternatives come back most-likely first, and match the argmax.
    auto s = bllm::computeLogprobs([&](int i) { return lg[i]; }, (int)lg.size(), 0, 3);
    CHECK(s.top.size() == 3);
    CHECK(s.top[0].first == 1);                        // token 1 has logit 5.0
    CHECK(s.top[0].second > s.top[1].second && s.top[1].second > s.top[2].second);
    CHECK(s.logprob < s.top[0].second);                // token 0 is not the most likely
    // Every logprob is <= 0, i.e. a real probability.
    for (const auto& t : s.top) CHECK(t.second <= 0.0);

    // top_n = 0 reports only the chosen token; a huge top_n clamps to the vocab.
    CHECK(bllm::computeLogprobs([&](int i) { return lg[i]; }, 5, 1, 0).top.empty());
    CHECK(bllm::computeLogprobs([&](int i) { return lg[i]; }, 5, 1, 99).top.size() == 5);

    // Large logits must not overflow: exp(900) is inf, so only the max-shift keeps this
    // finite. p(0) rounds to exactly 1 here, hence <= rather than <.
    std::vector<float> big = {900.0f, 800.0f};
    auto b = bllm::computeLogprobs([&](int i) { return big[i]; }, 2, 0, 2);
    CHECK(std::isfinite(b.logprob) && b.logprob <= 0.0);
    CHECK(std::isfinite(b.top[1].second) && b.top[1].second < -99.0);   // the loser: ~ -100
  }

  if (failures == 0) std::printf("all native-sampler tests passed\n");
  return failures == 0 ? 0 : 1;
}
