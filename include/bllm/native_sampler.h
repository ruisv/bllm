// bllm::Sampler — token sampling shared by the native engines, at parity with libxlm's
// (llama.cpp-style) sampler. Works on ANY logit source via a logit(i)->float functor, so
// it handles both int16-with-scale and float32 logit tensors, and only the top-k
// candidates are dequantized — the full vocab is never materialised.
//
// Knobs (all of libxlm's common_params_sampling_t): temperature, top-k, top-p, min-p,
// typical-p (min_keep floors every filter), and repetition / frequency / presence
// penalties over the last `penalty_last_n` tokens. Seeded for reproducibility.
//
// The candidate set is bounded to the top `max(top_k, 512, min_keep)` logits per step
// (nth_element, no full softmax): the excluded tail carries negligible probability, and
// penalties only push tokens down, so this matches the full sampler in practice while
// staying O(vocab) once instead of O(vocab·exp) per token.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <unordered_map>
#include <vector>

namespace bllm {

// `seed` sentinel: draw a fresh nondeterministic seed for each generation. Same value and
// meaning as llama.cpp's LLAMA_DEFAULT_SEED, and it is the DEFAULT — a fixed default seed
// silently makes temperature useless, because the sampler is rebuilt per generation and
// would replay the identical random stream every turn. (Measured before this was the
// default: three identical temp=0.8 requests returned byte-identical replies.) Pass any
// other value to get the reproducible behaviour back.
inline constexpr uint64_t kSeedRandom = 0xFFFFFFFFull;

// Sampling knobs. Defaults are the "disabled" values, so a zero-initialised params object
// samples greedily. Field names/semantics mirror libxlm's common_params_sampling_t.
struct NativeSamplingParams {
  float temp = 0.0f;             // <= 0 => greedy (argmax); else divides logits
  float top_p = 1.0f;            // 1.0 = disabled (nucleus)
  float min_p = 0.0f;            // 0.0 = disabled (keep tokens with p >= min_p * p_max)
  float typ_p = 1.0f;            // 1.0 = disabled (locally typical sampling)
  int top_k = 0;                 // <= 0 = disabled (bounded to the candidate cap)
  int min_keep = 1;              // floor on how many tokens survive each filter
  int max_new = 200;

  int penalty_last_n = 64;       // window the penalties look back over (<=0 = disabled)
  float rep_pen = 1.0f;          // 1.0 = disabled (llama.cpp penalty_repeat)
  float penalty_freq = 0.0f;     // 0.0 = disabled (subtract count * freq)
  float penalty_present = 0.0f;  // 0.0 = disabled (subtract present once)

  uint64_t seed = kSeedRandom;   // kSeedRandom => nondeterministic; any other value is exact
  std::vector<int> eos = {151645, 151643};
};

class Sampler {
 public:
  explicit Sampler(const NativeSamplingParams& p) : p_(p), rng_(seedFor(p.seed)) {}

  // Feed a just-emitted token so the penalty window tracks it.
  void record(int id) {
    if (p_.penalty_last_n == 0 || !has_penalty()) return;
    recent_.push_back(id);
    ++counts_[id];
    // penalty_last_n < 0 means "whole history"; otherwise bound the window.
    if (p_.penalty_last_n > 0) {
      while ((int)recent_.size() > p_.penalty_last_n) {
        const int old = recent_.front();
        recent_.pop_front();
        if (--counts_[old] <= 0) counts_.erase(old);
      }
    }
  }

  // temp<=0 is deterministic (argmax). Only when there is no penalty is it a pure raw
  // argmax we can shortcut; with penalties the argmax is over the penalised logits.
  bool greedy_raw() const { return p_.temp <= 0.0f && !has_penalty(); }

  // logit(i) returns the (dequantized) logit for token i; n = vocab size.
  template <class LogitFn>
  int pick(LogitFn logit, int n) {
    if (greedy_raw()) {
      int best = 0; float bv = logit(0);
      for (int i = 1; i < n; ++i) { const float v = logit(i); if (v > bv) { bv = v; best = i; } }
      return best;
    }

    // 1. Candidate set: the top `k` logits (penalties only lower tokens, so a token below
    //    this cut cannot win). Materialise (penalised logit, id).
    const int cap = std::max({p_.top_k > 0 ? p_.top_k : 0, 512, p_.min_keep});
    const int k = std::min(cap, n);
    if ((int)idx_.size() != n) { idx_.resize(n); for (int i = 0; i < n; ++i) idx_[i] = i; }
    std::nth_element(idx_.begin(), idx_.begin() + k, idx_.end(),
                     [&](int a, int b) { return logit(a) > logit(b); });

    std::vector<Cand> c(k);
    for (int i = 0; i < k; ++i) {
      const int id = idx_[i];
      c[i] = {penalise(id, logit(id)), 0.0, id};
    }
    std::sort(c.begin(), c.end(), [](const Cand& a, const Cand& b) { return a.lg > b.lg; });

    // temp<=0: deterministic argmax over the penalised logits (llama.cpp's greedy path —
    // penalties reorder, filters cannot change the max, so no softmax is needed).
    if (p_.temp <= 0.0f) return c[0].id;

    // 2. Temperature + softmax over the candidates → probabilities (c is lg-sorted).
    const float T = p_.temp;
    const double mx = c[0].lg;
    double sum = 0.0;
    for (auto& x : c) { x.p = std::exp((x.lg - mx) / T); sum += x.p; }
    for (auto& x : c) x.p /= sum;

    // 3. Filters, each keeping at least min_keep (already prob-sorted descending).
    int keep = k;
    keep = apply_top_k(keep);
    keep = apply_top_p(c, keep);
    keep = apply_min_p(c, keep);
    keep = apply_typical(c, keep);

    // 4. Sample from the survivors, renormalised.
    double ks = 0.0;
    for (int i = 0; i < keep; ++i) ks += c[i].p;
    std::uniform_real_distribution<double> U(0.0, ks);
    double r = U(rng_), acc = 0.0;
    for (int i = 0; i < keep; ++i) { acc += c[i].p; if (r <= acc) return c[i].id; }
    return c[keep - 1].id;
  }

 private:
  struct Cand { double lg; double p; int id; };

  bool has_penalty() const {
    return p_.rep_pen != 1.0f || p_.penalty_freq != 0.0f || p_.penalty_present != 0.0f;
  }

  // llama.cpp penalty on one token's logit, using its count in the window.
  double penalise(int id, double lg) const {
    if (!has_penalty()) return lg;
    auto it = counts_.find(id);
    if (it == counts_.end()) return lg;
    const int cnt = it->second;
    if (p_.rep_pen != 1.0f) lg = lg <= 0.0 ? lg * p_.rep_pen : lg / p_.rep_pen;
    lg -= cnt * (double)p_.penalty_freq + (double)p_.penalty_present;
    return lg;
  }

  int apply_top_k(int keep) const {
    if (p_.top_k <= 0 || p_.top_k >= keep) return keep;
    return std::max(p_.top_k, std::min(p_.min_keep, keep));
  }

  int apply_top_p(const std::vector<Cand>& c, int keep) const {
    if (p_.top_p >= 1.0f) return keep;
    double cum = 0.0;
    for (int i = 0; i < keep; ++i) {
      cum += c[i].p;
      if (cum >= p_.top_p) return std::max(i + 1, std::min(p_.min_keep, keep));
    }
    return keep;
  }

  int apply_min_p(const std::vector<Cand>& c, int keep) const {
    if (p_.min_p <= 0.0f || keep == 0) return keep;
    const double thresh = p_.min_p * c[0].p;   // c is prob-sorted, so c[0].p is the max
    int i = 0;
    while (i < keep && c[i].p >= thresh) ++i;
    return std::max(i, std::min(p_.min_keep, keep));
  }

  // Locally typical sampling: keep the tokens whose surprise -log(p) is closest to the
  // distribution's entropy, until their mass reaches typ_p.
  int apply_typical(std::vector<Cand>& c, int keep) const {
    if (p_.typ_p >= 1.0f || keep <= 1) return keep;
    double H = 0.0, norm = 0.0;
    for (int i = 0; i < keep; ++i) norm += c[i].p;
    for (int i = 0; i < keep; ++i) {
      const double p = c[i].p / norm;
      if (p > 0.0) H -= p * std::log(p);
    }
    std::vector<int> order(keep);
    for (int i = 0; i < keep; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return std::fabs(-std::log(c[a].p / norm) - H) < std::fabs(-std::log(c[b].p / norm) - H);
    });
    double cum = 0.0; int m = keep;
    for (int j = 0; j < keep; ++j) {
      cum += c[order[j]].p / norm;
      if (cum >= p_.typ_p) { m = j + 1; break; }
    }
    m = std::max(m, std::min(p_.min_keep, keep));
    std::vector<Cand> out(m);
    for (int j = 0; j < m; ++j) out[j] = c[order[j]];
    std::sort(out.begin(), out.end(), [](const Cand& a, const Cand& b) { return a.p > b.p; });
    for (int j = 0; j < m; ++j) c[j] = out[j];
    return m;
  }

  // mt19937 consumes 32 bits, so an explicit seed is narrowed; kSeedRandom instead pulls a
  // fresh value from the platform entropy source so each generation samples independently.
  static std::mt19937::result_type seedFor(uint64_t seed) {
    if (seed != kSeedRandom) return (std::mt19937::result_type)seed;
    std::random_device rd;
    return rd();
  }

  NativeSamplingParams p_;
  std::mt19937 rng_;
  std::deque<int> recent_;                 // penalty window, newest at the back
  std::unordered_map<int, int> counts_;    // token id -> occurrences in the window
  std::vector<int> idx_;
};

}  // namespace bllm
