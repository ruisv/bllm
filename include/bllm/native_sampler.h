// bllm::Sampler — token sampling shared by the native engines. Works on ANY logit
// source via a logit(i)->float functor (so it handles both int16-with-scale and
// float32 logit tensors). Supports greedy / temperature / top-k / top-p /
// repetition-penalty, seeded. Only the top-k candidates are dequantized, so the full
// vocab is never materialised.
#pragma once

#include "bllm/native_engine.h"   // NativeSamplingParams

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bllm {

class Sampler {
 public:
  explicit Sampler(const NativeSamplingParams& p) : p_(p), rng_(p.seed) {}

  void record(int id) { if (p_.rep_pen != 1.0f) seen_.insert(id); }

  bool greedy() const {
    return p_.temp <= 0.0f && p_.rep_pen == 1.0f && p_.top_k == 0 && p_.top_p >= 1.0f;
  }

  // logit(i) returns the (dequantized) logit for token i; n = vocab size.
  template <class LogitFn>
  int pick(LogitFn logit, int n) {
    if (greedy()) {
      int best = 0; float bv = logit(0);
      for (int i = 1; i < n; ++i) { float v = logit(i); if (v > bv) { bv = v; best = i; } }
      return best;
    }
    const int k = p_.top_k > 0 ? std::min(p_.top_k, n) : std::min(n, 512);
    if ((int)idx_.size() != n) { idx_.resize(n); for (int i = 0; i < n; ++i) idx_[i] = i; }
    std::nth_element(idx_.begin(), idx_.begin() + k, idx_.end(),
                     [&](int a, int b) { return logit(a) > logit(b); });
    std::vector<std::pair<float, int>> cand(k);
    for (int i = 0; i < k; ++i) {
      int id = idx_[i]; float v = logit(id);
      if (p_.rep_pen != 1.0f && seen_.count(id)) v = v > 0 ? v / p_.rep_pen : v * p_.rep_pen;
      cand[i] = {v, id};
    }
    std::sort(cand.begin(), cand.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
    const float T = p_.temp > 0 ? p_.temp : 1.0f;
    const float mx = cand[0].first;
    std::vector<double> pr(k); double sum = 0;
    for (int i = 0; i < k; ++i) { pr[i] = std::exp((cand[i].first - mx) / T); sum += pr[i]; }
    double cum = 0; int keep = k;
    for (int i = 0; i < k; ++i) { cum += pr[i] / sum; if (cum >= p_.top_p) { keep = i + 1; break; } }
    double ks = 0; for (int i = 0; i < keep; ++i) ks += pr[i];
    std::uniform_real_distribution<double> U(0.0, ks);
    double r = U(rng_), acc = 0;
    for (int i = 0; i < keep; ++i) { acc += pr[i]; if (r <= acc) return cand[i].second; }
    return cand[keep - 1].second;
  }

 private:
  NativeSamplingParams p_;
  std::mt19937 rng_;
  std::unordered_set<int> seen_;
  std::vector<int> idx_;
};

}  // namespace bllm
