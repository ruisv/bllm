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
// staying O(vocab) once instead of O(vocab·exp) per token. `logit_bias` is the one knob
// that can push a token UP, so biased ids are unioned into the candidate set explicitly.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bllm {

// `seed` sentinel: draw a fresh nondeterministic seed for each generation. Same value and
// meaning as llama.cpp's LLAMA_DEFAULT_SEED, and it is the DEFAULT — a fixed default seed
// silently makes temperature useless, because the sampler is rebuilt per generation and
// would replay the identical random stream every turn. (Measured before this was the
// default: three identical temp=0.8 requests returned byte-identical replies.) Pass any
// other value to get the reproducible behaviour back.
inline constexpr uint64_t kSeedRandom = 0xFFFFFFFFull;

// A restriction on WHICH tokens may be generated next — grammar-constrained decoding, in
// practice (bllm::GrammarConstraint in native_grammar.h). Kept as an interface here so the
// sampler stays free of the grammar machinery and of any tokenizer.
//
// It is stateful: accept() commits a token and moves the constraint forward, so the object
// belongs to the session (which resets it per turn), not to the params struct that is
// copied per generation.
struct TokenConstraint {
  virtual ~TokenConstraint() = default;
  virtual bool allows(int id) const = 0;   // may this token be generated at this position?
  virtual void accept(int id) = 0;         // commit a generated token
  virtual bool can_end() const = 0;        // is the constraint satisfied — may we stop?
};

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

  // token id -> additive logit bias, applied before every filter. OpenAI's `logit_bias`
  // (same [-100, 100] convention: ±100 effectively forces or bans a token, 0 = no effect).
  // Unlike the penalties this can raise a token, so it is also what forces a biased id
  // into the candidate set — see pick().
  std::unordered_map<int, float> logit_bias;

  // Grammar / structured-output constraint, NOT owned: the session holds it and resets it
  // per turn. null = unconstrained. See TokenConstraint and pick().
  TokenConstraint* constraint = nullptr;

  // OpenAI's logprobs: report each generated token's log probability plus this many
  // alternatives (its `top_logprobs`; 0 = the chosen token only). <0 = off, the default,
  // because the report costs two extra full-vocab passes per token. See computeLogprobs().
  int logprobs = -1;
};

// One generated token's probability report — OpenAI's `logprobs` / `top_logprobs`.
struct TokenLogprobs {
  int id = 0;                                // the token that was actually generated
  float logprob = 0.0f;                      // log p(id)
  std::vector<std::pair<int, float>> top;    // (id, logprob) alternatives, most likely first
};

// Exact log-softmax of the RAW logits at one position: logit[id] - logsumexp(all logits),
// accumulated in double over the whole vocab — no candidate truncation, so these are real
// probabilities and not the sampler's renormalised view of its top 512.
//
// "Raw" is deliberate: the report describes what the MODEL believes, so it is unaffected by
// temperature, the penalties, logit_bias and the top-k/p filters. That keeps it comparable
// across requests with different sampling settings, which is the point of asking for
// logprobs at all (confidence, scoring, classification). The consequence to document: the
// generated token is not necessarily the highest-logprob entry here — sampling drew it from
// the adjusted distribution, this reports the unadjusted one.
//
// Cost is two full-vocab passes (plus a partial sort for the alternatives) per token, on
// top of a decode step that already costs tens of ms. Measured on S100P / Qwen3.5-0.8B
// (151k vocab): 14.3 tok/s off, 13.9 with logprobs, 13.6 with 5 alternatives — 3-5%. Hence
// opt-in rather than always-on.
template <class LogitFn>
TokenLogprobs computeLogprobs(LogitFn logit, int n, int chosen, int top_n) {
  double mx = logit(0);
  for (int i = 1; i < n; ++i) { const double v = logit(i); if (v > mx) mx = v; }
  double se = 0.0;
  for (int i = 0; i < n; ++i) se += std::exp((double)logit(i) - mx);
  const double logZ = mx + std::log(se);        // max-shifted, so no overflow

  TokenLogprobs out;
  out.id = chosen;
  out.logprob = (float)((double)logit(chosen) - logZ);
  const int k = std::min(std::max(top_n, 0), n);
  if (k > 0) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logit(a) > logit(b); });
    out.top.reserve(k);
    for (int i = 0; i < k; ++i)
      out.top.emplace_back(idx[i], (float)((double)logit(idx[i]) - logZ));
  }
  return out;
}

class Sampler {
 public:
  explicit Sampler(const NativeSamplingParams& p) : p_(p), rng_(seedFor(p.seed)) {}

  // Feed a just-emitted token: the penalty window tracks it, and any constraint advances
  // over it. Call this for every token that is really generated.
  void record(int id) {
    if (p_.constraint) p_.constraint->accept(id);
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

  // temp<=0 is deterministic (argmax). Only when nothing rewrites the logits is it a pure
  // raw argmax we can shortcut; with penalties or a bias the argmax is over the adjusted
  // logits, which needs the candidate path.
  bool greedy_raw() const {
    return p_.temp <= 0.0f && !has_penalty() && p_.logit_bias.empty() && !p_.constraint;
  }

  // logit(i) returns the (dequantized) logit for token i; n = vocab size.
  // Returns -1 when a constraint allows nothing more — generation must stop there.
  template <class LogitFn>
  int pick(LogitFn logit, int n) {
    if (greedy_raw()) {
      int best = 0; float bv = logit(0);
      for (int i = 1; i < n; ++i) { const float v = logit(i); if (v > bv) { bv = v; best = i; } }
      return best;
    }

    // 1. Candidate set: the top `k` logits (penalties only lower tokens, so a token below
    //    this cut cannot win). Materialise (adjusted logit, id).
    const int cap = std::max({p_.top_k > 0 ? p_.top_k : 0, 512, p_.min_keep});
    const int k = std::min(cap, n);

    // Top-k by raw logit, via a bounded MIN-heap over a single sequential scan.
    //
    // This was std::nth_element over an index array with `logit(a) > logit(b)` as
    // the comparator, which makes every one of the ~n comparisons a random access
    // into the logits. On Qwen3.5's 248320-entry vocabulary that measured
    // 4.25 ms/token against 0.38 ms for the single sequential scan greedy does —
    // i.e. sampling cost 9% of a decode step purely in candidate selection.
    //
    // Scanning in order and keeping a k-sized min-heap touches each logit once and
    // in order; the heap only pays log(k) on an actual improvement, which happens
    // about k*ln(n/k) times, not n times.
    auto worse = [](const Scored& a, const Scored& b) { return a.v > b.v; };  // min-heap
    top_.clear();
    top_.reserve(k);
    for (int i = 0; i < n; ++i) {
      const float v = logit(i);
      if ((int)top_.size() < k) {
        top_.push_back({v, i});
        if ((int)top_.size() == k) std::make_heap(top_.begin(), top_.end(), worse);
      } else if (v > top_.front().v) {
        std::pop_heap(top_.begin(), top_.end(), worse);
        top_.back() = {v, i};
        std::push_heap(top_.begin(), top_.end(), worse);
      }
    }

    idx_.resize(top_.size());
    for (size_t i = 0; i < top_.size(); ++i) idx_[i] = top_[i].id;

    std::vector<Cand> c;
    c.reserve(k + p_.logit_bias.size());
    for (const Scored& t : top_) c.push_back({adjust(t.id, t.v), 0.0, t.id});
    // A bias can lift a token from anywhere in the tail above the cut, and the cut was
    // taken on RAW logits, so it cannot see that. Union the biased ids in (deduped against
    // the top-k, or their probability mass would be counted twice) — that restores the
    // invariant the truncation rests on: the candidate set contains everything that could
    // win. |logit_bias| is client-sized (tens of ids), so this is cheap and only runs when
    // a bias was actually given.
    if (!p_.logit_bias.empty()) {
      picked_.clear();
      picked_.insert(idx_.begin(), idx_.end());   // idx_ IS the top-k now
      for (const auto& kv : p_.logit_bias) {
        const int id = kv.first;
        if (id < 0 || id >= n || picked_.count(id)) continue;
        c.push_back({adjust(id, logit(id)), 0.0, id});
      }
    }
    // A constraint (a grammar) restricts WHICH tokens may come next. Filtering after the
    // cut is exact for what survives: every token outside the cut has a lower logit, so the
    // allowed tokens inside it are the highest-logit allowed tokens that exist. Only when
    // the cut holds none of them — the grammar wants "{" and the model wanted prose — does
    // the whole vocab have to be searched, which is why that path is the exception.
    if (p_.constraint) {
      // Once the constraint is satisfied, eos becomes legal again — otherwise a grammar
      // that CAN end but can also continue (`[0-9]+`) could never stop, because the
      // constraint itself never allows the eos token as text.
      const bool may_end = p_.constraint->can_end();
      c.erase(std::remove_if(c.begin(), c.end(),
                             [&](const Cand& x) { return !allowedBy(x.id, may_end); }),
              c.end());
      if (c.empty()) c = allowedFromWholeVocab(logit, n, may_end);
      if (c.empty()) return -1;      // nothing is allowed any more: stop generating
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
    int keep = (int)c.size();
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

  bool isEos(int id) const {
    for (int e : p_.eos) if (e == id) return true;
    return false;
  }

  bool allowedBy(int id, bool may_end) const {
    return (may_end && isEos(id)) || p_.constraint->allows(id);
  }

  // Every token the constraint allows, capped to the same candidate budget. One grammar
  // test per vocab entry, so this is the slow path — reached only when the top-k cut
  // contained nothing allowed.
  template <class LogitFn>
  std::vector<Cand> allowedFromWholeVocab(LogitFn logit, int n, bool may_end) {
    std::vector<Cand> c;
    for (int i = 0; i < n; ++i)
      if (allowedBy(i, may_end)) c.push_back({adjust(i, logit(i)), 0.0, i});
    const int cap = std::max({p_.top_k > 0 ? p_.top_k : 0, 512, p_.min_keep});
    if ((int)c.size() > cap) {
      std::nth_element(c.begin(), c.begin() + cap, c.end(),
                       [](const Cand& a, const Cand& b) { return a.lg > b.lg; });
      c.resize(cap);
    }
    return c;
  }

  bool has_penalty() const {
    return p_.rep_pen != 1.0f || p_.penalty_freq != 0.0f || p_.penalty_present != 0.0f;
  }

  // Everything that rewrites one token's logit: the llama.cpp penalties (from its count in
  // the window) then the client's bias. Bias last so it is not scaled by rep_pen — it is a
  // request-level override, not part of the repetition model.
  double adjust(int id, double lg) const {
    if (has_penalty()) {
      auto it = counts_.find(id);
      if (it != counts_.end()) {
        const int cnt = it->second;
        if (p_.rep_pen != 1.0f) lg = lg <= 0.0 ? lg * p_.rep_pen : lg / p_.rep_pen;
        lg -= cnt * (double)p_.penalty_freq + (double)p_.penalty_present;
      }
    }
    if (!p_.logit_bias.empty()) {
      auto it = p_.logit_bias.find(id);
      if (it != p_.logit_bias.end()) lg += (double)it->second;
    }
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
  struct Scored { float v; int id; };
  std::vector<Scored> top_;                // reused top-k buffer for candidate selection
  std::vector<int> idx_;                   // the top-k ids, for the logit_bias dedup below
  std::unordered_set<int> picked_;         // top-k ids, only built when a bias is set
};

}  // namespace bllm
