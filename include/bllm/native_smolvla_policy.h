// bllm::SmolVlaPolicy — SmolVLA as a policy, not as three graphs.
//
// `SmolVlaStack` takes patch matrices, an embedded prompt, a state token, two
// RoPE tables and two attention biases. Everything between that and "two camera
// frames, a state vector and an English sentence" lived in host Python that
// imported lerobot, transformers and the checkpoint. This class carries that
// seam onto the board so a user needs only the model package:
//
//   tokenize -> embed (mmap fp16 table) -> patchify -> normalise state ->
//   state_proj -> RoPE for the real token count -> SmolVlaStack -> unnormalise
//
// Five things here are load-bearing, and each was priced by injecting it into a
// working reference and measuring, rather than assumed. The costs quoted below
// are those measurements.
//
//   * the instruction gets a trailing "\n" before tokenization, which is what
//     `SmolVLANewLineProcessor` does in the shipped pipeline;
//   * the embedding table carries a `sqrt(hidden)` factor that `embed_prefix`
//     applies OUTSIDE `embed_language_tokens`. The package stores the table
//     pre-scaled so no consumer can forget, and this class refuses one that does
//     not claim it. Dropping it costs cosine 0.935 with gain 1.088 — invisible
//     to cosine alone, which is how π0.5 shipped it for three months;
//   * RoPE theta is **10000**, `apply_rope`'s hardcoded default, NOT the
//     `rope_theta: 100000` in the checkpoint's config.json. That one is worth
//     cosine 0.686;
//   * the action tokens rope at ABSOLUTE positions for the self-attention layers
//     (`real prefix tokens + i`) and at 0-based positions for the cross-attention
//     ones. The graph bakes the second and takes the first as an input; getting
//     them the wrong way round is cosine 0.315;
//   * the resize pads LEFT and TOP, not centred. SmolVLA and openpi differ here
//     and LIBERO's square frames hide it — it only appears on a non-square camera.
#pragma once

#include "bllm/model_config.h"
#include "bllm/native_embed.h"
#include "bllm/native_smolvla.h"
#include "bllm/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct SmolVlaObservation {
  // Interleaved RGB8, any size; resized with aspect-preserving LEFT/TOP padding.
  const uint8_t* image = nullptr;
  int image_h = 0, image_w = 0;
  const uint8_t* wrist_image = nullptr;
  int wrist_h = 0, wrist_w = 0;
  // Proprioception, `state_dim_real` values in the dataset's own units. Unlike
  // `pi05_libero`, SmolVLA genuinely reads this: `embed_prefix` projects it into
  // a prefix token, so a wrong or absent state is a different observation.
  const float* state = nullptr;
  std::string prompt;
};

class SmolVlaPolicy {
 public:
  explicit SmolVlaPolicy(const ModelConfig& cfg) : cfg_(cfg) {
    SmolVlaShape s;
    s.cameras = cfg.smolvla.cameras;
    s.prompt_len = cfg.smolvla.prompt_len;
    s.horizon = cfg.smolvla.horizon;
    s.steps = cfg.smolvla.steps;
    stack_ = std::make_unique<SmolVlaStack>(cfg.visual.c_str(), cfg.hbm.c_str(),
                                            cfg.smolvla.expert.c_str(), s);
    shape_ = s;
    tok_ = std::make_unique<Tokenizer>(Tokenizer::fromFile(cfg.tokenizer));
    embed_ = std::make_unique<EmbedTable>(cfg.embed, cfg.smolvla.vocab, s.hidden);
    if (!cfg.smolvla.embed_prescaled)
      throw std::runtime_error(
          "[smolvla] this runtime expects an embed table already multiplied by "
          "sqrt(hidden); repackage with export_smolvla_package.py");
    cond_ = readF32(cfg.smolvla.cond_table);
    if ((int)cond_.size() != s.steps * s.expert_width)
      throw std::runtime_error("[smolvla] cond_table is " + std::to_string(cond_.size()) +
                               " floats, expected steps*" + std::to_string(s.expert_width));
    const auto sp = readF32(cfg.smolvla.state_proj);
    const size_t want = (size_t)s.hidden * s.action_dim + s.hidden;
    if (sp.size() != want)
      throw std::runtime_error("[smolvla] state_proj is " + std::to_string(sp.size()) +
                               " floats, expected " + std::to_string(want));
    spW_.assign(sp.begin(), sp.begin() + (size_t)s.hidden * s.action_dim);
    spB_.assign(sp.begin() + (size_t)s.hidden * s.action_dim, sp.end());

    const int P = s.prefix(), H = s.horizon;
    patches_.resize((size_t)s.cameras * s.patch_tokens * s.patch_row);
    prompt_.resize((size_t)s.prompt_len * s.hidden);
    stateTok_.resize(s.hidden);
    bias_.resize((size_t)P * P);
    abias_.resize((size_t)H * (P + H));
    cos_.resize((size_t)P * s.head_dim);
    sin_.resize(cos_.size());
    acos_.resize((size_t)H * s.head_dim);
    asin_.resize(acos_.size());
    raw_.resize((size_t)H * s.action_dim);
    apos_.assign(H, 0);
  }

  const SmolVlaShape& shape() const { return shape_; }
  int actionDim() const { return cfg_.smolvla.action_dim_real; }
  int stateDim() const { return cfg_.smolvla.state_dim_real; }
  int horizon() const { return shape_.horizon; }

  // One observation -> [horizon, action_dim_real] in the dataset's own units.
  // `noise` is the flow-matching latent, [horizon, action_dim]; pass null to
  // draw one. SmolVLA is a flow model integrated from a Gaussian, so two calls
  // on the same observation legitimately differ — and by a lot here: the
  // reference disagrees with itself at position max|d| 0.408 across latents.
  // Pinning it is the only way to compare two implementations meaningfully.
  std::vector<float> act(const SmolVlaObservation& obs, const float* noise = nullptr) {
    const int PT = shape_.patch_tokens, PR = shape_.patch_row, C = shape_.hidden;
    const int NT = shape_.vision_tokens, cams = shape_.cameras, P = shape_.prefix();
    if (!obs.image || (cams > 1 && !obs.wrist_image))
      throw std::runtime_error("[smolvla] observation needs " + std::to_string(cams) +
                               " camera frames");
    preprocess(obs.image, obs.image_h, obs.image_w, patches_.data());
    if (cams > 1)
      preprocess(obs.wrist_image, obs.wrist_h, obs.wrist_w,
                 patches_.data() + (size_t)PT * PR);

    // `SmolVLANewLineProcessor`: the task string ends with a newline before it is
    // tokenized. One token of difference is a fluent but different instruction.
    std::string text = obs.prompt;
    if (text.empty() || text.back() != '\n') text.push_back('\n');
    std::vector<int> ids = tok_->encode(text);
    const int n = (int)ids.size();
    if (n > shape_.prompt_len)
      throw std::runtime_error("[smolvla] instruction is " + std::to_string(n) +
                               " tokens, the graph's language slot holds " +
                               std::to_string(shape_.prompt_len));
    std::fill(prompt_.begin(), prompt_.end(), 0.0f);
    for (int j = 0; j < n; ++j) embed_->lookup(ids[j], prompt_.data() + (size_t)j * C);

    stateToken(obs.state);

    // Positions are `cumsum(pad_mask) - 1`, so language padding does NOT advance
    // the counter: the state token sits at `cams*NT + n`, not at the slot count.
    const int nReal = cams * NT + n + 1;         // images + real language + state
    buildMasks(n);
    ropeTable(cos_.data(), sin_.data(), P, prefixPositions(n).data());
    for (int t = 0; t < shape_.horizon; ++t) apos_[t] = nReal + t;
    ropeTable(acos_.data(), asin_.data(), shape_.horizon, apos_.data());

    std::vector<float> z((size_t)shape_.horizon * shape_.action_dim);
    if (noise) {
      std::copy(noise, noise + z.size(), z.begin());
    } else {
      std::normal_distribution<float> g(0.0f, 1.0f);
      for (auto& v : z) v = g(rng_);
    }

    stack_->setObservation(patches_.data(), prompt_.data(), stateTok_.data(), cos_.data(),
                           sin_.data(), bias_.data());
    stack_->sampleActions(z.data(), cond_.data(), acos_.data(), asin_.data(), abias_.data(),
                          raw_.data());

    // MEAN_STD unnormalisation over the real dims only; the rest of the 32 are
    // padding the dataset never used. (π0.5 used quantiles here; SmolVLA's
    // `normalization_mapping` says MEAN_STD, and the two are not interchangeable.)
    const int A = cfg_.smolvla.action_dim_real;
    std::vector<float> out((size_t)shape_.horizon * A);
    for (int t = 0; t < shape_.horizon; ++t)
      for (int j = 0; j < A; ++j)
        out[(size_t)t * A + j] =
            raw_[(size_t)t * shape_.action_dim + j] * cfg_.smolvla.action_std[j] +
            cfg_.smolvla.action_mean[j];
    return out;
  }

  void seed(uint32_t s) { rng_.seed(s); }

 private:
  static constexpr float kMaskBias = -1e4f;      // clipped to the int16 floor in-graph
  static constexpr int kPatch = 16;

  static std::vector<float> readF32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("[smolvla] cannot open " + path);
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> v((size_t)n / 4);
    f.read((char*)v.data(), n);
    return v;
  }

  // `cumsum(pad_mask) - 1`: images all real, then the real language tokens, then
  // the padded slots repeat the last position, then the state.
  std::vector<int>& prefixPositions(int n) {
    const int NT = shape_.vision_tokens, cams = shape_.cameras, P = shape_.prefix();
    ppos_.assign(P, 0);
    int p = -1;
    for (int i = 0; i < P; ++i) {
      const bool real = (i < cams * NT) || (i < cams * NT + n) || (i == P - 1);
      if (real) ++p;
      ppos_[i] = p;
    }
    return ppos_;
  }

  // The two masks `make_att_2d_masks` produces, as additive biases.
  //
  // `att_masks` is 0 for every image and language token and 1 for the state, so
  // the image+language block is fully bidirectional and nothing attends to the
  // state except the state itself. The action tokens attend to the whole valid
  // prefix and causally among themselves — and NOT to the language padding,
  // which is the one that cost a chunk cosine of 0.957 when it was missed.
  void buildMasks(int n) {
    const int NT = shape_.vision_tokens, cams = shape_.cameras;
    const int P = shape_.prefix(), H = shape_.horizon;
    const int firstPad = cams * NT + n, lastPad = P - 1;   // [firstPad, lastPad)
    auto pad = [&](int i) { return i < firstPad || i == P - 1; };
    for (int i = 0; i < P; ++i)
      for (int j = 0; j < P; ++j) {
        const bool ar = pad(i) && pad(j) && (j != P - 1 || i == P - 1);
        bias_[(size_t)i * P + j] = ar ? 0.0f : kMaskBias;
      }
    for (int t = 0; t < H; ++t) {
      float* row = abias_.data() + (size_t)t * (P + H);
      for (int j = 0; j < P; ++j) row[j] = pad(j) ? 0.0f : kMaskBias;
      for (int j = 0; j < H; ++j) row[P + j] = (j <= t) ? 0.0f : kMaskBias;
    }
    (void)lastPad;
  }

  // theta 10000 in fp32, the way `apply_rope` computes it. No bf16 rounding: the
  // reference builds these tables in float32 and never rounds them, unlike
  // π0.5's `inv_freq`, which is a bf16 buffer in the checkpoint.
  void ropeTable(float* cos, float* sin, int rows, const int* positions) const {
    const int HD = shape_.head_dim, half = HD / 2;
    for (int i = 0; i < half; ++i) {
      const float ts = std::pow((float)cfg_.smolvla.rope_theta, (2.0f / HD) * (float)i);
      for (int t = 0; t < rows; ++t) {
        const double ang = (double)positions[t] / (double)ts;
        const float c = (float)std::cos(ang), s = (float)std::sin(ang);
        cos[(size_t)t * HD + i] = c;
        cos[(size_t)t * HD + i + half] = c;
        sin[(size_t)t * HD + i] = s;
        sin[(size_t)t * HD + i + half] = s;
      }
    }
  }

  // MEAN_STD normalise, pad to 32, then `state_proj` — one [32 -> 960] matmul,
  // the only prefix row the board cannot produce for itself.
  void stateToken(const float* state) {
    const int C = shape_.hidden, AD = shape_.action_dim;
    const int S = cfg_.smolvla.state_dim_real;
    std::vector<float> v((size_t)AD, 0.0f);
    if (state)
      for (int i = 0; i < S; ++i)
        v[i] = (state[i] - cfg_.smolvla.state_mean[i]) /
               (cfg_.smolvla.state_std[i] + 1e-8f);
    for (int o = 0; o < C; ++o) {
      float acc = spB_[o];
      const float* w = spW_.data() + (size_t)o * AD;
      for (int i = 0; i < AD; ++i) acc += w[i] * v[i];
      stateTok_[o] = acc;
    }
  }

  // Aspect-preserving resize into the tower's square input, padding LEFT and TOP
  // (`resize_with_pad`'s smolvla/xvla convention — openpi centres, and copying
  // that here would be wrong), then [0,255] -> [-1,1] and patchify. The pad value
  // is 0 before the scaling, so padding reads as -1.
  void preprocess(const uint8_t* src, int h, int w, float* dstPatches) const {
    const int PT = shape_.patch_tokens, PR = shape_.patch_row;
    const int side = (int)std::lround(std::sqrt((double)PT)) * kPatch;   // 32*16 = 512
    if (h <= 0 || w <= 0) throw std::runtime_error("[smolvla] image has no size");
    const double sc = std::min((double)side / h, (double)side / w);
    const int rh = std::max(1, (int)(h * sc));
    const int rw = std::max(1, (int)(w * sc));
    const int oy = side - rh, ox = side - rw;            // LEFT and TOP

    chw_.assign((size_t)3 * side * side, -1.0f);
    for (int y = 0; y < rh; ++y) {
      const double sy = (y + 0.5) * h / rh - 0.5;
      const int y0 = (int)std::floor(sy);
      const double fy = sy - y0;
      const int ya = std::min(std::max(y0, 0), h - 1);
      const int yb = std::min(std::max(y0 + 1, 0), h - 1);
      for (int x = 0; x < rw; ++x) {
        const double sx = (x + 0.5) * w / rw - 0.5;
        const int x0 = (int)std::floor(sx);
        const double fx = sx - x0;
        const int xa = std::min(std::max(x0, 0), w - 1);
        const int xb = std::min(std::max(x0 + 1, 0), w - 1);
        for (int c = 0; c < 3; ++c) {
          const double p00 = src[((size_t)ya * w + xa) * 3 + c];
          const double p01 = src[((size_t)ya * w + xb) * 3 + c];
          const double p10 = src[((size_t)yb * w + xa) * 3 + c];
          const double p11 = src[((size_t)yb * w + xb) * 3 + c];
          const double v = (p00 * (1 - fx) + p01 * fx) * (1 - fy) +
                           (p10 * (1 - fx) + p11 * fx) * fy;
          chw_[((size_t)c * side + oy + y) * side + ox + x] =
              (float)(v / 255.0 * 2.0 - 1.0);
        }
      }
    }
    const int g = side / kPatch;
    for (int py = 0; py < g; ++py)
      for (int px = 0; px < g; ++px) {
        float* row = dstPatches + (size_t)(py * g + px) * PR;
        int k = 0;
        for (int c = 0; c < 3; ++c)
          for (int dy = 0; dy < kPatch; ++dy)
            for (int dx = 0; dx < kPatch; ++dx)
              row[k++] = chw_[((size_t)c * side + py * kPatch + dy) * side +
                              px * kPatch + dx];
      }
  }

  ModelConfig cfg_;
  SmolVlaShape shape_;
  std::unique_ptr<SmolVlaStack> stack_;
  std::unique_ptr<Tokenizer> tok_;
  std::unique_ptr<EmbedTable> embed_;
  std::vector<float> cond_, patches_, prompt_, stateTok_, bias_, abias_;
  std::vector<float> cos_, sin_, acos_, asin_, raw_, spW_, spB_;
  std::vector<int> ppos_, apos_;
  mutable std::vector<float> chw_;
  std::mt19937 rng_{1234};
};

}  // namespace bllm
