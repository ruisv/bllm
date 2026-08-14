// bllm::Pi05Policy — π0.5 as a policy, not as three graphs.
//
// `Pi05Stack` (native_pi05.h) takes patch matrices, an embedded prompt, an
// attention bias, a RoPE table and a latent, and returns 32 normalised action
// dims. Everything between that and "two camera frames, a state vector and an
// English sentence" lived in host Python that imported openpi, its tokenizer,
// its 14 GB checkpoint and its LIBERO transforms. This class carries that seam
// onto the board so a user needs only the model package:
//
//   tokenize -> embed (mmap fp16 table) -> patchify -> RoPE for the real token
//   count -> Pi05Stack -> unnormalise -> [horizon, action_dim_real]
//
// Three things here are load-bearing and were each a bug at some point:
//
//   * the prompt embedding carries a `sqrt(hidden)` factor that `embed_prefix`
//     applies outside `embed_language_tokens`. The package stores the table
//     **pre-scaled** so no consumer can forget; `ModelConfig::embed_prescaled`
//     records it and this class refuses a table that claims otherwise.
//   * the action rows' RoPE positions start at `cameras*vision_tokens + n`,
//     the count of REAL prompt tokens, not the slot count.
//   * `inv_freq` must be rounded through bfloat16, because openpi builds the
//     model in bf16 and the reference has that loss baked in.
#pragma once

#include "bllm/model_config.h"
#include "bllm/native_embed.h"
#include "bllm/native_pi05.h"
#include "bllm/tokenizer.h"

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

struct Pi05Observation {
  // Interleaved RGB8, any size; resized with aspect-preserving zero padding.
  const uint8_t* image = nullptr;
  int image_h = 0, image_w = 0;
  const uint8_t* wrist_image = nullptr;
  int wrist_h = 0, wrist_w = 0;
  // Proprioception. `pi05_libero` is configured `discrete_state_input=False`
  // and `embed_suffix` applies `state_proj` only when `!pi05`, so openpi's own
  // LIBERO π0.5 does not read state at all. Accepted for API stability and for
  // configs that do tokenise it; ignored here.
  const float* state = nullptr;
  int state_dim = 0;
  std::string prompt;
};

class Pi05Policy {
 public:
  explicit Pi05Policy(const ModelConfig& cfg) : cfg_(cfg) {
    Pi05Shape s;
    s.cameras = cfg.pi05.cameras;
    s.prompt_len = cfg.pi05.prompt_len;
    s.horizon = cfg.pi05.horizon;
    s.steps = cfg.pi05.steps;
    stack_ = std::make_unique<Pi05Stack>(cfg.visual.c_str(), cfg.hbm.c_str(),
                                         cfg.pi05.expert.c_str(), s);
    shape_ = s;
    tok_ = std::make_unique<Tokenizer>(Tokenizer::fromFile(cfg.tokenizer));
    embed_ = std::make_unique<EmbedTable>(cfg.embed, cfg.pi05.vocab, s.hidden);
    if (!cfg.pi05.embed_prescaled)
      throw std::runtime_error(
          "[pi05] this runtime expects an embed table already multiplied by "
          "sqrt(hidden); repackage with export_pi05_package.py");
    cond_ = readF32(cfg.pi05.cond_table);
    if ((int)cond_.size() != s.steps * 1024)
      throw std::runtime_error("[pi05] cond_table is " + std::to_string(cond_.size()) +
                               " floats, expected steps*1024");
    q01_ = cfg.pi05.action_q01;
    q99_ = cfg.pi05.action_q99;
    if ((int)q01_.size() != cfg.pi05.action_dim_real || q01_.size() != q99_.size())
      throw std::runtime_error("[pi05] norm stats do not match action_dim_real");
    const int NT = s.vision_tokens, PR = s.patch_row;
    patches_.resize((size_t)s.cameras * NT * PR);
    prompt_.resize((size_t)s.prompt_len * s.hidden);
    bias_.resize(s.prefix());
    raw_.resize((size_t)s.horizon * s.action_dim);
    cos_.resize((size_t)s.horizon * s.head_dim);
    sin_.resize(cos_.size());
  }

  const Pi05Shape& shape() const { return shape_; }
  int actionDim() const { return cfg_.pi05.action_dim_real; }
  int horizon() const { return shape_.horizon; }

  // One observation -> [horizon, action_dim_real] in the dataset's own units.
  // `noise` is the flow-matching latent, [horizon, action_dim]; pass null to
  // draw one. Supplying it makes a run reproducible, which is the only way to
  // compare two implementations of a stochastic policy meaningfully.
  std::vector<float> act(const Pi05Observation& obs, const float* noise = nullptr) {
    const int NT = shape_.vision_tokens, PR = shape_.patch_row, C = shape_.hidden;
    const int cams = shape_.cameras;
    if (!obs.image || (cams > 1 && !obs.wrist_image))
      throw std::runtime_error("[pi05] observation needs " + std::to_string(cams) +
                               " camera frames");
    preprocess(obs.image, obs.image_h, obs.image_w, patches_.data());
    if (cams > 1)
      preprocess(obs.wrist_image, obs.wrist_h, obs.wrist_w,
                 patches_.data() + (size_t)NT * PR);

    // The tokenizer emits BOS itself; a prompt that overflows the compiled slot
    // is a hard error, never a silent truncation -- a clipped instruction reads
    // as a fluent but different task.
    std::vector<int> ids = tok_->encode(obs.prompt);
    const int n = (int)ids.size();
    if (n > shape_.prompt_len)
      throw std::runtime_error("[pi05] instruction is " + std::to_string(n) +
                               " tokens, the graph's prompt slot holds " +
                               std::to_string(shape_.prompt_len));
    std::fill(prompt_.begin(), prompt_.end(), 0.0f);
    for (int j = 0; j < n; ++j) embed_->lookup(ids[j], prompt_.data() + (size_t)j * C);

    std::fill(bias_.begin(), bias_.end(), 0.0f);
    for (int i = cams * NT + n; i < shape_.prefix(); ++i) bias_[i] = kMaskBias;

    ropeTable(cams * NT + n);

    std::vector<float> z((size_t)shape_.horizon * shape_.action_dim);
    if (noise) {
      std::copy(noise, noise + z.size(), z.begin());
    } else {
      std::normal_distribution<float> g(0.0f, 1.0f);
      for (auto& v : z) v = g(rng_);
    }

    stack_->setObservation(patches_.data(), prompt_.data(), bias_.data());
    stack_->sampleActions(z.data(), cond_.data(), cos_.data(), sin_.data(), raw_.data());

    // Quantile unnormalisation, openpi's `_unnormalize_quantile`, over the real
    // dims only; the rest of the 32 are padding the dataset never used.
    const int A = cfg_.pi05.action_dim_real;
    std::vector<float> out((size_t)shape_.horizon * A);
    for (int t = 0; t < shape_.horizon; ++t)
      for (int j = 0; j < A; ++j)
        out[(size_t)t * A + j] =
            (raw_[(size_t)t * shape_.action_dim + j] + 1.0f) * 0.5f *
                (q99_[j] - q01_[j] + 1e-6f) + q01_[j];
    return out;
  }

  void seed(uint32_t s) { rng_.seed(s); }

 private:
  static constexpr float kMaskBias = -1000.0f;   // clipped to the int16 floor in-graph

  static std::vector<float> readF32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("[pi05] cannot open " + path);
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> v((size_t)n / 4);
    f.read((char*)v.data(), n);
    return v;
  }

  // bf16 round-to-nearest-even, matching `torch.bfloat16` — see the class note.
  static float toBf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t r = (u >> 16) & 1u;
    u = (u + 0x7fffu + r) & 0xffff0000u;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
  }

  // cos/sin for rows `start .. start+horizon`, the positions openpi gives the
  // action tokens (`sum(prefix_pad_masks) + [0, horizon)`).
  void ropeTable(int start) {
    const int H = shape_.horizon, HD = shape_.head_dim, half = HD / 2;
    for (int i = 0; i < half; ++i) {
      const double inv = toBf16((float)(1.0 / std::pow((double)cfg_.rope_theta_pi05,
                                                       (double)(2 * i) / HD)));
      for (int t = 0; t < H; ++t) {
        const double ang = (double)(start + t) * inv;
        const float c = (float)std::cos(ang), s = (float)std::sin(ang);
        cos_[(size_t)t * HD + i] = c;
        cos_[(size_t)t * HD + i + half] = c;
        sin_[(size_t)t * HD + i] = s;
        sin_[(size_t)t * HD + i + half] = s;
      }
    }
  }

  // Aspect-preserving resize into the tower's square input with zero padding,
  // then [0,255] -> [-1,1] and patchify. openpi resizes with PIL bilinear; this
  // is a plain bilinear, which differs slightly under heavy downscaling. For
  // LIBERO (256 -> 224) the paired board-vs-reference check is unmoved by it.
  void preprocess(const uint8_t* src, int h, int w, float* dstPatches) const {
    const int NT = shape_.vision_tokens, PR = shape_.patch_row;
    const int side = (int)std::lround(std::sqrt((double)NT)) * kPatch;   // 16*14 = 224
    if (h <= 0 || w <= 0) throw std::runtime_error("[pi05] image has no size");
    const double sc = std::min((double)side / h, (double)side / w);
    const int rh = std::max(1, (int)std::lround(h * sc));
    const int rw = std::max(1, (int)std::lround(w * sc));
    const int oy = (side - rh) / 2, ox = (side - rw) / 2;

    chw_.assign((size_t)3 * side * side, -1.0f);   // zero pixel -> -1 after scaling
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
    // [3, side, side] -> [NT, 3*14*14], the conv2d-folded patch layout.
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

  static constexpr int kPatch = 14;

  ModelConfig cfg_;
  Pi05Shape shape_;
  std::unique_ptr<Pi05Stack> stack_;
  std::unique_ptr<Tokenizer> tok_;
  std::unique_ptr<EmbedTable> embed_;
  std::vector<float> cond_, patches_, prompt_, bias_, raw_, cos_, sin_;
  std::vector<float> q01_, q99_;
  mutable std::vector<float> chw_;
  std::mt19937 rng_{1234};
};

}  // namespace bllm
