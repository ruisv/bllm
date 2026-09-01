// SmolVLA on the BPU: SmolVLM vision tower -> SmolLM2 prefix -> flow-matching loop.
//
// Three graphs, chained, in the shape `native_pi05.h` established. Two things
// differ, and both were decided in the graph work rather than here:
//
//   * the prefix graph emits the **cross-attention layers' K/V already projected
//     through the expert's own k/v**, because that projection is constant across
//     the denoising steps. Even layers carry the raw VLM K/V (the expert
//     self-attends over them), odd layers the projected pair — 32 tensors either
//     way, and the expert does not re-project anything;
//   * the Euler step is **not** folded into the graph. The expert returns the
//     velocity and the host integrates, which is what the reference does and
//     what the numbers were gated against.
//
// The prefix KV is [177, 5, 64] x 32 = 7.25 MB. It is written into the expert's
// input buffers once per observation; only `x_t` and `cond` move per step.
#pragma once

#include "bllm/native_graph.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct SmolVlaShape {
  int cameras = 2;
  int vision_tokens = 64;     // 1024 patches -> pixel shuffle by 4
  int patch_tokens = 1024;    // 512px at patch 16
  int patch_row = 768;        // 3 * 16 * 16, conv2d's own layout
  int prompt_len = 48;        // `pad_language_to: max_length`
  int hidden = 960;
  int layers = 16;            // the first 16 of SmolLM2's 32
  int kv_heads = 5;
  int head_dim = 64;
  int expert_width = 720;
  int horizon = 50;
  int action_dim = 32;
  int steps = 10;

  int kvRow() const { return kv_heads * head_dim; }          // 320
  // images ++ language slots ++ one state token
  int prefix() const { return cameras * vision_tokens + prompt_len + 1; }
};

class SmolVlaStack {
 public:
  SmolVlaStack(const char* vision_hbm, const char* prefix_hbm, const char* expert_hbm,
               const SmolVlaShape& s)
      : shape_(s) {
    open(pVision_, vision_, vision_hbm, "smolvla_vision");
    open(pPrefix_, prefix_g_, prefix_hbm, "smolvla_prefix");
    open(pExpert_, expert_, expert_hbm, "smolvla_expert");
    const int want = kExpertKvBase + 2 * shape_.layers;
    if (expert_.inCount() != want)
      throw std::runtime_error("[smolvla] expert wants " + std::to_string(expert_.inCount()) +
                               " inputs, expected " + std::to_string(want));
    prefix_.resize((size_t)shape_.prefix() * shape_.hidden);
    kvbuf_.resize((size_t)shape_.prefix() * shape_.kvRow());
  }

  // --- one observation ----------------------------------------------------
  // `patches`  : `cameras` blocks of [patch_tokens, patch_row]
  // `prompt`   : [prompt_len, hidden], embedded on the host and ALREADY scaled by
  //              sqrt(hidden) -- `embed_prefix` applies that outside the table
  //              lookup, and a package that ships the raw row makes the policy
  //              45x deaf to its own instruction with every cosine still at
  //              0.9999 (see π0.5, which shipped that way for three months).
  // `state`    : [1, hidden], `state_proj(state)` on the host.
  // `cos`/`sin`: [prefix, head_dim] for positions `cumsum(pad_mask) - 1`.
  // `bias`     : [prefix, prefix] additive attention bias.
  void setObservation(const float* patches, const float* prompt, const float* state,
                      const float* cos, const float* sin, const float* bias) {
    const int NT = shape_.vision_tokens, C = shape_.hidden, P = shape_.prefix();
    for (int cam = 0; cam < shape_.cameras; ++cam) {
      vision_.writeRowsF32As(0, patches + (size_t)cam * shape_.patch_tokens * shape_.patch_row,
                             shape_.patch_tokens, shape_.patch_row);
      vision_.infer();
      vision_.readRowsF32As(0, prefix_.data() + (size_t)cam * NT * C, NT, C);
    }
    float* tail = prefix_.data() + (size_t)shape_.cameras * NT * C;
    std::copy(prompt, prompt + (size_t)shape_.prompt_len * C, tail);
    std::copy(state, state + C, tail + (size_t)shape_.prompt_len * C);

    prefix_g_.writeRowsF32As(0, prefix_.data(), P, C);
    prefix_g_.writeRowsF32As(1, cos, P, shape_.head_dim);
    prefix_g_.writeRowsF32As(2, sin, P, shape_.head_dim);
    prefix_g_.writeRowsF32As(3, bias, P, P);
    prefix_g_.infer();

    // The one copy that matters: 32 KV tensors moved once per observation
    // rather than once per denoising step.
    for (int i = 0; i < 2 * shape_.layers; ++i) {
      prefix_g_.readRowsF32As(i, kvbuf_.data(), P, shape_.kvRow());
      expert_.writeRowsF32As(kExpertKvBase + i, kvbuf_.data(), P, shape_.kvRow());
    }
  }

  // --- the flow-matching loop --------------------------------------------
  // `noise` is [horizon, action_dim]; `cond_table` is [steps, expert_width] and
  // is a build-time constant -- the time MLP is linear in its concatenation, so
  // the per-step term never needs the board.
  //
  // `acos`/`asin` are [horizon, head_dim] for the SELF-attention layers, built
  // at the **absolute** positions `sum(pad_mask) + [0, horizon)`. The
  // cross-attention layers rope at 0..horizon-1 and those tables are baked into
  // the graph, because that convention never moves.
  void sampleActions(const float* noise, const float* cond_table, const float* acos,
                     const float* asin, const float* bias, float* actions) {
    const int H = shape_.horizon, AD = shape_.action_dim, W = shape_.expert_width;
    const int S = shape_.prefix() + H;
    std::vector<float> x(noise, noise + (size_t)H * AD);
    std::vector<float> v((size_t)H * AD);
    expert_.writeRowsF32As(2, acos, H, shape_.head_dim);
    expert_.writeRowsF32As(3, asin, H, shape_.head_dim);
    expert_.writeRowsF32As(4, bias, H, S);
    for (int s = 0; s < shape_.steps; ++s) {
      expert_.writeRowsF32As(0, x.data(), H, AD);
      expert_.writeRowsF32As(1, cond_table + (size_t)s * W, 1, W);
      expert_.infer();
      expert_.readRowsF32As(0, v.data(), H, AD);
      const float dt = -1.0f / (float)shape_.steps;
      for (size_t i = 0; i < x.size(); ++i) x[i] += dt * v[i];
    }
    std::copy(x.begin(), x.end(), actions);
  }

  const SmolVlaShape& shape() const { return shape_; }
  native_detail::Graph& vision() { return vision_; }
  native_detail::Graph& prefix() { return prefix_g_; }
  native_detail::Graph& expert() { return expert_; }

 private:
  // x_t, cond, cos, sin, bias, then 2*layers KV
  static constexpr int kExpertKvBase = 5;

  // The handle has to outlive the graph — hbDNNRelease invalidates the model
  // handles it produced.
  void open(native_detail::PackedHbm& p, native_detail::Graph& g, const char* path,
            const char* name) {
    p.load({path});
    g.init(p, name);
  }

  SmolVlaShape shape_;
  native_detail::PackedHbm pVision_, pPrefix_, pExpert_;
  native_detail::Graph vision_, prefix_g_, expert_;
  std::vector<float> prefix_, kvbuf_;
};

}  // namespace bllm
