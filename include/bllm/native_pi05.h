// π0.5 on the BPU: SigLIP tower -> PaliGemma prefill -> flow-matching loop.
//
// Three graphs, chained. The shape is simpler than `native_beingh.h`'s because
// π0.5's prefix is entirely static — no rolled KV window, no per-call mask, no
// segment bookkeeping — so this class is mostly about *not* moving the prefix KV
// more than once.
//
// The expert reads all 36 prefix KV tensors on every denoising step, and at
// [532, 1, 256] f32 that is 19.6 MB. Uploading it ten times per chunk would cost
// more than the ten inferences do, so the KV is written into the expert's input
// buffers once per observation and only `x_t` moves afterwards.
//
// `cond` never depends on the observation, only on the step index, so the time
// MLP is not here at all: `cond_table` is a [steps, 1024] constant produced at
// build time.
#pragma once

#include "bllm/native_engine.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct Pi05Shape {
  int cameras = 2;
  int vision_tokens = 256;    // SigLIP So400m/14 at 224px
  int patch_row = 588;        // 3 * 14 * 14, conv2d's own layout
  int prompt_len = 20;
  int hidden = 2048;
  int layers = 18;
  int head_dim = 256;
  int horizon = 10;
  int action_dim = 32;
  int steps = 10;             // flow-matching denoising steps

  int prefix() const { return cameras * vision_tokens + prompt_len; }
};

class Pi05Stack {
 public:
  Pi05Stack(const char* vit_hbm, const char* prefill_hbm, const char* expert_hbm,
            const Pi05Shape& s)
      : shape_(s) {
    open(vit_, vit_hbm, "pi05_siglip");
    open(prefill_, prefill_hbm, "pi05_paligemma");
    open(expert_, expert_hbm, "pi05_expert");
    if (expert_.inCount() != kExpertKvBase + 2 * shape_.layers)
      throw std::runtime_error("[pi05] expert wants " + std::to_string(expert_.inCount()) +
                               " inputs, expected " + std::to_string(kExpertKvBase) + " + 2*" +
                               std::to_string(shape_.layers));
    prefix_.resize((size_t)shape_.prefix() * shape_.hidden);
  }

  // --- one observation ----------------------------------------------------
  // `patches` is `cameras` blocks of [vision_tokens, patch_row]; `prompt` is
  // [prompt_len, hidden] already embedded on the host, because the 257152-row
  // table is tied to lm_head and has no business on the board.
  // `bias` is [prefix] additive attention bias: 0 for a real token, a large
  // negative for a padded prompt slot. Instructions vary in length (LIBERO's run
  // 16-21 tokens) while the graph's prefix is fixed, and a padded slot that is
  // not masked both attracts attention here and leaves KV the expert will read.
  void setObservation(const float* patches, const float* prompt, const float* bias) {
    const int NT = shape_.vision_tokens, C = shape_.hidden;
    for (int cam = 0; cam < shape_.cameras; ++cam) {
      vit_.writeRowsF32As(0, patches + (size_t)cam * NT * shape_.patch_row, NT,
                          shape_.patch_row);
      vit_.infer();
      vit_.readRowsF32As(0, prefix_.data() + (size_t)cam * NT * C, NT, C);
    }
    std::copy(prompt, prompt + (size_t)shape_.prompt_len * C,
              prefix_.data() + (size_t)shape_.cameras * NT * C);

    prefill_.writeRowsF32As(0, prefix_.data(), shape_.prefix(), C);
    prefill_.writeRowsF32As(1, bias, 1, shape_.prefix());
    prefill_.infer();
    expert_.writeRowsF32As(2, bias, 1, shape_.prefix());

    // The one copy that matters: prefill's 36 outputs into the expert's 36
    // inputs, once per observation rather than once per denoising step.
    const int rows = shape_.prefix();
    std::vector<float> buf((size_t)rows * shape_.head_dim);
    for (int i = 0; i < 2 * shape_.layers; ++i) {
      prefill_.readRowsF32As(i, buf.data(), rows, shape_.head_dim);
      expert_.writeRowsF32As(kExpertKvBase + i, buf.data(), rows, shape_.head_dim);
    }
  }

  // --- the flow-matching loop --------------------------------------------
  // `noise` is [horizon, action_dim]; `cond_table` is [steps, 1024]. The Euler
  // step is folded into the graph, so this only carries x_t forward.
  //
  // `rope_cos` / `rope_sin` are [horizon, head_dim] and must be built for the
  // positions `cameras*vision_tokens + n + [0, horizon)`, where `n` is the
  // number of **real** prompt tokens -- openpi's `sum(prefix_pad_masks)`. They
  // are inputs rather than graph constants because `n` changes with the
  // instruction, and RoPE being relative, a wrong `n` shifts the action rows
  // against the image keys as well as the prompt keys. The host builds them:
  // it already knows `n`, and `rotary_emb.inv_freq` is a bf16 buffer in the
  // checkpoint, which is a correctness requirement rather than a rounding
  // preference (see `rope_tables` in `paligemma_native.py`).
  void sampleActions(const float* noise, const float* cond_table,
                     const float* rope_cos, const float* rope_sin, float* actions) {
    const int H = shape_.horizon, AD = shape_.action_dim, HD = shape_.head_dim;
    std::vector<float> x(noise, noise + (size_t)H * AD);
    expert_.writeRowsF32As(3, rope_cos, H, HD);
    expert_.writeRowsF32As(4, rope_sin, H, HD);
    for (int s = 0; s < shape_.steps; ++s) {
      expert_.writeRowsF32As(0, x.data(), H, AD);
      expert_.writeRowsF32As(1, cond_table + (size_t)s * kCondDim, 1, kCondDim);
      expert_.infer();
      expert_.readRowsF32As(0, x.data(), H, AD);
    }
    std::copy(x.begin(), x.end(), actions);
  }

  native_detail::Graph& vit() { return vit_; }
  native_detail::Graph& prefill() { return prefill_; }
  native_detail::Graph& expert() { return expert_; }

 private:
  static constexpr int kCondDim = 1024;      // gemma_300m width
  // x_t, cond, bias, rope_cos, rope_sin, then 2*layers KV
  static constexpr int kExpertKvBase = 5;

  void open(native_detail::Graph& g, const char* path, const char* name) {
    hbDNNPackedHandle_t packed = nullptr;
    const char* files[] = {path};
    BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&packed, files, 1));
    g.init(packed, name);
  }

  Pi05Shape shape_;
  native_detail::Graph vit_, prefill_, expert_;
  std::vector<float> prefix_;
};

}  // namespace bllm
