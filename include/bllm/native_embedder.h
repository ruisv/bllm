// bllm::NativeEmbedder — text embeddings on the BPU, no libxlm.
//
// An embedding model is far simpler than a chat model: no KV cache across steps, no
// decode loop, no sampling. One causal forward over the prompt, pool the last token's
// post-norm hidden state, L2-normalise. So the compiled `.hbm` is a single ENCODER
// graph, produced by replacing the recipe's `lm_head` with an identity:
//
//   in : tokens[1, seq] s32, positions[seq] s32, mask[seq, cache_len] s16, zeroed KV
//   out: hidden[1, seq, hidden] s16 or f32 (+ KV we ignore)
//
// The input template is not negotiable — it was recovered by diffing token ids against
// the model's own embedder, and getting it wrong silently returns a different vector:
//
//   <|im_start|>system\n{instruction}<|im_end|>\n
//   <|im_start|>user\n{text}<|im_end|>\n
//   <|im_start|>assistant\n<|endoftext|>
//                          ^^^^^^^^^^^^^ this token's hidden state is what gets pooled
//
// Matryoshka (MRL): the model was trained so a truncated prefix of the vector is still
// a valid embedding — truncate, then renormalise. `dim = 0` keeps the full width.
//
//
#pragma once

#include "bllm/model_config.h"
#include "bllm/native_engine.h"
#include "bllm/tokenizer.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

class NativeEmbedder {
 public:
  explicit NativeEmbedder(const std::string& model_dir) : NativeEmbedder(loadModelConfig(model_dir)) {}
  explicit NativeEmbedder(ModelConfig cfg)
      : cfg_(std::move(cfg)), tk_(Tokenizer::fromFile(cfg_.tokenizer)) {
    if (!cfg_.is_embed()) throw std::runtime_error("[bllm] NativeEmbedder needs arch=\"embed\"");
    const char* files[] = {cfg_.hbm.c_str()};
    BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&packed_, files, 1));
    g_.init(packed_, cfg_.graph.empty() ? "prefill" : cfg_.graph.c_str());

    nLayers_ = (g_.outCount() - 1) / 2;
    if (g_.inCount() != 3 + 2 * nLayers_)
      throw std::runtime_error("[embed] unexpected encoder graph: " + std::to_string(g_.inCount()) +
                               " inputs, " + std::to_string(g_.outCount()) + " outputs");
    const auto& ts = g_.inShape(0);
    seq_ = ts.dimensionSize[ts.numDimensions - 1];
    const auto& ms = g_.inShape(2);
    cacheLen_ = ms.dimensionSize[ms.numDimensions - 1];
    if (g_.inType(2) != HB_DNN_TENSOR_TYPE_S16)
      throw std::runtime_error("[embed] the encoder's mask input is not s16");
    const auto& os = g_.outShape(0);
    hidden_ = os.dimensionSize[os.numDimensions - 1];
    const int ht = g_.outType(0);
    if (ht != HB_DNN_TENSOR_TYPE_S16 && ht != HB_DNN_TENSOR_TYPE_F32)
      throw std::runtime_error("[embed] hidden output is neither s16 nor f32");
    hiddenIsF32_ = (ht == HB_DNN_TENSOR_TYPE_F32);
    rowBytes_ = g_.outStride(0, os.numDimensions - 2);   // one seq step, in bytes
    hiddenScale_ = g_.outScale(0);

    im_start_ = tk_.token_to_id("<|im_start|>");
    im_end_ = tk_.token_to_id("<|im_end|>");
    eot_ = tk_.token_to_id("<|endoftext|>");

    zeroCaches();
    writeMaskAndPositions();
  }
  ~NativeEmbedder() { if (packed_) hbDNNRelease(packed_); }
  NativeEmbedder(const NativeEmbedder&) = delete;
  NativeEmbedder& operator=(const NativeEmbedder&) = delete;

  int dimension() const { return hidden_; }
  int max_tokens() const { return seq_; }
  const ModelConfig& config() const { return cfg_; }
  const Tokenizer& tokenizer() const { return tk_; }
  void set_sched(const native_detail::BpuSched& s) { g_.sched = s; }

  // Token ids for `text` under the model's template. Exposed because reproducing this
  // sequence is the only way to check a port against the reference implementation.
  std::vector<int> build_ids(const std::string& text, const std::string& instruction = "") const {
    const std::string instr = instruction.empty() ? cfg_.chat.system : instruction;
    std::vector<int> ids;
    auto add = [&](const std::vector<int>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    add({im_start_}); add(tk_.encode("system\n" + instr)); add({im_end_}); add(tk_.encode("\n"));
    add({im_start_}); add(tk_.encode("user\n" + text));    add({im_end_}); add(tk_.encode("\n"));
    add({im_start_}); add(tk_.encode("assistant\n"));
    add({eot_});                                  // the pooled token
    return ids;
  }

  // L2-normalised embedding. `dim > 0` truncates first (Matryoshka), then renormalises.
  std::vector<float> embed(const std::string& text, int dim = 0,
                           const std::string& instruction = "") {
    return embed_ids(build_ids(text, instruction), dim);
  }

  std::vector<float> embed_ids(const std::vector<int>& ids, int dim = 0) {
    if (ids.empty()) throw std::runtime_error("[embed] empty input");
    if ((int)ids.size() > seq_)
      throw std::runtime_error("[embed] " + std::to_string(ids.size()) + " tokens exceed the "
                               "encoder's " + std::to_string(seq_) + "-token window; shorten the "
                               "text or compile a longer graph");
    const int out_dim = dim > 0 ? dim : hidden_;
    if (out_dim > hidden_) throw std::runtime_error("[embed] dim > model dimension");

    auto* tok = (int32_t*)g_.inPtr(0);
    std::memset(tok, 0, (size_t)seq_ * sizeof(int32_t));
    for (size_t i = 0; i < ids.size(); ++i) tok[i] = ids[i];
    for (int i = (int)ids.size(); i < seq_; ++i) tok[i] = eot_;   // pad; masked out anyway
    g_.inClean(0);
    g_.infer();

    // The pooled row is the last REAL token; attention is causal, so the padding after
    // it cannot have influenced it.
    const auto* row = (const uint8_t*)g_.outPtr(0) + (size_t)(ids.size() - 1) * rowBytes_;
    std::vector<float> v(out_dim);
    double n2 = 0;
    for (int i = 0; i < out_dim; ++i) {
      // A preserve_precision graph keeps the final norm in fp32, so the hidden comes back
      // as f32 with no scale; the quantized build returns s16 + a per-tensor scale.
      v[i] = hiddenIsF32_ ? ((const float*)row)[i] : ((const int16_t*)row)[i] * hiddenScale_;
      n2 += (double)v[i] * v[i];
    }
    const float inv = n2 > 0 ? (float)(1.0 / std::sqrt(n2)) : 0.0f;
    for (float& x : v) x *= inv;
    return v;
  }

 private:
  void zeroCaches() {
    for (int i = 3; i < g_.inCount(); ++i) {
      const auto& p = g_.in[i].properties;
      std::memset(g_.inPtr(i), 0, (size_t)(p.stride[0] * p.validShape.dimensionSize[0]));
      g_.inClean(i);
    }
  }
  // Positions and the causal mask never change: the graph always sees a full `seq`-token
  // chunk starting at position 0, occupying the last `seq` slots of the cache window.
  void writeMaskAndPositions() {
    auto* pos = (int32_t*)g_.inPtr(1);
    for (int i = 0; i < seq_; ++i) pos[i] = i;
    g_.inClean(1);
    auto* m = (int16_t*)g_.inPtr(2);
    const int base = cacheLen_ - seq_;
    for (int i = 0; i < seq_; ++i)
      for (int j = 0; j < cacheLen_; ++j)
        m[(size_t)i * cacheLen_ + j] = (j >= base && j <= base + i) ? 0 : -32768;
    g_.inClean(2);
  }

  ModelConfig cfg_;
  Tokenizer tk_;
  hbDNNPackedHandle_t packed_ = nullptr;
  native_detail::Graph g_;
  int nLayers_ = 0, seq_ = 0, cacheLen_ = 0, hidden_ = 0;
  int64_t rowBytes_ = 0;
  bool hiddenIsF32_ = false;
  float hiddenScale_ = 1.0f;
  int im_start_ = -1, im_end_ = -1, eot_ = -1;
};

}  // namespace bllm
