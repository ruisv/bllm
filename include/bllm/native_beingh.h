// bllm::BeingHStack — Being-H0.5's Mixture-of-Transformers stack driven as what
// it actually decomposes into: ordinary single-stream passes over one shared KV
// window.
//
// The checkpoint looks like two entangled transformers, but at inference the two
// streams occupy disjoint, contiguous position ranges under a plain causal mask.
// So a chunk is replayed as four calls into two compiled graphs:
//
//   A  und tokens before <state>   -> und graph   (system + user + vision)
//   s  the one gen state token     -> gen graph, N=1
//   B  und tokens after </state>   -> und graph   (instruction + assistant)
//   x  the action tokens           -> gen graph, N=chunk, run once per flow step
//
// Verified exact in fp32 against the packed reference (cosine 1.0, see
// host_toolchain/convert/beingh/P0_FINDINGS.md), so there is no fused
// dual-stream graph anywhere in this runtime.
//
// Both streams project into the same 16x128 q / 8x128 kv space, which is the
// whole reason one `[cache_len, 8, 128]` window per layer serves both graphs.
#pragma once

#include "bllm/native_graph.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct BeingHShape {
  int layers = 28;
  int n_heads = 16;
  int n_kv = 8;
  int head_dim = 128;
  int hidden_und = 2048;
  int hidden_gen = 1024;
  int cache_len = 512;
  float rope_theta = 1e6f;
};

// One compiled call shape: which graph, how many token slots it consumes, and
// whether it hands the rolled window back.
struct BeingHGraph {
  native_detail::Graph g;
  int n = 0;              // token slots per call
  int hidden = 0;
  bool emits_hidden = false;
  bool emits_cache = false;
  // Which cache generation this graph's input buffers already hold. Uploading
  // 28x2 [512,8,128] tensors is 117 MB of memcpy, and the action pass runs eight
  // times over a cache that never changes.
  uint64_t uploaded = (uint64_t)-1;

  int cacheOutBase() const { return emits_hidden ? 1 : 0; }
};

class BeingHStack {
 public:
  // `hbms` may repeat: each graph is looked up in the first file that has it.
  BeingHStack(const std::vector<std::string>& hbms, const BeingHShape& shape)
      : shape_(shape) {
    // Two of these graphs usually live in the same linked .hbm (gen1 and the
    // action pass share 441 M weights), so the caller naturally names the same
    // file twice. Handing a duplicate to hbDNNInitializeFromFiles gets the
    // second copy discarded with a warning and leaves the handle unusable.
    std::vector<std::string> uniq;
    for (const auto& s : hbms)
      if (std::find(uniq.begin(), uniq.end(), s) == uniq.end()) uniq.push_back(s);
    packed_.load(uniq);
    const int64_t slots = (int64_t)shape_.cache_len;
    k_.assign((size_t)shape_.layers, std::vector<float>((size_t)slots * shape_.n_kv * shape_.head_dim, 0.0f));
    v_ = k_;
    slotPos_.assign((size_t)slots, -1);
  }
  BeingHStack(const BeingHStack&) = delete;
  BeingHStack& operator=(const BeingHStack&) = delete;

  // `n` and the emit flags must match how the graph was compiled; they are
  // cross-checked against its input/output counts so a mismatched .hbm fails
  // here rather than producing plausible garbage.
  BeingHGraph& add(const std::string& key, const char* graph, int n, int hidden,
                   bool emits_hidden, bool emits_cache) {
    auto& e = graphs_[key];
    e.g.init(packed_, graph);
    e.n = n; e.hidden = hidden;
    e.emits_hidden = emits_hidden; e.emits_cache = emits_cache;
    const int wantIn = 4 + 2 * shape_.layers;
    const int wantOut = (emits_cache ? 2 * shape_.layers : 0) + (emits_hidden ? 1 : 0);
    if (e.g.inCount() != wantIn || e.g.outCount() != wantOut)
      throw std::runtime_error("[beingh] graph " + std::string(graph) + " has " +
                               std::to_string(e.g.inCount()) + " in / " +
                               std::to_string(e.g.outCount()) + " out, expected " +
                               std::to_string(wantIn) + " / " + std::to_string(wantOut));
    if (e.g.inShape(0).dimensionSize[0] != n ||
        e.g.inShape(0).dimensionSize[1] != hidden)
      throw std::runtime_error("[beingh] graph " + std::string(graph) + " input is not [" +
                               std::to_string(n) + "," + std::to_string(hidden) + "]");
    return e;
  }

  void reset() {
    for (auto& l : k_) std::fill(l.begin(), l.end(), 0.0f);
    for (auto& l : v_) std::fill(l.begin(), l.end(), 0.0f);
    std::fill(slotPos_.begin(), slotPos_.end(), -1);
    ++generation_;
  }

  // Run one segment. `x` holds `count` real rows of `hidden` floats, at absolute
  // positions [pos0, pos0+count); the graph's remaining slots are padding and
  // are masked out of every future call by leaving their slot position at -1.
  // Returns the hidden output (graph slots, `count` of them meaningful), or null.
  const float* run(BeingHGraph& e, const float* x, int count, int64_t pos0) {
    if (count > e.n) throw std::runtime_error("[beingh] segment longer than the graph");
    const int N = e.n, CL = shape_.cache_len, HD = shape_.head_dim;

    // x, padded with zeros
    std::vector<float> xin((size_t)N * e.hidden, 0.0f);
    std::memcpy(xin.data(), x, (size_t)count * e.hidden * sizeof(float));
    e.g.writeRowsF32(0, xin.data(), N, e.hidden);

    // rope tables for this segment's absolute positions
    std::vector<float> cos((size_t)N * HD, 0.0f), sin((size_t)N * HD, 0.0f);
    for (int i = 0; i < count; ++i)
      ropeRow(pos0 + i, cos.data() + (size_t)i * HD, sin.data() + (size_t)i * HD);
    e.g.writeRowsF32(1, cos.data(), N, HD);
    e.g.writeRowsF32(2, sin.data(), N, HD);

    // A rolling call moves the position ring exactly as the graph rolls the
    // window, then masks against it: a query at absolute p may read any slot
    // holding a real position <= p and nothing else, which is what keeps this
    // call's padding — and every earlier call's — permanently invisible.
    //
    // A non-rolling call leaves the ring alone; its graph scores the cache and
    // its own keys separately, so the mask is CL wide for the window plus N more
    // for this segment's own causal block.
    // Read the width off the graph rather than inferring it: a rolling graph
    // masks CL columns, a non-rolling one CL + N, and silently writing the wrong
    // count into the tensor is a buffer overrun, not a wrong answer.
    const int W = (int)e.g.inShape(3).dimensionSize[e.g.inShape(3).numDimensions - 1];
    if (W != (e.emits_cache ? CL : CL + N))
      throw std::runtime_error("[beingh] mask width " + std::to_string(W) +
                               " does not match cache_len " + std::to_string(CL) +
                               (e.emits_cache ? "" : " + N " + std::to_string(N)));
    if (e.emits_cache) {
      std::vector<int64_t> next((size_t)CL);
      std::copy(slotPos_.begin() + N, slotPos_.end(), next.begin());
      for (int i = 0; i < N; ++i) next[CL - N + i] = (i < count) ? pos0 + i : -1;
      slotPos_.swap(next);
    }
    std::vector<float> mask((size_t)N * W, kMaskNeg);
    for (int i = 0; i < count; ++i) {
      float* row = mask.data() + (size_t)i * W;
      const int64_t p = pos0 + i;
      for (int j = 0; j < CL; ++j)
        if (slotPos_[j] >= 0 && slotPos_[j] <= p) row[j] = 0.0f;
      if (!e.emits_cache)
        for (int j = 0; j <= i; ++j) row[CL + j] = 0.0f;   // causal within the segment
    }
    e.g.writeRowsF32(3, mask.data(), N, W);

    if (e.uploaded != generation_) {
      for (int l = 0; l < shape_.layers; ++l) {
        e.g.writeRowsF32(4 + 2 * l, k_[l].data(), CL, shape_.n_kv * HD);
        e.g.writeRowsF32(5 + 2 * l, v_[l].data(), CL, shape_.n_kv * HD);
      }
      e.uploaded = generation_;
    }

    e.g.infer();

    if (e.emits_cache) {
      const int base = e.cacheOutBase();
      for (int l = 0; l < shape_.layers; ++l) {
        readRows(e.g, base + 2 * l, k_[l].data(), CL, shape_.n_kv * HD);
        readRows(e.g, base + 2 * l + 1, v_[l].data(), CL, shape_.n_kv * HD);
      }
      ++generation_;   // every other graph's copy is now stale
    }
    if (!e.emits_hidden) return nullptr;
    hidden_.resize((size_t)N * e.hidden);
    readRows(e.g, 0, hidden_.data(), N, e.hidden);
    return hidden_.data();
  }

  BeingHGraph& graph(const std::string& key) {
    auto it = graphs_.find(key);
    if (it == graphs_.end()) throw std::runtime_error("[beingh] no graph " + key);
    return it->second;
  }
  const BeingHShape& shape() const { return shape_; }

 private:
  // What the graph's int16 mask can actually represent; -inf would saturate to
  // something the softmax cannot use.
  static constexpr float kMaskNeg = -30000.0f;

  void ropeRow(int64_t pos, float* cos, float* sin) const {
    const int HD = shape_.head_dim, half = HD / 2;
    for (int i = 0; i < half; ++i) {
      const double inv = std::pow((double)shape_.rope_theta, -(double)(2 * i) / (double)HD);
      const double a = (double)pos * inv;
      cos[i] = cos[i + half] = (float)std::cos(a);
      sin[i] = sin[i + half] = (float)std::sin(a);
    }
  }

  static void readRows(native_detail::Graph& g, int idx, float* dst, int64_t rows,
                       int64_t cols) {
    const int64_t pitch = g.outRowPitch(idx, cols);
    const auto* base = (const uint8_t*)g.outPtr(idx);
    if (pitch == cols * (int64_t)sizeof(float)) {
      std::memcpy(dst, base, (size_t)rows * cols * sizeof(float));
      return;
    }
    for (int64_t r = 0; r < rows; ++r)
      std::memcpy(dst + r * cols, base + r * pitch, (size_t)cols * sizeof(float));
  }

  BeingHShape shape_;
  native_detail::PackedHbm packed_;
  std::map<std::string, BeingHGraph> graphs_;
  std::vector<std::vector<float>> k_, v_;
  std::vector<int64_t> slotPos_;
  std::vector<float> hidden_;
  uint64_t generation_ = 0;
};

}  // namespace bllm
