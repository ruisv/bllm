// bllm::BeingHHost — the fp32 CPU half of Being-H0.5's action head.
//
// Everything here is deliberately NOT on the BPU. It is 33 M params, but it is
// the flow-matching head: a continuous velocity regression with no softmax to
// hide error and a 4-step Euler integration to accumulate it, so quantizing it
// buys nothing and risks the only output that matters.
//
// Loads the blob written by host_toolchain/convert/beingh/export_host_weights.py.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace bllm {

struct HostTensor {
  std::vector<int> shape;
  const float* data = nullptr;
  int rows() const { return shape.size() >= 2 ? shape[0] : 1; }
  int cols() const { return shape.empty() ? 0 : shape.back(); }
  size_t size() const {
    size_t n = 1;
    for (int d : shape) n *= (size_t)d;
    return n;
  }
};

class BeingHHost {
 public:
  explicit BeingHHost(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[beingh] cannot open " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "BHHW", 4) != 0)
      throw std::runtime_error("[beingh] " + path + " is not a host-weight blob");
    // Every field below sizes an allocation, and a 4-byte magic is weak
    // evidence. Unchecked, a truncated or half-written blob reaches resize()
    // with whatever the file happened to hold and dies in the allocator (or the
    // OOM killer) rather than with the diagnostic three lines down.
    auto bad = [&path](const char* what) {
      return std::runtime_error("[beingh] " + std::string(what) + " in host-weight blob " + path);
    };
    uint32_t count = 0;
    f.read((char*)&count, 4);
    if (!f || count == 0 || count > (1u << 20)) throw bad("implausible tensor count");
    std::vector<std::string> names(count);
    std::vector<std::vector<int>> shapes(count);
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t nl = 0;
      f.read((char*)&nl, 4);
      if (!f || nl == 0 || nl > 4096) throw bad("implausible tensor-name length");
      names[i].resize(nl);
      f.read(names[i].data(), nl);
      uint32_t nd = 0;
      f.read((char*)&nd, 4);
      if (!f || nd == 0 || nd > 8) throw bad("implausible tensor rank");
      shapes[i].resize(nd);
      for (uint32_t d = 0; d < nd; ++d) {
        uint32_t v = 0;
        f.read((char*)&v, 4);
        if (!f || v == 0 || v > (1u << 24)) throw bad("implausible dimension");
        shapes[i][d] = (int)v;
      }
    }
    size_t total = 0;
    for (const auto& s : shapes) {
      size_t n = 1;
      for (int d : s) n *= (size_t)d;
      total += n;
    }
    // Bound the allocation by what the file actually holds, before making it.
    const std::streampos dataPos = f.tellg();
    f.seekg(0, std::ios::end);
    const std::streamoff avail = f.tellg() - dataPos;
    f.seekg(dataPos);
    if (!f || avail < 0 || (uint64_t)avail < (uint64_t)total * sizeof(float))
      throw bad("truncated weight data");
    blob_.resize(total);
    f.read((char*)blob_.data(), total * sizeof(float));
    if (!f) throw bad("truncated weight data");
    size_t off = 0;
    for (uint32_t i = 0; i < count; ++i) {
      HostTensor t;
      t.shape = shapes[i];
      t.data = blob_.data() + off;
      off += t.size();
      w_[names[i]] = t;
    }
  }

  bool has(const std::string& n) const { return w_.count(n) != 0; }
  const HostTensor& at(const std::string& n) const {
    auto it = w_.find(n);
    if (it == w_.end()) throw std::runtime_error("[beingh] missing host tensor " + n);
    return it->second;
  }

  // ---- primitives -------------------------------------------------------
  // y[t, o] = sum_i x[t, i] * W[o, i] + b[o], torch's nn.Linear layout.
  //
  // NEON, split across cores. These are only 33 M params, but the flow loop runs
  // them 8-17 times per chunk — ~2.9 GFLOP — and the obvious scalar fp64 dot
  // product made them 1.9 s of a 2.4 s chunk, a serial dependency chain nothing
  // could vectorize. fp32 with four vector accumulators is what torch does
  // anyway, and the output rows are trivially independent.
  void linear(const float* x, int T, const std::string& wname, const std::string& bname,
              std::vector<float>& y) const {
    const auto& W = at(wname);
    const int O = W.shape[0], I = W.shape[1];
    const float* b = bname.empty() ? nullptr : at(bname).data;
    y.assign((size_t)T * O, 0.0f);
    const float* wd = W.data;
    float* yd = y.data();
    parallelFor(O, [&](int o0, int o1) {
      for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t)t * I;
        float* yr = yd + (size_t)t * O;
        for (int o = o0; o < o1; ++o) yr[o] = dot(xr, wd + (size_t)o * I, I) + (b ? b[o] : 0.0f);
      }
    });
  }

  static float dot(const float* a, const float* w, int n) {
#if defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 15 < n; i += 16) {
      s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(w + i));
      s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(w + i + 4));
      s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(w + i + 8));
      s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(w + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; ++i) s += a[i] * w[i];
    return s;
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
      s0 += a[i] * w[i];
      s1 += a[i + 1] * w[i + 1];
      s2 += a[i + 2] * w[i + 2];
      s3 += a[i + 3] * w[i + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s += a[i] * w[i];
    return s;
#endif
  }

  // Split [0, n) across the cores. Threads are created per call: a few tens of
  // microseconds against tens of milliseconds of work, and it keeps the header
  // free of a pool's lifetime.
  template <class F>
  static void parallelFor(int n, F&& f) {
    int nt = (int)std::thread::hardware_concurrency();
    nt = nt <= 1 ? 1 : (nt > 6 ? 6 : nt);
    if (nt == 1 || n < 256) { f(0, n); return; }
    std::vector<std::thread> ts;
    const int step = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
      const int a = t * step, b = a + step > n ? n : a + step;
      if (a >= b) break;
      ts.emplace_back([&f, a, b] { f(a, b); });
    }
    for (auto& t : ts) t.join();
  }

  static void relu(std::vector<float>& v) {
    for (float& x : v) x = x > 0 ? x : 0;
  }
  static void swish(std::vector<float>& v) {
    for (float& x : v) x = x / (1.0f + std::exp(-x));
  }

  void simpleMlp(const float* x, int T, const std::string& p, std::vector<float>& out) const {
    std::vector<float> h;
    linear(x, T, p + ".layer1.weight", p + ".layer1.bias", h);
    relu(h);
    linear(h.data(), T, p + ".layer2.weight", p + ".layer2.bias", out);
  }

  // ---- the action head --------------------------------------------------
  // state[200] -> the gen stream's state token embedding [1024]
  void encodeState(const float* state, std::vector<float>& out) const {
    simpleMlp(state, 1, "proprio_encoder_robot", out);
  }

  // The reference's SinusoidalPositionalEncoding: concat(sin, cos) over
  // half_dim frequencies of the INTEGER timestep bucket, not the continuous t.
  static void sinusoid(float t, int dim, float* out) {
    const int half = dim / 2;
    const double step = std::log(10000.0) / (double)half;
    for (int i = 0; i < half; ++i) {
      const double f = (double)t * std::exp(-(double)i * step);
      out[i] = (float)std::sin(f);
      out[half + i] = (float)std::cos(f);
    }
  }

  // actions [T, 200] at integer timestep bucket -> action features [T, 1024]
  void encodeActions(const float* actions, int T, int bucket, std::vector<float>& out) const {
    const int H = at("action_encoder.W1.weight").shape[0];
    std::vector<float> a;
    linear(actions, T, "action_encoder.W1.weight", "action_encoder.W1.bias", a);
    std::vector<float> tau((size_t)H);
    sinusoid((float)bucket, H, tau.data());
    std::vector<float> cat((size_t)T * 2 * H);
    for (int t = 0; t < T; ++t) {
      std::memcpy(cat.data() + (size_t)t * 2 * H, a.data() + (size_t)t * H, (size_t)H * sizeof(float));
      std::memcpy(cat.data() + (size_t)t * 2 * H + H, tau.data(), (size_t)H * sizeof(float));
    }
    std::vector<float> h;
    linear(cat.data(), T, "action_encoder.W2.weight", "action_encoder.W2.bias", h);
    swish(h);
    linear(h.data(), T, "action_encoder.W3.weight", "action_encoder.W3.bias", out);
  }

  // gen hidden [T, 1024] -> velocity [T, 200]
  void decode(const float* hidden, int T, std::vector<float>& out) const {
    simpleMlp(hidden, T, "action_decoder", out);
  }

  bool hasMpg() const { return has("mpg.obs_proj.weight"); }
  // The width the decoder emits — the robot's action dimension.
  int actionDim() const { return at("action_decoder.layer2.weight").shape[0]; }
  // The MPG gate's embedding width: the dimension `mpgEnhance`'s sliced-
  // Wasserstein directions have to be drawn in.
  int mpgEmbedDim() const { return at("mpg.obs_proj.weight").shape[0]; }

  // MPG refinement, exactly as the reference runs it at inference.
  //
  // Two details that look like bugs and are load-bearing: the observation half
  // reads the *und* buffer at the state position, which with an action expert
  // was never written, so it contributes a row of zeros to the Wasserstein
  // statistic and its "enhanced" result is written where nothing reads it. Only
  // the action half feeds back. Reproducing the zero row is required — it is
  // part of the distribution the gate is computed from.
  //
  // `dirs` supplies the 32 unit projection directions. The reference draws them
  // from the global RNG with no seed, so the model is stochastic; a deployment
  // has to fix them, and passing them in is how.
  void mpgEnhance(std::vector<float>& actionFeats, int T, const float* cleanEmb,
                  const float* dirs, int nDirs, float lambda, float gateTemp) const {
    // Three different widths meet in this function and they coincide only by
    // accident of the released checkpoint. Dv is the VLM stream — what the
    // suffix rows and the enhancement live in; De is the MPG embedding the gate
    // is computed in, so `dirs` has to be De-dimensional; the action-feature
    // width is whatever vlm_to_action_proj emits. Using one for all three
    // over-reads aVlm and mis-strides every row after the first, silently.
    const int Dv = at("action_to_vlm_proj.weight").shape[0];   // 2048
    const int De = at("mpg.obs_proj.weight").shape[0];         // 2048
    if (at("mpg.obs_proj.weight").shape[1] != Dv ||
        at("mpg.enhancement_proj.weight").shape[0] != Dv ||
        at("vlm_to_action_proj.weight").shape[1] != Dv ||
        at("mpg.action_proj.weight").shape[0] != De)
      throw std::runtime_error("[beingh] MPG projection dimensions disagree");

    std::vector<float> aVlm;
    linear(actionFeats.data(), T, "action_to_vlm_proj.weight", "action_to_vlm_proj.bias", aVlm);

    // suffix = [state row (zeros), action rows], in VLM dimension
    const int S = T + 1;
    std::vector<float> suffix((size_t)S * Dv, 0.0f);
    std::memcpy(suffix.data() + (size_t)Dv, aVlm.data(), (size_t)T * Dv * sizeof(float));

    std::vector<float> obsEmb, actEmb;
    linear(suffix.data(), S, "mpg.obs_proj.weight", "mpg.obs_proj.bias", obsEmb);
    linear(cleanEmb, T, "mpg.action_proj.weight", "mpg.action_proj.bias", actEmb);

    auto layerNorm = [De](std::vector<float>& v, int rows) {
      for (int r = 0; r < rows; ++r) {
        float* x = v.data() + (size_t)r * De;
        double mu = 0;
        for (int i = 0; i < De; ++i) mu += x[i];
        mu /= De;
        double var = 0;
        for (int i = 0; i < De; ++i) var += (x[i] - mu) * (x[i] - mu);
        var /= De;
        const double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int i = 0; i < De; ++i) x[i] = (float)((x[i] - mu) * inv);
      }
    };
    std::vector<float> obsN = obsEmb, actN = actEmb;
    layerNorm(obsN, S);
    layerNorm(actN, T);

    // pooled action, broadcast to every observation row
    std::vector<float> pooled((size_t)De, 0.0f);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < De; ++i) pooled[i] += actN[(size_t)t * De + i] / (float)T;

    // sliced Wasserstein-2: project, sort, mean squared difference
    double total = 0;
    std::vector<float> px((size_t)S), py((size_t)S);
    for (int k = 0; k < nDirs; ++k) {
      const float* dir = dirs + (size_t)k * De;
      double pooledProj = 0;
      for (int i = 0; i < De; ++i) pooledProj += (double)pooled[i] * dir[i];
      for (int r = 0; r < S; ++r) {
        double s = 0;
        const float* x = obsN.data() + (size_t)r * De;
        for (int i = 0; i < De; ++i) s += (double)x[i] * dir[i];
        px[r] = (float)s;
        py[r] = (float)pooledProj;   // every aligned row is the same pooled vector
      }
      std::sort(px.begin(), px.end());
      std::sort(py.begin(), py.end());
      double w2 = 0;
      for (int r = 0; r < S; ++r) w2 += (double)(px[r] - py[r]) * (px[r] - py[r]);
      total += w2 / S;
    }
    const float gate = (float)std::exp(-(total / nDirs) / gateTemp);

    std::vector<float> gated = obsEmb;
    for (float& x : gated) x *= gate;
    std::vector<float> enh;
    linear(gated.data(), S, "mpg.enhancement_proj.weight", "mpg.enhancement_proj.bias", enh);
    for (size_t i = 0; i < suffix.size(); ++i) suffix[i] += lambda * enh[i];

    // project the action rows back and overwrite the features
    linear(suffix.data() + (size_t)Dv, T, "vlm_to_action_proj.weight",
           "vlm_to_action_proj.bias", actionFeats);
  }

 private:
  std::vector<float> blob_;
  std::map<std::string, HostTensor> w_;
};

}  // namespace bllm
