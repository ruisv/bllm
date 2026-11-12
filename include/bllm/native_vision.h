// bllm::VisionTower — the Qwen2.5-Omni / Qwen2.5-VL vision encoder as an ordinary
// hbDNN graph, plus the host-side image preprocessing it expects.
//
// The compiled `Visual.hbm` is STATIC: it takes exactly [n_patch, 3*T*P*P] f32
// patches and returns [n_patch/merge², hidden] embedding rows. For the shipped
// Qwen2.5-Omni-3B that is (1024, 1176) -> (256, 2048), i.e. a fixed 448×448 image
// (32×32 patches of 14 px, 2×2-merged into a 16×16 grid of 256 vision tokens).
//
// Preprocessing reproduces HF's Qwen2VLImageProcessor exactly:
//   PIL-compatible bicubic resize -> rescale 1/255 -> CLIP mean/std normalize
//   -> replicate to `temporal_patch_size` frames -> patchify in merge order.
// Patch row r = ((blk_h*grid_blk_w + blk_w)*merge + mh)*merge + mw, and within a
// row the layout is (channel, frame, py, px) — matching the HF transpose.
//
// The official D-Robotics LLM 2.0 towers (Qwen3-VL, Qwen2.5-VL) use the same patch
// layout but differ at both ends: the image is stretched with OpenCV's INTER_CUBIC, the
// patches go in as scaled int16, and Qwen3-VL's tower also returns its DeepStack maps.
#pragma once

#include "bllm/native_graph.h"  // native_detail::Graph

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {
namespace vision_detail {

// PIL's bicubic kernel (Catmull-Rom, a = -0.5).
inline double bicubic(double x) {
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return ((x - 5.0) * a * x + 8.0 * a) * x - 4.0 * a;
  return 0.0;
}

// Per-output-pixel filter taps, matching PIL's precompute_coeffs(): the kernel is
// stretched by the downscale factor so shrinking antialiases.
struct Coeffs {
  int ksize = 0;
  std::vector<int> start;   // outLen
  std::vector<int> count;   // outLen
  std::vector<double> k;    // outLen * ksize
};

inline Coeffs makeCoeffs(int inLen, int outLen) {
  const double scale = (double)inLen / (double)outLen;
  const double filterscale = std::max(1.0, scale);
  const double support = 2.0 * filterscale;
  Coeffs c;
  c.ksize = (int)std::ceil(support) * 2 + 1;
  c.start.resize(outLen); c.count.resize(outLen); c.k.assign((size_t)outLen * c.ksize, 0.0);
  const double ss = 1.0 / filterscale;
  for (int xx = 0; xx < outLen; ++xx) {
    const double center = (xx + 0.5) * scale;
    int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
    int xmax = (int)(center + support + 0.5); if (xmax > inLen) xmax = inLen;
    xmax -= xmin;
    double* kk = c.k.data() + (size_t)xx * c.ksize;
    double sum = 0.0;
    for (int x = 0; x < xmax; ++x) { kk[x] = bicubic((x + xmin - center + 0.5) * ss); sum += kk[x]; }
    if (sum != 0.0) for (int x = 0; x < xmax; ++x) kk[x] /= sum;
    c.start[xx] = xmin; c.count[xx] = xmax;
  }
  return c;
}

// OpenCV's cv::resize(INTER_CUBIC) on interleaved 8-bit images, reproduced to the bit as
// OpenCV 4.x/5.x compute it on aarch64. The official D-Robotics LLM 2.0 runtime
// preprocesses with it, and pixel-identical input is what lets a VLM reply be
// token-identical too. Separable, as OpenCV's generic path is:
//   * taps: a = -0.75 Keys cubic at fx = (dx + 0.5) * scale - 0.5, rounded to Q11 shorts;
//     out-of-range source pixels clamp to the edge;
//   * horizontal pass: exact integer sum of pixel * coefficient (Q11);
//   * vertical pass: OpenCV's NEON kernel (VResizeCubicVec_32s8u) converts the four Q11
//     row sums to float, scales the coefficients by 2^-22, chains fused multiply-adds and
//     rounds half to even. Only the last (width mod 8) elements of a row take the scalar
//     fixed-point path, which rounds half up instead — mirrored here too.
inline void cubicTaps(int inLen, int outLen, std::vector<int>& ofs, std::vector<int16_t>& k) {
  const double scale = 1.0 / ((double)outLen / inLen);      // as cv::resize derives it
  ofs.resize(outLen); k.resize((size_t)outLen * 4);
  for (int d = 0; d < outLen; ++d) {
    float f = (float)((d + 0.5) * scale - 0.5);
    const int s = (int)std::floor(f);
    f -= (float)s;
    const float A = -0.75f;
    float c[4];
    c[0] = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A;
    c[1] = ((A + 2) * f - (A + 3)) * f * f + 1;
    c[2] = ((A + 2) * (1 - f) - (A + 3)) * (1 - f) * (1 - f) + 1;
    c[3] = 1.f - c[0] - c[1] - c[2];
    ofs[d] = s - 1;                                          // first of the four taps
    for (int j = 0; j < 4; ++j) {
      const long v = std::lrint(c[j] * 2048.0f);
      k[(size_t)d * 4 + j] = (int16_t)std::max(-32768L, std::min(32767L, v));
    }
  }
}

inline void resizeCubicCv(const uint8_t* src, int sw, int sh, int cn, uint8_t* dst, int dw,
                          int dh) {
  if (sw == dw && sh == dh) { std::memcpy(dst, src, (size_t)sw * sh * cn); return; }
  std::vector<int> xo, yo;
  std::vector<int16_t> xk, yk;
  cubicTaps(sw, dw, xo, xk);
  cubicTaps(sh, dh, yo, yk);
  const int W = dw * cn;
  // Horizontal pass, once per source row that any output row reads.
  std::vector<int> hrow((size_t)sh * W);
  std::vector<char> done(sh, 0);
  auto hpass = [&](int sy) {
    if (done[sy]) return;
    done[sy] = 1;
    const uint8_t* S = src + (size_t)sy * sw * cn;
    int* D = hrow.data() + (size_t)sy * W;
    for (int dx = 0; dx < dw; ++dx) {
      const int16_t* a = &xk[(size_t)dx * 4];
      int px[4];
      for (int j = 0; j < 4; ++j) px[j] = std::min(std::max(xo[dx] + j, 0), sw - 1);
      for (int c = 0; c < cn; ++c) {
        int v = 0;
        for (int j = 0; j < 4; ++j) v += (int)S[px[j] * cn + c] * a[j];
        D[dx * cn + c] = v;
      }
    }
  };
  const int vecEnd = W - W % 8;                              // 8 int16 lanes per NEON pass
  const float q = 1.0f / (2048.0f * 2048.0f);
  for (int dy = 0; dy < dh; ++dy) {
    const int* R[4];
    for (int j = 0; j < 4; ++j) {
      const int sy = std::min(std::max(yo[dy] + j, 0), sh - 1);
      hpass(sy);
      R[j] = hrow.data() + (size_t)sy * W;
    }
    const int16_t* b = &yk[(size_t)dy * 4];
    const float b0 = b[0] * q, b1 = b[1] * q, b2 = b[2] * q, b3 = b[3] * q;
    uint8_t* D = dst + (size_t)dy * W;
    for (int x = 0; x < vecEnd; ++x) {
      const float r = std::fma((float)R[0][x], b0,
                      std::fma((float)R[1][x], b1,
                      std::fma((float)R[2][x], b2, (float)R[3][x] * b3)));
      const long v = std::lrint(r);                          // half to even, as vcvtnq
      D[x] = (uint8_t)std::max(0L, std::min(255L, v));
    }
    for (int x = vecEnd; x < W; ++x) {
      const int v = (R[0][x] * b[0] + R[1][x] * b[1] + R[2][x] * b[2] + R[3][x] * b[3] +
                     (1 << 21)) >> 22;
      D[x] = (uint8_t)std::max(0, std::min(255, v));
    }
  }
}

}  // namespace vision_detail

// The square side the tower was compiled for is not a constant of the model — it is
// baked into the graph when the vision recipe is exported (the official Omni tower is
// 448², but a re-export at 224² costs 4x fewer decoder tokens per frame). So derive it
// from the graph's patch count instead of hardcoding it.
// Patch geometry and pixel normalization are NOT universal — they are per-tower
// constants that must come from the checkpoint that built the graph:
//   Qwen2.5-Omni / 2.5-VL : patch 14, CLIP mean/std   (the defaults below)
//   Qwen3.5               : patch 16, mean = std = 0.5
// Getting them wrong produces a graph-shaped tensor full of wrong numbers, which
// no shape check would catch — so model.json carries them (see ModelConfig::vision).
// Which order the graph expects its patch rows in. Qwen2VL-family towers want the
// 2x2 merge block contiguous because the merger fuses adjacent rows; InternViT
// (Being-H0.5) wants plain raster order because its pixel-shuffle happens inside
// the graph, on the token grid. Same patch content, different row permutation —
// and feeding the wrong one produces a correctly shaped tensor of wrong numbers.
enum class PatchOrder { Merged, RowMajor };

// How the image is brought to the tower's side. HF's processors resize with PIL bicubic
// (a = -0.5, antialiased when shrinking), which is what our own towers were checked
// against. The official LLM 2.0 runtime stretches with OpenCV INTER_CUBIC (a = -0.75, no
// antialiasing) instead — measured against its own tensor dump, every pixel equal — so its
// packages say so and get the same pixels.
enum class ResizeKind { PilBicubic, CvCubic };

struct VisionSpec {
  int image_size = 448;   // square side the graph was compiled for
  int patch = 14;
  int merge = 2;
  int temporal = 2;
  int channels = 3;
  PatchOrder order = PatchOrder::Merged;
  ResizeKind resize = ResizeKind::PilBicubic;
  // >0: the tower returns its tokens grouped in out_window × out_window windows (Qwen2.5-VL's
  // window-attention order; the official export leaves out HF's final reverse permutation)
  // and they are put back in raster order here. In merged tokens: 112 px / (14 * 2) = 4.
  int out_window = 0;
  float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

  int grid() const { return image_size / patch; }          // 448 -> 32
  int llm_grid() const { return grid() / merge; }          // 448 -> 16
  int n_patch() const { return grid() * grid(); }          // 448 -> 1024
  int n_token() const { return llm_grid() * llm_grid(); }  // 448 -> 256
  int row_len() const { return channels * temporal * patch * patch; }  // 1176
};

// Resize (bicubic, PIL semantics) + normalize an interleaved RGB8 image into a
// planar CHW float buffer of spec.image_size².
inline void preprocessToPlanar(const uint8_t* rgb, int w, int h, const VisionSpec& spec,
                               std::vector<float>& chw) {
  const int S = spec.image_size, C = spec.channels;
  chw.assign((size_t)C * S * S, 0.0f);
  if (spec.resize == ResizeKind::CvCubic) {
    // Normalised in float as (p * (1/255) - mean) * (1/std) — the form that reproduces the
    // official runtime's int16 input for all 3x256 values, CLIP normalisation included
    // (dividing by std instead moves a few exact .5 ties). Checked on its tensor dumps.
    std::vector<uint8_t> px((size_t)S * S * C);
    vision_detail::resizeCubicCv(rgb, w, h, C, px.data(), S, S);
    for (int c = 0; c < C; ++c) {
      float lut[256];
      const float inv = 1.0f / spec.std[c];
      for (int v = 0; v < 256; ++v) lut[v] = ((float)v * (1.0f / 255.0f) - spec.mean[c]) * inv;
      float* dst = chw.data() + (size_t)c * S * S;
      for (size_t i = 0; i < (size_t)S * S; ++i) dst[i] = lut[px[i * C + c]];
    }
    return;
  }
  const auto cx = vision_detail::makeCoeffs(w, S);
  const auto cy = vision_detail::makeCoeffs(h, S);

  // PIL resamples 8-bit images in two 8-bit passes; round+clip after each so the
  // patch tensor matches HF bit-for-bit-ish (cos > 0.9999).
  auto clip8 = [](double v) { v = std::rint(v); return v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v); };

  std::vector<float> tmp((size_t)C * h * S);  // horizontal pass: [C][h][S]
  for (int c = 0; c < C; ++c)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < S; ++x) {
        const double* kk = cx.k.data() + (size_t)x * cx.ksize;
        double acc = 0.0;
        for (int i = 0; i < cx.count[x]; ++i)
          acc += kk[i] * (double)rgb[((size_t)y * w + cx.start[x] + i) * C + c];
        tmp[((size_t)c * h + y) * S + x] = (float)clip8(acc);
      }

  for (int c = 0; c < C; ++c)
    for (int y = 0; y < S; ++y) {
      const double* kk = cy.k.data() + (size_t)y * cy.ksize;
      for (int x = 0; x < S; ++x) {
        double acc = 0.0;
        for (int i = 0; i < cy.count[y]; ++i)
          acc += kk[i] * (double)tmp[((size_t)c * h + cy.start[y] + i) * S + x];
        const double v = clip8(acc);
        chw[((size_t)c * S + y) * S + x] = (float)((v / 255.0 - spec.mean[c]) / spec.std[c]);
      }
    }
}

// Planar CHW frames (normalized) -> the graph's [n_patch, row_len] patch tensor.
// `frames` holds spec.temporal planes; a still image passes the same one twice,
// a video passes two consecutive frames (that is what temporal_patch_size means).
inline void patchify(const std::vector<float>* const* frames, const VisionSpec& spec, float* out) {
  const int S = spec.image_size, C = spec.channels, P = spec.patch;
  const int M = spec.merge, T = spec.temporal, GB = spec.llm_grid();
  const int RL = spec.row_len();
  const int G = spec.grid();
  auto emit = [&](float* row, int gy, int gx) {
    int k = 0;
    for (int c = 0; c < C; ++c)
      for (int t = 0; t < T; ++t)
        for (int py = 0; py < P; ++py)
          for (int px = 0; px < P; ++px)
            row[k++] = (*frames[t])[((size_t)c * S + gy * P + py) * S + gx * P + px];
  };

  if (spec.order == PatchOrder::RowMajor) {
    for (int gy = 0, r = 0; gy < G; ++gy)
      for (int gx = 0; gx < G; ++gx, ++r) emit(out + (size_t)r * RL, gy, gx);
    return;
  }
  int r = 0;
  for (int bh = 0; bh < GB; ++bh)
    for (int bw = 0; bw < GB; ++bw)
      for (int mh = 0; mh < M; ++mh)
        for (int mw = 0; mw < M; ++mw, ++r)
          emit(out + (size_t)r * RL, bh * M + mh, bw * M + mw);
}

// The vision encoder graph. encode()/encode_pair() return a pointer to
// n_token()×hidden floats owned by the tower (valid until the next call).
class VisionTower {
 public:
  // `spec` supplies the tower's own constants (patch size, merge, pixel
  // normalization); image_size is still DERIVED from the graph, so a re-export at
  // another resolution needs no manifest edit.
  explicit VisionTower(const std::string& hbm, const char* graph = "visual",
                       const VisionSpec& spec = {})
      : spec_(spec) {
    packed_.load({hbm});
    g_.init(packed_, graph);
    const auto& is = g_.inShape(0);
    const auto& os = g_.outShape(0);
    nPatch_ = is.dimensionSize[is.numDimensions - 2];
    rowLen_ = is.dimensionSize[is.numDimensions - 1];
    nToken_ = os.dimensionSize[os.numDimensions - 2];
    hidden_ = os.dimensionSize[os.numDimensions - 1];
    // Our towers take f32 patches; the official LLM 2.0 ones scaled int16, quantised on
    // the way in (writeRowsF32As divides by the input scale, as their runtime does).
    inType_ = g_.inType(0);
    if (inType_ != HB_DNN_TENSOR_TYPE_F32 && inType_ != HB_DNN_TENSOR_TYPE_S16)
      throw std::runtime_error("[vision] expected an f32 or scaled-s16 patch input");
    // Qwen3-VL also returns its DeepStack features — one [n_token, hidden] map per tapped
    // ViT layer, added into the decoder's early layers at the image positions. Found by
    // name, kept in index order.
    for (int i = 1; i < g_.outCount(); ++i) {
      const std::string& n = g_.outName(i);
      static const std::string kDeep = "deepstack_features_";
      if (n.compare(0, kDeep.size(), kDeep) != 0)
        throw std::runtime_error("[vision] unexpected extra tower output '" + n + "'");
      const int k = std::atoi(n.c_str() + kDeep.size());
      if (k != (int)deepOut_.size())
        throw std::runtime_error("[vision] deepstack outputs out of order at '" + n + "'");
      const auto& ds = g_.outShape(i);
      if (ds.dimensionSize[ds.numDimensions - 2] != nToken_ ||
          ds.dimensionSize[ds.numDimensions - 1] != hidden_)
        throw std::runtime_error("[vision] '" + n + "' is not [n_token, hidden]");
      deepOut_.push_back(i);
    }
    deep_.resize(deepOut_.size());
    if (spec_.out_window > 0) buildWindowPerm();

    // The rows are spliced straight into the decoder's embedding stream, so they
    // have to BE floats. A tower whose final linear quantizes its output comes back
    // as scaled int16 — reading that as float silently produces garbage rows and a
    // fluent, wrong answer, with every shape check still passing. Dequantize
    // instead, and refuse anything neither f32 nor scaled s16.
    outType_ = g_.outType(0);
    if (outType_ == HB_DNN_TENSOR_TYPE_S16) {
      outScale_ = g_.outScale(0);
      if (!(outScale_ > 0.0f))
        throw std::runtime_error("[vision] s16 tower output carries no scale");
    } else if (outType_ != HB_DNN_TENSOR_TYPE_F32) {
      throw std::runtime_error("[vision] unsupported tower output tensor type " +
                               std::to_string(outType_));
    }

    // grid² = n_patch, and the merger turns each merge² block into one token.
    const int grid = (int)std::lround(std::sqrt((double)nPatch_));
    if (grid * grid != nPatch_)
      throw std::runtime_error("[vision] non-square patch grid: " + std::to_string(nPatch_));
    spec_.image_size = grid * spec_.patch;
    if (rowLen_ != spec_.row_len() || nToken_ != spec_.n_token())
      throw std::runtime_error("[vision] graph shape is inconsistent: " + std::to_string(nPatch_) +
                               "x" + std::to_string(rowLen_) + " -> " + std::to_string(nToken_) +
                               " (derived " + std::to_string(spec_.image_size) + "px, expected " +
                               std::to_string(spec_.n_token()) + " tokens)");
  }
  VisionTower(const VisionTower&) = delete;
  VisionTower& operator=(const VisionTower&) = delete;

  void set_sched(const native_detail::BpuSched& s) { g_.sched = s; }
  const VisionSpec& spec() const { return spec_; }
  int image_size() const { return spec_.image_size; }
  int n_token() const { return nToken_; }
  int hidden() const { return hidden_; }
  // DeepStack maps the tower emits besides its features (3 for Qwen3-VL, else 0), and
  // map k of the last encode: n_token()×hidden floats, valid until the next call.
  int deepstack() const { return (int)deepOut_.size(); }
  const float* deepstack_rows(int k) const { return deep_.at(k).data(); }

  // A scaled-int16 tower feeding a scaled-int16 decoder input (the official 2.0 VLMs):
  // requantise each value as rint(q * (s_out / s_in)), the ratio taken in float — the
  // official runtime's formula, matched on its tensor dumps for the features and all
  // three DeepStack maps (dequantise-then-divide differs on ~1 value in 600 when the
  // two scales differ, as on Qwen3-VL-4B). The rows come back as t * s_in, which the
  // decoder's own rint(v / s_in) maps back to exactly t. A zero scale leaves that map
  // dequantised as before.
  void set_target_scales(float features, const std::vector<float>& deep) {
    if (outType_ != HB_DNN_TENSOR_TYPE_S16) return;
    if (deep.size() != deepOut_.size())
      throw std::runtime_error("[vision] one target scale per DeepStack map");
    tgt_.assign(1 + deep.size(), 0.0f);
    tgt_[0] = features;
    for (size_t k = 0; k < deep.size(); ++k) tgt_[1 + k] = deep[k];
  }

  // A still image: the same frame fills both temporal slots, as HF does.
  const float* encode(const uint8_t* rgb, int w, int h) { return encode_pair(rgb, rgb, w, h); }

  // An already-patchified tensor ([n_patch, row_len], HF's `pixel_values` layout).
  // Bypasses our resize/normalize/patchify entirely — the seam that lets a board
  // test tell a quantization problem apart from a preprocessing one.
  const float* encode_patches(const float* patches) {
    if (inType_ == HB_DNN_TENSOR_TYPE_F32) g_.writeRowsF32(0, patches, nPatch_, rowLen_);
    else g_.writeRowsF32As(0, patches, nPatch_, rowLen_);
    g_.infer();
    for (size_t k = 0; k < deepOut_.size(); ++k) {
      deep_[k].resize((size_t)nToken_ * hidden_);
      if (tgt_.size() > k + 1 && tgt_[k + 1] > 0.0f)
        readRequant(deepOut_[k], deep_[k].data(), tgt_[k + 1]);
      else
        g_.readRowsF32As(deepOut_[k], deep_[k].data(), nToken_, hidden_);
      if (!perm_.empty()) { toRaster(deep_[k].data(), win_); deep_[k].swap(win_); }
    }
    const float* r = rows();
    if (perm_.empty()) return r;
    toRaster(r, win_);
    return win_.data();
  }

  // Two consecutive video frames (same size) -> one temporal grid step of tokens.
  const float* encode_pair(const uint8_t* rgb0, const uint8_t* rgb1, int w, int h) {
    // Patchify densely, then hand it to writeRowsF32: the graph's rows may be
    // padded up to kAlign and writing straight into inPtr assumes they are not.
    patches_.resize((size_t)nPatch_ * rowLen_);
    preprocessToPlanar(rgb0, w, h, spec_, chw0_);
    if (rgb1 == rgb0) {
      const std::vector<float>* planes[2] = {&chw0_, &chw0_};
      patchify(planes, spec_, patches_.data());
    } else {
      preprocessToPlanar(rgb1, w, h, spec_, chw1_);
      const std::vector<float>* planes[2] = {&chw0_, &chw1_};
      patchify(planes, spec_, patches_.data());
    }
    return encode_patches(patches_.data());
  }

 private:
  // The graph's rows as floats, dequantizing if the tower emits scaled int16.
  // Output `idx` (scaled int16) requantised to `target` as set_target_scales describes.
  void readRequant(int idx, float* dst, float target) const {
    const int64_t pitch = g_.outRowPitch(idx, hidden_);
    const auto* base = (const uint8_t*)g_.out[idx].sysMem.virAddr;
    const float ratio = native_detail::Graph::scaleOf(g_.out[idx].properties) / target;
    for (int i = 0; i < nToken_; ++i) {
      const int16_t* q = (const int16_t*)(base + (int64_t)i * pitch);
      float* d = dst + (size_t)i * hidden_;
      for (int k = 0; k < hidden_; ++k) {
        const float t = std::rint((float)q[k] * ratio);
        d[k] = std::max(-32768.0f, std::min(32767.0f, t)) * target;
      }
    }
  }

  const float* rows() {
    if (!tgt_.empty() && tgt_[0] > 0.0f) {
      deq_.resize((size_t)nToken_ * hidden_);
      readRequant(0, deq_.data(), tgt_[0]);
      return deq_.data();
    }
    // Rank-aware: the row axis is numDimensions-2 (a [1, n_token, hidden] tower
    // is normal), so dimension 0's stride is the whole tensor, not a row.
    const int64_t pitch = g_.outRowPitch(0, hidden_);
    const auto* base = (const uint8_t*)g_.outPtr(0);
    const size_t n = (size_t)nToken_ * hidden_;
    // Only hand back the raw pointer when the rows really are contiguous; a
    // hidden size whose byte length misses kAlign is padded, and reading it as
    // dense silently shifts every row but the first.
    if (outType_ == HB_DNN_TENSOR_TYPE_F32 && pitch == (int64_t)hidden_ * 4)
      return (const float*)base;
    deq_.resize(n);
    for (int i = 0; i < nToken_; ++i) {
      if (outType_ == HB_DNN_TENSOR_TYPE_F32) {
        std::memcpy(deq_.data() + (size_t)i * hidden_, base + (int64_t)i * pitch,
                    (size_t)hidden_ * sizeof(float));
      } else {
        const int16_t* q = (const int16_t*)(base + (int64_t)i * pitch);
        for (int k = 0; k < hidden_; ++k) deq_[(size_t)i * hidden_ + k] = (float)q[k] * outScale_;
      }
    }
    return deq_.data();
  }

  // HF Qwen2.5-VL get_window_index for one image: windows of w×w tokens, windows and the
  // tokens inside each in raster order, a partial window at the right/bottom edge. Tower
  // row j is raster token window_index[j]; perm_[raster token] = its tower row.
  void buildWindowPerm() {
    const int G = spec_.llm_grid(), w = spec_.out_window, nw = (G + w - 1) / w;
    if (G * G != nToken_)
      throw std::runtime_error("[vision] window order needs a square token grid");
    perm_.assign((size_t)nToken_, -1);
    int j = 0;
    for (int wy = 0; wy < nw; ++wy)
      for (int wx = 0; wx < nw; ++wx)
        for (int r = 0; r < w; ++r)
          for (int c = 0; c < w; ++c) {
            const int y = wy * w + r, x = wx * w + c;
            if (y < G && x < G) perm_[(size_t)y * G + x] = j++;
          }
  }
  void toRaster(const float* src, std::vector<float>& dst) const {
    dst.resize((size_t)nToken_ * hidden_);
    for (int i = 0; i < nToken_; ++i)
      std::memcpy(dst.data() + (size_t)i * hidden_, src + (size_t)perm_[i] * hidden_,
                  (size_t)hidden_ * sizeof(float));
  }

  native_detail::PackedHbm packed_;
  native_detail::Graph g_;
  VisionSpec spec_;
  std::vector<int> perm_;                    // out_window: raster token -> tower row
  std::vector<float> tgt_;                   // decoder input scales: features, then maps
  std::vector<float> win_;                   // rows put back in raster order
  std::vector<float> chw0_, chw1_, deq_, patches_;
  std::vector<int> deepOut_;                 // output index of deepstack map k
  std::vector<std::vector<float>> deep_;     // their rows from the last encode
  int nPatch_ = 0, rowLen_ = 0, nToken_ = 0, hidden_ = 0;
  int inType_ = 0, outType_ = 0;
  float outScale_ = 1.0f;
};

}  // namespace bllm
