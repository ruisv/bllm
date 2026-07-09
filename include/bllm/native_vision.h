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
#pragma once

#include "bllm/native_engine.h"  // native_detail::Graph

#include <cmath>
#include <cstdint>
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

}  // namespace vision_detail

struct VisionSpec {
  int image_size = 448;   // square side the graph was compiled for
  int patch = 14;
  int merge = 2;
  int temporal = 2;
  int channels = 3;
  float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

  int grid() const { return image_size / patch; }          // 32
  int llm_grid() const { return grid() / merge; }          // 16
  int n_patch() const { return grid() * grid(); }          // 1024
  int n_token() const { return llm_grid() * llm_grid(); }  // 256
  int row_len() const { return channels * temporal * patch * patch; }  // 1176
};

// Resize (bicubic, PIL semantics) + normalize an interleaved RGB8 image into a
// planar CHW float buffer of spec.image_size².
inline void preprocessToPlanar(const uint8_t* rgb, int w, int h, const VisionSpec& spec,
                               std::vector<float>& chw) {
  const int S = spec.image_size, C = spec.channels;
  chw.assign((size_t)C * S * S, 0.0f);
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

// Planar CHW (normalized) -> the graph's [n_patch, row_len] patch tensor.
inline void patchify(const std::vector<float>& chw, const VisionSpec& spec, float* out) {
  const int S = spec.image_size, C = spec.channels, P = spec.patch;
  const int M = spec.merge, T = spec.temporal, GB = spec.llm_grid();
  const int RL = spec.row_len();
  int r = 0;
  for (int bh = 0; bh < GB; ++bh)
    for (int bw = 0; bw < GB; ++bw)
      for (int mh = 0; mh < M; ++mh)
        for (int mw = 0; mw < M; ++mw, ++r) {
          float* row = out + (size_t)r * RL;
          int k = 0;
          for (int c = 0; c < C; ++c)
            for (int t = 0; t < T; ++t)  // a still image is repeated across frames
              for (int py = 0; py < P; ++py)
                for (int px = 0; px < P; ++px) {
                  const int Y = (bh * M + mh) * P + py;
                  const int X = (bw * M + mw) * P + px;
                  row[k++] = chw[((size_t)c * S + Y) * S + X];
                }
        }
}

// The vision encoder graph. encode() returns a pointer to n_token()×hidden floats
// owned by the tower (valid until the next encode()).
class VisionTower {
 public:
  explicit VisionTower(const std::string& hbm, const char* graph = "visual") {
    const char* files[] = {hbm.c_str()};
    BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&packed_, files, 1));
    g_.init(packed_, graph);
    const auto& is = g_.inShape(0);
    const auto& os = g_.outShape(0);
    nPatch_ = is.dimensionSize[is.numDimensions - 2];
    rowLen_ = is.dimensionSize[is.numDimensions - 1];
    nToken_ = os.dimensionSize[os.numDimensions - 2];
    hidden_ = os.dimensionSize[os.numDimensions - 1];
    if (g_.inType(0) != HB_DNN_TENSOR_TYPE_F32)
      throw std::runtime_error("[vision] expected an f32 patch input");
    if (nPatch_ != spec_.n_patch() || rowLen_ != spec_.row_len() || nToken_ != spec_.n_token())
      throw std::runtime_error("[vision] graph shape does not match the 448×448 spec");
  }
  ~VisionTower() { if (packed_) hbDNNRelease(packed_); }
  VisionTower(const VisionTower&) = delete;
  VisionTower& operator=(const VisionTower&) = delete;

  const VisionSpec& spec() const { return spec_; }
  int n_token() const { return nToken_; }
  int hidden() const { return hidden_; }

  // rgb: interleaved RGB8, w×h. Returns n_token()×hidden() embedding rows.
  const float* encode(const uint8_t* rgb, int w, int h) {
    std::vector<float> chw;
    preprocessToPlanar(rgb, w, h, spec_, chw);
    patchify(chw, spec_, (float*)g_.inPtr(0));
    g_.inClean(0);
    g_.infer();
    return (const float*)g_.outPtr(0);
  }

 private:
  hbDNNPackedHandle_t packed_ = nullptr;
  native_detail::Graph g_;
  VisionSpec spec_;
  int nPatch_ = 0, rowLen_ = 0, nToken_ = 0, hidden_ = 0;
};

}  // namespace bllm
