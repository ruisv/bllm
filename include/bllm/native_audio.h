// bllm::AudioTower — the Qwen2.5-Omni audio encoder as an ordinary hbDNN graph,
// plus the Whisper-style log-mel front end it expects.
//
// The compiled `Audio.hbm` is STATIC and processes exactly ONE 2-second window:
//   in : mel      [1, 128, 200]  f32   log-mel, zero-padded on the right
//        valid    [1, 1, 200]    s32   1 = real mel frame, 0 = padding
//        attn     [1, 100, 100]  f32   additive mask over post-conv frames (0 | -1e4)
//   out: embeds   [50, 2048]     f32
// 200 mel frames -> conv stride 2 -> 100 -> avg-pool 2 -> 50 tokens. `n_window=100`
// in the config: the encoder attends only WITHIN a window and its positional
// embedding restarts each window, so long audio is just independent chunks — which
// is exactly why the graph could be compiled at a fixed 2 s.
//
// Two contract details established by probing the real graph:
//   * `valid` is boolean — only zero vs non-zero matters (all-1s and all-2s give a
//     bit-identical result).
//   * it does NOT gate the mel: conv leaks a garbage tail across the boundary
//     (cos 0.9925), so the padding must be ZEROED, in the MEL domain (matching HF,
//     which pads features with 0.0 — not the log floor a zero-PCM tail would give).
#pragma once

#include "bllm/native_graph.h"  // native_detail::Graph

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <complex>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct AudioSpec {
  int sample_rate = 16000;
  int n_fft = 400;
  int hop = 160;
  int n_mel = 128;
  int n_freq = 201;   // n_fft/2 + 1
};

// Raw 16 kHz mono PCM in [-1, 1]. The caller owns the samples.
struct AudioPCM {
  const float* samples = nullptr;
  size_t count = 0;
};

namespace audio_detail {

// transformers' mel_filter_bank(n_freq, n_mel, fmin, fmax, sr, norm="slaney",
// mel_scale="slaney") — the bank WhisperFeatureExtractor builds — as [n_mel][n_freq]
// row-major, the layout LogMel takes. In double, as numpy computes it.
inline std::vector<double> slaneyMelFilters(const AudioSpec& s, double fmin = 0.0,
                                            double fmax = -1.0) {
  if (fmax < 0) fmax = s.sample_rate / 2.0;
  const double logstep = std::log(6.4) / 27.0;           // slaney: linear below 1 kHz, log above
  auto toMel = [&](double f) { return f < 1000.0 ? 3.0 * f / 200.0 : 15.0 + std::log(f / 1000.0) / logstep; };
  auto toHz = [&](double m) { return m < 15.0 ? 200.0 * m / 3.0 : 1000.0 * std::exp(logstep * (m - 15.0)); };
  const int K = s.n_freq, M = s.n_mel;
  std::vector<double> fftHz(K), edge(M + 2);
  for (int k = 0; k < K; ++k) fftHz[k] = (double)(s.sample_rate / 2) * k / (K - 1);
  const double m0 = toMel(fmin), m1 = toMel(fmax);
  for (int i = 0; i < M + 2; ++i) edge[i] = toHz(m0 + (m1 - m0) * i / (M + 1));
  std::vector<double> fb((size_t)M * K);
  for (int m = 0; m < M; ++m) {
    const double lo = edge[m], mid = edge[m + 1], hi = edge[m + 2];
    const double enorm = 2.0 / (hi - lo);
    for (int k = 0; k < K; ++k) {
      const double down = (fftHz[k] - lo) / (mid - lo), up = (hi - fftHz[k]) / (hi - mid);
      fb[(size_t)m * K + k] = std::max(0.0, std::min(down, up)) * enorm;
    }
  }
  return fb;
}

// A complex FFT of one fixed length, any factorisation into 2/3/4/5 (n_fft = 400 is
// 4·4·5·5), in double — kissfft's recursive mixed-radix decimation in time. The direct DFT
// it replaces took 199 ms of an 11.4 s clip's front end; this is ~20x fewer multiply-adds
// per frame, with the same double precision the official runtime's features were matched at.
class FftD {
 public:
  using C = std::complex<double>;
  explicit FftD(int n) : n_(n), tw_(n) {
    for (int k = 0; k < n; ++k) tw_[k] = std::polar(1.0, -2.0 * M_PI * k / n);
    int m = n;
    for (int p : {4, 2, 3, 5}) {
      while (m % p == 0) { fac_.push_back(p); m /= p; }
    }
    if (m != 1) throw std::runtime_error("[audio] FFT length must factor into 2, 3, 4, 5");
    // stored as (p, m) pairs, outermost first
    std::vector<int> f;
    int rest = n;
    for (int p : fac_) { rest /= p; f.push_back(p); f.push_back(rest); }
    fac_ = f;
    scratch_.resize(8);
  }
  int size() const { return n_; }
  // out[k] = sum_i in[i] e^{-2 pi i k / n}
  void forward(const C* in, C* out) { work(out, in, 1, fac_.data()); }

 private:
  void work(C* out, const C* f, int fstride, const int* fac) {
    const int p = fac[0], m = fac[1];
    if (m == 1) {
      for (int q = 0; q < p; ++q) out[q] = f[(size_t)q * fstride];
    } else {
      for (int q = 0; q < p; ++q) work(out + (size_t)q * m, f + (size_t)q * fstride, fstride * p, fac + 2);
    }
    // Generic radix-p butterfly over m sub-transforms.
    if ((int)scratch_.size() < p) scratch_.resize(p);
    for (int u = 0; u < m; ++u) {
      for (int q = 0; q < p; ++q) scratch_[q] = out[u + q * m];
      for (int q1 = 0; q1 < p; ++q1) {
        const int k = u + q1 * m;
        // Written out: std::complex's operator* is a __muldc3 call without -ffast-math.
        double ar = scratch_[0].real(), ai = scratch_[0].imag();
        int twi = 0;
        const int step = fstride * k;
        for (int q = 1; q < p; ++q) {
          twi += step;
          if (twi >= n_) twi %= n_;
          const double xr = scratch_[q].real(), xi = scratch_[q].imag();
          const double wr = tw_[twi].real(), wi = tw_[twi].imag();
          ar += xr * wr - xi * wi;
          ai += xr * wi + xi * wr;
        }
        out[k] = C(ar, ai);
      }
    }
  }
  int n_;
  std::vector<C> tw_;
  std::vector<int> fac_;
  std::vector<C> scratch_;
};

// HF WhisperFeatureExtractor: hann window, |stft|², mel bank, log10, floor at
// (max - 8), then (x + 4) / 4. n_fft is 400 — not a power of two — hence FftD.
class LogMel {
 public:
  LogMel(const std::string& mel_filters_path, const AudioSpec& spec) : spec_(spec) {
    std::ifstream f(mel_filters_path);
    if (!f) throw std::runtime_error("[audio] cannot open mel filters: " + mel_filters_path);
    filt_.reserve((size_t)spec_.n_mel * spec_.n_freq);
    for (double v; f >> v;) filt_.push_back((double)(float)v);   // the file's float values
    init();
  }
  // The filterbank given directly, [n_mel][n_freq] row-major — a package that ships no
  // filter file builds it with audio_detail::slaneyMelFilters.
  LogMel(std::vector<double> filters, const AudioSpec& spec) : spec_(spec), filt_(std::move(filters)) {
    init();
  }

  // What lies past the last sample. HF pads a Whisper clip to 30 s, so zeros (the default,
  // and what Qwen2.5-Omni was matched against); a front end that takes the STFT of the clip
  // itself — the official Qwen3-ASR runtime, torch.stft(center=True) — reflects it.
  void set_reflect_end(bool on) { reflectEnd_ = on; }

 private:
  void init() {
    if (filt_.size() != (size_t)spec_.n_mel * spec_.n_freq)
      throw std::runtime_error("[audio] mel filters must be n_mel x n_freq (row-major)");

    win_.resize(spec_.n_fft);
    for (int i = 0; i < spec_.n_fft; ++i)  // periodic hann, as torch.hann_window
      win_[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / spec_.n_fft);

    fft_ = std::make_shared<FftD>(spec_.n_fft);
    // Each mel band is nonzero over a few bins only; the sum skips the zeros (adding an
    // exact 0 changes nothing, so this is the full sum, just shorter).
    band_.assign((size_t)2 * spec_.n_mel, 0);
    for (int m = 0; m < spec_.n_mel; ++m) {
      const double* fr = filt_.data() + (size_t)m * spec_.n_freq;
      int a = 0, b = 0;
      for (int k = 0; k < spec_.n_freq; ++k)
        if (fr[k] != 0.0) { if (b == 0) a = k; b = k + 1; }
      band_[2 * m] = a;
      band_[2 * m + 1] = b;
    }
  }

 public:
  int n_frames(size_t n_samples) const { return (int)(n_samples / spec_.hop); }

  // Returns [n_mel][T] row-major, T = n_samples / hop.
  std::vector<float> compute(const float* pcm, size_t n) const {
    const int F = spec_.n_fft, H = spec_.hop, K = spec_.n_freq, M = spec_.n_mel;
    const int pad = F / 2;
    const int T = n_frames(n);
    if (T < 1) throw std::runtime_error("[audio] clip too short");

    // torch.stft(center=True) reflect-pads the start; the end is zeros or a reflection, as
    // set_reflect_end says.
    std::vector<float> buf((size_t)pad + n + F, 0.0f);
    for (size_t i = 0; i < n; ++i) buf[pad + i] = pcm[i];
    for (int j = 0; j < pad; ++j) {
      const size_t src = (size_t)(pad - j);          // mirror about sample 0
      buf[j] = src < n ? pcm[src] : 0.0f;
    }
    if (reflectEnd_)
      for (int j = 1; j <= pad && (size_t)j < n; ++j)  // mirror about the last sample
        buf[pad + n - 1 + j] = pcm[n - 1 - j];

    // In double. A direct 400-term float sum loses the quiet bins to cancellation: against
    // the official Qwen3-ASR runtime's features (FFT-accurate) it was off by up to 21 LSB of
    // the graph's int16 input in 158 of 102400 values, which moved punctuation in the
    // transcripts; in double it matches numpy's float64 rfft.
    std::vector<double> mag((size_t)T * K);                 // frame-major
    std::vector<FftD::C> w(F), X(F);
    for (int t = 0; t < T; ++t) {
      const float* x = buf.data() + (size_t)t * H;
      for (int i = 0; i < F; ++i) w[i] = FftD::C((double)x[i] * win_[i], 0.0);
      fft_->forward(w.data(), X.data());
      double* row = mag.data() + (size_t)t * K;
      for (int k = 0; k < K; ++k) row[k] = X[k].real() * X[k].real() + X[k].imag() * X[k].imag();
    }

    std::vector<double> lg((size_t)M * T);
    double mx = -1e300;
    for (int m = 0; m < M; ++m) {
      const double* fr = filt_.data() + (size_t)m * K;
      const int k0 = band_[2 * m], k1 = band_[2 * m + 1];   // the band's nonzero bins
      for (int t = 0; t < T; ++t) {
        const double* row = mag.data() + (size_t)t * K;
        double acc = 0;
        for (int k = k0; k < k1; ++k) acc += fr[k] * row[k];
        const double v = std::log10(std::max(acc, 1e-10));
        lg[(size_t)m * T + t] = v;
        if (v > mx) mx = v;
      }
    }
    const double floor = mx - 8.0;
    std::vector<float> mel(lg.size());
    for (size_t i = 0; i < lg.size(); ++i) mel[i] = (float)((std::max(lg[i], floor) + 4.0) / 4.0);
    return mel;
  }

 private:
  AudioSpec spec_;
  std::vector<double> filt_, win_;
  std::shared_ptr<FftD> fft_;   // shared_ptr: compute() is const, and LogMel stays copyable
  std::vector<int> band_;       // [k0, k1) of each mel band's nonzero filter weights
  bool reflectEnd_ = false;
};

// HF Qwen2_5Omni _get_feat_extract_output_lengths, for one chunk of `mel` frames.
inline int cnnLen(int mel_len) { return (mel_len - 1) / 2 + 1; }
inline int tokenLen(int mel_len) {
  const int c = cnnLen(mel_len);
  return c < 2 ? 0 : (c - 2) / 2 + 1;
}

}  // namespace audio_detail

class AudioTower {
 public:
  AudioTower(const std::string& hbm, const std::string& mel_filters,
             const char* graph = "audio_tower")
      : mel_(mel_filters, spec_) {
    packed_.load({hbm});
    g_.init(packed_, graph);
    const auto& is = g_.inShape(0);
    const auto& os = g_.outShape(0);
    nMel_ = is.dimensionSize[is.numDimensions - 2];
    chunkFrames_ = is.dimensionSize[is.numDimensions - 1];
    cnnFrames_ = g_.inShape(2).dimensionSize[g_.inShape(2).numDimensions - 1];
    nToken_ = os.dimensionSize[os.numDimensions - 2];
    hidden_ = os.dimensionSize[os.numDimensions - 1];
    if (nMel_ != spec_.n_mel)
      throw std::runtime_error("[audio] graph mel bins != 128");
  }
  AudioTower(const AudioTower&) = delete;
  AudioTower& operator=(const AudioTower&) = delete;

  void set_sched(const native_detail::BpuSched& s) { g_.sched = s; }
  int hidden() const { return hidden_; }
  int chunk_frames() const { return chunkFrames_; }          // 200 (= 2 s)
  int tokens_per_chunk() const { return nToken_; }           // 50

  // How many decoder tokens a clip of `n_samples` will occupy.
  int token_count(size_t n_samples) const {
    int total = 0;
    for (int left = mel_.n_frames(n_samples); left > 0; left -= chunkFrames_)
      total += audio_detail::tokenLen(std::min(left, chunkFrames_));
    return total;
  }

  // 16 kHz mono PCM -> [n_token][hidden] embedding rows.
  std::vector<float> encode(const AudioPCM& pcm) {
    const std::vector<float> mel = mel_.compute(pcm.samples, pcm.count);
    const int T = mel_.n_frames(pcm.count);
    std::vector<float> out;
    for (int start = 0; start < T; start += chunkFrames_) {
      const int len = std::min(chunkFrames_, T - start);
      const int ntok = audio_detail::tokenLen(len);
      if (ntok <= 0) break;
      runChunk(mel.data(), T, start, len);
      const float* rows = (const float*)g_.outPtr(0);
      out.insert(out.end(), rows, rows + (size_t)ntok * hidden_);
    }
    return out;
  }

 private:
  // Fill one 2 s window: mel zero-padded on the right, plus the two masks.
  void runChunk(const float* mel, int T, int start, int len) {
    float* x = (float*)g_.inPtr(0);
    std::memset(x, 0, (size_t)nMel_ * chunkFrames_ * sizeof(float));
    for (int m = 0; m < nMel_; ++m)
      std::memcpy(x + (size_t)m * chunkFrames_, mel + (size_t)m * T + start, (size_t)len * sizeof(float));
    g_.inClean(0);

    int32_t* valid = (int32_t*)g_.inPtr(1);
    for (int t = 0; t < chunkFrames_; ++t) valid[t] = (t < len) ? 1 : 0;
    g_.inClean(1);

    const int cnn = audio_detail::cnnLen(len);
    float* attn = (float*)g_.inPtr(2);
    for (int i = 0; i < cnnFrames_; ++i)
      for (int j = 0; j < cnnFrames_; ++j)
        attn[(size_t)i * cnnFrames_ + j] = (i < cnn && j < cnn) ? 0.0f : -1e4f;
    g_.inClean(2);

    g_.infer();
  }

  native_detail::PackedHbm packed_;
  native_detail::Graph g_;
  AudioSpec spec_;
  audio_detail::LogMel mel_;
  int nMel_ = 0, chunkFrames_ = 0, cnnFrames_ = 0, nToken_ = 0, hidden_ = 0;
};

}  // namespace bllm
