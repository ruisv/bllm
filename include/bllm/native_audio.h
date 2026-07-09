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
// Two contract details established by probing the real graph (see docs/NATIVE_RUNTIME.md):
//   * `valid` is boolean — only zero vs non-zero matters (all-1s and all-2s give a
//     bit-identical result).
//   * it does NOT gate the mel: conv leaks a garbage tail across the boundary
//     (cos 0.9925), so the padding must be ZEROED, in the MEL domain (matching HF,
//     which pads features with 0.0 — not the log floor a zero-PCM tail would give).
#pragma once

#include "bllm/native_engine.h"  // native_detail::Graph

#include <algorithm>
#include <cmath>
#include <cstdint>
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

// HF WhisperFeatureExtractor: hann window, |stft|², mel bank, log10, floor at
// (max - 8), then (x + 4) / 4. n_fft is 400 — not a power of two — so a plain
// precomputed-table DFT is both simplest and fast enough (~160 k flops/frame).
class LogMel {
 public:
  LogMel(const std::string& mel_filters_path, const AudioSpec& spec) : spec_(spec) {
    std::ifstream f(mel_filters_path);
    if (!f) throw std::runtime_error("[audio] cannot open mel filters: " + mel_filters_path);
    filt_.reserve((size_t)spec_.n_mel * spec_.n_freq);
    for (double v; f >> v;) filt_.push_back((float)v);
    if (filt_.size() != (size_t)spec_.n_mel * spec_.n_freq)
      throw std::runtime_error("[audio] mel filters must be n_mel x n_freq (row-major)");

    win_.resize(spec_.n_fft);
    for (int i = 0; i < spec_.n_fft; ++i)  // periodic hann, as torch.hann_window
      win_[i] = 0.5f - 0.5f * (float)std::cos(2.0 * M_PI * i / spec_.n_fft);

    cos_.resize((size_t)spec_.n_freq * spec_.n_fft);
    sin_.resize((size_t)spec_.n_freq * spec_.n_fft);
    for (int k = 0; k < spec_.n_freq; ++k)
      for (int n = 0; n < spec_.n_fft; ++n) {
        const double a = -2.0 * M_PI * k * n / spec_.n_fft;
        cos_[(size_t)k * spec_.n_fft + n] = (float)std::cos(a);
        sin_[(size_t)k * spec_.n_fft + n] = (float)std::sin(a);
      }
  }

  int n_frames(size_t n_samples) const { return (int)(n_samples / spec_.hop); }

  // Returns [n_mel][T] row-major, T = n_samples / hop.
  std::vector<float> compute(const float* pcm, size_t n) const {
    const int F = spec_.n_fft, H = spec_.hop, K = spec_.n_freq, M = spec_.n_mel;
    const int pad = F / 2;
    const int T = n_frames(n);
    if (T < 1) throw std::runtime_error("[audio] clip too short");

    // torch.stft(center=True) reflect-pads; HF then zero-pads the clip to 30 s, so
    // everything past the end is zeros, not a reflection.
    std::vector<float> buf((size_t)pad + n + F, 0.0f);
    for (size_t i = 0; i < n; ++i) buf[pad + i] = pcm[i];
    for (int j = 0; j < pad; ++j) {
      const size_t src = (size_t)(pad - j);          // mirror about sample 0
      buf[j] = src < n ? pcm[src] : 0.0f;
    }

    std::vector<float> mag((size_t)K * T);
    std::vector<float> w(F);
    for (int t = 0; t < T; ++t) {
      const float* x = buf.data() + (size_t)t * H;
      for (int i = 0; i < F; ++i) w[i] = x[i] * win_[i];
      for (int k = 0; k < K; ++k) {
        const float* c = cos_.data() + (size_t)k * F;
        const float* s = sin_.data() + (size_t)k * F;
        float re = 0, im = 0;
        for (int i = 0; i < F; ++i) { re += w[i] * c[i]; im += w[i] * s[i]; }
        mag[(size_t)k * T + t] = re * re + im * im;
      }
    }

    std::vector<float> mel((size_t)M * T);
    float mx = -1e30f;
    for (int m = 0; m < M; ++m) {
      const float* fr = filt_.data() + (size_t)m * K;
      for (int t = 0; t < T; ++t) {
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += (double)fr[k] * mag[(size_t)k * T + t];
        float v = (float)std::log10(std::max(acc, 1e-10));
        mel[(size_t)m * T + t] = v;
        if (v > mx) mx = v;
      }
    }
    const float floor = mx - 8.0f;
    for (float& v : mel) v = (std::max(v, floor) + 4.0f) / 4.0f;
    return mel;
  }

 private:
  AudioSpec spec_;
  std::vector<float> filt_, win_, cos_, sin_;
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
    const char* files[] = {hbm.c_str()};
    BLLM_NATIVE_CK(hbDNNInitializeFromFiles(&packed_, files, 1));
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
  ~AudioTower() { if (packed_) hbDNNRelease(packed_); }
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

  hbDNNPackedHandle_t packed_ = nullptr;
  native_detail::Graph g_;
  AudioSpec spec_;
  audio_detail::LogMel mel_;
  int nMel_ = 0, chunkFrames_ = 0, cnnFrames_ = 0, nToken_ = 0, hidden_ = 0;
};

}  // namespace bllm
