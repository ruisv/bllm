// bllm::loadAudio16k — minimal WAV decoding for the CLI/demo paths.
//
// The runtime API (bllm::AudioTower / NativeVlm) takes 16 kHz mono float PCM, which
// is what a microphone pipeline hands you. This header exists so examples can also
// take a .wav path, via vendored dr_wav (public domain / MIT-0, no dependency).
//
// Anything that is not a WAV (mp4/mp3/…) should be transcoded first, e.g.
//   ffmpeg -i in.mp4 -ac 1 -ar 16000 -f wav out.wav
#pragma once

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct OwnedAudio {
  std::vector<float> samples;   // mono, 16 kHz, [-1, 1]
  int sample_rate = 16000;
  double seconds() const { return (double)samples.size() / sample_rate; }
};

// Decode a WAV to mono; linearly resample to `target_rate` if needed.
inline OwnedAudio loadAudio16k(const std::string& path, int target_rate = 16000) {
  unsigned channels = 0, rate = 0;
  drwav_uint64 frames = 0;
  float* pcm = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &rate, &frames, nullptr);
  if (!pcm) throw std::runtime_error("[bllm] cannot decode wav: " + path);

  std::vector<float> mono(frames);
  for (drwav_uint64 i = 0; i < frames; ++i) {
    double acc = 0;
    for (unsigned c = 0; c < channels; ++c) acc += pcm[i * channels + c];
    mono[i] = (float)(acc / channels);
  }
  drwav_free(pcm, nullptr);

  OwnedAudio out;
  out.sample_rate = target_rate;
  if ((int)rate == target_rate) { out.samples = std::move(mono); return out; }

  const double step = (double)rate / target_rate;
  const size_t n = (size_t)((double)mono.size() / step);
  out.samples.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const double sp = i * step;
    const size_t i0 = (size_t)sp;
    const size_t i1 = std::min(i0 + 1, mono.size() - 1);
    const float frac = (float)(sp - i0);
    out.samples[i] = mono[i0] * (1.0f - frac) + mono[i1] * frac;
  }
  return out;
}

}  // namespace bllm
