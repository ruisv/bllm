// Example-only video loading: shell out to ffmpeg to sample frames + soundtrack.
//
// The BLLM runtime deliberately knows nothing about containers — bllm::VideoClip
// takes already-decoded RGB8 frames and 16 kHz mono PCM, which is what a camera +
// microphone pipeline (or the board's hardware decoder) already produces. The demo
// CLIs need *some* way to open an .mp4, and ffmpeg is the least intrusive one.
#pragma once

#include "bllm/audio_io.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct DecodedVideo {
  std::vector<uint8_t> rgb;   // n_frames * height * width * 3, contiguous
  int width = 0, height = 0, n_frames = 0;
  double fps = 0;
  std::vector<float> audio;   // 16 kHz mono, empty if the file has no audio
};

namespace video_detail {

inline std::string shellQuote(const std::string& s) {
  std::string q = "'";
  for (char c : s) { if (c == '\'') q += "'\\''"; else q += c; }
  return q + "'";
}

inline std::string run(const std::string& cmd) {
  std::unique_ptr<FILE, int (*)(FILE*)> p(::popen(cmd.c_str(), "r"), ::pclose);
  if (!p) throw std::runtime_error("[bllm] popen failed: " + cmd);
  std::string out;
  char buf[512];
  while (std::fgets(buf, sizeof(buf), p.get())) out += buf;
  return out;
}

}  // namespace video_detail

// Sample `fps` frames per second, scaled to `size`x`size`, plus the soundtrack.
// `max_frames` caps how much of the clip is used (each PAIR of frames costs
// vision_tokens() of context, so long clips overflow the decoder window fast).
// ffmpeg does the square scale here, so VisionTower's own resize is a no-op; its
// bicubic is close to but not bit-identical with the PIL one HF uses.
inline DecodedVideo loadVideo(const std::string& path, double fps = 2.0, int size = 448,
                              int max_frames = 10, const std::string& ffmpeg = "ffmpeg") {
  using video_detail::run;
  using video_detail::shellQuote;

  DecodedVideo v;
  v.width = v.height = size;
  v.fps = fps;

  const std::string tmp_rgb = "/tmp/bllm_frames_" + std::to_string(::getpid()) + ".rgb";
  const std::string tmp_wav = "/tmp/bllm_audio_" + std::to_string(::getpid()) + ".wav";
  const std::string q = shellQuote(path);

  // Frames: fps sampling, letterbox-free square scale (matching the tower's 448²).
  const std::string vcmd = ffmpeg + " -hide_banner -loglevel error -y -i " + q +
                           " -vf fps=" + std::to_string(fps) + ",scale=" + std::to_string(size) +
                           ":" + std::to_string(size) + " -frames:v " + std::to_string(max_frames) +
                           " -pix_fmt rgb24 -f rawvideo " + tmp_rgb + " 2>&1";
  run(vcmd);
  FILE* f = std::fopen(tmp_rgb.c_str(), "rb");
  if (!f) throw std::runtime_error("[bllm] ffmpeg produced no frames for " + path);
  std::fseek(f, 0, SEEK_END);
  const size_t bytes = (size_t)std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  v.rgb.resize(bytes);
  if (std::fread(v.rgb.data(), 1, bytes, f) != bytes) { std::fclose(f); throw std::runtime_error("[bllm] short frame read"); }
  std::fclose(f);
  std::remove(tmp_rgb.c_str());
  const size_t frame_bytes = (size_t)size * size * 3;
  v.n_frames = (int)(bytes / frame_bytes);
  if (v.n_frames == 0) throw std::runtime_error("[bllm] no frames decoded from " + path);

  // Soundtrack (absent is fine — the clip then has no audio track).
  run(ffmpeg + " -hide_banner -loglevel error -y -i " + q + " -vn -ac 1 -ar 16000 -f wav " +
      tmp_wav + " 2>&1");
  FILE* a = std::fopen(tmp_wav.c_str(), "rb");
  if (a) {
    std::fclose(a);
    try { v.audio = loadAudio16k(tmp_wav).samples; } catch (...) {}
    std::remove(tmp_wav.c_str());
  }
  return v;
}

}  // namespace bllm
