// bllm_vlm_native — image+text chat on the self-built native runtime (no libxlm).
//
//   ./bllm_vlm_native --model /models/qwen2.5-omni-3b --image cat.jpg \
//                     --prompt "描述这张图片" [--max-new 128] [--temp 0.7]
//   ./bllm_vlm_native --model ... --audio clip.wav --prompt "听到了什么？"
//   ./bllm_vlm_native --model ... --video clip.mp4 [--fps 2] [--max-frames 10] [--mute]
//
// With no media it is a plain text chat, which is the cheapest way to check that
// the host embedding + mrope + mask plumbing matches the graph's expectations.
// Audio must be a WAV (any rate/channels). --video shells out to ffmpeg to sample
// frames + the soundtrack; by default the soundtrack is fed too (Qwen's
// `use_audio_in_video`), interleaved with the frames in 2-second chunks.
#include "bllm/audio_io.h"
#include "bllm/image_io.h"
#include "bllm/native_vlm.h"
#include "video_io.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* argOf(int argc, char** argv, const char* flag, const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model = argOf(argc, argv, "--model");
  if (!model) {
    std::fprintf(stderr,
                 "usage: %s --model <dir> [--image <f>]... [--audio <f.wav>]... [--video <f>]...\n"
                 "          [--prompt <text>] [--max-new N] [--temp T] [--top-p P] [--top-k K]\n"
                 "          [--seed S] [--fps F] [--max-frames N] [--mute]\n",
                 argv[0]);
    return 2;
  }
  const std::string prompt = argOf(argc, argv, "--prompt", "描述一下这张图片。");
  const int max_new = std::atoi(argOf(argc, argv, "--max-new", "128"));
  const float temp = (float)std::atof(argOf(argc, argv, "--temp", "0"));
  const float top_p = (float)std::atof(argOf(argc, argv, "--top-p", "1"));
  const int top_k = std::atoi(argOf(argc, argv, "--top-k", "0"));
  const uint64_t seed = (uint64_t)std::atoll(argOf(argc, argv, "--seed", "1234"));
  const double fps = std::atof(argOf(argc, argv, "--fps", "2"));
  const int max_frames = std::atoi(argOf(argc, argv, "--max-frames", "10"));
  bool mute = false;

  std::vector<std::string> image_paths, audio_paths, video_paths;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mute") == 0) mute = true;
    if (i + 1 >= argc) continue;
    if (std::strcmp(argv[i], "--image") == 0) image_paths.emplace_back(argv[i + 1]);
    if (std::strcmp(argv[i], "--audio") == 0) audio_paths.emplace_back(argv[i + 1]);
    if (std::strcmp(argv[i], "--video") == 0) video_paths.emplace_back(argv[i + 1]);
  }

  try {
    bllm::NativeVlm vlm(model);
    vlm.set_sampling(temp, top_p, top_k, 1.0f, seed);

    std::vector<bllm::OwnedImage> owned;
    std::vector<bllm::ImageRGB> images;
    owned.reserve(image_paths.size());
    for (const std::string& p : image_paths) {
      owned.push_back(bllm::loadImageRGB(p));
      std::fprintf(stderr, "[image] %s  %dx%d -> %d vision tokens\n", p.c_str(),
                   owned.back().width, owned.back().height, vlm.vision_tokens());
    }
    for (const auto& o : owned) images.push_back({o.rgb.data(), o.width, o.height});

    std::vector<bllm::OwnedAudio> clips;
    std::vector<bllm::AudioPCM> audios;
    clips.reserve(audio_paths.size());
    for (const std::string& p : audio_paths) {
      clips.push_back(bllm::loadAudio16k(p));
      std::fprintf(stderr, "[audio] %s  %.2f s @ 16 kHz mono\n", p.c_str(), clips.back().seconds());
    }
    for (const auto& c : clips) audios.push_back({c.samples.data(), c.samples.size()});

    std::vector<bllm::DecodedVideo> decoded;
    std::vector<bllm::VideoClip> videos;
    decoded.reserve(video_paths.size());
    for (const std::string& p : video_paths) {
      decoded.push_back(bllm::loadVideo(p, fps, 448, max_frames));
      const auto& d = decoded.back();
      const int pairs = (d.n_frames + 1) / 2;
      std::fprintf(stderr, "[video] %s  %d frames @ %.1f fps -> %d vision tokens%s\n",
                   p.c_str(), d.n_frames, d.fps, pairs * vlm.vision_tokens(),
                   (!mute && !d.audio.empty()) ? " + soundtrack" : "");
    }
    for (const auto& d : decoded) {
      bllm::VideoClip v;
      v.fps = d.fps;
      const size_t stride = (size_t)d.width * d.height * 3;
      for (int i = 0; i < d.n_frames; ++i) v.frames.push_back({d.rgb.data() + i * stride, d.width, d.height});
      if (!mute && !d.audio.empty()) v.audio = {d.audio.data(), d.audio.size()};
      videos.push_back(std::move(v));
    }

    std::cout << "> " << prompt << "\n";
    vlm.chat(prompt, images, audios, videos, max_new, [](const std::string& s) {
      std::cout << s << std::flush;
    });
    std::cout << "\n";
    std::fprintf(stderr, "[stats] ttft %.0f ms, decode %.1f tok/s\n",
                 vlm.last_ttft_ms(), vlm.last_decode_tps());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
