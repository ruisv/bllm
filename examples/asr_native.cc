// Speech recognition with an official D-Robotics Qwen3-ASR package (bllm::NativeAsr).
//
//   bllm_asr_native <model_dir> <audio.wav> [language] [context]
//
// <model_dir> is an arch="asr" dir: what `bllm.load()` synthesizes from the unzipped
// package, or `bllm-make-model-dir official <out> Qwen3-ASR-0.6B_w8/`. language is a name
// ("Chinese") or a code ("zh"); leave it out (or pass "") and the model names it.
#include "bllm/audio_io.h"
#include "bllm/native_asr.h"

#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model_dir> <audio.wav> [language] [context]\n", argv[0]);
    return 2;
  }
  try {
    bllm::NativeAsr asr(argv[1]);
    const bllm::OwnedAudio audio = bllm::loadAudio16k(argv[2]);
    bllm::AsrOptions opt;
    if (argc > 3) opt.language = argv[3];
    if (argc > 4) opt.context = argv[4];
    const bllm::AsrResult r = asr.transcribe(
        audio.samples.data(), audio.samples.size(), opt,
        [](const std::string& piece) { std::cout << piece << std::flush; });
    std::printf("\n[%s] %.1f s of audio in %.0f ms (RTF %.3f, first token %.0f ms, "
                "%.1f tok/s, %d chunk%s)\n",
                r.language.empty() ? "language not named" : r.language.c_str(),
                r.audio_seconds, r.e2e_ms, r.rtf(), r.ttft_ms, r.decode_tps, r.chunks,
                r.chunks == 1 ? "" : "s");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
