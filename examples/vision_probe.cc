// bllm_vision_probe — run a vision tower on the board and dump its rows, so the
// compiled graph can be scored against the HF fp32 reference.
//
// Two modes, and the split is the point: if end-to-end drifts but the graph is
// clean, the bug is in our host preprocessing (resize kernel, normalization
// constants, patch ordering), not in the quantized tower.
//
//   --patches ref_pixel_values.f32   feed HF's own pixel_values -> isolates the GRAPH
//   --image cat.jpg                  feed pixels -> tests PREPROCESSING + graph
//
//   ./bllm_vision_probe --visual qwen35_visual_448.hbm --patch-size 16 \
//        --mean 0.5 --std 0.5 --patches pv.f32 --out rows.f32
#include "bllm/image_io.h"   // defines STB_IMAGE_IMPLEMENTATION itself
#include "bllm/native_vision.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
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
  const char* visual = argOf(argc, argv, "--visual");
  if (!visual) {
    std::fprintf(stderr,
                 "usage: %s --visual <tower.hbm> [--graph visual]\n"
                 "          [--patch-size 16] [--merge 2] [--temporal 2]\n"
                 "          [--patch-order merged|rowmajor]  (default merged)\n"
                 "          [--mean M] [--std S]   one value for all channels, or three\n"
                 "                                 comma-separated for R,G,B\n"
                 "          (--patches <f32 file> | --image <file>) [--out <f32 file>]\n"
                 "          [--repeat N]\n", argv[0]);
    return 2;
  }
  bllm::VisionSpec spec;
  spec.patch = std::atoi(argOf(argc, argv, "--patch-size", "16"));
  spec.merge = std::atoi(argOf(argc, argv, "--merge", "2"));
  spec.temporal = std::atoi(argOf(argc, argv, "--temporal", "2"));
  if (const char* o = argOf(argc, argv, "--patch-order")) {
    // Silently falling back to Merged would defeat the point: both layouts give
    // a correctly shaped tensor and only the numbers differ, so a typo here
    // reads as a quantization problem in the one tool meant to tell them apart.
    if (std::strcmp(o, "rowmajor") == 0) spec.order = bllm::PatchOrder::RowMajor;
    else if (std::strcmp(o, "merged") == 0) spec.order = bllm::PatchOrder::Merged;
    else {
      std::fprintf(stderr, "unknown --patch-order '%s' (want merged|rowmajor)\n", o);
      return 2;
    }
  }
  // one value applies to all channels; three comma-separated set them per channel
  auto triple = [](const char* s, float* dst) {
    for (int i = 0; i < 3; ++i) {
      dst[i] = (float)std::atof(s);
      const char* c = std::strchr(s, ',');
      if (!c) { for (int j = i + 1; j < 3; ++j) dst[j] = dst[i]; return; }
      s = c + 1;
    }
  };
  if (const char* m = argOf(argc, argv, "--mean")) triple(m, spec.mean);
  if (const char* s = argOf(argc, argv, "--std")) triple(s, spec.std);

  try {
    bllm::VisionTower tower(visual, argOf(argc, argv, "--graph", "visual"), spec);
    const auto& sp = tower.spec();
    std::printf("[probe] %dpx  grid %dx%d  patches %d x %d -> %d rows x %d\n",
                sp.image_size, sp.grid(), sp.grid(), sp.n_patch(), sp.row_len(),
                tower.n_token(), tower.hidden());

    const float* rows = nullptr;
    const int repeat = std::atoi(argOf(argc, argv, "--repeat", "1"));
    double ms = 0;

    if (const char* pf = argOf(argc, argv, "--patches")) {
      // HF's pixel_values verbatim — every host-side preprocessing choice bypassed.
      const size_t want = (size_t)sp.n_patch() * sp.row_len();
      std::ifstream f(pf, std::ios::binary | std::ios::ate);
      if (!f) { std::fprintf(stderr, "cannot open %s\n", pf); return 1; }
      const size_t have = (size_t)f.tellg() / sizeof(float);
      if (have != want) {
        std::fprintf(stderr, "[probe] %s holds %zu floats, graph wants %zu\n", pf, have, want);
        return 1;
      }
      f.seekg(0);
      std::vector<float> pv(want);
      f.read((char*)pv.data(), want * sizeof(float));
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i) rows = tower.encode_patches(pv.data());
      ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / repeat;
      std::printf("[probe] fed HF pixel_values (graph only)\n");
    } else if (const char* imf = argOf(argc, argv, "--image")) {
      bllm::OwnedImage img = bllm::loadImageRGB(imf);
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i) rows = tower.encode(img.rgb.data(), img.width, img.height);
      ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / repeat;
      std::printf("[probe] fed %s (%dx%d) through host preprocessing + graph\n",
                  imf, img.width, img.height);
    } else {
      std::fprintf(stderr, "need --patches or --image\n");
      return 2;
    }
    std::printf("[probe] encode %.1f ms\n", ms);

    double s0 = 0;
    for (int i = 0; i < tower.n_token() * tower.hidden(); ++i) s0 += rows[i] * rows[i];
    std::printf("[probe] rms=%.6f  first row: %.4f %.4f %.4f %.4f\n",
                std::sqrt(s0 / (tower.n_token() * tower.hidden())),
                rows[0], rows[1], rows[2], rows[3]);

    if (const char* of = argOf(argc, argv, "--out")) {
      std::ofstream o(of, std::ios::binary);
      o.write((const char*)rows, (size_t)tower.n_token() * tower.hidden() * sizeof(float));
      std::printf("[probe] wrote %s\n", of);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
