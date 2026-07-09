// bllm::loadImageRGB — minimal image file decoding for the CLI/demo paths.
//
// The runtime API (bllm::VisionTower / NativeVlm) takes raw interleaved RGB8, which
// is what a camera pipeline hands you. This header exists so examples can also take
// a JPEG/PNG path, via vendored stb_image (public domain, no external dependency).
//
// Header-only and self-contained: stb is compiled `static` into whichever TU includes
// this, so including it from two translation units is harmless.
#pragma once

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct OwnedImage {
  std::vector<uint8_t> rgb;   // interleaved, 3 channels
  int width = 0, height = 0;
};

inline OwnedImage loadImageRGB(const std::string& path) {
  int w = 0, h = 0, c = 0;
  uint8_t* px = stbi_load(path.c_str(), &w, &h, &c, 3);
  if (!px) throw std::runtime_error("[bllm] cannot decode image: " + path +
                                    " (" + (stbi_failure_reason() ? stbi_failure_reason() : "?") + ")");
  OwnedImage img;
  img.width = w; img.height = h;
  img.rgb.assign(px, px + (size_t)w * h * 3);
  stbi_image_free(px);
  return img;
}

}  // namespace bllm
