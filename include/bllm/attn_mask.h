// Additive attention-mask construction, split out of the engine so it can be
// tested without a board.
//
// The graphs take the mask as [heads, ...rows..., cols] and read EVERY head
// plane — they do not broadcast one row over the head dimension. Filling only
// the first plane leaves the rest holding whatever the allocator handed back,
// which on the BPU is the previous tenant's data; the model then answers
// differently depending on what else the board had been running. The replicate
// step below is what keeps that from happening, and is the reason these two
// helpers exist as named, tested functions rather than as loops inline in the
// decode path.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bllm {

// The value that stands in for -inf. The graphs carry the mask as float and add
// it to the attention logits before the softmax; a true -inf would produce NaN
// for a fully masked row.
constexpr float kMaskNegInf = -1e9f;

// One query's mask row: columns in [lo, hi] are visible (0), the rest masked.
inline void maskRow(float* row, int cols, int lo, int hi) {
  for (int i = 0; i < cols; ++i) row[i] = (i >= lo && i <= hi) ? 0.0f : kMaskNegInf;
}

// Copy head plane 0 to heads 1..heads-1. `rowStride` is the byte pitch between
// query rows within a plane (ignored when rows == 1), `headStride` the pitch
// between planes, `cols` the number of floats actually used in a row.
inline void replicateHeadPlanes(uint8_t* base, int heads, int64_t headStride,
                                int rows, int64_t rowStride, int cols) {
  const size_t rowBytes = (size_t)cols * sizeof(float);
  for (int h = 1; h < heads; ++h)
    for (int r = 0; r < rows; ++r)
      std::memcpy(base + (size_t)h * headStride + (size_t)r * rowStride,
                  base + (size_t)r * rowStride, rowBytes);
}

}  // namespace bllm
