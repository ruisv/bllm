// Regression test for the attention mask, board-free.
//
//   c++ -std=c++17 -I include tests/test_attn_mask.cc -o /tmp/t && /tmp/t
//
// The bug this locks down: the mask tensor is [heads, ...rows..., cols] and the
// graph reads every head plane, but only plane 0 was ever written. The rest held
// whatever the BPU allocator returned, so the model's output depended on what had
// previously run on the board — clean memory read as "no causal mask at all",
// dirty memory as garbage. Both halves are asserted here: the row contents, and
// that EVERY head plane carries them.
#include "bllm/attn_mask.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);          \
      std::printf(__VA_ARGS__);                                 \
      std::printf("\n");                                        \
      ++failures;                                               \
    }                                                           \
  } while (0)

int main() {
  // --- maskRow ------------------------------------------------------------
  {
    std::vector<float> row(8, 12345.0f);
    bllm::maskRow(row.data(), 8, 3, 5);
    for (int i = 0; i < 8; ++i) {
      const float want = (i >= 3 && i <= 5) ? 0.0f : bllm::kMaskNegInf;
      CHECK(row[i] == want, "row[%d] = %g, want %g", i, row[i], want);
    }
  }
  // A fully masked row is legal (a padded query slot) and must not be left as -inf.
  {
    std::vector<float> row(4, 1.0f);
    bllm::maskRow(row.data(), 4, 2, 1);   // empty range
    for (int i = 0; i < 4; ++i) CHECK(row[i] == bllm::kMaskNegInf, "row[%d] not masked", i);
  }

  // --- replicateHeadPlanes: the decode shape, [heads, cols] ---------------
  {
    const int heads = 8, cols = 16;
    const int64_t headStride = cols * (int64_t)sizeof(float);
    std::vector<uint8_t> buf((size_t)heads * headStride, 0xAB);   // poisoned, as ION is
    bllm::maskRow(reinterpret_cast<float*>(buf.data()), cols, 4, cols - 1);
    bllm::replicateHeadPlanes(buf.data(), heads, headStride, 1, 0, cols);
    const auto* p0 = reinterpret_cast<const float*>(buf.data());
    for (int h = 0; h < heads; ++h) {
      const auto* ph = reinterpret_cast<const float*>(buf.data() + (size_t)h * headStride);
      for (int i = 0; i < cols; ++i)
        CHECK(ph[i] == p0[i], "head %d col %d = %g, want %g (head plane not written)",
              h, i, ph[i], p0[i]);
    }
  }

  // --- replicateHeadPlanes: the prefill shape, [heads, rows, cols], padded --
  {
    const int heads = 8, rows = 4, cols = 16;
    // A row pitch wider than cols, as the BPU's 32-byte row padding produces.
    const int64_t rowStride = 24 * (int64_t)sizeof(float);
    const int64_t headStride = rows * rowStride;
    std::vector<uint8_t> buf((size_t)heads * headStride, 0xAB);
    for (int r = 0; r < rows; ++r)
      bllm::maskRow(reinterpret_cast<float*>(buf.data() + r * rowStride), cols, 2, 2 + r);
    bllm::replicateHeadPlanes(buf.data(), heads, headStride, rows, rowStride, cols);
    for (int h = 0; h < heads; ++h)
      for (int r = 0; r < rows; ++r) {
        const auto* want = reinterpret_cast<const float*>(buf.data() + r * rowStride);
        const auto* got = reinterpret_cast<const float*>(
            buf.data() + (size_t)h * headStride + (size_t)r * rowStride);
        for (int i = 0; i < cols; ++i)
          CHECK(got[i] == want[i], "head %d row %d col %d = %g, want %g", h, r, i, got[i], want[i]);
      }
  }

  // --- a single head must be a no-op, not an out-of-bounds write ----------
  {
    const int cols = 8;
    std::vector<uint8_t> buf(cols * sizeof(float), 0);
    bllm::maskRow(reinterpret_cast<float*>(buf.data()), cols, 0, cols - 1);
    bllm::replicateHeadPlanes(buf.data(), 1, cols * (int64_t)sizeof(float), 1, 0, cols);
    const auto* p = reinterpret_cast<const float*>(buf.data());
    for (int i = 0; i < cols; ++i) CHECK(p[i] == 0.0f, "single-head col %d clobbered", i);
  }

  if (failures) { std::printf("%d check(s) failed\n", failures); return 1; }
  std::printf("test_attn_mask: all checks passed\n");
  return 0;
}
