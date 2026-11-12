// Regression test for embedding quantisation on official LLM 2.0 graphs. Board-only
// (native_graph.h needs the hobot headers and aarch64 __fp16), header-only otherwise:
//
//   c++ -std=c++17 -I include tests/test_quant_round.cc -o /tmp/t && /tmp/t
//
// The bug this locks down: BLLM quantised the host embedding as rint(x * (1/scale));
// the official runtime computes rint(x / scale). They are not the same number when
// x / scale sits a hair under .5 — and fp16 embedding values land there often. On
// Qwen3-4B w4 that was 54 elements off by one in a 19-token prompt, against the
// official runtime's own tensor dump, and greedy decoding took a different reply
// from the third token on. Dividing made the input byte-identical and the reply
// identical.
#include "bllm/native_graph.h"

#include <cstdio>

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
  using bllm::native_detail::quantS16;
  using bllm::native_detail::quantS16Div;

  // The measured case: Qwen3-4B's input_embeddings scale and an fp16 table value.
  const float scale = 4.16256762e-06f;
  const float x = 0.0294189453125f;               // exactly representable in fp16
  CHECK(quantS16Div(x, scale) == 7067, "x/scale must round to 7067, got %d",
        quantS16Div(x, scale));
  CHECK(quantS16Div(-x, scale) == -7067, "symmetric: got %d", quantS16Div(-x, scale));
  // ...and the reciprocal form really does disagree there, which is why it was wrong.
  CHECK(quantS16(x, 1.0f / scale) == 7068, "reciprocal form expected 7068, got %d",
        quantS16(x, 1.0f / scale));

  // Saturation on both ends, as before.
  CHECK(quantS16Div(1.0f, scale) == 32767, "positive saturation");
  CHECK(quantS16Div(-1.0f, scale) == -32768, "negative saturation");
  CHECK(quantS16Div(0.0f, scale) == 0, "zero");

  if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
  std::printf("test_quant_round: OK\n");
  return 0;
}
