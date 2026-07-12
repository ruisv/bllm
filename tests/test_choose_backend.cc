// Unit test for bllm::chooseBackend — native-only now (the libxlm backend was removed;
// the design notes, tag v-libxlm-final). Pure logic on a ModelConfig struct, no file IO
// and no hobot deps, so it builds and runs on the Mac.
//
//   c++ -std=c++17 -I include tests/test_choose_backend.cc -o /tmp/t && /tmp/t
#include "bllm/model_config.h"

#include <cstdio>
#include <string>

static int failures = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
  } while (0)

static bllm::ModelConfig cfg(const std::string& arch, const std::string& backend) {
  bllm::ModelConfig c;
  c.arch = arch;
  c.backend = backend;
  return c;
}

static bool throws(const bllm::ModelConfig& c) {
  try { bllm::chooseBackend(c); return false; } catch (const std::exception&) { return true; }
}

int main() {
  // Every arch and every non-libxlm field resolves to the one runtime: native.
  for (const char* arch : {"dense", "hybrid", "omni", "embed", ""}) {
    CHECK(bllm::chooseBackend(cfg(arch, "auto")) == "native");
    CHECK(bllm::chooseBackend(cfg(arch, "native")) == "native");
    // A leftover backend="libxlm" is a hard error — the backend is gone, not reinterpreted.
    CHECK(throws(cfg(arch, "libxlm")));
  }

  if (failures == 0) std::printf("all choose-backend tests passed\n");
  return failures == 0 ? 0 : 1;
}
