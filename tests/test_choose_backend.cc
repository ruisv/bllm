// Unit test for bllm::chooseBackend — the native/libxlm routing rule. Pure logic on a
// ModelConfig struct, no file IO and no hobot deps, so it builds and runs on the Mac.
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
  // Capability boundary: hybrid / omni / embed can only run native, whatever the field says.
  for (const char* arch : {"hybrid", "omni", "embed"}) {
    CHECK(bllm::chooseBackend(cfg(arch, "auto")) == "native");
    CHECK(bllm::chooseBackend(cfg(arch, "native")) == "native");
    // An explicit libxlm request on an arch libxlm cannot run is an error, not an override.
    CHECK(throws(cfg(arch, "libxlm")));
    // ...unless the caller opts out of the guard.
    CHECK(bllm::chooseBackend(cfg(arch, "libxlm"), /*throw_on_conflict=*/false) == "native");
  }

  // Dense runs on either; the field decides, defaulting to native.
  CHECK(bllm::chooseBackend(cfg("dense", "auto")) == "native");
  CHECK(bllm::chooseBackend(cfg("dense", "native")) == "native");
  CHECK(bllm::chooseBackend(cfg("dense", "libxlm")) == "libxlm");
  CHECK(!throws(cfg("dense", "libxlm")));

  // An unknown/empty arch is treated as dense (the common case), field still honoured.
  CHECK(bllm::chooseBackend(cfg("", "auto")) == "native");
  CHECK(bllm::chooseBackend(cfg("", "libxlm")) == "libxlm");

  if (failures == 0) std::printf("all choose-backend tests passed\n");
  return failures == 0 ? 0 : 1;
}
