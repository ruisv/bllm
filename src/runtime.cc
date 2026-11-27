#include "bllm/runtime.h"

#include <dlfcn.h>

#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>

namespace bllm {

namespace {
std::string orEmpty(const char* s) { return s ? s : ""; }
}  // namespace

RuntimeInfo runtimeInfo() {
  RuntimeInfo info;
  info.dnn_version = orEmpty(hbDNNGetVersion());
  info.ucp_version = orEmpty(hbUCPGetVersion());
  info.soc_name = orEmpty(hbUCPGetSocName());
  // Ask the dynamic linker which file provided the libdnn symbol we call.
  Dl_info dl{};
  if (dladdr(reinterpret_cast<void*>(&hbDNNGetVersion), &dl) != 0) {
    info.dnn_library = orEmpty(dl.dli_fname);
  }
  return info;
}

}  // namespace bllm
