#pragma once

#include <string>

namespace bllm {

/// The hobot runtime this process actually loaded.
///
/// BLLM does not ship the hobot SDK: it links against the copy that came with the
/// board's OS image, so the runtime always matches the board's firmware and BPU. When
/// results look wrong on one board but not another, these four fields say which runtime
/// was in play — in particular `dnn_library`, the file libdnn was resolved from (normally
/// under /usr/hobot/lib; anything else means another copy is shadowing the board's).
struct RuntimeInfo {
  std::string dnn_version;  ///< hbDNNGetVersion()
  std::string ucp_version;  ///< hbUCPGetVersion()
  std::string soc_name;     ///< hbUCPGetSocName(), e.g. "D-Robotics RDK S100P V1P0"
  std::string dnn_library;  ///< path libdnn was loaded from ("" if unknown)
};

/// Query the loaded hobot runtime. Cheap; safe to call before any model is loaded.
RuntimeInfo runtimeInfo();

}  // namespace bllm
