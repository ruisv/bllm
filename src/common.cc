#include "bllm/common.h"

#include <cstdio>

namespace bllm {
namespace detail {

void ThrowCheck(const char* expr, const char* file, int line,
                const std::string& msg, int code) {
  std::string what = "BLLM_CHECK(";
  what += expr;
  what += ") failed at ";
  what += file;
  what += ':';
  what += std::to_string(line);
  if (code != -1) {
    what += " [xlm ret=";
    what += std::to_string(code);
    what += ']';
  }
  if (!msg.empty()) {
    what += ": ";
    what += msg;
  }
  throw Error(std::move(what), code);
}

}  // namespace detail
}  // namespace bllm
