// BLLM — errors, version, small shared utilities.
//
// Kept free of any OE-LLM SDK (`xlm.h`) include so the public headers parse
// anywhere (Mac tooling, downstream consumers) without the board SDK present.

#ifndef BLLM_COMMON_H_
#define BLLM_COMMON_H_

#include <stdexcept>
#include <string>

#define BLLM_VERSION_MAJOR 0
#define BLLM_VERSION_MINOR 1
#define BLLM_VERSION_PATCH 0

namespace bllm {

constexpr const char* kVersion = "0.1.0";

// All BLLM failures surface as this exception. `code` carries the underlying
// `xlm_*` return code when the failure came from the runtime (else -1).
class Error : public std::runtime_error {
 public:
  explicit Error(std::string what, int code = -1)
      : std::runtime_error(std::move(what)), code_(code) {}
  int code() const noexcept { return code_; }

 private:
  int code_;
};

namespace detail {
// Builds "BLLM_CHECK(expr) failed at file:line: msg" and throws bllm::Error.
[[noreturn]] void ThrowCheck(const char* expr, const char* file, int line,
                             const std::string& msg, int code = -1);
}  // namespace detail

}  // namespace bllm

// Precondition / invariant guard. `msg` is a std::string or const char*.
#define BLLM_CHECK(cond, msg)                                             \
  do {                                                                    \
    if (!(cond))                                                          \
      ::bllm::detail::ThrowCheck(#cond, __FILE__, __LINE__, (msg));       \
  } while (0)

// Guard for an `xlm_*` C-API return code (0 == success). Reports the code.
#define BLLM_CHECK_XLM(ret, msg)                                          \
  do {                                                                    \
    int _bllm_ret = (ret);                                               \
    if (_bllm_ret != 0)                                                  \
      ::bllm::detail::ThrowCheck(#ret, __FILE__, __LINE__, (msg),        \
                                 _bllm_ret);                             \
  } while (0)

#endif  // BLLM_COMMON_H_
