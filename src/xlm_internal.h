// Internal (non-installed) helpers shared by the LlmSession / OmniSession
// wrappers: the xlm.h include, enum mappings, the streaming callback trampoline,
// and wall-clock stats. Everything here is inline and lives in bllm::detail.
//
// Not a public header — it pulls in xlm.h, which the public API deliberately
// hides. Only src/*.cc include this.

#ifndef BLLM_SRC_XLM_INTERNAL_H_
#define BLLM_SRC_XLM_INTERNAL_H_

#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>

#include "bllm/common.h"
#include "bllm/stop_match.h"
#include "bllm/types.h"
#include "xlm.h"  // OE-LLM runtime C API (board SDK only; located by CMake)

namespace bllm {
namespace detail {

using Clock = std::chrono::steady_clock;
using TokenSink = std::function<void(std::string_view)>;

// ── enum / struct mapping: BLLM public types -> xlm.h runtime types ──────────
inline xlm_model_type ToXlm(ModelType t) {
  return static_cast<xlm_model_type>(static_cast<int>(t));
}
inline xlm_infer_backend ToXlm(Backend b) {
  return static_cast<xlm_infer_backend>(static_cast<int>(b));
}
inline xlm_priority_type_t ToXlm(Priority p) {
  return static_cast<xlm_priority_type_t>(static_cast<int>(p));
}
inline common_params_sampling_t ToXlm(const SamplingParams& s) {
  common_params_sampling_t o;
  std::memset(&o, 0, sizeof(o));
  o.top_k = s.top_k;
  o.top_p = s.top_p;
  o.min_p = s.min_p;
  o.temp = s.temp;
  o.typ_p = s.typ_p;
  o.min_keep = s.min_keep;
  o.penalty_last_n = s.penalty_last_n;
  o.penalty_repeat = s.penalty_repeat;
  o.penalty_freq = s.penalty_freq;
  o.penalty_present = s.penalty_present;
  return o;
}

inline std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  BLLM_CHECK(f.good(), "cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Routed per generate()/generate-omni() through xlm's userdata. The C callback
// (registered once at xlm_init) casts userdata back to this and accumulates.
//
// The runtime does not expose its perf counters via the callback
// (result->performance is always zero here), so we time the stream ourselves.
struct CallContext {
  const TokenSink* on_token = nullptr;
  std::string accum;
  bool error = false;

  // Stop strings. libxlm has no cancellation API — xlm_infer blocks and its callback
  // returns void — so we cannot halt the runtime early; it keeps generating to its own
  // eos and we simply stop forwarding once a stop string appears. So the STREAM and the
  // RETURN value are trimmed correctly (with prefix holdback), but no compute is saved.
  StopMatcher stop;
  size_t emitted = 0;                 // bytes of `accum` already forwarded to on_token
  size_t cut = std::string::npos;     // where a stop string began, once seen

  Clock::time_point t_start;
  Clock::time_point t_first;
  Clock::time_point t_end;
  int32_t chunk_count = 0;
  bool have_first = false;

  // Stream a chunk through the stop matcher: emit only what is safe, hold back a live
  // stop-string prefix, and latch `cut` at the first complete stop.
  void ingest(std::string_view text) {
    if (cut != std::string::npos) return;    // already past a stop; drop the rest
    accum.append(text.data(), text.size());
    if (stop.empty()) {
      if (on_token && *on_token) (*on_token)(text);
      emitted = accum.size();
      return;
    }
    const auto sc = stop.scan(accum);
    const size_t show = sc.cut != std::string::npos ? sc.cut : sc.safe;
    if (on_token && *on_token && show > emitted)
      (*on_token)(std::string_view(accum).substr(emitted, show - emitted));
    if (show > emitted) emitted = show;
    cut = sc.cut;
  }

  // Called after xlm_infer returns: flush any held-back tail (no stop completed) and
  // trim the accumulated reply at the stop.
  void finish_stream() {
    if (cut != std::string::npos) { accum.resize(cut); return; }
    if (on_token && *on_token && accum.size() > emitted)
      (*on_token)(std::string_view(accum).substr(emitted));
  }
};

// The single C callback shared by every session; `userdata` selects the call.
inline void Trampoline(xlm_result_t* result, xlm_state_t state, void* userdata) {
  auto* ctx = static_cast<CallContext*>(userdata);
  if (!ctx) return;
  switch (state) {
    case XLM_STATE_START:
    case XLM_STATE_RUNNING:
      if (result && result->text && result->text[0] != '\0') {
        if (!ctx->have_first) {
          ctx->have_first = true;
          ctx->t_first = Clock::now();
        }
        ++ctx->chunk_count;
        ctx->ingest(result->text);
      }
      break;
    case XLM_STATE_END:
      ctx->t_end = Clock::now();
      break;
    case XLM_STATE_ERROR:
      ctx->error = true;
      ctx->t_end = Clock::now();
      break;
  }
}

inline GenerationStats ComputeStats(const CallContext& ctx) {
  GenerationStats s;
  if (!ctx.have_first) return s;
  auto ms = [](Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  s.ttft_ms = ms(ctx.t_first - ctx.t_start);
  s.decode_tokens = ctx.chunk_count;
  const double decode_ms = ms(ctx.t_end - ctx.t_first);
  if (ctx.chunk_count > 1 && decode_ms > 0.0) {
    s.decode_tps = (ctx.chunk_count - 1) * 1000.0 / decode_ms;
    s.tpot_ms = decode_ms / (ctx.chunk_count - 1);
  }
  return s;
}

}  // namespace detail
}  // namespace bllm

#endif  // BLLM_SRC_XLM_INTERNAL_H_
