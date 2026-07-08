// Python bindings for bllm::NativeEngine — the hbDNN LLM engine that runs a
// .hbm WITHOUT libxlm. Built as its own module `_bllm_native` linking only the
// generic hobot runtime (no OE-LLM SDK), so it is independent of the libxlm-based
// `_bllm` module. Speaks token ids; tokenization stays in Python.
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bllm/native_engine.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_bllm_native, m) {
  m.doc() = "BLLM native hbDNN LLM engine (no libxlm)";

  nb::class_<bllm::NativeSamplingParams>(m, "SamplingParams")
      .def(nb::init<>())
      .def_rw("temp", &bllm::NativeSamplingParams::temp)
      .def_rw("top_p", &bllm::NativeSamplingParams::top_p)
      .def_rw("top_k", &bllm::NativeSamplingParams::top_k)
      .def_rw("rep_pen", &bllm::NativeSamplingParams::rep_pen)
      .def_rw("max_new", &bllm::NativeSamplingParams::max_new)
      .def_rw("seed", &bllm::NativeSamplingParams::seed)
      .def_rw("eos", &bllm::NativeSamplingParams::eos);

  nb::class_<bllm::NativeStats>(m, "Stats")
      .def_ro("ttft_ms", &bllm::NativeStats::ttft_ms)
      .def_ro("decode_tps", &bllm::NativeStats::decode_tps)
      .def_ro("ntok", &bllm::NativeStats::ntok)
      .def("__repr__", [](const bllm::NativeStats& s) {
        char b[96];
        std::snprintf(b, sizeof(b), "Stats(ttft_ms=%.1f, decode_tps=%.2f, ntok=%d)",
                      s.ttft_ms, s.decode_tps, s.ntok);
        return std::string(b);
      });

  nb::class_<bllm::NativeEngine>(m, "NativeEngine")
      .def(nb::init<const std::string&>(), "hbm"_a,
           "Load a .hbm (prefill+decode) on hbDNN. libxlm is never touched.")
      .def("reset", &bllm::NativeEngine::reset, "Clear the conversation KV cache.")
      .def("feed", &bllm::NativeEngine::feed, "ids"_a,
           "Ingest token ids (a prompt or a new turn) into the cache.")
      .def(
          "generate",
          [](bllm::NativeEngine& e, const bllm::NativeSamplingParams& p, nb::object on_token) {
            std::function<void(int)> cb;
            if (!on_token.is_none())
              cb = [on_token](int t) { on_token(t); };
            return e.generate(p, cb);
          },
          "params"_a, "on_token"_a = nb::none(),
          "Generate up to params.max_new ids; stop on eos; on_token(id) streams.")
      .def("last_stats", &bllm::NativeEngine::last_stats)
      .def_prop_ro("n_layers", &bllm::NativeEngine::n_layers)
      .def_prop_ro("n_kv", &bllm::NativeEngine::n_kv)
      .def_prop_ro("head_dim", &bllm::NativeEngine::head_dim)
      .def_prop_ro("cache_len", &bllm::NativeEngine::cache_len)
      .def_prop_ro("vocab", &bllm::NativeEngine::vocab)
      .def_prop_ro("position", &bllm::NativeEngine::position);
}
