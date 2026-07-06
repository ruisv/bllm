// nanobind bindings for BLLM. Exposes bllm::LlmSession to Python; the GIL is
// released around xlm_infer and re-acquired only to invoke a Python token
// callback. The Pythonic `stream()` generator is layered on top in
// python/bllm/__init__.py.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <string>
#include <string_view>

#include "bllm/bllm.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_bllm, m) {
  m.doc() = "BPU LLM runtime (OE-LLM/libxlm) — Python bindings";
  m.attr("__version__") = bllm::kVersion;

  nb::class_<bllm::GenerationStats>(m, "GenerationStats")
      .def_ro("ttft_ms", &bllm::GenerationStats::ttft_ms)
      .def_ro("tpot_ms", &bllm::GenerationStats::tpot_ms)
      .def_ro("decode_tps", &bllm::GenerationStats::decode_tps)
      .def_ro("decode_tokens", &bllm::GenerationStats::decode_tokens)
      .def("__repr__", [](const bllm::GenerationStats& s) {
        return "GenerationStats(ttft_ms=" + std::to_string(s.ttft_ms) +
               ", decode_tps=" + std::to_string(s.decode_tps) +
               ", decode_tokens=" + std::to_string(s.decode_tokens) + ")";
      });

  nb::class_<bllm::LlmSession>(m, "LlmSession")
      // Keyword constructor mirroring SessionOptions. model_type is a friendly
      // string ("auto"/"qwen2.5"/"deepseek"/...). Only model_path + tokenizer_dir
      // are required; the template is auto-discovered from tokenizer_dir.
      .def(
          "__init__",
          [](bllm::LlmSession* self, const std::string& model_path,
             const std::string& tokenizer_dir, const std::string& model_type,
             const std::string& chat_template_path,
             const std::string& system_prompt, const std::string& config_path,
             int context_size) {
            bllm::SessionOptions o;
            o.model_path = model_path;
            o.tokenizer_dir = tokenizer_dir;
            o.model_type = bllm::ModelTypeFromString(model_type);
            o.chat_template_path = chat_template_path;
            o.system_prompt = system_prompt;
            o.config_path = config_path;
            o.context_size = context_size;
            nb::gil_scoped_release rel;  // xlm_init can be slow (loads .hbm)
            new (self) bllm::LlmSession(std::move(o));
          },
          "model_path"_a, "tokenizer_dir"_a, "model_type"_a = "auto",
          "chat_template_path"_a = "", "system_prompt"_a = "",
          "config_path"_a = "", "context_size"_a = 0)
      .def(
          "generate",
          [](bllm::LlmSession& self, std::string_view prompt,
             nb::object on_token) -> std::string {
            std::string out;
            if (on_token.is_none()) {
              nb::gil_scoped_release rel;
              out = self.generate(prompt);
              return out;
            }
            auto cb = [&on_token](std::string_view chunk) {
              nb::gil_scoped_acquire acq;
              on_token(nb::str(chunk.data(), chunk.size()));
            };
            nb::gil_scoped_release rel;
            out = self.generate(prompt, cb);
            return out;
          },
          "prompt"_a, "on_token"_a = nb::none(),
          "Generate a reply (blocking). If on_token is given, it is called with "
          "each streamed chunk. Returns the full reply.")
      .def("reset", &bllm::LlmSession::reset,
           "Clear history so the next generate() starts a fresh conversation.")
      .def_prop_ro("last_stats", &bllm::LlmSession::last_stats)
      .def_prop_ro("model_type", [](const bllm::LlmSession& self) {
        return std::string(bllm::ToString(self.options().model_type));
      });
}
