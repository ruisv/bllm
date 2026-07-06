// nanobind bindings for BLLM. Exposes bllm::LlmSession to Python; the GIL is
// released around xlm_infer and re-acquired only to invoke a Python token
// callback. The Pythonic `stream()` generator is layered on top in
// python/bllm/__init__.py.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "bllm/bllm.h"

namespace nb = nanobind;
using namespace nb::literals;

// Wrap a Python token callback as a bllm streaming callback (GIL re-acquired for
// the call; the surrounding generate() runs with the GIL released).
namespace {

bllm::Backend BackendFromStr(std::string_view s) {
  if (s == "any") return bllm::Backend::Any;
  if (s == "bpu0") return bllm::Backend::Bpu0;
  if (s == "bpu1") return bllm::Backend::Bpu1;
  if (s == "bpu2") return bllm::Backend::Bpu2;
  if (s == "bpu3") return bllm::Backend::Bpu3;
  throw std::invalid_argument("backend must be one of: any, bpu0..bpu3");
}

bllm::Priority PriorityFromStr(std::string_view s) {
  if (s == "normal") return bllm::Priority::Normal;
  if (s == "high") return bllm::Priority::High;
  if (s == "urgent") return bllm::Priority::Urgent;
  throw std::invalid_argument("priority must be one of: normal, high, urgent");
}

// Apply the Python-side sampling/backend/priority overrides onto any Options.
template <typename Options>
void ApplyCommon(Options& o, nb::object sampling, std::string_view backend,
                 std::string_view priority) {
  if (!sampling.is_none()) o.sampling = nb::cast<bllm::SamplingParams>(sampling);
  o.backend = BackendFromStr(backend);
  o.priority = PriorityFromStr(priority);
}

template <typename Session, typename Arg>
std::string StreamGenerate(Session& self, Arg&& arg, nb::object on_token) {
  std::string out;
  if (on_token.is_none()) {
    nb::gil_scoped_release rel;
    out = self.generate(std::forward<Arg>(arg));
    return out;
  }
  auto cb = [&on_token](std::string_view chunk) {
    nb::gil_scoped_acquire acq;
    on_token(nb::str(chunk.data(), chunk.size()));
  };
  nb::gil_scoped_release rel;
  out = self.generate(std::forward<Arg>(arg), cb);
  return out;
}
}  // namespace

NB_MODULE(_bllm, m) {
  m.doc() = "BPU LLM runtime (OE-LLM/libxlm) — Python bindings";
  m.attr("__version__") = bllm::kVersion;

  nb::class_<bllm::GenerationStats>(m, "GenerationStats")
      .def_ro("ttft_ms", &bllm::GenerationStats::ttft_ms)
      .def_ro("tpot_ms", &bllm::GenerationStats::tpot_ms)
      .def_ro("prefill_tps", &bllm::GenerationStats::prefill_tps)
      .def_ro("decode_tps", &bllm::GenerationStats::decode_tps)
      .def_ro("prefill_tokens", &bllm::GenerationStats::prefill_tokens)
      .def_ro("decode_tokens", &bllm::GenerationStats::decode_tokens)
      .def_ro("vit_ms", &bllm::GenerationStats::vit_ms)
      .def("__repr__", [](const bllm::GenerationStats& s) {
        return "GenerationStats(ttft_ms=" + std::to_string(s.ttft_ms) +
               ", decode_tps=" + std::to_string(s.decode_tps) +
               ", decode_tokens=" + std::to_string(s.decode_tokens) + ")";
      });

  // SamplingParams — construct, then set fields (defaults match the C++ struct).
  nb::class_<bllm::SamplingParams>(m, "SamplingParams")
      .def(nb::init<>())
      .def_rw("top_k", &bllm::SamplingParams::top_k)
      .def_rw("top_p", &bllm::SamplingParams::top_p)
      .def_rw("min_p", &bllm::SamplingParams::min_p)
      .def_rw("temp", &bllm::SamplingParams::temp)
      .def_rw("typ_p", &bllm::SamplingParams::typ_p)
      .def_rw("min_keep", &bllm::SamplingParams::min_keep)
      .def_rw("penalty_last_n", &bllm::SamplingParams::penalty_last_n)
      .def_rw("penalty_repeat", &bllm::SamplingParams::penalty_repeat)
      .def_rw("penalty_freq", &bllm::SamplingParams::penalty_freq)
      .def_rw("penalty_present", &bllm::SamplingParams::penalty_present);

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
             int context_size, nb::object sampling, const std::string& backend,
             const std::string& priority) {
            bllm::SessionOptions o;
            o.model_path = model_path;
            o.tokenizer_dir = tokenizer_dir;
            o.model_type = bllm::ModelTypeFromString(model_type);
            o.chat_template_path = chat_template_path;
            o.system_prompt = system_prompt;
            o.config_path = config_path;
            o.context_size = context_size;
            ApplyCommon(o, sampling, backend, priority);
            nb::gil_scoped_release rel;  // xlm_init can be slow (loads .hbm)
            new (self) bllm::LlmSession(std::move(o));
          },
          "model_path"_a, "tokenizer_dir"_a, "model_type"_a = "auto",
          "chat_template_path"_a = "", "system_prompt"_a = "",
          "config_path"_a = "", "context_size"_a = 0,
          "sampling"_a = nb::none(), "backend"_a = "any", "priority"_a = "normal")
      .def(
          "generate",
          [](bllm::LlmSession& self, std::string_view prompt,
             nb::object on_token) {
            return StreamGenerate(self, prompt, on_token);
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

  // ── Multimodal (Omni) ──────────────────────────────────────────────────────
  nb::class_<bllm::Content>(m, "Content")
      .def_static("text", &bllm::Content::Text, "text"_a)
      .def_static(
          "image",
          [](std::string p, int w, int h) {
            return bllm::Content::Image(std::move(p), w, h);
          },
          "path"_a, "width"_a = 448, "height"_a = 448)
      .def_static("audio", &bllm::Content::Audio, "path"_a)
      .def_static(
          "video",
          [](std::string p, int w, int h) {
            return bllm::Content::Video(std::move(p), w, h);
          },
          "path"_a, "width"_a = 448, "height"_a = 448);

  nb::class_<bllm::OmniSession>(m, "OmniSession")
      .def(
          "__init__",
          [](bllm::OmniSession* self, const std::string& text_model,
             const std::string& visual_model, const std::string& audio_model,
             const std::string& embed_tokens, const std::string& tokenizer_dir,
             const std::string& system_prompt, int context_size,
             nb::object sampling, const std::string& backend,
             const std::string& priority) {
            bllm::OmniOptions o;
            o.text_model_path = text_model;
            o.visual_model_path = visual_model;
            o.audio_model_path = audio_model;
            o.embed_tokens_path = embed_tokens;
            o.tokenizer_dir = tokenizer_dir;
            if (!system_prompt.empty()) o.system_prompt = system_prompt;
            o.context_size = context_size;
            ApplyCommon(o, sampling, backend, priority);
            nb::gil_scoped_release rel;
            new (self) bllm::OmniSession(std::move(o));
          },
          "text_model"_a, "visual_model"_a, "audio_model"_a, "embed_tokens"_a,
          "tokenizer_dir"_a, "system_prompt"_a = "", "context_size"_a = 0,
          "sampling"_a = nb::none(), "backend"_a = "any", "priority"_a = "normal")
      .def(
          "generate",
          [](bllm::OmniSession& self, std::vector<bllm::Content> content,
             nb::object on_token) {
            return StreamGenerate(self, content, on_token);
          },
          "content"_a, "on_token"_a = nb::none(),
          "Generate a reply from a list of Content parts (text/image/audio/"
          "video). Streams to on_token if given; returns the full reply.")
      .def(
          "ask",
          [](bllm::OmniSession& self, const std::string& text,
             nb::object on_token) {
            std::vector<bllm::Content> parts{bllm::Content::Text(text)};
            return StreamGenerate(self, parts, on_token);
          },
          "text"_a, "on_token"_a = nb::none(), "Convenience: a text-only turn.")
      .def("reset", &bllm::OmniSession::reset)
      .def_prop_ro("last_stats", &bllm::OmniSession::last_stats)
      .def_prop_ro("model_type", [](const bllm::OmniSession& self) {
        return std::string(bllm::ToString(self.options().model_type));
      });

  // ── Image VLM (InternVL / Qwen-VL) ─────────────────────────────────────────
  nb::class_<bllm::VlmSession>(m, "VlmSession")
      .def(
          "__init__",
          [](bllm::VlmSession* self, const std::string& model_path,
             const std::string& tokenizer_dir, const std::string& config_path,
             const std::string& model_type, const std::string& system_prompt,
             int context_size, nb::object sampling, const std::string& backend,
             const std::string& priority) {
            bllm::VlmOptions o;
            o.model_path = model_path;
            o.tokenizer_dir = tokenizer_dir;
            o.config_path = config_path;
            o.model_type = bllm::ModelTypeFromString(model_type);
            o.system_prompt = system_prompt;
            o.context_size = context_size;
            ApplyCommon(o, sampling, backend, priority);
            nb::gil_scoped_release rel;
            new (self) bllm::VlmSession(std::move(o));
          },
          "model_path"_a, "tokenizer_dir"_a, "config_path"_a = "",
          "model_type"_a = "auto", "system_prompt"_a = "", "context_size"_a = 0,
          "sampling"_a = nb::none(), "backend"_a = "any", "priority"_a = "normal")
      .def(
          "generate",
          [](bllm::VlmSession& self, std::vector<std::string> image_paths,
             std::string_view prompt, nb::object on_token) -> std::string {
            std::vector<bllm::VlmImage> imgs;
            imgs.reserve(image_paths.size());
            for (auto& p : image_paths)
              imgs.push_back(bllm::VlmImage::File(std::move(p)));
            std::string out;
            if (on_token.is_none()) {
              nb::gil_scoped_release rel;
              return self.generate(imgs, prompt);
            }
            auto cb = [&on_token](std::string_view c) {
              nb::gil_scoped_acquire acq;
              on_token(nb::str(c.data(), c.size()));
            };
            nb::gil_scoped_release rel;
            return self.generate(imgs, prompt, cb);
          },
          "images"_a, "prompt"_a, "on_token"_a = nb::none(),
          "Ask about one or more image files. Streams to on_token if given.")
      .def(
          "describe",
          [](bllm::VlmSession& self, const std::string& image_path,
             std::string_view prompt, nb::object on_token) -> std::string {
            if (on_token.is_none()) {
              nb::gil_scoped_release rel;
              return self.describe(image_path, prompt);
            }
            auto cb = [&on_token](std::string_view c) {
              nb::gil_scoped_acquire acq;
              on_token(nb::str(c.data(), c.size()));
            };
            nb::gil_scoped_release rel;
            return self.describe(image_path, prompt, cb);
          },
          "image_path"_a, "prompt"_a, "on_token"_a = nb::none())
      .def("reset", &bllm::VlmSession::reset)
      .def_prop_ro("last_stats", &bllm::VlmSession::last_stats)
      .def_prop_ro("model_type", [](const bllm::VlmSession& self) {
        return std::string(bllm::ToString(self.options().model_type));
      });
}
