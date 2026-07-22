#include "bllm/model_config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bllm {
namespace {
// join a base dir with a (possibly relative) path
std::string resolve(const std::string& dir, const std::string& p) {
  if (p.empty()) return p;
  if (p.front() == '/') return p;                // already absolute
  std::string d = dir;
  if (!d.empty() && d.back() != '/') d += '/';
  return d + p;
}
}  // namespace

ModelConfig loadModelConfig(const std::string& dir) {
  std::ifstream f(dir + "/model.json");
  if (!f) throw std::runtime_error("[bllm] no model.json in " + dir);
  nlohmann::json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    throw std::runtime_error("[bllm] bad model.json: " + std::string(e.what()));
  }
  ModelConfig c;
  c.dir = dir;
  c.name = j.value("name", "");
  c.backend = j.value("backend", "auto");
  c.arch = j.value("arch", "dense");
  c.graph = j.value("graph", c.is_hybrid() ? "qwen35" : (c.is_embed() ? "prefill" : "decode"));
  c.cache_len = j.value("cache_len", 0);
  c.bpu_cores = j.value("bpu_cores", 1);
  c.rope_theta = j.value("rope_theta", 1e7);
  c.hbm = resolve(dir, j.value("hbm", ""));
  c.hbm_prefill = resolve(dir, j.value("hbm_prefill", ""));
  c.tokenizer = resolve(dir, j.value("tokenizer", "tokenizer.json"));
  c.embed = resolve(dir, j.value("embed", ""));
  c.visual = resolve(dir, j.value("visual", ""));
  c.audio = resolve(dir, j.value("audio", ""));
  c.mel_filters = resolve(dir, j.value("mel_filters", ""));
  if (j.contains("eos")) c.eos = j.at("eos").get<std::vector<int>>();
  if (j.contains("mrope_section")) c.mrope_section = j.at("mrope_section").get<std::vector<int>>();
  c.mrope_interleaved = j.value("mrope_interleaved", false);
  if (j.contains("vision")) {
    const auto& v = j.at("vision");
    c.vision.patch = v.value("patch", c.vision.patch);
    c.vision.merge = v.value("merge", c.vision.merge);
    c.vision.temporal = v.value("temporal", c.vision.temporal);
    if (v.contains("mean")) {
      const auto m = v.at("mean").get<std::vector<float>>();
      if (m.size() != 3) throw std::runtime_error("[bllm] vision.mean must have 3 entries");
      for (int i = 0; i < 3; ++i) c.vision.mean[i] = m[i];
    }
    if (v.contains("std")) {
      const auto s = v.at("std").get<std::vector<float>>();
      if (s.size() != 3) throw std::runtime_error("[bllm] vision.std must have 3 entries");
      for (int i = 0; i < 3; ++i) c.vision.std[i] = s[i];
    }
  }
  if (j.contains("chat")) {
    const auto& ch = j.at("chat");
    c.chat.format = ch.value("format", "chatml");
    c.chat.im_start = ch.value("im_start", -1);
    c.chat.im_end = ch.value("im_end", -1);
    c.chat.bos = ch.value("bos", -1);
    c.chat.r_user = ch.value("r_user", -1);
    c.chat.r_assistant = ch.value("r_assistant", -1);
    c.chat.r_system = ch.value("r_system", -1);
    c.chat.r_end = ch.value("r_end", -1);
    c.chat.system = ch.value("system", c.chat.system);
  }
  if (c.hbm.empty()) throw std::runtime_error("[bllm] model.json missing 'hbm'");
  if (c.is_embed() && c.chat.format == "chatml" && c.chat.system.empty())
    throw std::runtime_error("[bllm] embed model.json needs chat.system (the instruction)");
  if ((c.is_hybrid() || c.is_omni()) && c.embed.empty())
    throw std::runtime_error("[bllm] " + c.arch + " model.json missing 'embed' (host embed table)");
  if (c.is_omni() && c.visual.empty())
    throw std::runtime_error("[bllm] omni model.json missing 'visual' (vision tower .hbm)");
  if (c.is_omni() && c.mrope_section.empty()) c.mrope_section = {16, 24, 24};
  if (c.has_vision() && c.mrope_section.empty())
    throw std::runtime_error("[bllm] a model.json with 'visual' needs 'mrope_section' — "
                             "image rows carry 3-D positions and plain rope would "
                             "collapse the whole patch grid onto one position");
  return c;
}

}  // namespace bllm
