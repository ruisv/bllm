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
  if (j.contains("pi05")) {
    const auto& p = j.at("pi05");
    c.pi05.expert = resolve(dir, p.value("expert", ""));
    c.pi05.cond_table = resolve(dir, p.value("cond_table", "cond_table.f32"));
    c.pi05.cameras = p.value("cameras", c.pi05.cameras);
    c.pi05.prompt_len = p.value("prompt_len", c.pi05.prompt_len);
    c.pi05.horizon = p.value("horizon", c.pi05.horizon);
    c.pi05.steps = p.value("steps", c.pi05.steps);
    c.pi05.vocab = p.value("vocab", c.pi05.vocab);
    c.pi05.action_dim_real = p.value("action_dim_real", c.pi05.action_dim_real);
    c.pi05.embed_prescaled = p.value("embed_prescaled", false);
    if (p.contains("action_q01")) c.pi05.action_q01 = p.at("action_q01").get<std::vector<float>>();
    if (p.contains("action_q99")) c.pi05.action_q99 = p.at("action_q99").get<std::vector<float>>();
    c.rope_theta_pi05 = p.value("rope_theta", c.rope_theta_pi05);
  }

  if (c.hbm.empty()) throw std::runtime_error("[bllm] model.json missing 'hbm'");
  if (j.contains("smolvla")) {
    const auto& p = j.at("smolvla");
    c.smolvla.expert = resolve(dir, p.value("expert", ""));
    c.smolvla.cond_table = resolve(dir, p.value("cond_table", "cond_table.f32"));
    c.smolvla.state_proj = resolve(dir, p.value("state_proj", "state_proj.f32"));
    c.smolvla.cameras = p.value("cameras", c.smolvla.cameras);
    c.smolvla.prompt_len = p.value("prompt_len", c.smolvla.prompt_len);
    c.smolvla.horizon = p.value("horizon", c.smolvla.horizon);
    c.smolvla.steps = p.value("steps", c.smolvla.steps);
    c.smolvla.vocab = p.value("vocab", c.smolvla.vocab);
    c.smolvla.state_dim_real = p.value("state_dim_real", c.smolvla.state_dim_real);
    c.smolvla.action_dim_real = p.value("action_dim_real", c.smolvla.action_dim_real);
    c.smolvla.embed_prescaled = p.value("embed_prescaled", false);
    c.smolvla.rope_theta = p.value("rope_theta", c.smolvla.rope_theta);
    for (const char* k : {"state_mean", "state_std", "action_mean", "action_std"}) {
      if (!p.contains(k)) continue;
      auto v = p.at(k).get<std::vector<float>>();
      if (std::string(k) == "state_mean") c.smolvla.state_mean = v;
      else if (std::string(k) == "state_std") c.smolvla.state_std = v;
      else if (std::string(k) == "action_mean") c.smolvla.action_mean = v;
      else c.smolvla.action_std = v;
    }
  }

  if (c.is_smolvla()) {
    // `hbm` is the SmolLM2 prefix pass, `visual` the SmolVLM tower,
    // `smolvla.expert` the interleaved action expert.
    if (c.visual.empty())
      throw std::runtime_error("[bllm] smolvla model.json missing 'visual' (vision tower .hbm)");
    if (c.smolvla.expert.empty())
      throw std::runtime_error("[bllm] smolvla model.json missing 'smolvla.expert'");
    if (c.embed.empty())
      throw std::runtime_error("[bllm] smolvla model.json missing 'embed' (host embed table)");
    if (!c.smolvla.embed_prescaled)
      throw std::runtime_error(
          "[bllm] smolvla model.json must set smolvla.embed_prescaled — `embed_prefix` "
          "scales the tied row by sqrt(hidden) outside the table lookup, and a package "
          "that ships the raw row makes the prompt 31x too small while every cosine "
          "still looks perfect. Repackage with export_smolvla_package.py.");
    if ((int)c.smolvla.action_mean.size() != c.smolvla.action_dim_real ||
        c.smolvla.action_mean.size() != c.smolvla.action_std.size())
      throw std::runtime_error("[bllm] smolvla model.json needs action_mean/action_std of "
                               "length action_dim_real");
    if ((int)c.smolvla.state_mean.size() != c.smolvla.state_dim_real ||
        c.smolvla.state_mean.size() != c.smolvla.state_std.size())
      throw std::runtime_error("[bllm] smolvla model.json needs state_mean/state_std of "
                               "length state_dim_real");
    return c;   // smolvla carries no mrope/chat requirements
  }

  if (c.is_pi05()) {
    // `hbm` is the PaliGemma prefill, `visual` the SigLIP tower, `pi05.expert`
    // the action expert. Each of the three is separately fatal if missing, and
    // so are the pieces that make the prompt path correct: without the table the
    // policy has no instruction, and without the quantiles its actions come out
    // in normalised units that look plausible and drive the arm nowhere.
    if (c.visual.empty())
      throw std::runtime_error("[bllm] pi05 model.json missing 'visual' (SigLIP tower .hbm)");
    if (c.pi05.expert.empty())
      throw std::runtime_error("[bllm] pi05 model.json missing 'pi05.expert' (action expert .hbm)");
    if (c.embed.empty())
      throw std::runtime_error("[bllm] pi05 model.json missing 'embed' (host embed table)");
    if (!c.pi05.embed_prescaled)
      throw std::runtime_error(
          "[bllm] pi05 model.json must set pi05.embed_prescaled — `embed_prefix` scales "
          "the tied row by sqrt(hidden), and a table without it makes the prompt 45x too "
          "small, which no cosine against a golden built the same way can see. Repackage "
          "with export_pi05_package.py.");
    if ((int)c.pi05.action_q01.size() != c.pi05.action_dim_real ||
        c.pi05.action_q01.size() != c.pi05.action_q99.size())
      throw std::runtime_error("[bllm] pi05 model.json needs action_q01/action_q99 of "
                               "length action_dim_real");
    return c;   // pi05 carries no mrope/chat requirements
  }
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
