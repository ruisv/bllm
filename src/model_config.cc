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
  c.graph = j.value("graph", c.is_hybrid() ? "qwen35" : "decode");
  c.cache_len = j.value("cache_len", 0);
  c.rope_theta = j.value("rope_theta", 1e7);
  c.hbm = resolve(dir, j.value("hbm", ""));
  c.tokenizer = resolve(dir, j.value("tokenizer", "tokenizer.json"));
  c.embed = resolve(dir, j.value("embed", ""));
  if (j.contains("eos")) c.eos = j.at("eos").get<std::vector<int>>();
  if (j.contains("chat")) {
    const auto& ch = j.at("chat");
    c.chat.format = ch.value("format", "chatml");
    c.chat.im_start = ch.value("im_start", -1);
    c.chat.im_end = ch.value("im_end", -1);
    c.chat.system = ch.value("system", c.chat.system);
  }
  if (c.hbm.empty()) throw std::runtime_error("[bllm] model.json missing 'hbm'");
  if (c.is_hybrid() && c.embed.empty())
    throw std::runtime_error("[bllm] hybrid model.json missing 'embed' (host embed table)");
  return c;
}

}  // namespace bllm
