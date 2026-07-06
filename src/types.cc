#include "bllm/types.h"

#include <algorithm>
#include <cctype>

namespace bllm {
namespace {

std::string Lower(std::string_view s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

bool Has(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

ModelType ModelTypeFromString(std::string_view name) {
  const std::string s = Lower(name);
  // Order matters: check the most specific tokens first.
  if (Has(s, "qwen2.5") || Has(s, "qwen2_5") || Has(s, "qwen2.5")) return ModelType::Qwen2_5;
  if (Has(s, "omni")) return ModelType::Omni;
  if (Has(s, "qwen-vl") || Has(s, "qwenvl") || Has(s, "qwen_vl")) return ModelType::QwenVL;
  if (Has(s, "deepseek")) return ModelType::DeepSeek;
  if (Has(s, "internvl")) return ModelType::InternVL;
  if (Has(s, "internlm")) return ModelType::InternLM2;
  if (Has(s, "llama")) return ModelType::Llama;
  if (Has(s, "qwen")) return ModelType::Qwen;
  return ModelType::Auto;
}

ModelType ModelTypeFromPath(std::string_view path) {
  // Same matcher as the name form — the path basename usually carries the family.
  return ModelTypeFromString(path);
}

const char* ToString(ModelType t) {
  switch (t) {
    case ModelType::Auto: return "auto";
    case ModelType::InternVL: return "internvl";
    case ModelType::DeepSeek: return "deepseek";
    case ModelType::Qwen: return "qwen";
    case ModelType::Llama: return "llama";
    case ModelType::InternLM2: return "internlm2";
    case ModelType::Omni: return "omni";
    case ModelType::QwenVL: return "qwen-vl";
    case ModelType::Qwen2_5: return "qwen2.5";
  }
  return "unknown";
}

}  // namespace bllm
