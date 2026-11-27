// bllm::asr_detail — the text and signal bookkeeping around Qwen3-ASR, with no runtime
// behind it: the encoder's token count, upstream qwen-asr's quiet-point splitting and
// repetition fix, language names, and how chunk transcripts join. Pure, so it builds and
// is tested anywhere (tests/test_asr_util.cc); bllm::NativeAsr (native_asr.h) uses it.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace bllm {

namespace asr_detail {

// Qwen3-ASR's _get_feat_extract_output_lengths: 13 tokens per 100 mel frames, plus three
// stride-2 convolutions over the remainder.
inline int tokensFor(int frames) {
  if (frames <= 0) return 0;
  const int leave = frames % 100;
  int t = (frames / 100) * 13;
  if (leave > 0) {
    const int f = (leave - 1) / 2 + 1;
    t += ((f - 1) / 2 + 1 - 1) / 2 + 1;
  }
  return t;
}

// Upstream split_audio_into_chunks: while more than `maxLen` samples remain, cut near
// start+maxLen — at the quietest sample of the quietest `win`-sample window within
// ±`expand` of it. No overlap; the pieces concatenate back to the input exactly.
inline std::vector<std::pair<size_t, size_t>> splitAtQuiet(const float* pcm, size_t n,
                                                           size_t maxLen, size_t expand,
                                                           size_t win) {
  std::vector<std::pair<size_t, size_t>> out;
  size_t start = 0;
  while (n - start > maxLen) {
    const size_t cut = start + maxLen;
    const size_t left = std::max(start, cut > expand ? cut - expand : 0);
    const size_t right = std::min(n, cut + expand);
    size_t boundary = cut;
    if (right - left > win) {
      double sum = 0, best = 0;
      size_t bestAt = left;
      for (size_t i = left; i < left + win; ++i) sum += std::fabs(pcm[i]);
      best = sum;
      for (size_t i = left + 1; i + win <= right; ++i) {
        sum += std::fabs(pcm[i + win - 1]) - std::fabs(pcm[i - 1]);
        if (sum < best) { best = sum; bestAt = i; }
      }
      size_t inner = bestAt;
      for (size_t i = bestAt; i < bestAt + win; ++i)
        if (std::fabs(pcm[i]) < std::fabs(pcm[inner])) inner = i;
      boundary = inner;
    }
    boundary = std::min(std::max(boundary, start + 1), n);
    out.emplace_back(start, boundary);
    start = boundary;
  }
  out.emplace_back(start, n);
  return out;
}

inline std::u32string toU32(const std::string& s) {
  std::u32string o;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = (unsigned char)s[i];
    const int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    char32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    o.push_back(cp);
    i += len;
  }
  return o;
}

inline std::string toUtf8(const std::u32string& s) {
  std::string o;
  for (char32_t c : s) {
    if (c < 0x80) o += (char)c;
    else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) {
      o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F));
    } else {
      o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F));
      o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F));
    }
  }
  return o;
}

// Upstream detect_and_fix_repetitions(threshold=20), on code points as Python's str is: a
// run of one character longer than `thresh` becomes one copy; a 1..`maxLen`-character
// pattern repeated at least `thresh` times in a row becomes one copy. A decoder stuck in
// a loop (it happens on noise) then ends as one copy rather than the whole budget of it.
inline std::string fixRepetitions(const std::string& text, int thresh = 20, int maxLen = 20) {
  std::u32string s = toU32(text), a;
  for (size_t i = 0; i < s.size();) {                      // character runs
    size_t j = i + 1;
    while (j < s.size() && s[j] == s[i]) ++j;
    if ((int)(j - i) > thresh) a.push_back(s[i]); else a.append(s, i, j - i);
    i = j;
  }
  std::u32string b;
  for (size_t i = 0; i < a.size();) {                      // pattern runs
    bool hit = false;
    for (int k = 1; k <= maxLen && i + (size_t)k * thresh <= a.size(); ++k) {
      size_t reps = 1;
      while (i + (reps + 1) * k <= a.size() && a.compare(i + reps * k, k, a, i, k) == 0) ++reps;
      if ((int)reps >= thresh) {
        b.append(a, i, k);
        i += reps * k;
        hit = true;
        break;
      }
    }
    if (!hit) b.push_back(a[i++]);
  }
  return toUtf8(b);
}

inline std::string trim(const std::string& s) {
  const char* ws = " \t\r\n";
  const size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

// The languages Qwen3-ASR transcribes, as the model writes them after "language".
inline const std::vector<std::string>& qwen3AsrLanguages() {
  static const std::vector<std::string> v = {
      "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
      "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
      "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
      "Persian", "Greek", "Hungarian", "Macedonian", "Romanian"};
  return v;
}

// Upstream normalize_language_name ("chinese" -> "Chinese"), plus the short codes the
// official runtime accepts ("zh", "en", ...).
inline std::string normalizeLanguage(std::string lang) {
  lang = trim(lang);
  if (lang.empty()) return lang;
  static const std::pair<const char*, const char*> codes[] = {
      {"zh", "Chinese"}, {"en", "English"}, {"yue", "Cantonese"}, {"ja", "Japanese"},
      {"ko", "Korean"},  {"fr", "French"},  {"de", "German"},     {"es", "Spanish"},
      {"pt", "Portuguese"}, {"it", "Italian"}, {"ru", "Russian"}, {"ar", "Arabic"},
      {"th", "Thai"},    {"vi", "Vietnamese"}, {"id", "Indonesian"}, {"tr", "Turkish"},
      {"hi", "Hindi"},   {"ms", "Malay"},   {"nl", "Dutch"}};
  std::string low;
  for (char c : lang) low += (char)std::tolower((unsigned char)c);
  for (const auto& kv : codes)
    if (low == kv.first) return kv.second;
  std::string out = low;
  out[0] = (char)std::toupper((unsigned char)out[0]);
  return out;
}

// Whether two chunk transcripts need a space where they meet: between two Latin-script
// words, yes ("go." + "She"); next to CJK, no. Upstream joins with "" outright, which
// glues English chunks together.
inline bool needsSpace(const std::string& left, const std::string& right) {
  if (left.empty() || right.empty()) return false;
  const unsigned char a = (unsigned char)left.back(), b = (unsigned char)right.front();
  return a < 0x80 && b < 0x80 && !std::isspace(a) && !std::isspace(b);
}

}  // namespace asr_detail

}  // namespace bllm
