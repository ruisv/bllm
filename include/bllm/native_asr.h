// bllm::NativeAsr — speech recognition on an official D-Robotics LLM 2.0 Qwen3-ASR package.
//
// A Qwen3-ASR package is two graphs and a table:
//   *_audio_chunk_800_*.hbm      graph `audio`:  input_features [1,128,800] s16,
//                                                attention_mask [1,1,104,104] s16
//                                             -> audio_features [1,104,1024] s16
//   *_language_chunk_512_*.hbm   graphs `prefill` / `decode`: the official LLM 2.0 dense
//                                contract with [3,1,T] position ids (as Qwen3-VL's)
//   *_embed_tokens.bin           fp16 [vocab, hidden]
// The pipeline below reproduces what the official runtime feeds those graphs, checked
// against its tensor dumps (OE_LLM_DUMP_MODEL_IO=1):
//   * log-mel: Whisper's recipe (128 Slaney bins, global max-8 floor over the real audio
//     only — no 30 s padding — then (x+4)/4), T = samples/160 frames;
//   * the encoder runs one 800-frame window at a time, the last zero-padded, with an
//     all-zero mask even on a partial window; a window of n frames contributes
//     Qwen3-ASR's _get_feat_extract_output_lengths(n) tokens (800 -> 104);
//   * the features enter the decoder requantised as rint(q * f32(s_out / s_in));
//   * the prompt is the package's chat template: system = context, user = the audio
//     between <|audio_start|> and <|audio_end|>, then `language X<asr_text>` when the
//     language is forced; positions are sequential and identical on all three rows.
// With no language given, upstream qwen-asr appends nothing and lets the model write
// `language X<asr_text>`. This compiled model opens such a reply with a stray <|im_start|>,
// or answers `None` (no speech) and stops, far too often for plain speech; BLLM gives it the
// word `language`, takes the most likely of the 30 language names it knows (so never None),
// and decodes the transcript as if that language had been forced. (The official runtime appends
// `language <asr_text>` with an EMPTY name, never learns the language, and loses ~1 clip in
// 10 to an empty reply; the forced-language prompts are the same in all three.)
// Even with the language settled the model now and then stops before the first word of
// plain speech (about 1 clip in 500 on AISHELL-1, the official runtime too). An empty
// reply is decoded once more past that stop and kept only if it reads as a confident
// sentence — rescueEmpty() below; silence and noise stay empty.
//
// The decoder's cache is short (1024 tokens at ~13 audio tokens a second), so a long clip
// is cut at quiet points into chunks that each fit, transcribed independently and joined
// — upstream's split_audio_into_chunks, with the chunk length set by the cache instead of
// its 20 minutes.
#pragma once

#include "bllm/asr_util.h"
#include "bllm/model_config.h"
#include "bllm/native_audio.h"
#include "bllm/native_graph.h"
#include "bllm/stop_match.h"
#include "bllm/text_engine.h"
#include "bllm/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bllm {

// The Qwen3-ASR audio encoder graph: one 800-frame log-mel window in, its tokens out,
// already requantised into the decoder's embedding scale.
class Qwen3AsrTower {
 public:
  explicit Qwen3AsrTower(const std::string& hbm, const char* graph = "audio")
      : mel_(audio_detail::slaneyMelFilters(spec_), spec_) {
    packed_.load({hbm});
    g_.init(packed_, graph);
    if (g_.inCount() != 2 || g_.outCount() != 1)
      throw std::runtime_error("[asr] the audio graph must take (input_features, attention_mask) "
                               "and return one tensor");
    const auto& is = g_.inShape(0);
    const auto& ms = g_.inShape(1);
    const auto& os = g_.outShape(0);
    if (is.numDimensions != 3 || g_.inType(0) != HB_DNN_TENSOR_TYPE_S16 ||
        g_.inType(1) != HB_DNN_TENSOR_TYPE_S16 || g_.outType(0) != HB_DNN_TENSOR_TYPE_S16)
      throw std::runtime_error("[asr] expected scaled-int16 audio graph I/O (an S100P package)");
    nMel_ = is.dimensionSize[1];
    frames_ = is.dimensionSize[2];
    nToken_ = os.dimensionSize[os.numDimensions - 2];
    hidden_ = os.dimensionSize[os.numDimensions - 1];
    if (nMel_ != spec_.n_mel) throw std::runtime_error("[asr] audio graph mel bins != 128");
    mel_.set_reflect_end(true);
    if (asr_detail::tokensFor(frames_) != nToken_ ||
        ms.dimensionSize[ms.numDimensions - 1] != nToken_)
      throw std::runtime_error("[asr] audio graph window (" + std::to_string(frames_) +
                               " frames -> " + std::to_string(nToken_) +
                               " tokens) does not follow Qwen3-ASR's length formula");
    // The official runtime never masks anything here, not even a partial window.
    const auto& mp = g_.in[1].properties;
    std::memset(g_.inPtr(1), 0, (size_t)(mp.stride[0] * mp.validShape.dimensionSize[0]));
    g_.inClean(1);
    window_.assign((size_t)nMel_ * frames_, 0.0f);
  }
  Qwen3AsrTower(const Qwen3AsrTower&) = delete;
  Qwen3AsrTower& operator=(const Qwen3AsrTower&) = delete;

  void set_sched(const native_detail::BpuSched& s) { g_.sched = s; }
  // The decoder's embedding scale: rows come back as rint(q * f32(s_out / target)) * target,
  // which the decoder's own rint(v / target) maps back to exactly that integer.
  void set_target_scale(float s) { target_ = s; }
  int hidden() const { return hidden_; }
  int window_frames() const { return frames_; }
  int tokens_per_window() const { return nToken_; }
  int frames(size_t n_samples) const { return mel_.n_frames(n_samples); }
  int token_count(size_t n_samples) const {
    int total = 0;
    for (int left = frames(n_samples); left > 0; left -= frames_)
      total += asr_detail::tokensFor(std::min(left, frames_));
    return total;
  }

  // 16 kHz mono PCM -> [token_count][hidden] rows.
  std::vector<float> encode(const float* pcm, size_t n) {
    if (target_ <= 0) throw std::runtime_error("[asr] set_target_scale() first");
    const int T = mel_.n_frames(n);
    if (T < 1) throw std::runtime_error("[asr] clip too short (under one 10 ms frame)");
    const std::vector<float> mel = mel_.compute(pcm, n);
    std::vector<float> out;
    out.reserve((size_t)token_count(n) * hidden_);
    for (int start = 0; start < T; start += frames_) {
      const int len = std::min(frames_, T - start);
      std::fill(window_.begin(), window_.end(), 0.0f);
      for (int m = 0; m < nMel_; ++m)
        std::memcpy(window_.data() + (size_t)m * frames_, mel.data() + (size_t)m * T + start,
                    (size_t)len * sizeof(float));
      g_.writeRowsF32As(0, window_.data(), nMel_, frames_);
      g_.infer();
      const int ntok = asr_detail::tokensFor(len);
      const size_t at = out.size();
      out.resize(at + (size_t)ntok * hidden_);
      readRequant(out.data() + at, ntok);
    }
    return out;
  }

 private:
  void readRequant(float* dst, int ntok) const {
    const int64_t pitch = g_.outRowPitch(0, hidden_);
    const auto* base = (const uint8_t*)g_.out[0].sysMem.virAddr;
    const float ratio = native_detail::Graph::scaleOf(g_.out[0].properties) / target_;
    for (int i = 0; i < ntok; ++i) {
      const int16_t* q = (const int16_t*)(base + (int64_t)i * pitch);
      float* d = dst + (size_t)i * hidden_;
      for (int k = 0; k < hidden_; ++k) {
        const float t = std::rint((float)q[k] * ratio);
        d[k] = std::max(-32768.0f, std::min(32767.0f, t)) * target_;
      }
    }
  }

  AudioSpec spec_;
  audio_detail::LogMel mel_;
  native_detail::PackedHbm packed_;
  native_detail::Graph g_;
  int nMel_ = 0, frames_ = 0, nToken_ = 0, hidden_ = 0;
  float target_ = 0.0f;
  std::vector<float> window_;
};

struct AsrOptions {
  std::string language;      // "" = detect (the model names it); "Chinese", "en", ...
  std::string context;       // optional text the model may lean on (names, terms)
  int max_new = 0;           // per chunk; 0 = whatever the cache has left
  double chunk_seconds = 0;  // longest piece of a long clip; 0 = the session default
  // An empty reply is not always silence: now and then the model stops before the first
  // word of plain speech. Such a reply is decoded once more past that stop and kept only
  // if what comes out is a sentence the model is sure of (see NativeAsr::rescueEmpty).
  // false = take every empty reply as given, as the official runtime does.
  bool rescue_empty = true;
};

struct AsrResult {
  std::string text;
  std::string language;      // the forced one, or what the model wrote ("Chinese,English"
                             // across chunks); "" when it named none
  std::string raw;           // what the model wrote, header included, chunks joined by \n
  int chunks = 0;
  int ntok = 0;              // tokens generated, all chunks
  double audio_seconds = 0, e2e_ms = 0, ttft_ms = 0, decode_tps = 0;
  double rtf() const { return audio_seconds > 0 ? e2e_ms / 1000.0 / audio_seconds : 0; }
};

class NativeAsr {
 public:
  using OnText = std::function<void(const std::string&)>;
  static constexpr int kSampleRate = 16000;

  explicit NativeAsr(const std::string& model_dir) : NativeAsr(loadModelConfig(model_dir)) {}
  explicit NativeAsr(ModelConfig cfg)
      : cfg_(std::move(cfg)), tk_(Tokenizer::fromFile(cfg_.tokenizer)) {
    if (!cfg_.is_asr())
      throw std::runtime_error("[bllm] '" + cfg_.name + "' is arch=\"" + cfg_.arch +
                               "\", not a speech recogniser (arch=\"asr\")");
    if (cfg_.audio.empty() || cfg_.embed.empty())
      throw std::runtime_error("[bllm] an arch=\"asr\" model needs \"audio\" and \"embed\"");
    text_ = std::make_unique<DenseTextEngine>(cfg_.hbm, cfg_.embed, cfg_.rope_theta,
                                              std::vector<int>{}, false, "prefill", "decode");
    float es = 0;
    std::vector<float> xs;
    if (!text_->input_scales(es, xs) || !xs.empty())
      throw std::runtime_error("[asr] the language graph is not an official LLM 2.0 graph "
                               "(scaled-int16 embeddings, no extra prefill inputs)");
    tower_ = std::make_unique<Qwen3AsrTower>(cfg_.audio);
    if (tower_->hidden() != text_->hidden())
      throw std::runtime_error("[asr] audio tower width != decoder hidden size");
    tower_->set_target_scale(es);
    auto need = [&](const char* t) {
      const int id = tk_.token_to_id(t);
      if (id < 0) throw std::runtime_error(std::string("[asr] tokenizer has no ") + t);
      return id;
    };
    imStart_ = need("<|im_start|>");
    imEnd_ = need("<|im_end|>");
    audioStart_ = need("<|audio_start|>");
    audioEnd_ = need("<|audio_end|>");
    asrText_ = need("<asr_text>");
    // Detecting picks among the languages the model knows: the first token of each name,
    // as written after "language", lifted far above everything else — so `None`, a lone
    // space (the official runtime's empty name), `<asr_text>` or EOS cannot win.
    for (const std::string& l : asr_detail::qwen3AsrLanguages()) {
      const std::vector<int> ids = tk_.encode(" " + l);
      if (ids.empty() || langByFirst_.count(ids[0])) continue;
      langByFirst_[ids[0]] = {l, ids};
      langBias_[ids[0]] = 1e4f;
    }
    if (cfg_.eos.empty()) cfg_.eos = {imEnd_, tk_.token_to_id("<|endoftext|>")};
    // Qwen's special and added tokens all sit above <|endoftext|>: none belongs in a
    // transcript, so stepping past a stop must not land on one either.
    const int firstSpecial = tk_.token_to_id("<|endoftext|>");
    if (firstSpecial > 0)
      for (int id = firstSpecial; id < text_->vocab(); ++id) noSpecial_[id] = -1e4f;
    for (int id : cfg_.eos) noSpecial_[id] = -1e4f;
    // Budget per chunk: the cache less the prompt (~18 tokens) and room for the reply at a
    // fast talker's ~7 tokens a second next to the audio's 13.
    const int cache = text_->context_left();
    defaultChunkSec_ = std::max(5.0, std::min(40.0, (cache - 32) / 20.0));
  }
  NativeAsr(const NativeAsr&) = delete;
  NativeAsr& operator=(const NativeAsr&) = delete;

  const ModelConfig& config() const { return cfg_; }
  double default_chunk_seconds() const { return defaultChunkSec_; }
  int tokens_per_second() const { return 13; }
  void set_sched(const native_detail::BpuSched& s) {
    text_->set_sched(s);
    native_detail::BpuSched m;
    m.priority = s.priority;
    tower_->set_sched(m);
  }

  // 16 kHz mono float PCM in [-1, 1]. `on_text` streams the transcript as it is decoded
  // (never the `language X<asr_text>` header); the returned text is the final one, after
  // the repetition fix.
  AsrResult transcribe(const float* pcm, size_t n, const AsrOptions& opt = {},
                       const OnText& on_text = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    AsrResult r;
    r.audio_seconds = (double)n / kSampleRate;
    const std::string forced = asr_detail::normalizeLanguage(opt.language);
    const double sec = opt.chunk_seconds > 0 ? opt.chunk_seconds : defaultChunkSec_;
    const size_t minLen = kSampleRate / 2;                 // upstream pads to 0.5 s
    const auto pieces = asr_detail::splitAtQuiet(pcm, n, (size_t)(sec * kSampleRate),
                                                 5 * kSampleRate, kSampleRate / 10);
    std::vector<std::string> langs;
    std::vector<float> padded;
    for (size_t c = 0; c < pieces.size(); ++c) {
      const float* p = pcm + pieces[c].first;
      size_t len = pieces[c].second - pieces[c].first;
      if (pieces.size() > 1 && len < minLen) {
        padded.assign(p, p + len);
        padded.resize(minLen, 0.0f);
        p = padded.data();
        len = minLen;
      }
      std::string lang;
      OnText chunkText;
      if (on_text) {
        const bool lead = !r.text.empty();
        chunkText = [&, lead, first = true](const std::string& s) mutable {
          if (first && lead) {
            // The joining space, decided on the chunk's first visible text.
            if (asr_detail::needsSpace(r.text, s)) on_text(" ");
            first = false;
          }
          on_text(s);
        };
      }
      std::string text = transcribeOne(p, len, forced, opt, chunkText, lang, r);
      if (!lang.empty() && (langs.empty() || langs.back() != lang)) langs.push_back(lang);
      if (!text.empty()) {
        if (asr_detail::needsSpace(r.text, text)) r.text += " ";
        r.text += text;
      }
      ++r.chunks;
    }
    for (size_t i = 0; i < langs.size(); ++i) r.language += (i ? "," : "") + langs[i];
    r.e2e_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
  }

 private:
  // One piece that fits the cache: fresh KV, prompt + audio, greedy decode.
  std::string transcribeOne(const float* pcm, size_t n, const std::string& forced,
                            const AsrOptions& opt, const OnText& on_text, std::string& lang,
                            AsrResult& r) {
    text_->reset();
    text_->mark_turn_start();
    const std::vector<float> audio = tower_->encode(pcm, n);
    const int nAudio = (int)(audio.size() / hidden());

    // The package's template, rendered: segments between special tokens are tokenized as
    // the rendered string would be.
    std::vector<int> pre = {imStart_};
    append(pre, tk_.encode("system\n" + opt.context));
    pre.push_back(imEnd_);
    append(pre, tk_.encode("\n"));
    pre.push_back(imStart_);
    append(pre, tk_.encode("user\n"));
    pre.push_back(audioStart_);
    std::vector<int> post = {audioEnd_, imEnd_};
    append(post, tk_.encode("\n"));
    post.push_back(imStart_);
    // Detecting: the prompt ends at the word `language` and the name is picked below.
    // (Ending at `assistant\n`, upstream's prompt, this compiled model often opens with a
    // stray <|im_start|> or stops at once.)
    append(post, tk_.encode(forced.empty() ? std::string("assistant\nlanguage")
                                           : "assistant\nlanguage " + forced));
    if (!forced.empty()) post.push_back(asrText_);

    const int total = (int)pre.size() + nAudio + (int)post.size();
    if (total + 8 > text_->context_left())
      throw std::runtime_error("[asr] " + std::to_string(nAudio) + " audio tokens do not fit the " +
                               std::to_string(text_->context_left()) + "-token cache; use a "
                               "shorter chunk_seconds");
    rows_.resize((size_t)total * hidden());
    pos_.resize(total);
    int at = 0;
    for (int id : pre) embedAt(at++, id);
    std::memcpy(rows_.data() + (size_t)at * hidden(), audio.data(), audio.size() * sizeof(float));
    for (int i = 0; i < nAudio; ++i, ++at) pos_[at] = Pos3::scalar(at);
    for (int id : post) embedAt(at++, id);
    text_->feed_embeds(rows_.data(), total, pos_.data(), nullptr);

    // Detecting: first the language name alone, over the same prefill. Left to itself
    // this compiled model answers `None` (no speech) or skips straight to <asr_text> on
    // about one clip in six of plain speech, and then stops with nothing — AISHELL-1 clean
    // went to 16% CER against 4% with the language forced (the official runtime, which
    // gives an empty name, 10%). So the name is picked with those choices banned — one
    // token, sampled once — and the transcript is then decoded exactly as if forced.
    std::string name;
    if (forced.empty()) {
      NativeSamplingParams s1;                            // one token, so the biased path is free
      s1.temp = 0.0f;
      s1.max_new = 1;
      s1.logit_bias = langBias_;
      int pick = -1;
      text_->generate(s1, [&](int id) { pick = id; return true; });
      const auto it = langByFirst_.find(pick);
      if (it == langByFirst_.end()) throw std::runtime_error("[asr] no language name was generated");
      name = it->second.first;
      std::vector<int> feed = it->second.second;          // the whole name, then <asr_text>
      feed.push_back(asrText_);
      std::vector<float> rows(feed.size() * (size_t)hidden());
      std::vector<Pos3> pos(feed.size());
      const int p0 = text_->next_pos();
      for (size_t i = 0; i < feed.size(); ++i) {
        text_->embed(feed[i], rows.data() + i * hidden());
        pos[i] = Pos3::scalar(p0 + (int)i);
      }
      text_->feed_embeds(rows.data(), (int)feed.size(), pos.data(), nullptr);
      r.ntok += (int)feed.size() - 1;
    }

    std::vector<int> body;
    size_t emitted = 0;
    std::string full;
    auto on_token = [&](int id) -> bool {
      body.push_back(id);
      if (on_text) {
        full = tk_.decode(body);
        const size_t show = utf8SafeEnd(full, full.size());
        if (show > emitted) on_text(full.substr(emitted, show - emitted));
        emitted = std::max(emitted, show);
      }
      return false;
    };
    NativeSamplingParams sp;                              // greedy, as the package says
    sp.temp = 0.0f;
    sp.eos = cfg_.eos;
    const int room = text_->context_left();
    sp.max_new = opt.max_new > 0 ? std::min(opt.max_new, room) : room;
    text_->generate(sp, on_token);
    double ttft = text_->ttft_ms(), tps = text_->decode_tps();
    if (body.empty() && opt.rescue_empty && sp.max_new > 1) {
      const std::vector<int> kept = rescueEmpty(sp, ttft, tps);
      for (int id : kept) on_token(id);
    }
    if (r.chunks == 0) r.ttft_ms = ttft;
    r.decode_tps = tps;
    r.ntok += (int)body.size();

    const std::string text = tk_.decode(body);
    if (!r.raw.empty()) r.raw += "\n";
    r.raw += forced.empty() ? "language " + name + "<asr_text>" + text : text;
    // Nothing said, no language: silence is not "English" because a name had to be picked.
    lang = !forced.empty() ? forced : asr_detail::trim(text).empty() ? "" : name;
    if (on_text && !body.empty() && full.size() > emitted)   // flush any held-back bytes
      on_text(full.substr(emitted));
    return asr_detail::trim(asr_detail::fixRepetitions(text));
  }

  // The reply stopped before its first token. Step past that stop once: the most likely
  // token that is neither a stop nor special, then greedy to the next stop. Keep it only
  // if it reads as a sentence the model is sure of: kRescueDistinct or more distinct
  // tokens, a mean log-probability of at least kRescueMeanLp after the forced first one,
  // and a stop of its own at the end. How sure the model was of the first stop does not
  // tell the two apart (speech it skipped: p(stop) 0.21–0.999; silence and noise: 0.05–1.0).
  // What follows does: a skipped sentence comes back whole and confident, silence, noise,
  // hum and tones give a filler ("嗯。", "The."), a code fence, a chant of one token, or a
  // hesitant hallucination (mean log-probability -1.6 and below). Measured: 2400 AISHELL-1
  // decodes (clean/noisy, forced/detected) left 5 replies empty and all 5 came back; 96 of
  // silence, noise, room tone, hum and tones left 76 empty and none came back.
  // On the way out ttft/tps describe the kept tokens; on a refusal they are left as given.
  std::vector<int> rescueEmpty(const NativeSamplingParams& sp, double& ttft, double& tps) {
    static constexpr int kRescueDistinct = 6;
    static constexpr float kRescueMeanLp = -1.0f;
    NativeSamplingParams s1 = sp;
    s1.max_new = 1;
    s1.eos.clear();
    s1.logit_bias = noSpecial_;
    const std::vector<int> first = text_->generate(s1, [](int) { return false; });  // fed
    if (first.empty()) return {};
    const double t1 = text_->ttft_ms();
    NativeSamplingParams s2 = sp;
    s2.max_new = std::min(sp.max_new - 1, text_->context_left());
    s2.logprobs = 0;
    const std::vector<int> rest = text_->generate(s2, [](int) { return false; });
    if (rest.empty() || (int)rest.size() >= s2.max_new) return {};   // no stop of its own
    double sum = 0;
    for (const TokenLogprobs& t : text_->raw().last_logprobs()) sum += t.logprob;
    std::vector<int> all = first;
    all.insert(all.end(), rest.begin(), rest.end());
    std::vector<int> distinct = all;
    std::sort(distinct.begin(), distinct.end());
    const auto nDistinct = std::unique(distinct.begin(), distinct.end()) - distinct.begin();
    if (nDistinct < kRescueDistinct || sum / rest.size() < kRescueMeanLp) return {};
    ttft = t1;
    tps = text_->decode_tps();
    return all;
  }

  int hidden() const { return text_->hidden(); }
  void embedAt(int i, int id) {
    text_->embed(id, rows_.data() + (size_t)i * hidden());
    pos_[i] = Pos3::scalar(i);
  }
  static void append(std::vector<int>& v, const std::vector<int>& w) { v.insert(v.end(), w.begin(), w.end()); }

  ModelConfig cfg_;
  Tokenizer tk_;
  std::unique_ptr<DenseTextEngine> text_;
  std::unique_ptr<Qwen3AsrTower> tower_;
  int imStart_ = -1, imEnd_ = -1, audioStart_ = -1, audioEnd_ = -1, asrText_ = -1;
  std::unordered_map<int, std::pair<std::string, std::vector<int>>> langByFirst_;
  std::unordered_map<int, float> langBias_;
  std::unordered_map<int, float> noSpecial_;     // bans every stop and special token
  double defaultChunkSec_ = 30;
  std::vector<float> rows_;
  std::vector<Pos3> pos_;
};

}  // namespace bllm
