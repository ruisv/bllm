// bllm::NativeVlm — image+text chat on the self-built native runtime (no libxlm).
//
// This is the multimodal sibling of bllm::NativeLlm. It loads a model directory
// (model.json, arch="omni") holding a text tower, a vision tower and the host
// embedding table, and answers a turn of text + images:
//
//   bllm::NativeVlm vlm("/models/qwen2.5-omni-3b");
//   std::string a = vlm.chat("这张图里有什么?", {img}, 256, [](auto s){ std::cout << s; });
//
// How multimodality actually works here: the text tower's graph takes an input
// EMBEDDING (not a token id) and host-supplied rope cos/sin. So the session
//   1. tokenizes the ChatML turn and embeds each token on the host (EmbedTable),
//   2. runs the vision tower and splices its rows into that same stream where the
//      image sits — HF's <|IMAGE|> placeholders are never needed,
//   3. assigns 3-D mrope positions — scalar for text, (t, t+row, t+col) for image
//      patches — and feeds the whole stream to prefill/decode.
//
// See docs/NATIVE_RUNTIME.md (SE5).
#pragma once

#include "bllm/model_config.h"
#include "bllm/native_audio.h"
#include "bllm/native_embed.h"
#include "bllm/native_engine.h"
#include "bllm/native_vision.h"
#include "bllm/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bllm {

// A decoded, interleaved RGB8 image. The caller owns the pixels.
struct ImageRGB {
  const uint8_t* rgb = nullptr;
  int width = 0, height = 0;
};

// Frames already sampled at `fps` (all the same size), optionally with the
// soundtrack. Qwen's vision tower consumes frames in PAIRS (temporal_patch_size=2),
// so K = ceil(frames/2) grid steps of vision_tokens() each. When `audio` is set,
// Qwen2.5-Omni's `use_audio_in_video` layout applies: video and audio tokens are
// interleaved in 2-second chunks so TMRoPE keeps them time-aligned.
struct VideoClip {
  std::vector<ImageRGB> frames;
  double fps = 2.0;
  AudioPCM audio;
};

class NativeVlm {
 public:
  using OnText = std::function<void(const std::string&)>;

  explicit NativeVlm(const std::string& model_dir) : NativeVlm(loadModelConfig(model_dir)) {}
  explicit NativeVlm(ModelConfig cfg)
      : cfg_(std::move(cfg)), tk_(Tokenizer::fromFile(cfg_.tokenizer)) {
    if (!cfg_.is_omni()) throw std::runtime_error("[bllm] NativeVlm needs arch=\"omni\"");
    text_ = std::make_unique<NativeEngine>(cfg_.hbm, "text_model_prefill", "text_model_decode");
    if (!text_->embed_input())
      throw std::runtime_error("[bllm] the omni text tower must take an embedding input");
    text_->set_rope(cfg_.rope_theta, cfg_.mrope_section);
    vision_ = std::make_unique<VisionTower>(cfg_.visual);
    embed_ = std::make_unique<EmbedTable>(cfg_.embed, text_->vocab(), text_->hidden());
    text_->set_embedder([this](int id, float* dst) { embed_->lookup(id, dst); });

    if (vision_->hidden() != text_->hidden())
      throw std::runtime_error("[bllm] vision/text hidden size mismatch");

    im_start_ = tk_.token_to_id("<|im_start|>");
    im_end_ = tk_.token_to_id("<|im_end|>");
    vision_bos_ = tk_.token_to_id("<|vision_bos|>");
    vision_eos_ = tk_.token_to_id("<|vision_eos|>");
    audio_bos_ = tk_.token_to_id("<|audio_bos|>");
    audio_eos_ = tk_.token_to_id("<|audio_eos|>");
    if (cfg_.eos.empty()) cfg_.eos = {im_end_, tk_.token_to_id("<|endoftext|>")};
  }

  const ModelConfig& config() const { return cfg_; }
  const Tokenizer& tokenizer() const { return tk_; }
  int vision_tokens() const { return vision_->n_token(); }
  // The square side this tower was compiled for — sample frames at it to
  // avoid resizing twice.
  int vision_image_size() const { return vision_->image_size(); }
  bool has_audio() const { return !cfg_.audio.empty() && !cfg_.mel_filters.empty(); }
  double last_decode_tps() const { return text_->last_stats().decode_tps; }
  double last_ttft_ms() const { return text_->last_stats().ttft_ms; }

  void set_sampling(float temp, float top_p, int top_k, float rep_pen, uint64_t seed) {
    sampling_.temp = temp; sampling_.top_p = top_p; sampling_.top_k = top_k;
    sampling_.rep_pen = rep_pen; sampling_.seed = seed;
  }

  void reset() {
    text_->reset();
    first_turn_ = true;
    streaming_ = false;
    frames_.clear(); pcm_.clear();
  }

  // One ChatML user turn. Media precedes `text`, as the Qwen chat template does.
  std::string chat(const std::string& text, const std::vector<ImageRGB>& images = {},
                   const std::vector<AudioPCM>& audios = {},
                   const std::vector<VideoClip>& videos = {},
                   int max_new = 256, const OnText& on_text = {}) {
    if (streaming_) throw std::runtime_error("[bllm] a stream is open; use stream_ask()");
    text_->mark_turn_start();                    // TTFT counts the media encode too
    Stream s(text_->hidden(), text_->next_pos());
    openTurn(s);
    for (const ImageRGB& im : images) appendImage(s, im);
    for (const VideoClip& v : videos) appendVideo(s, v);
    for (const AudioPCM& au : audios) appendAudio(s, au);
    appendText(s, tk_.encode(text));
    closeTurn(s);
    feedStream(s);
    return streamDecode(max_new, on_text);
  }

  // ── live streaming: push camera frames / microphone PCM as they arrive ──────
  //
  //   vlm.stream_begin(2.0);                 // frames will arrive at 2 fps
  //   while (grabbing) { vlm.stream_frame(f); vlm.stream_audio(pcm, n); }
  //   std::string a = vlm.stream_ask("刚才发生了什么？");
  //
  // Media is encoded and pushed into the KV cache as each 2-second chunk completes,
  // so by the time the question arrives most of the prefill is already paid for.
  //
  // BUDGET: the KV window is the whole context (see NativeEngine::context_left).
  // A frame-pair spans temporal_patch_size (2) frames, so at 2 fps video costs
  // vision_tokens() per second of wall-clock and audio costs 25. A 2048-slot model
  // therefore holds 8 s of video with the stock 448px tower (32 s with a 224px
  // re-export), or ~80 s of audio alone. stream_frame/stream_audio throw once the
  // budget is spent rather than evict; context_left() lets a caller see it coming.

  // fps <= 0 means audio only (no video block).
  void stream_begin(double fps = 2.0, bool with_audio = true) {
    if (streaming_) throw std::runtime_error("[bllm] stream already open");
    if (fps <= 0 && !with_audio) throw std::runtime_error("[bllm] stream needs video, audio, or both");
    fps_ = fps; withVideo_ = fps > 0; withAudio_ = with_audio;
    framesPerChunk_ = withVideo_ ? std::max(2, (int)std::lround(fps * kSecondsPerChunk)) : 0;
    frames_.clear(); pcm_.clear();
    pairIdx_ = 0; audioIdx_ = 0; maxRel_ = -1; frameW_ = frameH_ = 0;

    Stream s(text_->hidden(), text_->next_pos());
    openTurn(s);
    if (withVideo_) appendText(s, {vision_bos_});
    if (withAudio_) appendText(s, {audio_bos_});
    st_ = s.nextPos;
    feedStream(s);
    streaming_ = true;
  }

  // One frame; all frames must share a size. Copied, since encoding is deferred.
  void stream_frame(const ImageRGB& f) {
    requireStream();
    if (!withVideo_) throw std::runtime_error("[bllm] stream opened without video (fps <= 0)");
    if (!f.rgb || f.width <= 0 || f.height <= 0) throw std::runtime_error("[bllm] empty frame");
    if (frames_.empty()) { frameW_ = f.width; frameH_ = f.height; }
    else if (f.width != frameW_ || f.height != frameH_)
      throw std::runtime_error("[bllm] streamed frames must all be the same size");
    frames_.emplace_back(f.rgb, f.rgb + (size_t)f.width * f.height * 3);
    drainChunks();
  }

  void stream_audio(const float* samples, size_t count) {
    requireStream();
    if (!withAudio_) throw std::runtime_error("[bllm] stream opened without audio");
    pcm_.insert(pcm_.end(), samples, samples + count);
    drainChunks();
  }

  // Flush whatever is buffered, close the media block, ask, and generate.
  std::string stream_ask(const std::string& text, int max_new = 256, const OnText& on_text = {}) {
    requireStream();
    // TTFT is measured from the question, not from stream_begin: everything the
    // capture loop already pushed was encoded while the camera was still running.
    text_->mark_turn_start();
    Stream s(text_->hidden(), text_->next_pos());
    emitChunk(s, /*final=*/true);
    s.nextPos = st_ + std::max(maxRel_, 0) + 1;
    if (withAudio_) appendText(s, {audio_eos_});
    if (withVideo_) appendText(s, {vision_eos_});
    appendText(s, tk_.encode(text));
    closeTurn(s);
    feedStream(s);
    streaming_ = false;
    return streamDecode(max_new, on_text);
  }

  bool streaming() const { return streaming_; }
  int context_left() const { return text_->context_left(); }
  int tokens_used() const { return text_->position(); }
  // How many decoder tokens a second of video/audio costs at the current settings.
  double video_tokens_per_second() const { return withVideo_ ? fps_ * vision_->n_token() / 2.0 : 0.0; }
  static constexpr double kAudioTokensPerSecond = 25.0;

 private:
  // The turn being assembled: one hidden row + one mrope position per token.
  struct Stream {
    Stream(int hidden, int next) : H(hidden), nextPos(next) {}
    void push(const float* row, Pos3 p) {
      rows.insert(rows.end(), row, row + H);
      pos.push_back(p);
      ++n;
    }
    int H, nextPos, n = 0;
    std::vector<float> rows;
    std::vector<Pos3> pos;
  };

  void appendText(Stream& s, const std::vector<int>& ids) {
    std::vector<float> row(s.H);
    for (int id : ids) {
      embed_->lookup(id, row.data());
      s.push(row.data(), Pos3::scalar(s.nextPos++));
    }
  }

  void openTurn(Stream& s) {
    if (first_turn_) {
      appendText(s, {im_start_});
      appendText(s, tk_.encode("system\n" + cfg_.chat.system));
      appendText(s, {im_end_});
      appendText(s, tk_.encode("\n"));
      first_turn_ = false;
    } else {
      appendText(s, {im_end_});                  // close the previous assistant turn
      appendText(s, tk_.encode("\n"));
    }
    appendText(s, {im_start_});
    appendText(s, tk_.encode("user\n"));
  }
  void closeTurn(Stream& s) {
    appendText(s, {im_end_});
    appendText(s, tk_.encode("\n"));
    appendText(s, {im_start_});
    appendText(s, tk_.encode("assistant\n"));
  }
  void feedStream(Stream& s) {
    if (s.n) text_->feed_embeds(s.rows.data(), s.n, s.pos.data());
  }

  // ── live-stream internals ─────────────────────────────────────────────────
  static constexpr int kSampleRate = 16000;
  void requireStream() const { if (!streaming_) throw std::runtime_error("[bllm] no stream open"); }

  bool chunkReady() const {
    if (withVideo_ && (int)frames_.size() < framesPerChunk_) return false;
    if (withAudio_ && pcm_.size() < (size_t)kSampleRate * kSecondsPerChunk) return false;
    return true;
  }
  void drainChunks() {
    while (chunkReady()) {
      Stream s(text_->hidden(), 0);
      emitChunk(s, /*final=*/false);
      feedStream(s);
    }
  }

  // One 2-second slice: the chunk's video frame-pairs, then its audio tokens — the
  // same order the offline `use_audio_in_video` interleave produces.
  void emitChunk(Stream& s, bool final) {
    if (withVideo_ && !frames_.empty()) {
      int nf = final ? (int)frames_.size() : framesPerChunk_;
      if (final && (nf & 1)) frames_.push_back(frames_.back()), ++nf;   // pad the odd tail
      const int G = vision_->spec().llm_grid(), TOK = vision_->n_token();
      for (int p = 0; p < nf / 2; ++p, ++pairIdx_) {
        const float* e = vision_->encode_pair(frames_[2 * p].data(), frames_[2 * p + 1].data(), frameW_, frameH_);
        const int t = (int)(pairIdx_ * (2.0 / fps_) * kPositionIdPerSecond);
        for (int idx = 0; idx < TOK; ++idx)
          s.push(e + (size_t)idx * s.H, Pos3{st_ + t, st_ + idx / G, st_ + idx % G});
        maxRel_ = std::max(maxRel_, std::max(t, G - 1));
      }
      frames_.erase(frames_.begin(), frames_.begin() + nf);
    }
    if (withAudio_ && !pcm_.empty()) {
      const size_t want = (size_t)kSampleRate * kSecondsPerChunk;
      const size_t ns = final ? pcm_.size() : want;
      if (ns > kSampleRate / 20) {                                       // < 50 ms is not codeable
        const std::vector<float> rows = audioTower().encode({pcm_.data(), ns});
        const int n = (int)(rows.size() / s.H);
        for (int i = 0; i < n; ++i) s.push(rows.data() + (size_t)i * s.H, Pos3::scalar(st_ + audioIdx_++));
        maxRel_ = std::max(maxRel_, audioIdx_ - 1);
      }
      pcm_.erase(pcm_.begin(), pcm_.begin() + std::min(ns, pcm_.size()));
    }
  }

  // <|vision_bos|> <256 vision rows> <|vision_eos|>. The <|IMAGE|> placeholder ids
  // that HF's template expands never reach the graph — since the host owns the
  // embedding stream, the tower's rows go straight in where they would have sat.
  void appendImage(Stream& s, const ImageRGB& im) {
    if (!im.rgb || im.width <= 0 || im.height <= 0)
      throw std::runtime_error("[bllm] empty image");
    appendText(s, {vision_bos_});
    const float* emb = vision_->encode(im.rgb, im.width, im.height);
    const int G = vision_->spec().llm_grid();
    const int st = s.nextPos;
    for (int r = 0; r < G; ++r)
      for (int c = 0; c < G; ++c)
        s.push(emb + (size_t)(r * G + c) * s.H, Pos3{st, st + r, st + c});
    s.nextPos = st + G;                          // next text token continues after the block
    appendText(s, {vision_eos_});
  }

  // <|audio_bos|> <one row per 20 ms of audio> <|audio_eos|>. Audio is 1-D, so its
  // tokens take ordinary scalar rope positions.
  void appendAudio(Stream& s, const AudioPCM& au) {
    if (!au.samples || au.count == 0) throw std::runtime_error("[bllm] empty audio");
    appendText(s, {audio_bos_});
    const std::vector<float> emb = audioTower().encode(au);
    const int n = (int)(emb.size() / s.H);
    for (int i = 0; i < n; ++i) s.push(emb.data() + (size_t)i * s.H, Pos3::scalar(s.nextPos++));
    appendText(s, {audio_eos_});
  }

  // Qwen2.5-Omni TMRoPE constants (thinker config): a video grid step advances the
  // temporal position by seconds*25, and audio/video interleave in 2 s chunks.
  static constexpr int kPositionIdPerSecond = 25;
  static constexpr int kSecondsPerChunk = 2;
  static constexpr int kTokensPerChunk = kPositionIdPerSecond * kSecondsPerChunk;

  // HF's get_chunked_index: split a sorted position list into [start, end) runs, one
  // per chunk boundary crossed. Note it advances the boundary by a single chunk per
  // yield even when a value skips several — mirrored here on purpose.
  static std::vector<std::pair<int, int>> chunkRanges(const std::vector<int>& vals, int per) {
    std::vector<std::pair<int, int>> out;
    int start = 0, chunk = 1;
    for (int i = 0; i < (int)vals.size(); ++i)
      if (vals[i] >= chunk * per) { out.emplace_back(start, i); start = i; ++chunk; }
    out.emplace_back(start, (int)vals.size());
    return out;
  }

  // <|vision_bos|>[<|audio_bos|>] <video (+ interleaved audio) tokens> [<|audio_eos|>]<|vision_eos|>
  void appendVideo(Stream& s, const VideoClip& v) {
    if (v.frames.empty()) throw std::runtime_error("[bllm] empty video");
    if (v.fps <= 0) throw std::runtime_error("[bllm] video fps must be > 0");
    const bool with_audio = v.audio.samples && v.audio.count > 0;

    appendText(s, {vision_bos_});
    if (with_audio) appendText(s, {audio_bos_});
    const int st = s.nextPos;

    // Frames -> K temporal grid steps of vision_tokens() rows each.
    const int G = vision_->spec().llm_grid(), TOK = vision_->n_token();
    const int T = vision_->spec().temporal;
    const int K = ((int)v.frames.size() + T - 1) / T;
    std::vector<float> vrows((size_t)K * TOK * s.H);
    for (int k = 0; k < K; ++k) {
      const ImageRGB& f0 = v.frames[std::min((size_t)(k * T), v.frames.size() - 1)];
      const ImageRGB& f1 = v.frames[std::min((size_t)(k * T + 1), v.frames.size() - 1)];
      if (f1.width != f0.width || f1.height != f0.height)
        throw std::runtime_error("[bllm] video frames must all be the same size");
      const float* e = vision_->encode_pair(f0.rgb, f1.rgb, f0.width, f0.height);
      std::memcpy(vrows.data() + (size_t)k * TOK * s.H, e, (size_t)TOK * s.H * sizeof(float));
    }
    // Temporal position of grid step k, in rope units.
    const double sec_per_step = (double)T / v.fps;
    std::vector<int> tRel(K);
    for (int k = 0; k < K; ++k) tRel[k] = (int)(k * sec_per_step * kPositionIdPerSecond);

    auto pushVideo = [&](int lo, int hi) {
      for (int j = lo; j < hi; ++j) {
        const int k = j / TOK, idx = j % TOK;
        s.push(vrows.data() + (size_t)j * s.H,
               Pos3{st + tRel[k], st + idx / G, st + idx % G});
      }
    };

    int maxRel = K ? std::max(tRel[K - 1], G - 1) : 0;
    if (!with_audio) {
      pushVideo(0, K * TOK);
    } else {
      const std::vector<float> arows = audioTower().encode(v.audio);
      const int A = (int)(arows.size() / s.H);
      auto pushAudio = [&](int lo, int hi) {
        for (int j = lo; j < hi; ++j)
          s.push(arows.data() + (size_t)j * s.H, Pos3::scalar(st + j));
      };
      std::vector<int> vVals((size_t)K * TOK), aVals(A);
      for (int j = 0; j < K * TOK; ++j) vVals[j] = tRel[j / TOK];
      for (int j = 0; j < A; ++j) aVals[j] = j;
      const auto vc = chunkRanges(vVals, kTokensPerChunk);
      const auto ac = chunkRanges(aVals, kTokensPerChunk);
      for (size_t j = 0; j < std::max(vc.size(), ac.size()); ++j) {
        if (j < vc.size()) pushVideo(vc[j].first, vc[j].second);
        if (j < ac.size()) pushAudio(ac[j].first, ac[j].second);
      }
      maxRel = std::max(maxRel, A - 1);
    }
    s.nextPos = st + maxRel + 1;

    if (with_audio) appendText(s, {audio_eos_});
    appendText(s, {vision_eos_});
  }

  AudioTower& audioTower() {
    if (!audio_) {
      if (!has_audio())
        throw std::runtime_error("[bllm] model.json has no 'audio' + 'mel_filters' — audio unsupported");
      audio_ = std::make_unique<AudioTower>(cfg_.audio, cfg_.mel_filters);
      if (audio_->hidden() != text_->hidden())
        throw std::runtime_error("[bllm] audio/text hidden size mismatch");
    }
    return *audio_;
  }

  std::string streamDecode(int max_new, const OnText& on_text) {
    std::vector<int> gen;
    std::string emitted;
    auto on_token = [&](int id) {
      gen.push_back(id);
      std::string full = tk_.decode(gen);
      if (on_text && full.size() > emitted.size()) on_text(full.substr(emitted.size()));
      emitted = std::move(full);
    };
    NativeSamplingParams sp = sampling_;
    sp.max_new = max_new;
    sp.eos = cfg_.eos;
    text_->generate(sp, on_token);
    return tk_.decode(gen);
  }

  ModelConfig cfg_;
  Tokenizer tk_;
  std::unique_ptr<NativeEngine> text_;
  std::unique_ptr<VisionTower> vision_;
  std::unique_ptr<AudioTower> audio_;           // lazy: only built when audio is fed
  std::unique_ptr<EmbedTable> embed_;
  NativeSamplingParams sampling_;
  bool first_turn_ = true;
  int im_start_ = -1, im_end_ = -1, vision_bos_ = -1, vision_eos_ = -1;
  int audio_bos_ = -1, audio_eos_ = -1;

  // live stream state
  bool streaming_ = false, withVideo_ = false, withAudio_ = false;
  double fps_ = 2.0;
  int framesPerChunk_ = 0, st_ = 0, pairIdx_ = 0, audioIdx_ = 0, maxRel_ = -1;
  int frameW_ = 0, frameH_ = 0;
  std::vector<std::vector<uint8_t>> frames_;
  std::vector<float> pcm_;
};

}  // namespace bllm
