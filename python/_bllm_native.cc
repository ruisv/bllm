// Python bindings for bllm::NativeEngine — the hbDNN LLM engine that runs a
// .hbm WITHOUT libxlm. Built as its own module `_bllm_native` linking only the
// generic hobot runtime (no OE-LLM SDK), so it is independent of the libxlm-based
// `_bllm` module. Speaks token ids; tokenization stays in Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <deque>

#include "bllm/native_engine.h"
#ifdef BLLM_HAVE_TOKENIZERS
#include "bllm/audio_io.h"     // dr_wav-backed decode, for wav paths
#include "bllm/image_io.h"     // stb-backed decode, for image paths
#include "bllm/native_llm.h"   // unified string-in/out session (needs the C++ tokenizer)
#include "bllm/native_vlm.h"   // multimodal session (Qwen2.5-Omni)
#endif

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_bllm_native, m) {
  m.doc() = "BLLM native hbDNN LLM engine (no libxlm)";

  m.attr("SEED_RANDOM") = bllm::kSeedRandom;

  nb::class_<bllm::NativeSamplingParams>(m, "SamplingParams")
      .def(nb::init<>())
      .def_rw("temp", &bllm::NativeSamplingParams::temp)
      .def_rw("top_p", &bllm::NativeSamplingParams::top_p)
      .def_rw("top_k", &bllm::NativeSamplingParams::top_k)
      .def_rw("min_p", &bllm::NativeSamplingParams::min_p)
      .def_rw("typ_p", &bllm::NativeSamplingParams::typ_p)
      .def_rw("min_keep", &bllm::NativeSamplingParams::min_keep)
      .def_rw("rep_pen", &bllm::NativeSamplingParams::rep_pen)
      .def_rw("penalty_last_n", &bllm::NativeSamplingParams::penalty_last_n)
      .def_rw("penalty_freq", &bllm::NativeSamplingParams::penalty_freq)
      .def_rw("penalty_present", &bllm::NativeSamplingParams::penalty_present)
      .def_rw("max_new", &bllm::NativeSamplingParams::max_new)
      .def_rw("seed", &bllm::NativeSamplingParams::seed)
      .def_rw("eos", &bllm::NativeSamplingParams::eos)
      .def_rw("logit_bias", &bllm::NativeSamplingParams::logit_bias);

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
            std::function<bool(int)> cb;
            if (!on_token.is_none())
              cb = [on_token](int t) {   // a truthy return stops after this token
                nb::object r = on_token(t);
                return !r.is_none() && nb::cast<bool>(r);
              };
            return e.generate(p, cb);
          },
          "params"_a, "on_token"_a = nb::none(),
          "Generate up to params.max_new ids; stop on eos; on_token(id) streams. "
          "on_token may return True to stop early.")
      .def("last_stats", &bllm::NativeEngine::last_stats)
      .def_prop_ro("n_layers", &bllm::NativeEngine::n_layers)
      .def_prop_ro("n_kv", &bllm::NativeEngine::n_kv)
      .def_prop_ro("head_dim", &bllm::NativeEngine::head_dim)
      .def_prop_ro("cache_len", &bllm::NativeEngine::cache_len)
      .def_prop_ro("vocab", &bllm::NativeEngine::vocab)
      .def_prop_ro("position", &bllm::NativeEngine::position);

#ifdef BLLM_HAVE_TOKENIZERS
  // The unified, string-in/out session: load a model DIRECTORY (model.json),
  // auto-pick the hybrid/dense engine + tokenizer + ChatML, chat in one call.
  auto run = [](auto method, auto& llm, const std::string& text, int max_new, nb::object on_text,
                const std::vector<std::string>& stop) {
    std::function<void(const std::string&)> cb;
    if (!on_text.is_none())
      cb = [&on_text](const std::string& s) { nb::gil_scoped_acquire g; on_text(s); };
    std::string out;
    { nb::gil_scoped_release r; out = method(llm, text, max_new, cb, stop); }
    return out;
  };
  nb::class_<bllm::NativeLlm>(m, "NativeLlm")
      .def(nb::init<const std::string&>(), "model_dir"_a,
           "Load a model directory (model.json) — native engine + C++ tokenizer, no libxlm.")
      .def(
          "chat",
          [run](bllm::NativeLlm& llm, const std::string& msg, int max_new, nb::object on_text,
                const std::vector<std::string>& stop) {
            return run([](bllm::NativeLlm& l, const std::string& t, int n,
                          const std::function<void(const std::string&)>& c,
                          const std::vector<std::string>& st) { return l.chat(t, n, c, st); },
                       llm, msg, max_new, on_text, stop);
          },
          "message"_a, "max_new"_a = 400, "on_text"_a = nb::none(), "stop"_a = std::vector<std::string>(),
          "ChatML multi-turn (context persists); on_text(str) streams decoded deltas. "
          "stop=[...] ends the turn on any of those strings (trimmed from the reply).")
      .def(
          "generate",
          [run](bllm::NativeLlm& llm, const std::string& prompt, int max_new, nb::object on_text,
                const std::vector<std::string>& stop) {
            return run([](bllm::NativeLlm& l, const std::string& t, int n,
                          const std::function<void(const std::string&)>& c,
                          const std::vector<std::string>& st) { return l.generate(t, n, c, st); },
                       llm, prompt, max_new, on_text, stop);
          },
          "prompt"_a, "max_new"_a = 256, "on_text"_a = nb::none(), "stop"_a = std::vector<std::string>(),
          "Raw completion; on_text(str) streams decoded deltas. stop=[...] ends generation "
          "on any of those strings (trimmed from the reply).")
      .def("reset", &bllm::NativeLlm::reset, "Start a fresh conversation.")
      .def(
          "last_timing",
          [](bllm::NativeLlm& llm) {
            const auto t = llm.last_timing();
            return nb::make_tuple(t[0], t[1], t[2]);
          },
          "Per-token (prep_ms, infer_ms, sample_ms) of the last turn — host prep, "
          "BPU submit+wait, and vocabulary scan. Hybrid models only.")
      .def(
          "perplexity",
          [](bllm::NativeLlm& llm, const std::string& text) {
            double v; { nb::gil_scoped_release r; v = llm.perplexity(text); }
            return v;
          },
          "text"_a,
          "Teacher-forced perplexity of raw text (no chat template). Resets the "
          "conversation. libxlm's xlm_ppl on the native path.")
      .def("save_state", &bllm::NativeLlm::save_state, "path"_a,
           "Save the conversation KV(+SSM) state to a file (libxlm's path_prompt_cache): "
           "evaluate a shared prefix once, reload it later to skip re-prefill.")
      .def("load_state", &bllm::NativeLlm::load_state, "path"_a,
           "Restore a state saved by save_state(); generate() resumes bit-identically.")
      .def(
          "set_prefix",
          [](bllm::NativeLlm& llm, const std::string& shared_context) {
            nb::gil_scoped_release r; llm.set_prefix(shared_context);
          },
          "shared_context"_a = std::string(),
          "Prefill the system prompt + optional shared_context ONCE and snapshot it in memory. "
          "Then ask() each query against it without re-prefilling the prefix (RAG over a "
          "document, fixed system/tool context, few-shot exemplars). One-time cost here.")
      .def(
          "ask",
          [run](bllm::NativeLlm& llm, const std::string& query, int max_new, nb::object on_text,
                const std::vector<std::string>& stop) {
            return run([](bllm::NativeLlm& l, const std::string& t, int n,
                          const std::function<void(const std::string&)>& c,
                          const std::vector<std::string>& st) { return l.ask(t, n, c, st); },
                       llm, query, max_new, on_text, stop);
          },
          "query"_a, "max_new"_a = 400, "on_text"_a = nb::none(), "stop"_a = std::vector<std::string>(),
          "Answer query against the set_prefix() prefix, INDEPENDENTLY (does not accumulate "
          "across asks). Restores the prefix snapshot, so the prefix is never re-prefilled.")
      .def("clear_prefix", &bllm::NativeLlm::clear_prefix, "Drop the cached prefix.")
      .def_prop_ro("has_prefix", &bllm::NativeLlm::has_prefix)
      .def("set_sampling", &bllm::NativeLlm::set_sampling,
           "temp"_a = 0.0f, "top_p"_a = 1.0f, "top_k"_a = 0, "rep_pen"_a = 1.0f, "seed"_a = bllm::kSeedRandom,
           "min_p"_a = 0.0f, "typ_p"_a = 1.0f, "min_keep"_a = 1, "penalty_last_n"_a = 64,
           "penalty_freq"_a = 0.0f, "penalty_present"_a = 0.0f,
           "logit_bias"_a = std::unordered_map<int, float>{}, "logprobs"_a = -1,
           "Set sampling (temp<=0 => greedy). Full libxlm parity: top_k/top_p/min_p/typ_p "
           "(min_keep floors each filter) + repeat/freq/presence penalties over penalty_last_n. "
           "seed defaults to SEED_RANDOM (a fresh draw per generation); pass a value to make "
           "sampling reproducible. logit_bias={id: bias} forces (+) or bans (-) tokens, "
           "OpenAI's [-100, 100]. logprobs>=0 records per-token log probabilities with that "
           "many alternatives, read back with last_logprobs(). "
           "Applies to the next generate()/chat().")
      .def("token_bytes",
           [](bllm::NativeLlm& l, int id) {
             const std::string b = l.token_bytes(id);
             return nb::bytes(b.data(), b.size());
           },
           "id"_a,
           "The exact bytes token `id` contributes, as bytes (empty for control tokens). "
           "Unlike decode([id]) this survives a token that holds only part of a multi-byte "
           "character — decode() would return U+FFFD there.")
      .def("last_logprobs",
           [](bllm::NativeLlm& l) {
             nb::list out;
             for (const auto& t : l.last_logprobs()) {
               nb::list alts;
               for (const auto& a : t.top) alts.append(nb::make_tuple(a.first, a.second));
               nb::dict d;
               d["id"] = t.id;
               d["logprob"] = t.logprob;
               d["top"] = alts;
               out.append(d);
             }
             return out;
           },
           "Per-token logprobs of the last turn: [{id, logprob, top: [(id, logprob), ...]}], "
           "in generated order. Empty unless set_sampling(logprobs=N) was set for that turn. "
           "These are the RAW model log-probabilities (an exact full-vocab log-softmax), so "
           "temperature/penalties/logit_bias do not move them and the generated token need "
           "not be the top entry.")
      .def("set_grammar", &bllm::NativeLlm::set_grammar, "gbnf"_a, "root"_a = "root",
           "Constrain generation to a GBNF grammar (llama.cpp's format): only tokens that "
           "keep the grammar satisfiable can be sampled, so the reply is structurally "
           "guaranteed. \"\" clears it. Applies from the next turn; the grammar restarts at "
           "its root each turn.")
      .def("clear_grammar", &bllm::NativeLlm::clear_grammar, "Drop the grammar constraint.")
      .def_prop_ro("has_grammar", &bllm::NativeLlm::has_grammar)
      .def("set_stop", &bllm::NativeLlm::set_stop, "stop"_a,
           "Persistent stop strings for every following turn (a per-call stop=[...] overrides).")
      .def("set_thinking", &bllm::NativeLlm::set_thinking, "enabled"_a,
           "Qwen3.5 reasoning: enabled=False prefills an empty <think></think> so the model "
           "answers directly (faster, fewer tokens). enabled=True (default) reasons and shows "
           "the <think>...</think>. (/no_think in-message is unreliable; this prefill is.)")
      .def_prop_ro("thinking", &bllm::NativeLlm::thinking)
      .def("set_bpu_priority", &bllm::NativeLlm::set_bpu_priority, "priority"_a,
           "BPU queue priority 0..255 (default 0 = lowest, yields to a vision pipeline). "
           "Orders the queue; does not preempt a running graph.")
      .def_prop_ro("last_decode_tps", &bllm::NativeLlm::last_decode_tps)
      .def_prop_ro("name", [](bllm::NativeLlm& l) { return l.config().name; })
      .def_prop_ro("arch", [](bllm::NativeLlm& l) { return l.config().arch; })
      .def_prop_ro("vocab_size", [](bllm::NativeLlm& l) { return l.tokenizer().vocab_size(); })
      .def("encode", [](bllm::NativeLlm& l, const std::string& s) { return l.tokenizer().encode(s); }, "text"_a)
      .def("decode", [](bllm::NativeLlm& l, const std::vector<int>& ids) { return l.tokenizer().decode(ids); }, "ids"_a)
      // ---- serving primitives: many states, one engine (session pool × prefix cache) ----
      // The payload is binary (KV/SSM rows + raw logits), so it crosses as bytes — a str
      // would send it through a UTF-8 decode it cannot survive.
      .def(
          "snapshot",
          [](bllm::NativeLlm& l) {
            std::string b;
            { nb::gil_scoped_release r; b = l.snapshot(); }
            return nb::bytes(b.data(), b.size());
          },
          "In-memory snapshot of the current context (same payload as save_state, no file). "
          "Cache it and restore() later to skip re-prefilling that prefix.")
      .def(
          "restore",
          [](bllm::NativeLlm& l, nb::bytes blob) {
            std::string s(blob.c_str(), blob.size());
            nb::gil_scoped_release r;
            l.restore(s);
          },
          "blob"_a,
          "Restore a snapshot(); generation resumes from it bit-identically. Rejected if the "
          "blob does not match this model's shape.")
      .def(
          "feed_ids",
          [](bllm::NativeLlm& l, const std::vector<int>& ids) {
            nb::gil_scoped_release r;
            l.feed_ids(ids);
          },
          "ids"_a,
          "Ingest token ids into the context WITHOUT decoding (the prefill half of "
          "generate()). Pair with decode_stream(); snapshot() in between to cache the prompt.")
      .def(
          "decode_stream",
          [](bllm::NativeLlm& llm, int max_new, nb::object on_text,
             const std::vector<std::string>& stop) {
            std::function<void(const std::string&)> cb;
            if (!on_text.is_none())
              cb = [&on_text](const std::string& s) { nb::gil_scoped_acquire g; on_text(s); };
            std::string out;
            { nb::gil_scoped_release r; out = llm.decode_stream(max_new, cb, stop); }
            return out;
          },
          "max_new"_a = 256, "on_text"_a = nb::none(), "stop"_a = std::vector<std::string>(),
          "Decode from the current context (the generation half of generate()). Feed the "
          "prompt with feed_ids() first.")
      .def_prop_ro("context_left", &bllm::NativeLlm::context_left,
                   "Tokens still free in the KV/SSM window. The window IS the context: "
                   "overflow raises rather than silently evicting.")
      .def_prop_ro("prefill_chunk", &bllm::NativeLlm::prefill_chunk,
                   "Prefill chunk width, 0 if this engine has no chunked prefill. Cache "
                   "snapshots at multiples of it: the dense engine batch-prefills runs of "
                   "this width and decode-steps the rest, and the two graphs differ "
                   "numerically under int8, so an unaligned split can change the reply.")
      .def("request_cancel", &bllm::NativeLlm::request_cancel,
           "Stop the in-flight generation at the next token boundary. Thread-safe — call it "
           "from another thread while a generation runs (a server dropping a disconnected "
           "client). The context stays consistent; nothing is generated past the cancel.")
      .def_prop_ro("canceled", &bllm::NativeLlm::canceled,
                   "Whether the last generation ended on request_cancel().");

  // The multimodal session. Media arrives either as a path (decoded by stb/dr_wav)
  // or as a raw array — an HxWx3 uint8 frame, or 16 kHz mono float PCM — which is
  // what a camera/microphone pipeline already holds. A video is a dict:
  //   {"frames": [...], "fps": 2.0, "audio": pcm|path|None}
  using RgbArray = nb::ndarray<const uint8_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
  using PcmArray = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  nb::class_<bllm::NativeVlm>(m, "NativeVlm")
      .def(nb::init<const std::string&>(), "model_dir"_a,
           "Load an omni model directory (model.json: text + visual/audio towers + embed table).")
      .def(
          "chat",
          [](bllm::NativeVlm& vlm, const std::string& text, nb::list images, nb::list audios,
             nb::list videos, int max_new, nb::object on_text, const std::vector<std::string>& stop) {
            // deques: decoded media must not move, since the *Ref structs point into it.
            std::deque<bllm::OwnedImage> ownedImg;
            std::deque<bllm::OwnedAudio> ownedAu;
            auto asImage = [&](nb::handle h) -> bllm::ImageRGB {
              if (nb::isinstance<nb::str>(h)) {
                ownedImg.push_back(bllm::loadImageRGB(nb::cast<std::string>(h)));
                const auto& o = ownedImg.back();
                return {o.rgb.data(), o.width, o.height};
              }
              RgbArray a = nb::cast<RgbArray>(h);
              if (a.shape(2) != 3) throw std::runtime_error("[bllm] image array must be HxWx3 uint8");
              return {a.data(), (int)a.shape(1), (int)a.shape(0)};
            };
            auto asAudio = [&](nb::handle h) -> bllm::AudioPCM {
              if (nb::isinstance<nb::str>(h)) {
                ownedAu.push_back(bllm::loadAudio16k(nb::cast<std::string>(h)));
                const auto& o = ownedAu.back();
                return {o.samples.data(), o.samples.size()};
              }
              PcmArray a = nb::cast<PcmArray>(h);
              return {a.data(), a.shape(0)};
            };

            std::vector<bllm::ImageRGB> imgs;
            for (nb::handle h : images) imgs.push_back(asImage(h));
            std::vector<bllm::AudioPCM> aus;
            for (nb::handle h : audios) aus.push_back(asAudio(h));
            std::vector<bllm::VideoClip> vids;
            for (nb::handle h : videos) {
              nb::dict d = nb::cast<nb::dict>(h);
              bllm::VideoClip v;
              if (d.contains("fps")) v.fps = nb::cast<double>(d["fps"]);
              if (!d.contains("frames")) throw std::runtime_error("[bllm] video dict needs 'frames'");
              for (nb::handle fh : nb::cast<nb::list>(d["frames"])) v.frames.push_back(asImage(fh));
              if (d.contains("audio") && !d["audio"].is_none()) v.audio = asAudio(d["audio"]);
              vids.push_back(std::move(v));
            }

            std::function<void(const std::string&)> cb;
            if (!on_text.is_none())
              cb = [&on_text](const std::string& s) { nb::gil_scoped_acquire g; on_text(s); };
            std::string out;
            { nb::gil_scoped_release r; out = vlm.chat(text, imgs, aus, vids, max_new, cb, stop); }
            return out;
          },
          "text"_a, "images"_a = nb::list(), "audios"_a = nb::list(), "videos"_a = nb::list(),
          "max_new"_a = 256, "on_text"_a = nb::none(), "stop"_a = std::vector<std::string>(),
          "Answer a turn of text + images (paths or HxWx3 uint8) + audio (wav paths or "
          "16 kHz mono float32) + videos ({'frames':[...], 'fps':2.0, 'audio':pcm|path}); "
          "on_text(str) streams. stop=[...] ends the answer on any of those strings.")
      .def(
          "append_turn",
          [](bllm::NativeVlm& vlm, const std::string& text, nb::list images,
             const std::string& reply) {
            std::deque<bllm::OwnedImage> ownedImg;
            std::vector<bllm::ImageRGB> imgs;
            for (nb::handle h : images) {
              if (nb::isinstance<nb::str>(h)) {
                ownedImg.push_back(bllm::loadImageRGB(nb::cast<std::string>(h)));
                const auto& o = ownedImg.back();
                imgs.push_back({o.rgb.data(), o.width, o.height});
              } else {
                RgbArray a = nb::cast<RgbArray>(h);
                if (a.shape(2) != 3) throw std::runtime_error("[bllm] image array must be HxWx3 uint8");
                imgs.push_back({a.data(), (int)a.shape(1), (int)a.shape(0)});
              }
            }
            nb::gil_scoped_release r;
            vlm.append_turn(text, imgs, reply);
          },
          "text"_a, "images"_a = nb::list(), "reply"_a,
          "Replay a turn (text + images) we already know the reply to — prefill only, "
          "no decode. For a stateless HTTP server replaying history it did not just "
          "generate itself: skips the wasted (and, since sampling is stochastic, "
          "possibly non-reproducing) decode of a reply the caller already has.")
      // Live streaming: push camera frames / mic PCM as they arrive, ask any time.
      .def("stream_begin", &bllm::NativeVlm::stream_begin, "fps"_a = 2.0, "with_audio"_a = true,
           "Open a live media turn. fps<=0 means audio only.")
      .def(
          "stream_frame",
          [](bllm::NativeVlm& vlm, RgbArray a) {
            if (a.shape(2) != 3) throw std::runtime_error("[bllm] frame must be HxWx3 uint8");
            nb::gil_scoped_release r;
            vlm.stream_frame({a.data(), (int)a.shape(1), (int)a.shape(0)});
          },
          "frame"_a, "Push one HxWx3 uint8 frame.")
      .def(
          "stream_audio",
          [](bllm::NativeVlm& vlm, PcmArray a) {
            nb::gil_scoped_release r;
            vlm.stream_audio(a.data(), a.shape(0));
          },
          "pcm"_a, "Push 16 kHz mono float32 samples.")
      .def(
          "stream_ask",
          [](bllm::NativeVlm& vlm, const std::string& text, int max_new, nb::object on_text,
             const std::vector<std::string>& stop) {
            std::function<void(const std::string&)> cb;
            if (!on_text.is_none())
              cb = [&on_text](const std::string& s) { nb::gil_scoped_acquire g; on_text(s); };
            std::string out;
            { nb::gil_scoped_release r; out = vlm.stream_ask(text, max_new, cb, stop); }
            return out;
          },
          "text"_a, "max_new"_a = 256, "on_text"_a = nb::none(), "stop"_a = std::vector<std::string>(),
          "Close the live media block, ask, and generate. stop=[...] ends on any of those strings.")
      .def_prop_ro("streaming", &bllm::NativeVlm::streaming)
      .def_prop_ro("context_left", &bllm::NativeVlm::context_left)
      .def_prop_ro("tokens_used", &bllm::NativeVlm::tokens_used)
      .def_prop_ro("video_tokens_per_second", &bllm::NativeVlm::video_tokens_per_second)
      .def("reset", &bllm::NativeVlm::reset, "Start a fresh conversation.")
      .def("set_sampling", &bllm::NativeVlm::set_sampling,
           "temp"_a = 0.0f, "top_p"_a = 1.0f, "top_k"_a = 0, "rep_pen"_a = 1.0f, "seed"_a = bllm::kSeedRandom,
           "min_p"_a = 0.0f, "typ_p"_a = 1.0f, "min_keep"_a = 1, "penalty_last_n"_a = 64,
           "penalty_freq"_a = 0.0f, "penalty_present"_a = 0.0f,
           "logit_bias"_a = std::unordered_map<int, float>{})
      .def("set_stop", &bllm::NativeVlm::set_stop, "stop"_a,
           "Persistent stop strings for every following turn (a per-call stop=[...] overrides).")
      .def("set_thinking", &bllm::NativeVlm::set_thinking, "enabled"_a,
           "Qwen3.5 reasoning: enabled=False prefills an empty <think></think> so the model "
           "answers directly (faster, fewer tokens). enabled=True (default) reasons and shows "
           "the <think>...</think>. Same mechanism as NativeLlm.set_thinking.")
      .def_prop_ro("thinking", &bllm::NativeVlm::thinking)
      .def("set_bpu_priority", &bllm::NativeVlm::set_bpu_priority, "priority"_a,
           "BPU queue priority 0..255 for the text + vision + audio graphs.")
      .def_prop_ro("vision_tokens", &bllm::NativeVlm::vision_tokens)
      .def_prop_ro("vision_image_size", &bllm::NativeVlm::vision_image_size)
      .def_prop_ro("has_audio", &bllm::NativeVlm::has_audio)
      .def_prop_ro("last_decode_tps", &bllm::NativeVlm::last_decode_tps)
      .def_prop_ro("last_ttft_ms", &bllm::NativeVlm::last_ttft_ms)
      .def("encode", &bllm::NativeVlm::encode, "text"_a,
           "Plain tokenizer encode (no ChatML wrapping) — a token count for usage "
           "accounting, not a way to feed text.")
      .def_prop_ro("name", [](bllm::NativeVlm& v) { return v.config().name; });
#endif
}
