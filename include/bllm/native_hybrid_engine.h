// bllm::NativeHybridEngine — the self-built autoregressive engine for HYBRID
// (Gated-DeltaNet + attention) models like Qwen3.5, driving a single-graph
// `.hbm` directly on hbDNN/hbUCP, WITHOUT libxlm. Unlike NativeEngine (KV-only
// dual-graph), this drives ONE seq_len==1 decode graph whose I/O is:
//   inputs : x[1,H] (host-embedded), cos[ROT], sin[ROT], mask[nH,CL], then N
//            mixed per-layer caches (GDN: state+conv; attn: K+V) as graph tensors
//   outputs: logits[1,vocab], then the N updated caches (the graph rolls/overwrites
//            internally, so cache update is a uniform output->input hand-off)
//
// Perf design (to match/beat libxlm):
//   * Zero-copy cache ping-pong: every cache is double-buffered; each step binds
//     one buffer set as inputs and the other as outputs, then swaps. No per-token
//     memcpy of the (large) mixed caches.
//   * Caches are BPU-write then BPU-read only — the CPU never touches them, so
//     they need NO cache flush. Only the small CPU-written inputs (x/cos/sin/mask)
//     are cleaned and the logits output invalidated.
//   * Embedding lookup is host-side from an fp16 table (aarch64 native __fp16).
//
#pragma once

#include "bllm/native_graph.h"    // native_detail::{Mem, check, elemSize, alignUp, kAlign}
#include "bllm/native_sampler.h"  // bllm::Sampler (temperature/top-k/top-p/rep-pen)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bllm {

struct HybridStats {
  double decode_tps = 0, prefill_tps = 0, ttft_ms = 0;
  int ntok = 0;
  // Per-token host/BPU split over the last generate(), in ms. The BPU graph time is
  // measurable independently (hrt_model_exec), so knowing where the REST goes is the
  // only way to tell a host-side cost from a graph-side one instead of guessing.
  double prep_ms = 0;     // embed lookup + rope + mask + cache clean
  double infer_ms = 0;    // submit + wait  (should track the profiler's number)
  double sample_ms = 0;   // scanning/sorting the vocab to pick a token
};

class NativeHybridEngine {
  using Mem = native_detail::Mem;

 public:
  // hbm: the hybrid model; embed_fp16: raw fp16 embedding table [vocab*hidden];
  // model_name: the graph name inside the .hbm (default "qwen35").
  // `hbm_prefill` is optional: the seq_len=N prefill graph, either already linked
  // into `hbm` as a second graph or shipped as its own file. hbDNN packs several
  // files into one handle, so loading it separately means an existing decode .hbm
  // never has to be relinked to gain a prefill graph.
  NativeHybridEngine(const std::string& hbm, const std::string& embed_fp16,
                     const std::string& model_name = "qwen35", double rope_theta = 1e7,
                     const std::string& hbm_prefill = "")
      : theta_(rope_theta), decodeName_(model_name) {
    std::vector<std::string> files{hbm};
    if (!hbm_prefill.empty()) files.push_back(hbm_prefill);
    packed_.load(files);
    BLLM_NATIVE_CK(hbDNNGetModelHandle(&h_, packed_, model_name.c_str()));
    int ic = 0, oc = 0;
    BLLM_NATIVE_CK(hbDNNGetInputCount(&ic, h_));
    BLLM_NATIVE_CK(hbDNNGetOutputCount(&oc, h_));
    if (ic < 5) throw std::runtime_error("[hybrid] unexpected input count");
    in_.resize(ic); out_.resize(oc);
    nCache_ = ic - kFixedIn;                 // caches after x,cos,sin,mask
    if (oc != 1 + nCache_) throw std::runtime_error("[hybrid] output count != 1 + nCache");

    // --- inputs ---
    for (int i = 0; i < ic; ++i) {
      auto& p = in_[i].properties;
      BLLM_NATIVE_CK(hbDNNGetInputTensorProperties(&p, h_, i));
      fixStride(p);
      inBytes_.push_back(tensorBytes(p));
    }
    H_   = dim(in_[0].properties, in_[0].properties.validShape.numDimensions - 1);
    ROT_ = dim(in_[1].properties, in_[1].properties.validShape.numDimensions - 1);
    { const auto& ms = in_[3].properties.validShape;
      nHead_ = ms.dimensionSize[0]; CL_ = ms.dimensionSize[1]; }
    // fixed CPU-written inputs get one buffer each
    for (int i = 0; i < kFixedIn; ++i) { fixedMem_.emplace_back(); fixedMem_[i].alloc(inBytes_[i]); in_[i].sysMem = fixedMem_[i].m; }
    // caches: two buffers each (ping-pong), zero-initialised
    bufA_.resize(nCache_); bufB_.resize(nCache_);
    for (int c = 0; c < nCache_; ++c) {
      uint64_t b = inBytes_[kFixedIn + c];
      bufA_[c].alloc(b); bufB_[c].alloc(b);
      std::memset(bufA_[c].p(), 0, b); bufA_[c].clean();
      std::memset(bufB_[c].p(), 0, b); bufB_[c].clean();
    }

    // Classify each cache for snapshotting. This model is a MIX: the GDN layers hold a
    // fixed-size recurrent state (state + conv) that summarises all history and cannot
    // be trimmed, while the attention layers hold a K/V window whose leading dimension
    // IS the cache window — so only the occupied slots are worth saving. Measured on
    // Qwen3.5: attention is 70% of the cache set at ctx2048 and 83% at ctx4096, so
    // trimming it takes a short conversation's snapshot from ~122 MB to ~27 MB.
    //
    // tensorBytes() is stride[0] * dim[0], so stride[0] is exactly the bytes per
    // position — no padding to reason about. Anything whose leading dimension is not
    // the window is dumped whole.
    kvRow_.assign(nCache_, 0);
    for (int c = 0; c < nCache_; ++c) {
      const auto& p = in_[kFixedIn + c].properties;
      const auto& vs = p.validShape;
      if (vs.numDimensions >= 2 && vs.dimensionSize[0] == CL_ &&
          (uint64_t)p.stride[0] * CL_ == inBytes_[kFixedIn + c])
        kvRow_[c] = (uint64_t)p.stride[0];
    }

    // --- outputs ---
    for (int i = 0; i < oc; ++i) {
      BLLM_NATIVE_CK(hbDNNGetOutputTensorProperties(&out_[i].properties, h_, i));
    }
    logitMem_.alloc(out_[0].properties.alignedByteSize);
    out_[0].sysMem = logitMem_.m;
    const auto& ls = out_[0].properties.validShape;
    vocab_ = ls.dimensionSize[ls.numDimensions - 1];
    logitType_ = out_[0].properties.tensorType;
    logitScale_ = out_[0].properties.scale.scaleData && out_[0].properties.scale.scaleLen > 0
                      ? out_[0].properties.scale.scaleData[0] : 1.0f;

    // rope inv-freq: inv[i] = theta^(-2i/ROT)
    inv_.resize(ROT_ / 2);
    for (int i = 0; i < ROT_ / 2; ++i) inv_[i] = std::pow(theta_, -(double)(2 * i) / ROT_);

    // embedding table (fp16)
    std::ifstream f(embed_fp16, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("[hybrid] cannot open embed file: " + embed_fp16);
    size_t sz = f.tellg(); f.seekg(0);
    emb_.resize(sz / 2);
    f.read(reinterpret_cast<char*>(emb_.data()), sz);
    if (emb_.size() < (size_t)vocab_ * H_) throw std::runtime_error("[hybrid] embed table too small");

    attachPrefill(model_name + "_prefill");
    reset();
  }
  NativeHybridEngine(const NativeHybridEngine&) = delete;
  NativeHybridEngine& operator=(const NativeHybridEngine&) = delete;

  void set_sched(const native_detail::BpuSched& s) { sched_ = s; }
  const native_detail::BpuSched& sched() const { return sched_; }

  // mrope split across (t,h,w). Text-only models leave this empty and get plain
  // 1-D rope; a VLM build passes the checkpoint's section. Qwen3.5 is INTERLEAVED
  // ({11,11,10}, [THWTHW...]), which is a different layout from Omni's contiguous
  // {16,24,24} — see native_detail::mropeAxisMap. Nothing here touches the compiled
  // graph: cos/sin are host-computed vectors fed in per step.
  void set_mrope(const std::vector<int>& section, bool interleaved) {
    sec_ = native_detail::mropeAxisMap(ROT_ / 2, section, interleaved);
  }

  // Start the clock for time-to-first-token. Call before feeding a turn so TTFT
  // includes the prefill (and, for a VLM, the vision encode) rather than just the
  // first decode step.
  void mark_turn_start() { tTurn_ = std::chrono::steady_clock::now(); turnMarked_ = true; }

  // Look a token id up in the engine's own fp16 embedding table. Exposed so a VLM
  // session can build text rows from the SAME table the engine decodes against,
  // instead of mapping a second 500 MB copy.
  void embed_row(int id, float* dst) const {
    if (id < 0 || id >= vocab_) throw std::runtime_error("[hybrid] embed id out of range");
    const __fp16* row = emb_.data() + (size_t)id * H_;
    for (int i = 0; i < H_; ++i) dst[i] = (float)row[i];
  }

  int hidden() const { return H_; }
  int vocab() const { return vocab_; }
  int n_cache() const { return nCache_; }
  int cache_len() const { return CL_; }
  int position() const { return P_; }
  // Tokens that still fit. The attention layers hold a cache_len window (the GDN
  // layers are O(1) in length), and rolling past it evicts the earliest tokens —
  // which turns a full-attention model into gibberish. Refuse rather than degrade.
  int context_left() const { return CL_ - P_; }
  const HybridStats& last_stats() const { return stats_; }

  // Per-token logprobs for the last generate(), in order; empty unless it ran with
  // params.logprobs >= 0. Same contract as NativeEngine::last_logprobs().
  const std::vector<TokenLogprobs>& last_logprobs() const { return logprobs_; }

  void reset() {
    for (int c = 0; c < nCache_; ++c) {
      std::memset(bufA_[c].p(), 0, inBytes_[kFixedIn + c]); bufA_[c].clean();
      std::memset(bufB_[c].p(), 0, inBytes_[kFixedIn + c]); bufB_[c].clean();
    }
    P_ = 0; maxPos_ = -1; phase_ = false; curLogits_ = nullptr;
  }

  // Save/restore the mixed SSM+KV state — libxlm's path_prompt_cache for the hybrid engine.
  // The current cache set (GDN state+conv and attention KV, all layers) plus the last logits
  // are saved, so generate() resumes bit-identically. Bound to this model's cache layout
  // (rejected on load if it differs). File variants wrap the stream/in-memory core; the
  // in-memory snapshot()/restore() back the cheap prefix-reuse path (no temp file).
  //
  // The attention K/V is saved only for the OCCUPIED part of the window, which is most of
  // the payload — a serving prefix cache holds many of these, and a blob that ignored how
  // little context is actually used would bound the cache by the window rather than by the
  // conversation. The GDN state is fixed-size by nature and always saved whole.
  void save_state(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[hybrid] cannot write state: " + path);
    writeState(f);
    if (!f) throw std::runtime_error("[hybrid] failed writing state: " + path);
  }
  void load_state(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[hybrid] cannot read state: " + path);
    readState(f);
  }
  // In-memory snapshot of the current context — same payload as save_state, as a byte blob.
  // For prefix reuse: snapshot() a prefilled prefix once, restore() it before each query so
  // the shared prefix is never re-prefilled. Cheap (a memcpy of the caches), no file I/O.
  std::string snapshot() const {
    std::ostringstream os(std::ios::binary);
    writeState(os);
    return os.str();
  }
  void restore(const std::string& blob) {
    std::istringstream is(blob, std::ios::binary);
    readState(is);
  }

 private:
  void writeState(std::ostream& f) const {
    if (!curLogits_valid_) throw std::runtime_error("[hybrid] no state to save — feed a prompt first");
    const int32_t logitBytes = (int32_t)out_[0].properties.alignedByteSize;
    // CL_ goes in the header so a blob carries the window it was trimmed against.
    const int32_t hdr[] = {kStateMagic, nCache_, vocab_, P_, logitBytes, CL_, maxPos_};
    f.write((const char*)hdr, sizeof(hdr));
    for (int c = 0; c < nCache_; ++c) {
      const int32_t b = (int32_t)inBytes_[kFixedIn + c];
      f.write((const char*)&b, sizeof(b));
    }
    // Attention K/V is right-aligned in the window — step() masks slots [CL_-valid, CL_)
    // — so the occupied part is the LAST D positions. GDN state is written whole.
    const int D = P_ < CL_ ? P_ : CL_;
    const std::vector<Mem>& cur = phase_ ? bufB_ : bufA_;   // holds the latest post-step state
    for (int c = 0; c < nCache_; ++c) {
      const_cast<Mem&>(cur[c]).inval();                     // BPU-written; make the CPU read current
      if (kvRow_[c] && D < CL_)
        f.write((const char*)cur[c].p() + (size_t)(CL_ - D) * kvRow_[c],
                (size_t)D * kvRow_[c]);
      else
        f.write((const char*)cur[c].p(), (size_t)inBytes_[kFixedIn + c]);
    }
    const_cast<Mem&>(logitMem_).inval();
    f.write((const char*)logitMem_.p(), (size_t)logitBytes);
  }
  void readState(std::istream& f) {
    int32_t hdr[7];
    f.read((char*)hdr, sizeof(hdr));
    const int32_t logitBytes = (int32_t)out_[0].properties.alignedByteSize;
    if (!f || hdr[0] != kStateMagic || hdr[1] != nCache_ || hdr[2] != vocab_ ||
        hdr[4] != logitBytes || hdr[5] != CL_)
      throw std::runtime_error("[hybrid] state does not match this model");
    for (int c = 0; c < nCache_; ++c) {
      int32_t b; f.read((char*)&b, sizeof(b));
      if (b != (int32_t)inBytes_[kFixedIn + c])
        throw std::runtime_error("[hybrid] state cache layout mismatch");
    }
    reset();                                            // phase_=false → current set is bufA_
    // reset() zeroed everything, so a trimmed K/V only needs its occupied tail written
    // back at the same right-aligned offset it was taken from.
    const int savedP = hdr[3];
    const int D = savedP < CL_ ? savedP : CL_;
    for (int c = 0; c < nCache_; ++c) {
      if (kvRow_[c] && D < CL_)
        f.read((char*)bufA_[c].p() + (size_t)(CL_ - D) * kvRow_[c], (size_t)D * kvRow_[c]);
      else
        f.read((char*)bufA_[c].p(), (size_t)inBytes_[kFixedIn + c]);
      bufA_[c].clean();                                 // CPU-written; make the BPU read it
    }
    f.read((char*)logitMem_.p(), (size_t)logitBytes);
    if (!f) throw std::runtime_error("[hybrid] truncated / corrupt state");
    P_ = hdr[3]; maxPos_ = hdr[6]; phase_ = false;
    curLogits_ = logitMem_.p(); curLogits_valid_ = true;
  }

 public:

  // ingest prompt tokens (updates caches; keeps the last logits for sampling).
  void feed(const std::vector<int>& ids) {
    if ((int)ids.size() > context_left())
      throw std::runtime_error("[hybrid] context overflow: " + std::to_string(ids.size()) +
                               " tokens do not fit in the remaining " + std::to_string(context_left()) +
                               " of " + std::to_string(CL_) +
                               " cache slots. reset(), shorten the turn, or use a longer-context build.");
    if (!ph_) { for (int id : ids) step(id); return; }
    // With a prefill graph, embed on the host and go through the chunked path —
    // this is where a long prompt (or an image's vision rows) stops costing one
    // full weight stream per token.
    std::vector<float> rows((size_t)ids.size() * H_);
    for (size_t i = 0; i < ids.size(); ++i) {
      const int id = ids[i];
      if (id < 0 || id >= vocab_) throw std::runtime_error("[hybrid] token id out of range");
      const __fp16* src = emb_.data() + (size_t)id * H_;
      float* dst = rows.data() + i * H_;
      for (int k = 0; k < H_; ++k) dst[k] = (float)src[k];
    }
    feed_embeds(rows.data(), (int)ids.size(), nullptr);
  }

  // greedy-generate up to max_new; stop on any eos; call on_token(id) per token.
  // on_token may return true to stop after that token (session-level stop string).
  std::vector<int> generate(int max_new, const std::vector<int>& eos,
                            const std::function<bool(int)>& on_token = {}) {
    std::vector<int> gen;
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (int i = 0; i < max_new && curLogits_valid_ && P_ < CL_; ++i) {
      int tok = argmax();
      bool stop = false; for (int e : eos) if (e == tok) { stop = true; break; }
      if (stop) break;
      gen.push_back(tok);
      if (on_token && on_token(tok)) break;
      step(tok);
    }
    double dt = std::chrono::duration<double>(clk::now() - t0).count();
    stats_.ntok = (int)gen.size();
    stats_.decode_tps = dt > 0 ? gen.size() / dt : 0.0;
    return gen;
  }

  // sample up to p.max_new (temperature/top-k/top-p/rep-pen, seeded); stop on p.eos.
  // on_token may return true to stop after that token (session-level stop string).
  std::vector<int> generate(const NativeSamplingParams& p,
                            const std::function<bool(int)>& on_token = {}) {
    Sampler s(p);
    std::unordered_set<int> eos(p.eos.begin(), p.eos.end());
    auto logit = [this](int i) -> float {
      return logitType_ == HB_DNN_TENSOR_TYPE_F32
                 ? reinterpret_cast<const float*>(curLogits_)[i]
                 : reinterpret_cast<const int16_t*>(curLogits_)[i] * logitScale_;
    };
    std::vector<int> gen;
    logprobs_.clear();
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    accPrep_ = accInfer_ = accSample_ = 0;
    for (int i = 0; i < p.max_new && curLogits_valid_ && P_ < CL_; ++i) {
      const auto tSamp0 = clk::now();
      int tok = s.pick(logit, vocab_);
      accSample_ += std::chrono::duration<double, std::milli>(clk::now() - tSamp0).count();
      if (tok < 0) break;                 // a constraint (grammar) has nothing left to allow
      // TTFT is measured from mark_turn_start(), so it includes the prefill (and a
      // VLM's vision encode) — not just this decode step.
      if (i == 0 && turnMarked_)
        stats_.ttft_ms = std::chrono::duration<double, std::milli>(clk::now() - tTurn_).count();
      if (eos.count(tok)) break;
      gen.push_back(tok);
      // Report against the logits this token came out of, before step() advances them.
      if (p.logprobs >= 0) logprobs_.push_back(computeLogprobs(logit, vocab_, tok, p.logprobs));
      if (on_token && on_token(tok)) break;
      s.record(tok);
      step(tok);
    }
    double dt = std::chrono::duration<double>(clk::now() - t0).count();
    stats_.ntok = (int)gen.size();
    stats_.decode_tps = dt > 0 ? gen.size() / dt : 0.0;
    if (!gen.empty()) {
      const double n = (double)gen.size();
      stats_.prep_ms = accPrep_ / n;
      stats_.infer_ms = accInfer_ / n;
      stats_.sample_ms = accSample_ / n;
    }
    return gen;
  }

  // run one decode step for `token`; leaves logits available for argmax()/generate().
  void step(int token) {
    float* x = reinterpret_cast<float*>(fixedMem_[0].p());
    const __fp16* row = emb_.data() + (size_t)token * H_;   // fp16 table -> f32 input
    for (int i = 0; i < H_; ++i) x[i] = (float)row[i];
    stepRow(nullptr, Pos3::scalar(maxPos_ + 1));
  }

  // Feed one already-embedded row at an explicit 3-D mrope position. This is the
  // whole multimodal seam: the graph's input[0] was always an embedding, so a
  // vision row goes in exactly like a token row does — only the position differs.
  // `x == nullptr` means the caller already wrote into the input buffer.
  void stepRow(const float* x, Pos3 pos) {
    const auto tPrep0 = std::chrono::steady_clock::now();
    if (x) std::memcpy(fixedMem_[0].p(), x, (size_t)H_ * sizeof(float));
    fixedMem_[0].clean();
    // cos/sin for `pos`, per the mrope axis map (plain rope when t==h==w)
    float* cs = reinterpret_cast<float*>(fixedMem_[1].p());
    float* sn = reinterpret_cast<float*>(fixedMem_[2].p());
    const int half = ROT_ / 2;
    const int32_t pv[3] = {pos.t, pos.h, pos.w};
    for (int i = 0; i < half; ++i) {
      double ang = (double)pv[sec_.empty() ? 0 : sec_[i]] * inv_[i];
      float c = (float)std::cos(ang), s = (float)std::sin(ang);
      cs[i] = cs[i + half] = c; sn[i] = sn[i + half] = s;
    }
    fixedMem_[1].clean(); fixedMem_[2].clean();
    // causal mask [1, CL]: valid = last min(P+1,CL) slots. Head-independent (the
    // causal window doesn't vary by head), so the graph carries one row and
    // broadcasts it over nHead_ internally — one write instead of nHead_ copies.
    float* mask = reinterpret_cast<float*>(fixedMem_[3].p());
    int valid = P_ + 1; if (valid > CL_) valid = CL_;
    const int lo = CL_ - valid;
    for (int j = 0; j < CL_; ++j) mask[j] = (j >= lo) ? 0.0f : -1e9f;
    fixedMem_[3].clean();
    // bind ping-pong caches: cur -> inputs, next -> outputs
    std::vector<Mem>& cur = phase_ ? bufB_ : bufA_;
    std::vector<Mem>& nxt = phase_ ? bufA_ : bufB_;
    for (int c = 0; c < nCache_; ++c) { in_[kFixedIn + c].sysMem = cur[c].m; out_[1 + c].sysMem = nxt[c].m; }
    // infer
    const auto tInfer0 = std::chrono::steady_clock::now();
    accPrep_ += std::chrono::duration<double, std::milli>(tInfer0 - tPrep0).count();
    hbUCPTaskHandle_t task = nullptr;
    BLLM_NATIVE_CK(hbDNNInferV2(&task, out_.data(), in_.data(), h_));
    native_detail::submitAndWait(task, sched_);
    hbUCPReleaseTask(task);
    logitMem_.inval();                          // logits are CPU-read
    accInfer_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tInfer0).count();
    curLogits_ = logitMem_.p(); curLogits_valid_ = true;
    phase_ = !phase_;
    ++P_;
    // P_ counts cache slots consumed; maxPos_ tracks how far rope has advanced.
    // They differ once an image is in the stream: 196 rows share one t but span a
    // 14x14 h/w grid, so the next text token continues from the grid's far corner,
    // not from P_.
    if (pos.max() > maxPos_) maxPos_ = pos.max();
  }

  // Splice a run of already-embedded rows in at explicit positions — vision rows,
  // or text rows the caller embedded itself. `pos == nullptr` continues the plain
  // scalar sequence.
  void feed_embeds(const float* rows, int n, const Pos3* pos) {
    if (n > context_left())
      throw std::runtime_error("[hybrid] context overflow: " + std::to_string(n) +
                               " rows do not fit in the remaining " +
                               std::to_string(context_left()) + " of " +
                               std::to_string(CL_) + " cache slots.");
    std::vector<Pos3> seq;
    if (!pos) {                                  // continue the plain scalar sequence
      seq.resize(n);
      for (int i = 0; i < n; ++i) seq[i] = Pos3::scalar(maxPos_ + 1 + i);
      pos = seq.data();
    }
    int i = 0;
    // Whole chunks through the prefill graph (one weight stream per N_ rows), then
    // the tail one row at a time. With no prefill graph N_ is 0 and this is a no-op,
    // leaving exactly today's behaviour.
    for (; ph_ && i + N_ <= n; i += N_)
      prefillChunk(rows + (size_t)i * H_, pos + i);
    for (; i < n; ++i) stepRow(rows + (size_t)i * H_, pos[i]);
  }

  // Does this model carry a prefill graph, and how wide is it? (0 = chunk=1 only.)
  int prefill_chunk() const { return ph_ ? N_ : 0; }

  // The next plain-rope position — what a text token following the stream so far
  // should use. Not the same as position() once an image has been fed.
  int next_pos() const { return maxPos_ + 1; }

  int argmax() const {
    if (!curLogits_valid_) return -1;
    if (logitType_ == HB_DNN_TENSOR_TYPE_F32) {
      const float* lg = reinterpret_cast<const float*>(curLogits_);
      int best = 0; float bv = lg[0];
      for (int i = 1; i < vocab_; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
      return best;
    } else {  // S16 logits
      const int16_t* lg = reinterpret_cast<const int16_t*>(curLogits_);
      return native_detail::argmaxS16(lg, vocab_);
    }
  }

  // Teacher-forced perplexity of `tokens` on a FRESH context (this reset()s the session):
  // exp( mean over i of -log p(token_i | token_<i) ). libxlm's xlm_ppl equivalent, costing
  // ~N decode steps (per-position log-probs need one token at a time).
  double perplexity(const std::vector<int>& tokens) {
    const int N = (int)tokens.size();
    if (N < 2) throw std::runtime_error("[hybrid] perplexity needs >= 2 tokens");
    reset();
    if (N > CL_)
      throw std::runtime_error("[hybrid] perplexity: " + std::to_string(N) +
                               " tokens exceed cache_len " + std::to_string(CL_));
    double lp = 0.0;
    int count = 0;
    step(tokens[0]);
    for (int i = 1; i < N && curLogits_valid_ && P_ < CL_; ++i) {
      lp += logProbOfNext(tokens[i]);
      ++count;
      if (i + 1 < N) step(tokens[i]);
    }
    return count > 0 ? std::exp(-lp / count) : 0.0;
  }

 private:
  double logProbOfNext(int tok) const {
    auto L = [this](int i) -> double {
      return logitType_ == HB_DNN_TENSOR_TYPE_F32
                 ? reinterpret_cast<const float*>(curLogits_)[i]
                 : reinterpret_cast<const int16_t*>(curLogits_)[i] * (double)logitScale_;
    };
    double m = L(0);
    for (int v = 1; v < vocab_; ++v) { const double x = L(v); if (x > m) m = x; }
    double se = 0.0;
    for (int v = 0; v < vocab_; ++v) se += std::exp(L(v) - m);
    return L(tok) - (m + std::log(se));
  }

  // ── optional prefill graph ────────────────────────────────────────────────
  //
  // A second graph in the same .hbm, identical mathematics but seq_len = N. Decode
  // here is weight-bandwidth-bound, so feeding one token at a time streams the whole
  // model once per token; this consumes N per weight load. Its absence is NOT an
  // error — an older .hbm simply keeps today's chunk=1 behaviour.
  void attachPrefill(const std::string& name) {
    // Ask what graphs the .hbm actually holds first. Calling GetModelHandle for a
    // missing graph works (it returns non-zero) but the DNN layer logs it at ERROR
    // level, so every load of a model without a prefill graph — which is every
    // model today — printed a red "Model not exists" line that looks like a fault.
    const char** names = nullptr;
    int32_t count = 0;
    if (hbDNNGetModelNameList(&names, &count, packed_) != 0 || !names) { ph_ = nullptr; return; }
    bool present = false;
    for (int32_t i = 0; i < count; ++i)
      if (names[i] && name == names[i]) { present = true; break; }
    if (!present) { ph_ = nullptr; return; }
    if (hbDNNGetModelHandle(&ph_, packed_, name.c_str()) != 0) { ph_ = nullptr; return; }
    int ic = 0, oc = 0;
    if (hbDNNGetInputCount(&ic, ph_) != 0 || hbDNNGetOutputCount(&oc, ph_) != 0 ||
        ic != (int)in_.size() || oc != (int)out_.size()) {
      ph_ = nullptr;                       // shape-incompatible: ignore rather than guess
      return;
    }
    pin_.resize(ic); pout_.resize(oc);
    for (int i = 0; i < ic; ++i) {
      auto& p = pin_[i].properties;
      BLLM_NATIVE_CK(hbDNNGetInputTensorProperties(&p, ph_, i));
      fixStride(p);
      pinBytes_.push_back(tensorBytes(p));
    }
    // x is [N, H]; every cache must match decode's exactly, since the two graphs
    // hand the same buffers back and forth.
    const auto& xs = pin_[0].properties.validShape;
    N_ = xs.numDimensions >= 2 ? xs.dimensionSize[xs.numDimensions - 2] : 0;
    if (N_ < 2 || dim(pin_[0].properties, xs.numDimensions - 1) != H_) { ph_ = nullptr; return; }
    for (int c = 0; c < nCache_; ++c)
      if (pinBytes_[kFixedIn + c] != inBytes_[kFixedIn + c]) { ph_ = nullptr; return; }
    { const auto& ms = pin_[3].properties.validShape;   // mask [1, N, CL] (broadcast over heads)
      if (ms.numDimensions != 3 || ms.dimensionSize[0] != nHead_ ||
          ms.dimensionSize[1] != N_ || ms.dimensionSize[2] != CL_) { ph_ = nullptr; return; } }

    for (int i = 0; i < kFixedIn; ++i) {
      pFixedMem_.emplace_back();
      pFixedMem_[i].alloc(pinBytes_[i]);
      pin_[i].sysMem = pFixedMem_[i].m;
    }
    for (int i = 0; i < oc; ++i)
      BLLM_NATIVE_CK(hbDNNGetOutputTensorProperties(&pout_[i].properties, ph_, i));
    pLogitMem_.alloc(pout_[0].properties.alignedByteSize);
    pout_[0].sysMem = pLogitMem_.m;
    // The prefill graph emits only the LAST row's logits ([1, vocab]) — the other
    // N-1 rows are never read, and materialising them would cost 127 MB of ION at
    // N=128 plus N times the 254 MB lm_head matmul. Accept either shape: take the
    // last row whatever the leading dimension is.
    { const auto& ls = pout_[0].properties.validShape;
      pLogitRows_ = ls.numDimensions >= 2 ? ls.dimensionSize[ls.numDimensions - 2] : 1; }
    pLogitStride_ = pout_[0].properties.stride[pout_[0].properties.validShape.numDimensions - 2];
    // One row of prefill logits must fit the decode logit buffer we copy it into.
    if (pLogitStride_ <= 0 ||
        (uint64_t)pLogitStride_ > (uint64_t)out_[0].properties.alignedByteSize ||
        pout_[0].properties.tensorType != logitType_) {
      ph_ = nullptr;
      return;
    }
  }

  // One chunk of exactly N_ already-embedded rows through the prefill graph.
  void prefillChunk(const float* rows, const Pos3* pos) {
    // These inputs are multi-ROW, unlike decode's [1,H]: hbDNN pads each leading
    // dimension up to its alignment, so index by the tensor's own stride rather
    // than assuming rows are tight. (They usually are at these sizes, which is
    // exactly what makes the assumption dangerous to rely on.)
    const int64_t xStride = rowStride(pin_[0].properties);
    const int64_t cStride = rowStride(pin_[1].properties);
    const int64_t sStride = rowStride(pin_[2].properties);
    auto* xBase = reinterpret_cast<uint8_t*>(pFixedMem_[0].p());
    for (int j = 0; j < N_; ++j)
      std::memcpy(xBase + (size_t)j * xStride, rows + (size_t)j * H_,
                  (size_t)H_ * sizeof(float));
    pFixedMem_[0].clean();

    auto* csBase = reinterpret_cast<uint8_t*>(pFixedMem_[1].p());
    auto* snBase = reinterpret_cast<uint8_t*>(pFixedMem_[2].p());
    const int half = ROT_ / 2;
    for (int j = 0; j < N_; ++j) {
      const int32_t pv[3] = {pos[j].t, pos[j].h, pos[j].w};
      float* cj = reinterpret_cast<float*>(csBase + (size_t)j * cStride);
      float* sj = reinterpret_cast<float*>(snBase + (size_t)j * sStride);
      for (int i = 0; i < half; ++i) {
        const double ang = (double)pv[sec_.empty() ? 0 : sec_[i]] * inv_[i];
        const float c = (float)std::cos(ang), sv = (float)std::sin(ang);
        cj[i] = cj[i + half] = c; sj[i] = sj[i + half] = sv;
      }
    }
    pFixedMem_[1].clean(); pFixedMem_[2].clean();

    // mask[1, N, CL] — TWO conditions, not one (see CHUNKED_PREFILL.md):
    //   * the slot holds a real token: the window is right-aligned, so with
    //     T = P_ + N_ tokens after this chunk, index i is occupied iff
    //     i >= CL_ - min(T, CL_);
    //   * causality within the chunk: new token j sits at CL_ - N_ + j, so query j
    //     must not see i > CL_ - N_ + j.
    // Writing only the causal half lets an early chunk attend to zeroed slots.
    // Head-independent (the window doesn't vary by head), so the graph carries one
    // head-row and broadcasts it internally — one write instead of nHead_ copies.
    auto* maskBase = reinterpret_cast<uint8_t*>(pFixedMem_[3].p());
    const auto& mp = pin_[3].properties;                 // [1, N, CL]
    const int T = P_ + N_;
    const int lo = CL_ - (T < CL_ ? T : CL_);
    for (int j = 0; j < N_; ++j) {
      const int hi = CL_ - N_ + j;                       // last index query j may see
      auto* row = reinterpret_cast<float*>(maskBase + (size_t)j * mp.stride[1]);
      for (int i = 0; i < CL_; ++i) row[i] = (i >= lo && i <= hi) ? 0.0f : -1e9f;
    }
    pFixedMem_[3].clean();

    std::vector<Mem>& cur = phase_ ? bufB_ : bufA_;
    std::vector<Mem>& nxt = phase_ ? bufA_ : bufB_;
    for (int c = 0; c < nCache_; ++c) {
      pin_[kFixedIn + c].sysMem = cur[c].m;
      pout_[1 + c].sysMem = nxt[c].m;
    }
    hbUCPTaskHandle_t task = nullptr;
    BLLM_NATIVE_CK(hbDNNInferV2(&task, pout_.data(), pin_.data(), ph_));
    native_detail::submitAndWait(task, sched_);
    hbUCPReleaseTask(task);
    pLogitMem_.inval();
    // Only the LAST row's logits matter — they seed the next sampled token. Copy
    // them into the decode graph's logit buffer rather than pointing at the prefill
    // one: save_state()/snapshot() serialize logitMem_ by design, so leaving
    // curLogits_ pointing elsewhere would silently snapshot the previous step's
    // logits and make a restored prefix resume from the wrong distribution.
    std::memcpy(logitMem_.p(),
                (const uint8_t*)pLogitMem_.p() + (size_t)(pLogitRows_ - 1) * pLogitStride_,
                (size_t)pLogitStride_);
    curLogits_ = logitMem_.p();
    curLogits_valid_ = true;
    phase_ = !phase_;
    P_ += N_;
    for (int j = 0; j < N_; ++j)
      if (pos[j].max() > maxPos_) maxPos_ = pos[j].max();
  }

  static constexpr int kFixedIn = 4;            // x, cos, sin, mask
  // Bytes between consecutive rows of a [rows, cols] input.
  static int64_t rowStride(const hbDNNTensorProperties& p) {
    return p.stride[p.validShape.numDimensions - 2];
  }
  static int dim(const hbDNNTensorProperties& p, int idx) { return p.validShape.dimensionSize[idx]; }
  static void fixStride(hbDNNTensorProperties& p) {
    const int nd = p.validShape.numDimensions;
    if (nd > 0 && p.stride[nd - 1] == -1) p.stride[nd - 1] = (int64_t)native_detail::elemSize(p.tensorType);
    for (int d = nd - 2; d >= 0; --d)
      if (p.stride[d] == -1)
        p.stride[d] = native_detail::alignUp(p.stride[d + 1] * p.validShape.dimensionSize[d + 1], native_detail::kAlign);
  }
  static uint64_t tensorBytes(const hbDNNTensorProperties& p) {
    const int nd = p.validShape.numDimensions;
    return nd > 0 ? (uint64_t)(p.stride[0] * p.validShape.dimensionSize[0]) : 0;
  }

  native_detail::PackedHbm packed_;
  hbDNNHandle_t h_ = nullptr;
  std::vector<hbDNNTensor> in_, out_;
  std::vector<uint64_t> inBytes_;
  // optional seq_len=N_ prefill graph; ph_ == nullptr means chunk=1 as before
  hbDNNHandle_t ph_ = nullptr;
  std::vector<hbDNNTensor> pin_, pout_;
  std::vector<uint64_t> pinBytes_;
  std::vector<Mem> pFixedMem_;
  Mem pLogitMem_;
  int64_t pLogitStride_ = 0;
  int pLogitRows_ = 1;
  int N_ = 0;
  std::string decodeName_;
  std::vector<uint64_t> kvRow_;   // per cache: bytes/position if window-indexed, else 0
  std::vector<Mem> fixedMem_, bufA_, bufB_;
  Mem logitMem_;
  std::vector<__fp16> emb_;
  std::vector<double> inv_;
  native_detail::BpuSched sched_;
  int H_ = 0, ROT_ = 0, nHead_ = 0, CL_ = 0, nCache_ = 0, vocab_ = 0, logitType_ = 0;
  float logitScale_ = 1.0f;
  double theta_ = 1e7;
  std::vector<uint8_t> sec_;      // per rope frequency: which of (t,h,w) it reads
  int P_ = 0, maxPos_ = -1;
  std::chrono::steady_clock::time_point tTurn_{};
  bool turnMarked_ = false;
  double accPrep_ = 0, accInfer_ = 0, accSample_ = 0;   // ms, reset per generate()
  bool phase_ = false, curLogits_valid_ = false;
  void* curLogits_ = nullptr;
  static constexpr int32_t kStateMagic = 0x424C4B4A;   // "BLKJ" — hybrid prompt-cache tag
  // Bumped from "BLKH" when K/V trimming changed the payload, then to "BLKJ" when
  // maxPos_ joined the header. A restored VLM context has maxPos_ != P_-1 (an image
  // occupies 196 slots but advances rope by only 14), so deriving it would silently
  // mis-position every token after a restore. Old blobs are rejected, not misread.
  HybridStats stats_;
  std::vector<TokenLogprobs> logprobs_;    // last generate()'s, when asked for
};

}  // namespace bllm
