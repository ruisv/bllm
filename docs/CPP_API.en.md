<!-- English C++ API docs; 中文（默认）见 CPP_API.zh.md -->

# BLLM C++ API Reference (English)

The C++ library, namespace `bllm`. The public headers under
[`include/bllm/`](../include/bllm/) are the source of truth — this document
summarizes them with usage. For the Python bindings see [`API.en.md`](API.en.md);
the names map 1:1 (Python `snake_case` ⇄ the same C++ method names).
中文：[CPP_API.zh.md](CPP_API.zh.md).

- [Linking](#linking) · [Conventions](#conventions)
- Model: [ModelConfig](#modelconfig) · [Tokenizer](#tokenizer)
- Sessions: [NativeLlm — text chat](#nativellm--text-chat) ·
  [NativeVlm — multimodal](#nativevlm--multimodal) ·
  [NativeEmbedder — text embeddings](#nativeembedder--text-embeddings)
- Generation control: [Sampling parameters](#sampling-parameters) ·
  [GBNF grammar constraints](#gbnf-grammar-constraints)
- Under the sessions: [NativeEngine and NativeHybridEngine](#nativeengine-and-nativehybridengine) ·
  [Media loading helpers](#media-loading-helpers)
- [Worked example](#worked-example) · [Python mapping](#python-mapping)

## Linking

```cpp
#include "bllm/native_llm.h"      // text chat
#include "bllm/native_vlm.h"      // multimodal (image / audio / video)
#include "bllm/native_embedder.h" // text embeddings
```

One CMake target, `bllm::native`, needing only the board's generic hobot runtime
(plus `tokenizers-cpp` for string I/O):

```cmake
find_package(bllm REQUIRED)
target_link_libraries(app PRIVATE bllm::native)
```

For the conda packages and the on-board build see the
[README](../README.en.md#build-from-source). Everything here runs **only on an
RDK S100 / S100P / S600 board** — the hobot runtime and the BPU are board-only.
The ION carveout must be enlarged (and the board rebooted) first, or the runtime
segfaults where it allocates the weights — see [`MODELS.en.md`](MODELS.en.md).

## Conventions

- **Namespace** `bllm`; headers are `.h`, split by area, with no umbrella header.
- **Errors** are `std::runtime_error` (messages start with `[bllm]` / `[native]` /
  `[embed]`), covering the wrong model class, an overflowed KV window, a snapshot
  that does not match the model, and so on. Wrap calls in `try/catch`.
- **A model is a directory** (`model.json` + `.hbm` + `tokenizer.json`). Every
  session class takes either that path or an already-parsed `ModelConfig`.
- **Session classes are non-copyable** — each owns its BPU graphs and caches.
- **One generation at a time per session.** The only method safe to call from
  another thread is `NativeLlm::request_cancel()`.
- **The KV / SSM window is the context**: overflowing throws rather than silently
  evicting the oldest tokens (eviction turns this family of models into garbage).
  `context_left()` lets you see it coming.

---

## ModelConfig

`bllm/model_config.h` — the `model.json` manifest that makes a model
self-describing: arch, the `.hbm`, tokenizer, the (hybrid) embed table, eos ids,
cache_len, rope and the chat template. It is the seam between offline conversion
(which emits `model.json`) and the runtime (which loads a model in one call
instead of guessing).

```cpp
bllm::ModelConfig cfg = bllm::loadModelConfig("/models/qwen3.5-2b");
// relative paths are resolved against the directory
```

| Field | Meaning |
|---|---|
| `dir` / `name` | model directory / model name. |
| `arch` | `"dense"` / `"hybrid"` (SSM) / `"omni"` (multimodal) / `"embed"` (encoder). |
| `hbm` | absolute path to the text tower's `.hbm`. |
| `tokenizer` | path to `tokenizer.json`. |
| `embed` | host-side embedding table (needed by hybrid and omni). |
| `visual` / `audio` / `mel_filters` | vision tower / audio tower / mel filters (multimodal). |
| `graph` | graph name inside the `.hbm` (hybrid; defaults to `"qwen35"`). |
| `hbm_prefill` | optional `seq_len=N` chunked prefill graph. **Its absence is not an error** — the engine falls back to token-by-token ingestion. |
| `eos` | stop-token ids. |
| `mrope_section` / `mrope_interleaved` | how the VLM's rope dims split across (t,h,w), and the frequency layout of that split. |
| `vision` | `VisionCfg{patch, merge, temporal, mean[3], std[3]}`. |
| `cache_len` | informational. |
| `bpu_cores` | BPU cores the `.hbm` was compiled for; **> 1 must be submitted to specific cores**, and the session sets the core mask at construction. |
| `rope_theta` | rope base. |
| `chat` | `ChatConfig{format, im_start, im_end, bos, r_user, r_assistant, r_system, r_end, system}`. |

Predicates: `is_hybrid()`, `is_omni()`, `is_embed()`, `has_vision()`.

> **Multimodality is a property of the PACKAGE, not of the decoder family.**
> `has_vision()` (a `visual` tower in the directory) is what decides whether a
> model can answer about images: `"omni"` is a dense text tower with vision and
> audio, Qwen3.5 is a hybrid one with vision, and both run through `NativeVlm`.

> **The `VisionCfg` constants have to be carried by the manifest.** Nothing about
> a compiled graph reveals the patch size or the pixel normalization — a patch-14
> preprocessor feeding a patch-16 graph produces correctly-**shaped** garbage.

## Tokenizer

`bllm/tokenizer.h` — a thin wrapper over HuggingFace `tokenizers` (via
mlc-ai/tokenizers-cpp). This is what lets the native engine speak **strings**
rather than just token ids; encode/decode match HF exactly.

```cpp
bllm::Tokenizer tk = bllm::Tokenizer::fromFile("/models/x/tokenizer.json");
// or bllm::Tokenizer::fromBlob(json_string)
std::vector<int> ids = tk.encode("hello");
std::string text     = tk.decode(ids);
int n = tk.vocab_size();
int id = tk.token_to_id("<|im_start|>");
std::string tok = tk.id_to_token(id);
```

Each session holds one; `tokenizer()` returns a const reference to it.

---

## NativeLlm — text chat

`bllm/native_llm.h` — the unified string-in / string-out session over the
self-built native engines. It loads a model directory, builds the right engine
(hybrid Gated-DeltaNet for Qwen3.5, dense KV for Qwen2.5/DeepSeek/…), holds a C++
tokenizer, and exposes `generate()` / `chat()` / `reset()` on strings, with
streaming.

```cpp
#include "bllm/native_llm.h"

bllm::NativeLlm llm("/models/qwen3.5-2b");
std::string reply = llm.chat("Hello", 256,
                             [](const std::string& s){ std::cout << s << std::flush; });
```

Handing it an `arch="omni"` or `arch="embed"` directory **throws with the class
you should have used**, rather than building a broken session. A multi-core
`.hbm` (`bpu_cores > 1`) gets its core mask set at construction — the runtime
rejects `HB_UCP_CORE_ANY` for multi-core tasks.

### Generation

| Method | Meaning |
|---|---|
| `std::string generate(prompt, max_new=256, on_text={}, stop={})` | Raw completion, no chat template. |
| `std::string chat(msg, max_new=256, on_text={}, stop={})` | ChatML / Phi multi-turn; the context persists across calls, so you feed only the new turn. |
| `void reset()` | Fresh conversation. |
| `double last_decode_tps() const` | Last turn's decode tok/s. |
| `std::array<double,3> last_timing() const` | Last turn's per-token `{prep, infer, sample}` in ms. **Hybrid only**, zeros on dense. The graph's own time is measurable with `hrt_model_exec`, so this is what makes the REMAINDER attributable instead of guessed at. |

`on_text` is a `std::function<void(const std::string&)>` receiving **deltas**;
the return value is always the full reply.

### Sampling

```cpp
llm.set_sampling(/*temp=*/0.7f, /*top_p=*/0.9f, /*top_k=*/0, /*rep_pen=*/1.0f,
                 /*seed=*/bllm::kSeedRandom);
// the full signature also takes: min_p, typ_p, min_keep, penalty_last_n,
//                                penalty_freq, penalty_present, logit_bias, logprobs
```

- `temp <= 0` is greedy (argmax).
- `logit_bias` is an `std::unordered_map<int,float>` — OpenAI's parameter, same
  `[-100, 100]` convention, forcing or banning a token. An empty map clears it.
- `logprobs >= 0` makes the **next turn** record per-token log probabilities
  (that many alternatives each), read afterwards with `last_logprobs()`; `-1`
  (the default) skips the two extra full-vocab passes.

| Method | Meaning |
|---|---|
| `const std::vector<TokenLogprobs>& last_logprobs() const` | Last turn's per-token log probabilities in generated order, **trimmed** to the tokens that survive into the returned text (a stop string must not leave the report longer than the content it describes). |
| `std::string token_bytes(int id) const` | The exact bytes a token contributes; empty for control tokens. **`decode({id})` cannot answer this**: a token holding half a multi-byte character comes back as U+FFFD, so reassembling text per token would corrupt any CJK the tokenizer split — which is what OpenAI's `bytes` field exists for. |

`last_logprobs()` reports the **raw** model distribution (exact full-vocab
log-softmax). It does not move with temperature, penalties or `logit_bias`, so
the generated token is not necessarily `top[0]`.

### Stop strings and thinking

| Method | Meaning |
|---|---|
| `void set_stop(std::vector<std::string>)` | Persistent stop strings: end the turn as soon as one appears, and return the reply **without** it. Complements eos-token stopping (a stop string can span tokens and need not be a token). Empty clears; a per-call `stop` overrides for that call only. |
| `void set_thinking(bool)` / `bool thinking() const` | Qwen3.5 reasoning control. `false` prefills a closed empty `<think></think>` after the assistant marker (== the official template's `enable_thinking=False`) so the model answers **directly** — faster, fewer tokens. `true` (the default) reasons and shows the `<think>…</think>` block. Note the in-message `/no_think` soft switch is **not reliable** on this checkpoint; this prefill is. |

While streaming, a stop string's own prefix is **held back** until it either
completes or is ruled out, so a client never sees half a stop string and then has
it taken away.

### Context, priority, perplexity

| Method | Meaning |
|---|---|
| `int context_left() const` | Tokens that still fit in this conversation's KV / SSM window. |
| `void set_bpu_priority(int)` | BPU queue priority 0..255 (0, the default, is lowest and lets a co-resident vision pipeline jump the queue). It **orders the queue; it does not preempt** a graph that is already running — switching inside a graph additionally needs `max_time_per_fc` at compile time. The call preserves the core mask set at load. |
| `double perplexity(const std::string&)` | Teacher-forced perplexity of raw text (no chat template). **Resets the conversation.** |
| `const ModelConfig& config() const` / `const Tokenizer& tokenizer() const` | Read-only access. |

### State and reusable prefixes

```cpp
llm.save_state("ctx.bin");     // KV (+SSM) state + last logits
llm.load_state("ctx.bin");     // generation resumes bit-identically
```

A state is bound to its model; loading one from another model is rejected.

For workloads that ask **many independent questions against one shared context**
(RAG over a document, a fixed system prompt + tool defs, few-shot exemplars),
prefill the shared prefix **once** and restore the snapshot per question instead
of re-prefilling it.

```cpp
llm.set_prefix(document_text);          // prefill once, snapshot in memory
std::string a1 = llm.ask("What does the report conclude?");
std::string a2 = llm.ask("Which risks does it name?");   // reuses the prefix
```

| Method | Meaning |
|---|---|
| `void set_prefix(const std::string& shared_context = "")` | Prefill the system block + optional shared context and snapshot it as the reusable base. |
| `std::string ask(query, max_new=256, on_text={}, stop={})` | Answer against the fixed prefix, **independently** — unlike `chat()`, it does not accumulate. |
| `bool has_prefix() const` / `void clear_prefix()` | Whether a prefix is set / drop it. |

### Serving primitives

`set_prefix()`/`ask()` serve **one** fixed prefix. A server juggling many
conversations holds many states and picks one per request, so it drives the
snapshot itself:

| Method | Meaning |
|---|---|
| `std::string snapshot() const` | Snapshot the current context to a byte string (same payload as `save_state()`, in memory). |
| `void restore(const std::string& blob)` | Restore it; generation resumes bit-identically. Cheap relative to a prefill but **not free** — `restore()` clears the whole cache window before copying rows back. |
| `int prefill_chunk() const` | Prefill chunk width, or `0` when this engine does not chunk. |
| `void feed_ids(const std::vector<int>& ids)` | Ingest tokens without decoding (the prefill half of `generate()`). It takes ids rather than text because the server has already tokenized — and re-encoding a text delta could tokenize differently at the seam than the ids the cache was keyed on. |
| `std::string decode_stream(max_new=256, on_text={}, stop={})` | Decode from the current context (the generation half of `generate()`). |
| `void request_cancel()` / `bool canceled() const` | Stop an in-flight generation cleanly at the next token boundary. **Thread-safe** — call it from another thread (a server cancelling a disconnected client). The context stays consistent: nothing is generated past the cancel point. The flag is one-shot, so a cancel that lands between turns is dropped rather than killing the next one. |

> ⚠️ **Snapshot only at multiples of `prefill_chunk()`.** The dense engine ingests
> a long enough run of tokens through the batch prefill graph and decode-steps the
> remainder, and the two graphs are **not numerically identical under int8**.
> Splitting a prompt at an arbitrary point therefore changes which tokens went
> through which graph — and so can change the reply. Snapshot at those multiples
> and the segmentation matches a straight-through prefill, making reuse exact.
> `prefill_chunk() == 0` means any split is exact (the hybrid engine ingests one
> token at a time unless it carries an `hbm_prefill` graph).

A worked implementation is in `docker/serving/` (`cache.py` for the prefix cache,
`engine.py` for the single worker thread).

---

## NativeVlm — multimodal

`bllm/native_vlm.h` — the multimodal sibling of `NativeLlm`: it loads a model
directory carrying a vision tower and answers a turn of text + images (+ audio /
video).

How multimodality actually works here: the text tower's graph takes an input
**embedding** (not a token id) plus host-supplied rope cos/sin. So the session

1. tokenizes the ChatML turn and embeds each token on the host (`EmbedTable`),
2. runs the vision tower and **splices** its rows into that same stream where the
   image sits — HF's `<|IMAGE|>` placeholders are never needed,
3. assigns 3-D mrope positions — scalar for text, `(t, t+row, t+col)` for image
   patches — and feeds the whole stream to prefill/decode.

```cpp
#include "bllm/native_vlm.h"
#include "bllm/image_io.h"

bllm::NativeVlm vlm("/models/qwen3.5-0.8b-vlm448");
bllm::OwnedImage img = bllm::loadImageRGB("bears.jpg");
std::string a = vlm.chat("What is in this image?", {{img.rgb.data(), img.width, img.height}},
                         {}, {}, 256,
                         [](const std::string& s){ std::cout << s << std::flush; });
```

Data types:

```cpp
struct ImageRGB { const uint8_t* rgb; int width, height; };  // interleaved RGB8, caller owns
struct AudioPCM { const float* samples; size_t count; };     // 16 kHz mono float
struct VideoClip { std::vector<ImageRGB> frames; double fps = 2.0; AudioPCM audio; };
```

| Method | Meaning |
|---|---|
| `std::string chat(text, images={}, audios={}, videos={}, max_new=256, on_text={}, stop={})` | One turn; media precedes `text`, as the Qwen chat template does. |
| `void append_turn(text, images, reply)` | Replay a turn whose reply is **already known** — prefill only (`openTurn`+media+text+`closeTurn`+reply text), no decode loop. Leaves context exactly where a real `chat()` call would have; for a serving layer replaying history it just generated itself and the client resent — see `_serve_vlm` in `docker/serving/engine.py`. |
| `std::vector<int> encode(text) const` | Plain tokenizer encode (no ChatML wrapping) — a token count for usage accounting. |
| `int vision_tokens() const` | Tokens per frame-pair / per image. |
| `int vision_image_size() const` | The square side this tower was compiled for — sample frames at it to avoid resizing twice. |
| `bool has_audio() const` | Whether this package carries an audio tower. |
| `double last_ttft_ms() const` | Last turn's first-token latency, **including the media encode**. |
| `double last_decode_tps() const` · `int context_left() const` · `int tokens_used() const` | As on `NativeLlm`. |
| `set_sampling(...)` / `set_stop(...)` / `set_bpu_priority(...)` / `reset()` | Same meaning as on `NativeLlm`; `set_bpu_priority` covers the text, vision and audio graphs alike. |

Construction checks that the vision and text hidden sizes agree, and throws if
they do not — a mismatch there is a silent garbage splice.

### Live streaming

Each 2-second chunk is encoded into the KV cache as it arrives, so by the time
the question comes most of the prefill is already paid for.

```cpp
vlm.stream_begin(/*fps=*/2.0);            // fps <= 0 means audio only
while (grabbing) {
  vlm.stream_frame(frame);                // ImageRGB; all frames must share a size
  vlm.stream_audio(pcm, n);               // 16 kHz mono float
}
std::string a = vlm.stream_ask("What just happened?");
bool live = vlm.streaming();
```

> **The budget is the KV window.** A frame-pair spans `temporal_patch_size` (2)
> frames, so at 2 fps video costs `vision_tokens()` per second of wall-clock and
> audio costs about 25. A 2048-slot model therefore holds ~8 s of video with the
> stock 448px tower (32 s with a 224px re-export), or ~80 s of audio alone.
> `stream_frame` / `stream_audio` **throw once the budget is spent** rather than
> evict; `context_left()` lets a caller see it coming, and
> `video_tokens_per_second()` reports the current cost.

> **Video is Omni-only.** The video path here implements Qwen2.5-Omni's TMRoPE (a
> temporal position of `seconds*25` per grid step, audio interleaved with video in
> 2 s chunks). Qwen3.5 separates frames with explicit **timestamp tokens**, which
> is a different position assignment entirely, so passing `videos=` or
> `stream_begin(fps>0)` on a hybrid package **throws** rather than quietly
> producing a plausible-looking, mis-ordered answer. Still images are unaffected:
> their 3-D layout IS shared, and it is checked against the HF reference.

## NativeEmbedder — text embeddings

`bllm/native_embedder.h` — text embeddings on the BPU. An embedding model is far
simpler than a chat model: no KV cache across steps, no decode loop, no sampling.
One causal forward over the prompt, pool the last token's post-norm hidden state,
L2-normalise. So the compiled `.hbm` is a single **encoder** graph (produced by
replacing the recipe's `lm_head` with an identity).

```cpp
#include "bllm/native_embedder.h"

bllm::NativeEmbedder emb("/models/qwen3-vl-embedding");
std::vector<float> v = emb.embed("Paris is the capital of France.");
std::vector<float> small = emb.embed("same text", /*dim=*/256);   // Matryoshka
```

| Method | Meaning |
|---|---|
| `std::vector<float> embed(text, dim=0, instruction="")` | L2-normalised embedding. `dim > 0` truncates first (Matryoshka) and renormalises; `0` keeps the full width. |
| `std::vector<float> embed_ids(ids, dim=0)` | Same, from token ids. |
| `std::vector<int> build_ids(text, instruction="") const` | The token ids under the model's template. **Exposed because reproducing this sequence is the only way to check a port against the reference implementation.** |
| `int dimension() const` / `int max_tokens() const` | Vector width / encoder window. |

> **The input template is not negotiable.** It was recovered by diffing token ids
> against the model's own embedder, and getting it wrong **silently returns a
> different vector**:
>
> ```
> <|im_start|>system\n{instruction}<|im_end|>\n
> <|im_start|>user\n{text}<|im_end|>\n
> <|im_start|>assistant\n<|endoftext|>
>                        ^^^^^^^^^^^^^ this token's hidden state is what gets pooled
> ```

Input longer than the encoder window throws (shorten the text or compile a longer
graph) rather than truncating. A directory whose `arch` is not `"embed"` is
rejected.

For CLI usage see [`examples/embed_native.cc`](../examples/embed_native.cc), whose
`--verify` is a **smoke test, not a ship gate** (the reference sentences are near
duplicates whose pairwise similarities are nearly tied, so their ranking flips on
noise — what separates a good build from a bad one is the cosine, not the ranking).

---

## Sampling parameters

`bllm/native_sampler.h` — what a session's `set_sampling(...)` writes into, and
what you construct directly if you drive an engine yourself.

```cpp
struct NativeSamplingParams {
  float temp = 0.0f;             // <= 0 => greedy (argmax)
  float top_p = 1.0f;            // 1.0 = disabled (nucleus)
  float min_p = 0.0f;            // 0.0 = disabled
  float typ_p = 1.0f;            // 1.0 = disabled (locally typical)
  int   top_k = 0;               // <= 0 = disabled
  int   min_keep = 1;            // floor on survivors per filter
  int   max_new = 200;

  int   penalty_last_n = 64;     // penalty look-back window (<=0 = disabled)
  float rep_pen = 1.0f;          // 1.0 = disabled
  float penalty_freq = 0.0f;     // 0.0 = disabled
  float penalty_present = 0.0f;  // 0.0 = disabled

  uint64_t seed = kSeedRandom;   // kSeedRandom => nondeterministic; any other value is exact
  std::vector<int> eos;
  std::unordered_map<int, float> logit_bias;   // OpenAI's, [-100, 100]
  TokenConstraint* constraint = nullptr;       // grammar constraint, not owned
  int logprobs = -1;                           // <0 = off
};

struct TokenLogprobs {
  int id;                                      // the token actually generated
  float logprob;                               // log p(id)
  std::vector<std::pair<int, float>> top;      // (id, logprob) alternatives, most likely first
};
```

`seed` defaults to `bllm::kSeedRandom`, i.e. a fresh seed per generation — the
sampler is constructed per generation, so a fixed seed replays the same random
stream every turn and makes temperature look like it does nothing.

## GBNF grammar constraints

`bllm/native_grammar.h` — grammar-constrained decoding in llama.cpp's GBNF
format: **structured output that is guaranteed rather than requested**. At every
step, only tokens that keep the grammar satisfiable can be sampled.

```cpp
llm.set_grammar("root ::= \"yes\" | \"no\"\n");
std::string a = llm.chat("Is this sentence positive?");   // can only be yes or no
llm.clear_grammar();
bool on = llm.has_grammar();
```

- An empty string clears it; it applies from the **next turn** on, and the grammar
  restarts at its root each turn.
- Cost: a token→bytes table over the whole vocabulary, built once per session on
  first use (~315 ms on the board), plus one grammar test per candidate per step
  (measured 14.3 → 13.4 tok/s, ~6%).
- Control tokens (eos, the chat markers, anything shaped like `<|...|>`) are blank
  in that table, so a grammar **cannot** produce them.
- The underlying `bllm::Grammar` and `bllm::GrammarConstraint` can be used
  directly; `Grammar` rejects left recursion.

> The JSON Schema → GBNF converter currently lives on the Python side
> (`bllm.json_grammar(schema)`). From C++, write GBNF directly or generate it with
> the Python helper and pass the string in.

---

## NativeEngine and NativeHybridEngine

`bllm/native_engine.h` (dense KV) and `bllm/native_hybrid_engine.h`
(Gated-DeltaNet / SSM) — the layer under the session classes. Most applications
never touch them; reach for them when you need to control ingestion and state
yourself.

Shared surface:

| Member | Meaning |
|---|---|
| `void feed(const std::vector<int>& ids)` | Ingest tokens (leaving the last logits for sampling). |
| `std::vector<int> generate(const NativeSamplingParams&, on_token)` | The decode loop; `on_token` returning `true` breaks it. |
| `void reset()` | Clear the context. |
| `int position()` / `int context_left()` / `int cache_len()` | Used / remaining / total window. |
| `std::string snapshot() const` / `void restore(const std::string&)` | In-memory state. |
| `void save_state(path)` / `void load_state(path)` | The on-disk version. |
| `double perplexity(const std::vector<int>&)` | Teacher-forced perplexity. |
| `const std::vector<TokenLogprobs>& last_logprobs() const` | Last turn's per-token log probabilities. |
| `void set_sched(const native_detail::BpuSched&)` / `sched()` | BPU priority and core mask. |
| `void mark_turn_start()` | Make TTFT count the media encode too. |
| `void feed_embeds(const float* rows, int n, const Pos3* pos)` | Feed embedding rows + 3-D positions directly (the multimodal path). |

Family-specific: `NativeEngine::chunk()` (batch prefill width), `n_layers()`,
`n_kv()`, `head_dim()`, `vocab()`, `embed_input()`;
`NativeHybridEngine::prefill_chunk()`, `n_cache()`, `embed_row()`. A session's
`prefill_chunk()` forwards to whichever it holds.

`native_detail::BpuSched{priority, core_mask}` is the submit parameter. A
multi-core `.hbm` (nash-p / S600) **must** be bound to specific cores — the
runtime rejects `HB_UCP_CORE_ANY` for multi-core tasks. The mask is `(1<<N)-1`
(`HB_UCP_BPU_CORE_k == 1<<k`); single-core keeps the `CORE_ANY` default.

## Media loading helpers

Minimal file decoding for the CLI/demo paths — header-only and self-contained, no
external dependency. The runtime API (`VisionTower` / `AudioTower` / `NativeVlm`)
itself takes raw RGB8 and 16 kHz float PCM, which is what a camera or microphone
pipeline hands you.

```cpp
#include "bllm/image_io.h"    // vendored stb_image
#include "bllm/audio_io.h"    // vendored dr_wav

bllm::OwnedImage img = bllm::loadImageRGB("photo.jpg");   // {rgb, width, height}
bllm::OwnedAudio au  = bllm::loadAudio16k("clip.wav");    // {samples, sample_rate}, .seconds()
```

`loadAudio16k` mixes to mono and linearly resamples if needed. **Transcode
anything that is not a WAV first**:
`ffmpeg -i in.mp4 -ac 1 -ar 16000 -f wav out.wav`.

---

## Worked example

A minimal streaming chat program (the shape of
[`examples/chat_native.cc`](../examples/chat_native.cc)):

```cpp
#include "bllm/native_llm.h"
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
  try {
    bllm::NativeLlm llm(argv[1]);              // a model directory
    llm.set_sampling(0.7f, 0.9f, 0, 1.0f, bllm::kSeedRandom);
    llm.set_stop({"\n\n"});

    for (std::string line; std::getline(std::cin, line) && !line.empty(); ) {
      llm.chat(line, 400, [](const std::string& s){ std::cout << s << std::flush; });
      std::printf("\n[%.1f tok/s, %d context left]\n", llm.last_decode_tps(), llm.context_left());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bllm error: %s\n", e.what());
    return 1;
  }
}
```

More runnable programs are in [`examples/`](../examples/): `chat_native.cc` (a
REPL), `qwen35_native.cc`, `vlm_native.cc` (image / audio / video),
`embed_native.cc`, `tokenizer_probe.cc`, `vision_probe.cc`.

## Python mapping

| C++ | Python |
|---|---|
| `bllm::NativeLlm` | `bllm.NativeSession` |
| `bllm::NativeVlm` | `bllm.NativeVlmSession` |
| `bllm::NativeEmbedder` | no binding yet (C++ API only) |
| `bllm::loadModelConfig(dir)` | done inside `bllm.load(dir)` |
| `llm.chat(msg, max_new, on_text)` | `llm.chat(msg, max_new, on_text)` / `llm.stream_chat(msg)` |
| `llm.set_sampling(...)` | `llm.set_sampling(...)` (same keyword arguments) |
| `llm.set_grammar(gbnf)` | `llm.set_grammar(gbnf)` + `bllm.json_grammar(schema)` |
| `llm.snapshot()` / `restore()` | `llm.snapshot()` / `llm.restore()` |
| `llm.prefill_chunk()` | `llm.prefill_chunk` / `llm.cache_align(n)` |
| `tk.encode/decode` | `llm.encode/decode` |

Python additionally offers `cache_align(n)`, which hands back the largest prefix
length that is safe to snapshot; from C++, round down to a multiple of
`prefill_chunk()` yourself.
