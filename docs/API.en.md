<!-- English API docs; 中文: API.zh.md -->

# BLLM API Reference (English)

Python package `bllm`. The matching C++ type is noted at the end of each section.
中文: [API.zh.md](API.zh.md).

## Top-level functions

### `bllm.load(path, *, backend="auto", **kwargs)`

Load a model and return the right session object.

- `path` is a **model directory** (with `model.json`) → `NativeSession` for dense/hybrid,
  `NativeVlmSession` for `arch="omni"`.
- `path` is a **bare `.hbm`** and you pass `tokenizer_dir=<dir>` → the runtime synthesizes the
  manifest and returns a `NativeSession`.
- Extra keyword args are forwarded to the session constructor.

```python
llm = bllm.load("/models/qwen2.5-1.5b")
llm = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="Qwen2.5_1.5B_config/")
```

### `bllm.available_backends() -> tuple[str, ...]`

Which runtimes are available — `("native",)` in a normal build (`()` on a dev host with no
extension).

### `bllm.load_video(path, fps=2.0, size=448, max_frames=10, with_audio=True, ffmpeg="ffmpeg") -> dict`

Sample a video file into the dict `NativeVlmSession.chat(videos=...)` expects (shells out to
ffmpeg for frames + audio).

## `bllm.NativeSession` — text chat

```python
llm = bllm.NativeSession(model_dir)          # or bllm.load(...)
```

**Properties**: `name`, `arch` (`"dense"` / `"hybrid"`), `vocab_size`, `last_decode_tps`
(decode tok/s of the last turn).

**Methods** (most return `self`, chainable):

| Method | Description |
|---|---|
| `set_sampling(temp=0.0, top_p=1.0, top_k=0, rep_pen=1.0, seed=SEED_RANDOM, min_p=0.0, typ_p=1.0, min_keep=1, penalty_last_n=64, penalty_freq=0.0, penalty_present=0.0, logit_bias=None, logprobs=-1)` | Configure sampling; `temp<=0` is greedy. `logit_bias={token_id: bias}` forces (+) or bans (-) tokens on OpenAI's [-100, 100]; `logprobs>=0` records per-token log probabilities (plus that many alternatives), read back with `last_logprobs()`. |
| `last_logprobs() -> list[dict]` | Last turn's per-token log probabilities, `[{id, logprob, top: [(id, logprob), ...]}]`. Empty unless `logprobs` was set. These are the **raw** model distribution (an exact full-vocab log-softmax), unmoved by temperature / penalties / `logit_bias` — so the generated token need not be `top[0]`. |
| `set_grammar(gbnf, root="root")` / `clear_grammar()` / `has_grammar` | **GBNF grammar-constrained decoding** (llama.cpp's format): at every step only tokens that keep the grammar satisfiable may be sampled, so the structure is *guaranteed* rather than requested. `""` clears it; the grammar restarts at its root each turn. First use builds a token->bytes table over the whole vocabulary (~315 ms, once per session); after that it is one grammar test per candidate per step (measured 14.3 -> 13.4 tok/s). |
| `bllm.json_grammar(schema=None) -> str` | JSON Schema -> GBNF. No schema means "any valid JSON". Supports `type`, `enum`, `const`, `properties` + `required`, `items`, `anyOf` / `oneOf`, `$ref` into `$defs`. Two deliberate narrowings: optional properties must appear in schema order, and `additionalProperties` is treated as false. |
| `set_stop(stop: list[str])` | Persistent stop strings: end the turn as soon as any appears, trimmed from stream and return value. |
| `set_bpu_priority(priority: int)` | BPU queue priority 0..255 (default 0 = lowest, so a co-resident vision pipeline preempts). |
| `reset()` | Clear multi-turn history; start fresh. |
| `chat(message, max_new=400, on_text=None, stop=None) -> str` | ChatML multi-turn; returns the full reply, streams deltas to `on_text`. |
| `generate(prompt, max_new=256, on_text=None, stop=None) -> str` | Raw completion (no chat template). |
| `stream_chat(message, max_new=400, stop=None) -> Iterator[str]` | Generator form of `chat`. |
| `stream(prompt, max_new=256, stop=None) -> Iterator[str]` | Generator form of `generate`. |
| `chat_async(...)` / `generate_async(...)` -> `concurrent.futures.Future[str]` | Non-blocking submit (releases the GIL); one generation at a time per session. |
| `perplexity(text) -> float` | Teacher-forced perplexity of raw text (no template); resets the session. |
| `save_state(path)` / `load_state(path)` | Save/restore the session KV(+SSM) state + last logits; resumes bit-identically (skips re-prefill). |
| `set_prefix(shared_context="")` | Prefill the system prompt + optional `shared_context` (a document / fixed tool context / few-shot exemplars) ONCE and snapshot it in memory; each `ask()` reuses it. |
| `ask(query, max_new=400, on_text=None, stop=None) -> str` | Answer `query` against the `set_prefix` prefix, INDEPENDENTLY (each call restores the prefix; never re-prefills it). |
| `stream_ask(query, max_new=400, stop=None) -> Iterator[str]` | Generator form of `ask`. |
| `has_prefix -> bool` / `clear_prefix()` | Whether a prefix is set / drop the cached prefix. |
| `encode(text) -> list[int]` / `decode(ids) -> str` | Tokenize / detokenize with the model's C++ tokenizer. |
| `set_thinking(enabled)` | Qwen3.5 reasoning toggle. `False` prefills an empty `<think></think>` so the model **answers directly** (faster, fewer tokens); `True` (default) reasons and shows the `<think>...</think>`. (The in-message `/no_think` soft switch is unreliable on this checkpoint; this prefill is.) |

```python
llm.set_thinking(False)                 # no reasoning: direct answers (much faster on 4B)
```

**Prefix reuse** (many questions over one shared document; the prefix is prefilled once):

```python
llm = bllm.load("/models/qwen3.5-4b")
llm.set_prefix(open("report.txt").read())     # one-time prefill of the document
print(llm.ask("What are the conclusions?"))
print(llm.ask("Which risks are mentioned?"))  # reuses the prefix, no re-prefill
```

C++: `bllm::NativeLlm` (`bllm/native_llm.h`); `set_prefix`/`ask` are backed by the engine
in-memory `snapshot()`/`restore()` (`bllm::NativeHybridEngine` and `bllm::NativeEngine`).


### Serving primitives (rolling your own session pool / prefix cache)

`set_prefix()`/`ask()` serve ONE fixed prefix. To hold many conversation states and pick
one per request — an OpenAI-compatible server, say — drive the snapshot yourself:

| Method | Meaning |
|---|---|
| `snapshot() -> bytes` | Snapshot the current context (same payload as `save_state`, no file). |
| `restore(blob)` | Restore it; generation resumes bit-identically. Raises on a shape mismatch. |
| `feed_ids(ids)` | Ingest tokens without decoding (the prefill half of `generate()`). |
| `decode_stream(max_new=256, on_text=None, stop=None) -> str` | Decode from the current context (the generation half). |
| `stream_decode(max_new=256, stop=None) -> Iterator[str]` | The same, as a lazy generator. |
| `cache_align(n) -> int` | **Largest prefix length that is safe to snapshot** — see the warning. |
| `prefill_chunk -> int` | Prefill chunk width; `0` when the engine does not chunk (hybrid). |
| `context_left -> int` | Tokens free in the KV/SSM window. The window IS the context — overflow raises rather than silently dropping turns. |
| `request_cancel()` / `canceled -> bool` | Stop an in-flight generation at a token boundary. **Thread-safe** — call it from another thread (a server dropping a disconnected client). The context stays consistent, so the session is immediately reusable. |

> ⚠️ **Snapshots must be chunk-aligned.** The dense engine batch-prefills long enough runs
> and decode-steps the remainder, and those two graphs are **not numerically identical**
> under int8. Segmentation depends only on how many tokens remain, so the un-cached path's
> chunk boundaries sit at multiples of `prefill_chunk`; snapshot at one of those and the
> tail is segmented exactly as it would have been. Measured on-board (Qwen2.5-1.5B, greedy):
> aligned splits reproduced the original answer **5/5**, unaligned **1/18** — one turning
> "Venus is the hottest" into "Earth is the hottest". Use `cache_align()`; do not compute it
> yourself. The hybrid engine does not chunk (`prefill_chunk == 0`), so any point is safe.

```python
ids  = sess.encode(prompt)
blob, n = cache.match(ids)             # your cache: longest ALIGNED prefix hit
sess.restore(blob) if blob else sess.reset()

at = sess.cache_align(len(ids))        # a safe snapshot point
sess.feed_ids(ids[n:at])               # feed up to the boundary
cache.put(ids[:at], sess.snapshot())   # state here == ids[:at]
sess.feed_ids(ids[at:])                # then the rest
for piece in sess.stream_decode(256):
    print(piece, end="", flush=True)
```

A working implementation lives in `docker/serving/` (`cache.py` for the cache,
`engine.py` for the single-worker engine).

## `bllm.NativeVlmSession` — multimodal (Qwen2.5-Omni)

```python
vlm = bllm.NativeVlmSession(model_dir)        # or bllm.load(...) with arch="omni"
```

**Properties**: `name`, `vision_tokens`, `vision_image_size`, `has_audio`, `last_decode_tps`,
`last_ttft_ms` (first-token latency, includes media encode), `streaming`, `tokens_used`,
`context_left` (KV-window budget remaining).

**Offline Q&A** (media as file paths or raw arrays: `HxWx3` uint8 frames / 16 kHz mono float PCM):

| Method | Description |
|---|---|
| `chat(text, images=(), audios=(), videos=(), max_new=256, on_text=None, stop=None) -> str` | One turn over text + image/audio/video. |
| `stream_chat(text, images=(), audios=(), videos=(), max_new=256, stop=None) -> Iterator[str]` | Streaming form. |
| `set_sampling(...)` / `set_stop(...)` / `set_bpu_priority(...)` / `reset()` | Same as `NativeSession`. |

**Live input** (encode as it arrives):

| Method | Description |
|---|---|
| `stream_begin(fps=2.0, with_audio=True)` | Start a live session. |
| `stream_frame(frame)` | Feed one `HxWx3` uint8 frame. |
| `stream_audio(pcm)` | Feed a chunk of 16 kHz mono float PCM. |
| `stream_ask(text, max_new=256, on_text=None) -> str` | Ask about the media encoded so far. |

> **Context is the constraint**: the KV window *is* the context. At 2 fps, video costs about one
> frame-pair's vision tokens per second and audio ~25 tok/s; overflowing `context_left` raises
> rather than silently evicting (evicting the earliest tokens turns these full-attention models
> into gibberish). A vision tower's compiled resolution sets the video budget.

C++: `bllm::NativeVlm` (`bllm/native_vlm.h`).

## Sampling parameters at a glance

`temp<=0` → greedy (argmax). `top_k` (0 = vocab size), `top_p`, `min_p`, `typ_p` are candidate
filters, with `min_keep` a floor per filter. `rep_pen` (repeat), `penalty_freq` (frequency) and
`penalty_present` (presence) act over the last `penalty_last_n` tokens. `seed` defaults to
`bllm.SEED_RANDOM`, which draws a fresh seed for every generation — the sampler is built per
generation, so a fixed seed would replay the same random stream each turn and temperature
would have no visible effect. Pass an explicit `seed` to make sampling reproducible.
