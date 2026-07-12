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
| `set_sampling(temp=0.0, top_p=1.0, top_k=0, rep_pen=1.0, seed=1234, min_p=0.0, typ_p=1.0, min_keep=1, penalty_last_n=64, penalty_freq=0.0, penalty_present=0.0)` | Configure sampling; `temp<=0` is greedy. |
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

C++: `bllm::NativeLlm` (`bllm/native_llm.h`).

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
`penalty_present` (presence) act over the last `penalty_last_n` tokens. `seed` fixes randomness.
