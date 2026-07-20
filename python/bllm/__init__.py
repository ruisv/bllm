"""BLLM — BPU LLM runtime for RDK S100 (Python API).

One runtime: the self-built hbDNN native engine. ``bllm.load(path)`` loads any model — a
native ``model.json`` package, or a bare ``.hbm`` + a tokenizer dir (an OE-LLM package,
whose manifest is synthesized on the fly) — and hands back a uniform chat/stream session.

    import bllm
    llm = bllm.load("qwen3.5-0.8b")          # hybrid SSM package
    for chunk in llm.stream_chat("你好"):     # streaming generator
        print(chunk, end="", flush=True)

    llm = bllm.load("Qwen2.5_1.5B.hbm",       # bare .hbm + tokenizer dir
                    tokenizer_dir="Qwen2.5_1.5B_config/")
    print(llm.chat("你好"))

The old libxlm (OE-LLM SDK) backend was removed once the native runtime proved a validated
superset.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Iterator, Optional

# The native engine + unified NativeSession need only the hobot runtime + the C++
# tokenizer. The extension is absent on a dev host, so import it tolerantly.
__version__ = "0.1.6"   # keep in sync with CMakeLists.txt project(... VERSION ...)
try:
    from ._bllm_native import (  # noqa: F401
        NativeEngine,
        NativeLlm as _NativeLlm,
        NativeVlm as _NativeVlm,
    )
    _HAVE_NATIVE = True
except ImportError:
    _HAVE_NATIVE = False

# `__all__` is built from what actually imported: on a dev host the extension is absent
# and `from bllm import *` should still degrade quietly rather than name a missing symbol.
__all__ = ["__version__"]


def _run_async(fn):
    """Run a blocking generate on a background thread and hand back a Future. The native
    call releases the GIL, so this is a real async submission — libxlm's xlm_infer_async
    intent, at the Python level. One generation at a time per session (a session serialises
    on its single BPU handle); use separate sessions for true concurrency."""
    from concurrent.futures import Future
    fut: "Future[str]" = Future()

    def run() -> None:
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(fn())
        except BaseException as e:  # noqa: BLE001 — surfaced through the Future
            fut.set_exception(e)

    threading.Thread(target=run, daemon=True).start()
    return fut


def _stream(generate_call) -> "Iterator[str]":
    """Turn a blocking generate(on_token=...) into a lazy generator: run it on a
    worker thread (the native call releases the GIL) and drain a queue here."""
    q: "queue.Queue[object]" = queue.Queue()
    sentinel = object()
    error: list[BaseException] = []

    def run() -> None:
        try:
            generate_call(q.put)
        except BaseException as e:  # surface on the consumer thread
            error.append(e)
        finally:
            q.put(sentinel)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item  # type: ignore[misc]
    if error:
        raise error[0]


if _HAVE_NATIVE:

    class NativeSession:
        """Unified native LLM session — no libxlm, no manual tokenization.

        Load a model DIRECTORY (with a ``model.json`` manifest); the right engine
        (hybrid Gated-DeltaNet for Qwen3.5, or dense KV for Qwen2.5/DeepSeek/…) and
        the tokenizer + ChatML template are all resolved automatically.

            import bllm
            llm = bllm.NativeSession("/path/to/model_dir")
            for chunk in llm.stream_chat("你好"):
                print(chunk, end="", flush=True)
        """

        def __init__(self, model_dir: str) -> None:
            self._llm = _NativeLlm(model_dir)

        @property
        def name(self) -> str:
            return self._llm.name

        @property
        def arch(self) -> str:
            return self._llm.arch

        @property
        def vocab_size(self) -> int:
            return self._llm.vocab_size

        def reset(self) -> None:
            """Start a fresh conversation."""
            self._llm.reset()

        def set_sampling(self, temp: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                         rep_pen: float = 1.0, seed: int = 1234, min_p: float = 0.0,
                         typ_p: float = 1.0, min_keep: int = 1, penalty_last_n: int = 64,
                         penalty_freq: float = 0.0, penalty_present: float = 0.0) -> "NativeSession":
            """Configure sampling (temp<=0 => greedy). Full libxlm parity: top_k / top_p /
            min_p / typ_p (min_keep floors each filter) and repeat / frequency / presence
            penalties over the last penalty_last_n tokens. Returns self for chaining."""
            self._llm.set_sampling(temp, top_p, top_k, rep_pen, seed, min_p, typ_p,
                                   min_keep, penalty_last_n, penalty_freq, penalty_present)
            return self

        def set_thinking(self, enabled: bool) -> "NativeSession":
            """Qwen3.5 reasoning toggle. enabled=False prefills an empty <think></think> so the
            model answers DIRECTLY (faster, fewer tokens, clean output). enabled=True (default)
            reasons and shows the <think>...</think>. The in-message /no_think soft switch is
            unreliable on this checkpoint; this prefill is."""
            self._llm.set_thinking(bool(enabled))
            return self

        @property
        def thinking(self) -> bool:
            return self._llm.thinking

        def set_stop(self, stop: "list[str]") -> "NativeSession":
            """Persistent stop strings: end each turn as soon as any appears in the reply,
            trimmed from both the stream and the return value. A per-call ``stop=`` on
            chat()/generate() overrides. Returns self for chaining."""
            self._llm.set_stop(list(stop))
            return self

        def set_bpu_priority(self, priority: int) -> "NativeSession":
            """BPU queue priority 0..255 (default 0 = lowest, so a co-resident vision
            pipeline wins the queue).

            Priority is relative, and it only bites where the running graph offers
            preemption points. Protecting a vision pipeline's latency needs BOTH the
            LLM below it in priority AND an .hbm compiled with `--max_time_per_fc`.
            Measured on S100P (yolo26s p99, LLM decoding): 41.3 ms with neither,
            13.5 ms with priority alone, 2.95 ms with both (2.06 ms when CV runs alone).
            """
            self._llm.set_bpu_priority(priority)
            return self

        @property
        def last_decode_tps(self) -> float:
            return self._llm.last_decode_tps

        def chat(self, message: str, max_new: int = 400,
                 on_text: "Optional[Callable[[str], None]]" = None,
                 stop: "Optional[list[str]]" = None) -> str:
            """ChatML multi-turn; returns the full reply (and streams to on_text).
            stop=[...] ends the turn on any of those strings (trimmed from the reply)."""
            return self._llm.chat(message, max_new, on_text, list(stop) if stop else [])

        def generate(self, prompt: str, max_new: int = 256,
                     on_text: "Optional[Callable[[str], None]]" = None,
                     stop: "Optional[list[str]]" = None) -> str:
            """Raw completion; returns the full text (and streams to on_text).
            stop=[...] ends generation on any of those strings (trimmed)."""
            return self._llm.generate(prompt, max_new, on_text, list(stop) if stop else [])

        def stream_chat(self, message: str, max_new: int = 400,
                        stop: "Optional[list[str]]" = None) -> "Iterator[str]":
            """Chat as a lazy generator of decoded text chunks."""
            s = list(stop) if stop else []
            return _stream(lambda cb: self._llm.chat(message, max_new, cb, s))

        def stream(self, prompt: str, max_new: int = 256,
                   stop: "Optional[list[str]]" = None) -> "Iterator[str]":
            """Raw completion as a lazy generator of decoded text chunks."""
            s = list(stop) if stop else []
            return _stream(lambda cb: self._llm.generate(prompt, max_new, cb, s))

        def chat_async(self, message: str, max_new: int = 400,
                       on_text: "Optional[Callable[[str], None]]" = None,
                       stop: "Optional[list[str]]" = None):
            """Non-blocking chat → concurrent.futures.Future[str] (libxlm's xlm_infer_async
            intent). One generation at a time per session."""
            s = list(stop) if stop else []
            return _run_async(lambda: self._llm.chat(message, max_new, on_text, s))

        def generate_async(self, prompt: str, max_new: int = 256,
                           on_text: "Optional[Callable[[str], None]]" = None,
                           stop: "Optional[list[str]]" = None):
            """Non-blocking raw completion → concurrent.futures.Future[str]."""
            s = list(stop) if stop else []
            return _run_async(lambda: self._llm.generate(prompt, max_new, on_text, s))

        def perplexity(self, text: str) -> float:
            """Teacher-forced perplexity of raw `text` under the model (no chat template).
            Resets the conversation. The native equivalent of libxlm's xlm_ppl."""
            return self._llm.perplexity(text)

        def save_state(self, path: str) -> "NativeSession":
            """Save the conversation's KV(+SSM) state to a file — libxlm's path_prompt_cache.
            Feed a shared prefix once, save, and reload it later to skip re-prefill."""
            self._llm.save_state(path)
            return self

        def load_state(self, path: str) -> "NativeSession":
            """Restore a state written by save_state(); chat()/generate() resume from it
            bit-identically. The state is bound to this model."""
            self._llm.load_state(path)
            return self

        def set_prefix(self, shared_context: str = "") -> "NativeSession":
            """Prefill the system prompt + optional `shared_context` ONCE and snapshot it in
            memory. Then ask() each query against it without re-prefilling the (possibly long)
            prefix — RAG over a document, a fixed system/tool context, few-shot exemplars.
            The one-time prefix prefill runs here; chunked prefill (roadmap) speeds that up."""
            self._llm.set_prefix(shared_context)
            return self

        def ask(self, query: str, max_new: int = 400,
                on_text: "Optional[Callable[[str], None]]" = None,
                stop: "Optional[list[str]]" = None) -> str:
            """Answer `query` against the set_prefix() prefix, INDEPENDENTLY (each ask restores
            the prefix — no accumulation). Returns the full reply (and streams to on_text)."""
            return self._llm.ask(query, max_new, on_text, list(stop) if stop else [])

        def stream_ask(self, query: str, max_new: int = 400,
                       stop: "Optional[list[str]]" = None) -> "Iterator[str]":
            """ask() as a lazy generator of decoded text chunks."""
            s = list(stop) if stop else []
            return _stream(lambda cb: self._llm.ask(query, max_new, cb, s))

        @property
        def has_prefix(self) -> bool:
            """Whether a reusable prefix is set (via set_prefix())."""
            return self._llm.has_prefix

        def clear_prefix(self) -> "NativeSession":
            """Drop the cached prefix set by set_prefix()."""
            self._llm.clear_prefix()
            return self

        def encode(self, text: str) -> "list[int]":
            """Tokenize `text` to token ids with the model's C++ tokenizer."""
            return self._llm.encode(text)

        def decode(self, ids: "list[int]") -> str:
            """Detokenize `ids` back to text with the model's C++ tokenizer."""
            return self._llm.decode(list(ids))

        # ---- serving primitives ------------------------------------------------------
        # set_prefix()/ask() serve ONE fixed prefix. A server juggling many conversations
        # holds many states and picks one per request, so it drives the snapshot itself:
        #
        #     ids = sess.encode(prompt)
        #     blob, n = cache.match(ids)          # longest cached prefix of ids
        #     sess.restore(blob) if blob else sess.reset()
        #     sess.feed_ids(ids[n:])              # prefill only what is new
        #     cache.put(ids, sess.snapshot())     # state == the whole prompt
        #     for piece in sess.stream_decode(max_new): ...

        def snapshot(self) -> bytes:
            """Snapshot the current context as a byte blob (no file, unlike save_state)."""
            return self._llm.snapshot()

        def restore(self, blob: bytes) -> "NativeSession":
            """Restore a snapshot(); generation resumes from it bit-identically. Raises if
            the blob does not match this model."""
            self._llm.restore(blob)
            return self

        def feed_ids(self, ids: "list[int]") -> "NativeSession":
            """Ingest token ids without decoding (the prefill half of generate())."""
            self._llm.feed_ids(list(ids))
            return self

        @property
        def context_left(self) -> int:
            """Tokens still free in the KV/SSM window. The window IS the context — going
            past it raises rather than silently dropping the oldest turns."""
            return self._llm.context_left

        def cache_align(self, n: int) -> int:
            """Largest prefix of `n` tokens that is safe to snapshot, i.e. the largest
            multiple of prefill_chunk that is <= n (just `n` when the engine does not
            chunk). Snapshotting anywhere else can change the reply — see prefill_chunk.

            Measured on an S100P (Qwen2.5-1.5B, chunk=256, greedy): aligned splits
            reproduced the un-cached reply 5/5, unaligned ones 1/18, and one unaligned
            split turned "Venus is the hottest" into "Earth is the hottest".
            """
            c = self.prefill_chunk
            return n if c <= 0 else (n // c) * c

        @property
        def prefill_chunk(self) -> int:
            """Prefill chunk width; 0 when the engine has no chunked prefill.

            Cache snapshots at multiples of this. The dense engine batch-prefills runs of
            `prefill_chunk` tokens and decode-steps the remainder, and those two graphs are
            not numerically identical under int8 — so splitting a prompt at an arbitrary
            point changes which tokens went through which graph, and can change the reply.
            Aligned splits reproduce a straight-through prefill exactly.
            """
            return self._llm.prefill_chunk

        def decode_stream(self, max_new: int = 256,
                          on_text: "Optional[Callable[[str], None]]" = None,
                          stop: "Optional[list[str]]" = None) -> str:
            """Decode from the current context (the generation half of generate())."""
            return self._llm.decode_stream(max_new, on_text, list(stop) if stop else [])

        def stream_decode(self, max_new: int = 256,
                          stop: "Optional[list[str]]" = None) -> "Iterator[str]":
            """decode_stream() as a lazy generator of decoded text chunks."""
            s = list(stop) if stop else []
            return _stream(lambda cb: self._llm.decode_stream(max_new, cb, s))

        def request_cancel(self) -> None:
            """Stop the in-flight generation at the next token boundary. Thread-safe: call
            it from another thread while a generation runs. The context stays consistent."""
            self._llm.request_cancel()

        @property
        def canceled(self) -> bool:
            """Whether the last generation ended on request_cancel()."""
            return self._llm.canceled

    def load_video(path: str, fps: float = 2.0, size: int = 448, max_frames: int = 10,
                   with_audio: bool = True, ffmpeg: str = "ffmpeg") -> dict:
        """Sample a video file into the ``videos=`` argument of NativeVlmSession.chat.

        BLLM itself knows nothing about containers — the runtime takes decoded frames
        and PCM, which a camera + microphone pipeline already has. This helper exists
        so demos can open an .mp4; it shells out to ffmpeg and needs numpy.

        Each PAIR of frames costs ``vision_tokens()`` of context, so ``max_frames``
        is what keeps a long clip from overflowing the decoder window.
        """
        import subprocess
        import numpy as np

        vf = f"fps={fps},scale={size}:{size}"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-vf", vf,
               "-frames:v", str(max_frames), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)
        if len(frames) == 0:
            raise RuntimeError(f"[bllm] no frames decoded from {path}")

        pcm = None
        if with_audio:
            acmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-vn",
                    "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
            out = subprocess.run(acmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
            if out:
                pcm = np.frombuffer(out, np.float32).copy()
        return {"frames": [np.ascontiguousarray(f) for f in frames], "fps": fps, "audio": pcm}

    class NativeVlmSession:
        """Unified native multimodal session — image + text, no libxlm.

        Load an ``arch="omni"`` model directory. Media may be file paths or raw
        arrays — HxWx3 uint8 frames and 16 kHz mono float32 PCM, what a camera or
        microphone pipeline already has — so no file round-trip is needed.

            import bllm
            vlm = bllm.NativeVlmSession("/path/to/qwen2.5-omni-3b")
            for chunk in vlm.stream_chat("描述这张图片。", ["bus.jpg"]):
                print(chunk, end="", flush=True)
            vlm.reset()
            print(vlm.chat("这段音频里是什么声音？", audios=["clip.wav"]))
            vlm.reset()
            print(vlm.chat("描述这段视频。", videos=[bllm.load_video("clip.mp4")]))
        """

        def __init__(self, model_dir: str) -> None:
            self._vlm = _NativeVlm(model_dir)

        @property
        def name(self) -> str:
            return self._vlm.name

        @property
        def vision_tokens(self) -> int:
            """Decoder tokens one image (or one video frame-pair) occupies."""
            return self._vlm.vision_tokens

        @property
        def vision_image_size(self) -> int:
            """Square side the tower was compiled for; sample frames at it."""
            return self._vlm.vision_image_size

        @property
        def last_decode_tps(self) -> float:
            return self._vlm.last_decode_tps

        @property
        def last_ttft_ms(self) -> float:
            """Time to first token, including the vision encode."""
            return self._vlm.last_ttft_ms

        def reset(self) -> None:
            """Start a fresh conversation."""
            self._vlm.reset()

        def set_sampling(self, temp: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                         rep_pen: float = 1.0, seed: int = 1234, min_p: float = 0.0,
                         typ_p: float = 1.0, min_keep: int = 1, penalty_last_n: int = 64,
                         penalty_freq: float = 0.0, penalty_present: float = 0.0) -> "NativeVlmSession":
            self._vlm.set_sampling(temp, top_p, top_k, rep_pen, seed, min_p, typ_p,
                                   min_keep, penalty_last_n, penalty_freq, penalty_present)
            return self

        def set_stop(self, stop: "list[str]") -> "NativeVlmSession":
            """Persistent stop strings for every following turn (see NativeSession.set_stop)."""
            self._vlm.set_stop(list(stop))
            return self

        def set_bpu_priority(self, priority: int) -> "NativeVlmSession":
            """BPU queue priority 0..255 for the text + vision + audio graphs."""
            self._vlm.set_bpu_priority(priority)
            return self

        @property
        def has_audio(self) -> bool:
            """Whether this model directory ships an audio tower + mel filters."""
            return self._vlm.has_audio

        def chat(self, text: str, images=(), audios=(), videos=(), max_new: int = 256,
                 on_text: "Optional[Callable[[str], None]]" = None,
                 stop: "Optional[list[str]]" = None) -> str:
            """Answer a turn of text + images + audio + videos; returns the full reply.
            stop=[...] ends the answer on any of those strings (trimmed)."""
            return self._vlm.chat(text, list(images), list(audios), list(videos), max_new,
                                  on_text, list(stop) if stop else [])

        def stream_chat(self, text: str, images=(), audios=(), videos=(),
                        max_new: int = 256, stop: "Optional[list[str]]" = None) -> "Iterator[str]":
            """The same turn as a lazy generator of decoded text chunks."""
            imgs, aus, vids = list(images), list(audios), list(videos)
            s = list(stop) if stop else []
            return _stream(lambda cb: self._vlm.chat(text, imgs, aus, vids, max_new, cb, s))

        # ── live capture: push frames / mic PCM as they arrive, ask any time ──
        #
        # The KV window IS the context, so budget matters: a frame-pair costs
        # `vision_tokens` (256) and a second of audio costs 25. At 2 fps a
        # frame-pair spans 2 frames, so at 2 fps video costs `vision_tokens` per
        # second: a 2048-slot model holds 8 s of 448px video (32 s at 224px), or
        # ~80 s of audio alone.
        # Pushing past that raises rather than silently evicting — evicting the
        # earliest tokens collapses these full-attention models into gibberish.

        def stream_begin(self, fps: float = 2.0, with_audio: bool = True) -> None:
            """Open a live media turn. fps<=0 means audio only."""
            self._vlm.stream_begin(fps, with_audio)

        def stream_frame(self, frame) -> None:
            """Push one HxWx3 uint8 frame."""
            self._vlm.stream_frame(frame)

        def stream_audio(self, pcm) -> None:
            """Push 16 kHz mono float32 samples."""
            self._vlm.stream_audio(pcm)

        def stream_ask(self, text: str, max_new: int = 256,
                       on_text: "Optional[Callable[[str], None]]" = None) -> str:
            """Close the live media block, ask, and generate."""
            return self._vlm.stream_ask(text, max_new, on_text)

        @property
        def streaming(self) -> bool:
            return self._vlm.streaming

        @property
        def tokens_used(self) -> int:
            return self._vlm.tokens_used

        @property
        def context_left(self) -> int:
            """KV slots still free in this conversation."""
            return self._vlm.context_left

        @property
        def video_tokens_per_second(self) -> float:
            return self._vlm.video_tokens_per_second


def _reject_libxlm(backend_field: str) -> None:
    """The libxlm backend is gone (tag v-libxlm-final); everything runs
    native. A leftover backend="libxlm" pin — in a call or a model.json — is a hard error
    so it's noticed, not silently reinterpreted."""
    if backend_field == "libxlm":
        raise ValueError("the libxlm backend was removed — everything runs on the native "
                         "runtime now (recover it at tag v-libxlm-final). Drop backend='libxlm'.")


def _read_manifest(path: str) -> "Optional[dict]":
    """Return the parsed model.json if `path` is a native model package (a directory with
    one, or the manifest file itself); None if it looks like a bare .hbm / OE-LLM package."""
    import json
    import os
    if os.path.isdir(path):
        mj = os.path.join(path, "model.json")
        if os.path.isfile(mj):
            with open(mj) as f:
                return json.load(f)
        return None
    if path.endswith(".json") and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def load(path: str, *, backend: str = "auto", **kwargs):
    """Load a model and return a native session.

    Routing:
      * a native package (a directory with model.json) → NativeSession (dense/hybrid) or
        NativeVlmSession (omni), by its `arch`;
      * a bare `.hbm` + tokenizer_dir (an OE-LLM package) → NativeSession, by synthesizing a
        dense model.json from the `.hbm` + tokenizer dir (bllm._modelmeta).
    Extra kwargs go to the chosen session's constructor.

        s = bllm.load("qwen3.5-0.8b")      # -> NativeSession (hybrid SSM)
        s = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="Qwen2.5_1.5B_config/")  # -> dense
        s.chat("你好")

    The libxlm backend was removed; `backend="libxlm"` is a hard error.
    """
    _reject_libxlm(backend)
    if not _HAVE_NATIVE:
        raise RuntimeError("this install has no native backend built (dev host?).")

    manifest = _read_manifest(path)
    if manifest is not None:
        _reject_libxlm(manifest.get("backend", "auto"))
        arch = manifest.get("arch", "dense")
        if arch == "omni":
            return NativeVlmSession(path, **kwargs)
        if arch == "embed":
            raise ValueError("arch='embed' is an encoder, not a chat session; use the "
                             "NativeEmbedder C++ API (no Python binding yet)")
        return NativeSession(path, **kwargs)   # dense or hybrid

    # Not a native package → a bare `.hbm` / OE-LLM package (a `.hbm` + a tokenizer dir).
    # Native consumes it directly: synthesize a dense model.json from the `.hbm` + tokenizer
    # dir (bllm._modelmeta) and load that.
    tokenizer_dir = kwargs.pop("tokenizer_dir", "") or kwargs.pop("tokenizer", "")
    if not tokenizer_dir:
        raise ValueError(
            "loading a bare .hbm needs tokenizer_dir=<dir> (a directory with tokenizer.json, "
            "ideally generation_config.json too). A hybrid (Qwen3.5) or omni package instead "
            "needs a model.json built with scripts/make_model_dir.py — its host embed table "
            "isn't inside the .hbm.")
    from ._modelmeta import native_dir_for
    system = kwargs.pop("system_prompt", None) or kwargs.pop("system", None)
    d = native_dir_for(path, tokenizer_dir, name=kwargs.pop("name", None),
                       eos=kwargs.pop("eos", None),
                       system=system or "You are a helpful assistant.")
    return NativeSession(str(d), **kwargs)


# ── public surface ───────────────────────────────────────────────────────────
__all__ += ["load", "available_backends"]
if _HAVE_NATIVE:
    __all__ += ["NativeSession", "NativeVlmSession", "load_video"]


def available_backends() -> "tuple[str, ...]":
    """Which runtimes this install can drive. Native-only now (libxlm was removed)."""
    return ("native",) if _HAVE_NATIVE else ()
