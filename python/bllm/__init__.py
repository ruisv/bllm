"""BLLM — BPU LLM runtime for RDK S100 (Python API).

Two runtimes behind one front door. ``bllm.load(path)`` picks the right one by the model:
a native model.json package → the hbDNN native engine; a bare libxlm ``.hbm`` / OE-LLM
package → libxlm. Either way you get a uniform chat/stream surface.

    import bllm
    llm = bllm.load("qwen3.5-0.8b")          # native (hybrid SSM — libxlm can't run it)
    for chunk in llm.stream_chat("你好"):     # streaming generator
        print(chunk, end="", flush=True)

    llm = bllm.load("Qwen2.5_1.5B.hbm",       # bare libxlm .hbm → LlmSession
                    tokenizer_dir="Qwen2.5_1.5B_config/")
    print(llm.chat("你好"))

The concrete session classes (LlmSession / NativeSession / NativeVlmSession) remain
available if you want to pick a backend explicitly.
"""

from __future__ import annotations

import queue
import threading
import warnings
from typing import Callable, Iterator, Optional


def _warn_libxlm_deprecated(what: str) -> None:
    """libxlm is being retired (docs/MIGRATION.md); the native runtime is a validated
    superset (docs/PARITY.md). One release of overlap, then the libxlm path is removed."""
    warnings.warn(
        f"{what} uses the legacy libxlm backend, which is deprecated and will be removed. "
        "Use bllm.load(<.hbm>, tokenizer_dir=...) — it now runs the native runtime, a "
        "validated superset (see docs/PARITY.md).",
        DeprecationWarning, stacklevel=3)

# The libxlm-backed sessions need the OE-LLM SDK; the native engine + unified
# NativeSession need only the hobot runtime + the C++ tokenizer. Either may be
# absent in a given install, so import each independently.
__version__ = "0.1.0"
try:
    from ._bllm import (  # noqa: F401
        Content,
        GenerationStats,
        LlmSession as _LlmSession,
        OmniSession as _OmniSession,
        SamplingParams,
        VlmSession as _VlmSession,
        __version__,
    )
    _HAVE_LIBXLM = True
except ImportError:
    _HAVE_LIBXLM = False

try:
    from ._bllm_native import (  # noqa: F401
        NativeEngine,
        NativeLlm as _NativeLlm,
        NativeVlm as _NativeVlm,
    )
    _HAVE_NATIVE = True
except ImportError:
    _HAVE_NATIVE = False

# `__all__` is built from what actually imported: an install may carry either backend,
# both, or (on a dev host) neither. Advertising a name that does not exist would make
# `from bllm import *` fail on exactly the machines where it should degrade quietly.
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


class LlmSession:
    """A loaded on-board LLM. Thin Python facade over the native session that
    adds the idiomatic ``stream()`` generator.

    Only ``model_path`` and ``tokenizer_dir`` are required. ``model_type``
    defaults to ``"auto"`` (inferred from the path); the chat template is
    auto-discovered from ``tokenizer_dir`` when present.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_dir: str = "",
        *,
        model_type: str = "auto",
        chat_template_path: str = "",
        system_prompt: str = "",
        config_path: str = "",
        context_size: int = 0,
        sampling: "SamplingParams | None" = None,
        backend: str = "any",
        priority: str = "normal",
    ) -> None:
        _warn_libxlm_deprecated("LlmSession")
        self._s = _LlmSession(
            model_path,
            tokenizer_dir,
            model_type=model_type,
            chat_template_path=chat_template_path,
            system_prompt=system_prompt,
            config_path=config_path,
            context_size=context_size,
            sampling=sampling,
            backend=backend,
            priority=priority,
        )

    def generate(
        self, prompt: str, on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """Generate a reply, blocking until done. If ``on_token`` is given it is
        called with each streamed chunk. Returns the full reply."""
        return self._s.generate(prompt, on_token)

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield reply chunks as they decode."""
        return _stream(lambda on_token: self._s.generate(prompt, on_token))

    def generate_async(self, prompt: str, on_token: Optional[Callable[[str], None]] = None):
        """Non-blocking generate → concurrent.futures.Future[str] (libxlm's xlm_infer_async
        intent). on_token still streams chunks. One at a time per session."""
        return _run_async(lambda: self._s.generate(prompt, on_token))

    # chat()/stream_chat() are aliases: libxlm applies the chat template inside the
    # runtime, so generate() on an Instruct model already IS a chat turn. Having them
    # lets bllm.load() hand back either backend behind one surface.
    def chat(self, message: str, on_token: Optional[Callable[[str], None]] = None) -> str:
        """A chat turn (libxlm manages the template + history internally)."""
        return self._s.generate(message, on_token)

    def stream_chat(self, message: str) -> Iterator[str]:
        """A chat turn as a lazy generator of decoded chunks."""
        return _stream(lambda on_token: self._s.generate(message, on_token))

    def set_stop(self, stop: "list[str]") -> "LlmSession":
        """Stop strings: end a reply as soon as any appears, trimmed from stream and
        return value. NOTE: libxlm has no cancellation hook, so it still generates to
        its own eos — the reply is correct but no compute is saved. The native backend
        (``NativeSession``) stops early. Returns self for chaining."""
        self._s.set_stop(list(stop))
        return self

    def reset(self) -> None:
        """Clear history so the next generate()/stream() starts fresh."""
        self._s.reset()

    @property
    def last_stats(self) -> GenerationStats:
        return self._s.last_stats

    @property
    def model_type(self) -> str:
        return self._s.model_type


class OmniSession:
    """Multimodal (text + image + audio + video) session for Qwen2.5-Omni.

    Build content parts with ``bllm.Content.text/image/audio/video`` and pass a
    list to ``generate()``; ``ask()`` is a text-only shortcut.

        omni = bllm.OmniSession(text_hbm, visual_hbm, audio_hbm, embed_bin, cfg)
        for chunk in omni.stream([bllm.Content.image("cat.jpg"),
                                  bllm.Content.text("这是什么？")]):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        text_model: str,
        visual_model: str,
        audio_model: str,
        embed_tokens: str,
        tokenizer_dir: str,
        *,
        system_prompt: str = "",
        context_size: int = 0,
        sampling: "SamplingParams | None" = None,
        backend: str = "any",
        priority: str = "normal",
    ) -> None:
        _warn_libxlm_deprecated("OmniSession")
        self._s = _OmniSession(
            text_model, visual_model, audio_model, embed_tokens, tokenizer_dir,
            system_prompt=system_prompt, context_size=context_size,
            sampling=sampling, backend=backend, priority=priority,
        )

    def generate(self, content, on_token=None) -> str:
        return self._s.generate(list(content), on_token)

    def ask(self, text: str, on_token=None) -> str:
        return self._s.ask(text, on_token)

    def stream(self, content) -> Iterator[str]:
        parts = list(content)
        return _stream(lambda on_token: self._s.generate(parts, on_token))

    def reset(self) -> None:
        self._s.reset()

    @property
    def last_stats(self) -> GenerationStats:
        return self._s.last_stats

    @property
    def model_type(self) -> str:
        return self._s.model_type


class VlmSession:
    """Image + text vision-language session (InternVL / Qwen-VL).

        vlm = bllm.VlmSession(hbm, tokenizer_dir, config_path=cfg)
        print(vlm.describe("cat.jpg", "这是什么动物？"))

    Note: the SDK 1.0.0 ships no pre-compiled InternVL/Qwen-VL model — this is
    API-ready but not yet board-validated end to end (see docs/MODELS.md).
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_dir: str,
        *,
        config_path: str = "",
        model_type: str = "auto",
        system_prompt: str = "",
        context_size: int = 0,
        sampling: "SamplingParams | None" = None,
        backend: str = "any",
        priority: str = "normal",
    ) -> None:
        _warn_libxlm_deprecated("VlmSession")
        self._s = _VlmSession(
            model_path, tokenizer_dir, config_path=config_path,
            model_type=model_type, system_prompt=system_prompt,
            context_size=context_size,
            sampling=sampling, backend=backend, priority=priority,
        )

    def generate(self, images, prompt: str, on_token=None) -> str:
        return self._s.generate(list(images), prompt, on_token)

    def describe(self, image_path: str, prompt: str, on_token=None) -> str:
        return self._s.describe(image_path, prompt, on_token)

    def stream(self, images, prompt: str) -> Iterator[str]:
        paths = list(images)
        return _stream(lambda on_token: self._s.generate(paths, prompt, on_token))

    def reset(self) -> None:
        self._s.reset()

    @property
    def last_stats(self) -> GenerationStats:
        return self._s.last_stats

    @property
    def model_type(self) -> str:
        return self._s.model_type


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


def _choose_backend(arch: str, backend_field: str) -> str:
    """Mirror of C++ bllm::chooseBackend (include/bllm/model_config.h) — kept in step by
    tests/test_choose_backend.cc on the C++ side and test_backend_routing.py here.

    A capability boundary, not a preference: hybrid / omni / embed can ONLY run native
    (libxlm's op set can't express them, and it has no host-embedding entry point); dense
    runs on either, so the manifest's `backend` field decides, defaulting to native."""
    dense = arch in ("", "dense")
    if not dense:
        if backend_field == "libxlm":
            raise ValueError(f"model.json requests backend='libxlm' but arch='{arch}' "
                             "only runs on the native runtime")
        return "native"
    return "libxlm" if backend_field == "libxlm" else "native"


def _read_manifest(path: str) -> "Optional[dict]":
    """Return the parsed model.json if `path` is a native model package (a directory with
    one, or the manifest file itself); None if it looks like a bare libxlm .hbm/package."""
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
    """Load a model and return a session on the right runtime, chosen automatically.

    Routing (see bllm.chooseBackend rule):
      * a native package (a directory with model.json) → NativeSession (dense/hybrid) or
        NativeVlmSession (omni), by its `arch`;
      * a bare `.hbm` + tokenizer_dir (an OE-LLM package) → NativeSession, by synthesizing a
        dense model.json from the `.hbm` + tokenizer dir (bllm._modelmeta). Pass
        `backend="libxlm"` to force the legacy runtime instead.
    `backend="native"|"libxlm"` forces one and errors if it is impossible. Extra kwargs go
    to the chosen session's constructor.

        s = bllm.load("qwen3.5-0.8b")      # -> NativeSession (hybrid; libxlm can't run it)
        s = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="Qwen2.5_1.5B_config/")  # -> native
        s = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="...", backend="libxlm")  # -> libxlm
        s.chat("你好")                      # uniform surface either way
    """
    manifest = _read_manifest(path)
    if manifest is not None:
        arch = manifest.get("arch", "dense")
        chosen = backend if backend != "auto" else _choose_backend(arch, manifest.get("backend", "auto"))
        if backend == "libxlm" and arch not in ("", "dense"):
            raise ValueError(f"backend='libxlm' cannot run arch='{arch}' (native only)")
        if chosen == "libxlm":
            if not _HAVE_LIBXLM:
                raise RuntimeError("this install has no libxlm backend (available: "
                                   f"{available_backends()})")
            # A native package pinned to libxlm: drive its .hbm through libxlm, using the
            # package dir as the tokenizer dir (it holds tokenizer.json + config).
            import os
            hbm = manifest.get("hbm", "")
            hbm_path = hbm if os.path.isabs(hbm) else os.path.join(path, hbm)
            return LlmSession(hbm_path, tokenizer_dir=path, **kwargs)
        if not _HAVE_NATIVE:
            raise RuntimeError("this install has no native backend (available: "
                               f"{available_backends()})")
        if arch == "omni":
            return NativeVlmSession(path, **kwargs)
        if arch == "embed":
            raise ValueError("arch='embed' is an encoder, not a chat session; use the "
                             "NativeEmbedder C++ API (no Python binding yet)")
        return NativeSession(path, **kwargs)   # dense or hybrid

    # Not a native package → a bare `.hbm` / OE-LLM package (a `.hbm` + a tokenizer dir).
    # Blocker A: native consumes this directly now — synthesize a dense model.json from the
    # `.hbm` + tokenizer dir (bllm._modelmeta) and load it — so `bllm.load(<.hbm>)` runs
    # native, not libxlm. `backend="libxlm"` remains the explicit escape hatch.
    tokenizer_dir = kwargs.pop("tokenizer_dir", "") or kwargs.pop("tokenizer", "")

    want_libxlm = backend == "libxlm" or (backend == "auto" and not _HAVE_NATIVE)
    if want_libxlm:
        if not _HAVE_LIBXLM:
            raise RuntimeError("this install has no libxlm backend (available: "
                               f"{available_backends()})")
        return LlmSession(path, tokenizer_dir, **kwargs)

    if not _HAVE_NATIVE:                      # backend == "native" but it isn't built
        raise RuntimeError(f"this install has no native backend (available: {available_backends()})")
    if not tokenizer_dir:
        raise ValueError(
            "loading a bare .hbm on the native backend needs tokenizer_dir=<dir> "
            "(a directory with tokenizer.json, ideally generation_config.json too). "
            "A hybrid (Qwen3.5) or omni package instead needs a model.json built with "
            "scripts/make_model_dir.py — its host embed table isn't inside the .hbm.")
    from ._modelmeta import native_dir_for
    system = kwargs.pop("system_prompt", None) or kwargs.pop("system", None)
    d = native_dir_for(path, tokenizer_dir, name=kwargs.pop("name", None),
                       eos=kwargs.pop("eos", None),
                       system=system or "You are a helpful assistant.")
    return NativeSession(str(d), **kwargs)


# ── public surface, per backend actually present ─────────────────────────────
__all__ += ["load", "available_backends"]
if _HAVE_LIBXLM:
    __all__ += ["LlmSession", "OmniSession", "VlmSession",
                "Content", "SamplingParams", "GenerationStats"]
if _HAVE_NATIVE:
    __all__ += ["NativeSession", "NativeVlmSession", "load_video"]


def available_backends() -> "tuple[str, ...]":
    """Which runtimes this install can actually drive: "libxlm" and/or "native"."""
    return tuple(n for n, ok in (("libxlm", _HAVE_LIBXLM), ("native", _HAVE_NATIVE)) if ok)
