"""BLLM — BPU LLM runtime for RDK S100 (Python API).

Wraps the D-Robotics OE-LLM runtime (libxlm.so) behind an ergonomic session:

    import bllm
    llm = bllm.LlmSession(
        "Qwen2.5_1.5B_Instruct_1024.hbm",
        "Qwen2.5_1.5B_Instruct_config/",
    )                                   # model_type + chat template auto-detected
    for chunk in llm.stream("你好"):     # streaming generator
        print(chunk, end="", flush=True)
    print(llm.last_stats.decode_tps)
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Iterator, Optional

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
    from ._bllm_native import NativeLlm as _NativeLlm, NativeEngine  # noqa: F401
    _HAVE_NATIVE = True
except ImportError:
    _HAVE_NATIVE = False

__all__ = [
    "NativeSession",
    "LlmSession",
    "OmniSession",
    "VlmSession",
    "Content",
    "SamplingParams",
    "GenerationStats",
    "__version__",
]


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

        def chat(self, message: str, max_new: int = 400,
                 on_text: "Optional[Callable[[str], None]]" = None) -> str:
            """ChatML multi-turn; returns the full reply (and streams to on_text)."""
            return self._llm.chat(message, max_new, on_text)

        def generate(self, prompt: str, max_new: int = 256,
                     on_text: "Optional[Callable[[str], None]]" = None) -> str:
            """Raw completion; returns the full text (and streams to on_text)."""
            return self._llm.generate(prompt, max_new, on_text)

        def stream_chat(self, message: str, max_new: int = 400) -> "Iterator[str]":
            """Chat as a lazy generator of decoded text chunks."""
            return _stream(lambda cb: self._llm.chat(message, max_new, cb))

        def stream(self, prompt: str, max_new: int = 256) -> "Iterator[str]":
            """Raw completion as a lazy generator of decoded text chunks."""
            return _stream(lambda cb: self._llm.generate(prompt, max_new, cb))
