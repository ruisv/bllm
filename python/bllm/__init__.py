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

from ._bllm import GenerationStats, LlmSession as _LlmSession, __version__

__all__ = ["LlmSession", "GenerationStats", "__version__"]


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
        tokenizer_dir: str,
        *,
        model_type: str = "auto",
        chat_template_path: str = "",
        system_prompt: str = "",
        config_path: str = "",
        context_size: int = 0,
    ) -> None:
        self._s = _LlmSession(
            model_path,
            tokenizer_dir,
            model_type=model_type,
            chat_template_path=chat_template_path,
            system_prompt=system_prompt,
            config_path=config_path,
            context_size=context_size,
        )

    def generate(
        self, prompt: str, on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """Generate a reply, blocking until done. If ``on_token`` is given it is
        called with each streamed chunk. Returns the full reply."""
        return self._s.generate(prompt, on_token)

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield reply chunks as they decode. Runs generate() on a worker thread
        (the native call releases the GIL) and drains a queue on this thread."""
        q: "queue.Queue[object]" = queue.Queue()
        sentinel = object()
        error: list[BaseException] = []

        def run() -> None:
            try:
                self._s.generate(prompt, q.put)
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

    def reset(self) -> None:
        """Clear history so the next generate()/stream() starts fresh."""
        self._s.reset()

    @property
    def last_stats(self) -> GenerationStats:
        return self._s.last_stats

    @property
    def model_type(self) -> str:
        return self._s.model_type
