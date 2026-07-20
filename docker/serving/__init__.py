"""BLLM serving layer — engine (worker thread + queue) and prefix cache.

Split out of ``server.py``, which is now just the OpenAI-protocol facade. ``cache``
imports nothing from bllm and is unit-testable on any host; ``engine`` needs the board.
"""
from .cache import PrefixCache, Stats  # noqa: F401

__all__ = ["PrefixCache", "Stats"]
