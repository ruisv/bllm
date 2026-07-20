#!/usr/bin/env python3
"""Unit tests for the serving prefix cache. Pure logic — runs on any host, no board."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))

from serving.cache import PrefixCache  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def blob(n: int, fill: bytes = b"x") -> bytes:
    return fill * n


def test_miss_on_empty():
    c = PrefixCache()
    assert c.match([1, 2, 3]) is None
    assert c.stats.misses == 1
    assert c.stats.prefilled_tokens == 3


def test_hit_returns_longest_strict_prefix():
    c = PrefixCache()
    c.put([1, 2], blob(10))
    c.put([1, 2, 3, 4], blob(20))
    got = c.match([1, 2, 3, 4, 5])
    assert got is not None
    b, n = got
    assert n == 4 and b == blob(20)


def test_exact_full_match_is_not_reused():
    """An entry covering the entire prompt leaves nothing to feed."""
    c = PrefixCache()
    c.put([1, 2, 3], blob(10))
    assert c.match([1, 2, 3]) is None


def test_divergent_history_falls_back_to_shorter_entry():
    """A client that edits an old turn must not get another conversation's state."""
    c = PrefixCache()
    c.put([1, 2], blob(10))
    c.put([1, 2, 9, 9], blob(20))
    got = c.match([1, 2, 7, 7, 7])
    assert got is not None
    assert got[1] == 2  # the [1,2,9,9] entry is not a prefix, so it is not used


def test_completely_different_prompt_misses():
    c = PrefixCache()
    c.put([1, 2, 3], blob(10))
    assert c.match([9, 8, 7, 6]) is None


def test_duplicate_put_is_ignored():
    c = PrefixCache()
    assert c.put([1, 2], blob(10)) is True
    assert c.put([1, 2], blob(10)) is False
    assert len(c) == 1


def test_oversized_blob_is_rejected_not_thrashed():
    """A single hybrid snapshot (~72 MB) can exceed a small budget outright."""
    c = PrefixCache(max_bytes=1000)
    assert c.put([1, 2, 3], blob(5000)) is False
    assert len(c) == 0
    assert c.stats.rejected_too_big == 1


def test_byte_budget_evicts():
    c = PrefixCache(max_bytes=250)
    c.put([1] * 10, blob(100))
    c.put([2] * 10, blob(100))
    c.put([3] * 10, blob(100))
    assert c.nbytes <= 250
    assert len(c) == 2


def test_eviction_prefers_keeping_long_recent_prefixes():
    clock = Clock()
    c = PrefixCache(max_bytes=250, clock=clock)
    c.put(list(range(100)), blob(100))  # long
    clock.advance(1)
    c.put([1, 2], blob(100))            # short, newer
    clock.advance(1)
    c.put(list(range(200, 400)), blob(100))  # longest, newest -> forces an eviction
    remaining = [len(e.ids) for e in c._entries]
    assert 2 not in remaining, "the short prefix should have been evicted first"


def test_ttl_expires_entries():
    clock = Clock()
    c = PrefixCache(ttl_s=100, clock=clock)
    c.put([1, 2, 3], blob(10))
    clock.advance(50)
    assert c.match([1, 2, 3, 4]) is not None
    clock.advance(101)
    assert c.match([1, 2, 3, 4]) is None
    assert c.stats.expirations >= 1


def test_hit_refreshes_last_used():
    clock = Clock()
    c = PrefixCache(ttl_s=100, clock=clock)
    c.put([1, 2, 3], blob(10))
    clock.advance(80)
    assert c.match([1, 2, 3, 4]) is not None  # refresh
    clock.advance(80)
    assert c.match([1, 2, 3, 4]) is not None  # would have expired without the refresh


def test_max_entries_cap():
    c = PrefixCache(max_bytes=1 << 30, max_entries=2)
    for i in range(5):
        c.put([i] * (i + 1), blob(10))
    assert len(c) <= 2


def test_drop_removes_a_poisoned_blob():
    c = PrefixCache()
    b = blob(10)
    c.put([1, 2, 3], b)
    c.drop(b)
    assert c.match([1, 2, 3, 4]) is None


def test_stats_track_reuse():
    c = PrefixCache()
    c.put([1, 2, 3, 4], blob(10))
    c.match([1, 2, 3, 4, 5, 6])   # hit: 4 reused, 2 prefilled
    c.match([9, 9])               # miss: 2 prefilled
    s = c.stats.as_dict(c)
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["reused_tokens"] == 4
    assert s["prefilled_tokens"] == 4
    assert s["hit_rate"] == 0.5


def test_empty_inputs_are_ignored():
    c = PrefixCache()
    assert c.put([], blob(10)) is False
    assert c.put([1], b"") is False
    assert c.match([]) is None
