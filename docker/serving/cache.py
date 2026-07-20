"""Token-prefix snapshot cache for the BLLM serving layer.

The OpenAI protocol is stateless: a client re-sends the whole ``messages[]`` every
turn, so turn N would re-prefill turns 1..N-1. This cache keeps engine snapshots
keyed by the token ids they cover, so a turn only prefills what is new.

**Matching is exact token-prefix comparison, and that is the whole safety argument.**
A cached entry is used only when its ids are a strict prefix of the incoming ids.
Anything else — the client edited history, re-tokenization differed, a different
system prompt — simply matches a shorter entry or misses entirely and falls back to
a full prefill. A wrong reuse is not possible, so this layer cannot produce a wrong
answer; at worst it is slow.

**Callers must only insert CHUNK-ALIGNED prefixes** (``NativeSession.cache_align``).
The dense engine batch-prefills runs of ``prefill_chunk`` tokens and decode-steps the
remainder, and those graphs are not numerically identical under int8 — so a snapshot
taken off a chunk boundary changes the reply. Measured on an S100P: aligned splits
reproduced the un-cached reply 5/5, unaligned 1/18, one of them turning "Venus is the
hottest" into "Earth is the hottest". This module cannot verify alignment itself (it
never sees the engine), so it is the caller's contract — see ``engine.py``.

No bllm import: this is pure logic and runs in CI on any host.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


def snapshot_point(n_total: int, covered: int, align: int) -> int | None:
    """Where to snapshot this prompt, or None to cache nothing.

    `align` is ``session.cache_align(n_total)`` — the largest prefix that is safe to
    snapshot. Snapshot there whenever it is past what the cache already covers.

    The subtle case is ``align == n_total``, which is the NORMAL case for the hybrid
    engine (it does not chunk, so every point is safe). Treating that as "nothing to
    do" silently disables the cache for exactly the models it helps most — hybrid
    prefill is ~68 ms/token, so this is the 25x case. Caching the whole prompt is both
    safe and useful: it is a strict prefix of the next turn's prompt, which is what
    match() will be asked for.
    """
    if align <= covered:
        return None
    return align


@dataclass
class Entry:
    ids: tuple[int, ...]
    blob: bytes
    last_used: float
    created: float
    hits: int = 0

    @property
    def nbytes(self) -> int:
        return len(self.blob)


@dataclass
class Stats:
    hits: int = 0
    misses: int = 0
    reused_tokens: int = 0
    prefilled_tokens: int = 0
    evictions: int = 0
    expirations: int = 0
    inserts: int = 0
    rejected_too_big: int = 0

    def as_dict(self, cache: "PrefixCache") -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "reused_tokens": self.reused_tokens,
            "prefilled_tokens": self.prefilled_tokens,
            "token_reuse_rate": (
                round(self.reused_tokens / (self.reused_tokens + self.prefilled_tokens), 4)
                if (self.reused_tokens + self.prefilled_tokens)
                else 0.0
            ),
            "entries": len(cache),
            "bytes": cache.nbytes,
            "max_bytes": cache.max_bytes,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "inserts": self.inserts,
            "rejected_too_big": self.rejected_too_big,
        }


class PrefixCache:
    """Snapshots keyed by the token ids they cover, bounded by bytes and by age.

    Sized in BYTES, not entries: a dense snapshot is ~22 KB/token and grows with the
    context, while a hybrid one is a flat ~72 MB regardless of length (it dumps the
    whole cache set). An entry count would mean wildly different memory per model.
    """

    def __init__(self, max_bytes: int = 512 << 20, ttl_s: float = 1800.0,
                 max_entries: int = 64, clock=None) -> None:
        self.max_bytes = int(max_bytes)
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._clock = clock or (lambda: __import__("time").monotonic())
        self._entries: list[Entry] = []
        self._lock = threading.Lock()
        self.stats = Stats()

    # -- introspection ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        """A cache object is always truthy. Without this, __len__ makes an EMPTY cache
        falsy, so `if self.cache:` would skip inserts — and the cache, starting empty,
        could never fill. (It shipped that way for one test run.)"""
        return True

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self._entries)

    # -- lookup ----------------------------------------------------------------
    def match(self, ids) -> tuple[bytes, int] | None:
        """Longest cached entry that is a STRICT prefix of `ids` -> (blob, covered).

        Strict: an entry equal to the whole of `ids` is not usable, because the engine
        needs at least one token to feed before decoding — restoring a state whose
        logits already predict the next token is fine, but the request would then
        prefill nothing and the snapshot/decode split has no token to hang TTFT on.
        """
        key = tuple(ids)
        now = self._clock()
        with self._lock:
            self._expire(now)
            best: Entry | None = None
            for e in self._entries:
                n = len(e.ids)
                if n < len(key) and key[:n] == e.ids and (best is None or n > len(best.ids)):
                    best = e
            if best is None:
                self.stats.misses += 1
                self.stats.prefilled_tokens += len(key)
                return None
            best.last_used = now
            best.hits += 1
            self.stats.hits += 1
            self.stats.reused_tokens += len(best.ids)
            self.stats.prefilled_tokens += len(key) - len(best.ids)
            return best.blob, len(best.ids)

    # -- insertion -------------------------------------------------------------
    def put(self, ids, blob: bytes) -> bool:
        """Cache `blob` as the state covering `ids`. Returns whether it was stored.

        `ids` MUST be a chunk-aligned prefix (see the module docstring).
        """
        key = tuple(ids)
        if not key or not blob:
            return False
        if len(blob) > self.max_bytes:
            # A single hybrid snapshot can exceed a small budget outright; say so via
            # the counter rather than thrashing the cache trying to fit it.
            self.stats.rejected_too_big += 1
            return False
        now = self._clock()
        with self._lock:
            self._expire(now)
            for e in self._entries:
                if e.ids == key:  # already have this exact prefix
                    e.last_used = now
                    return False
            self._entries.append(Entry(ids=key, blob=blob, last_used=now, created=now))
            self.stats.inserts += 1
            self._evict_to_fit()
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def drop(self, blob: bytes) -> None:
        """Forget the entry holding `blob` — used when a generation failed against it,
        so a possibly-poisoned state is never handed to another request."""
        with self._lock:
            self._entries = [e for e in self._entries if e.blob is not blob]

    # -- eviction --------------------------------------------------------------
    def _expire(self, now: float) -> None:
        if self.ttl_s <= 0:
            return
        keep = [e for e in self._entries if now - e.last_used <= self.ttl_s]
        self.stats.expirations += len(self._entries) - len(keep)
        self._entries = keep

    def _value(self, e: Entry, now: float) -> float:
        """Higher is more worth keeping: long prefixes cost more to rebuild, recent
        ones are likelier to be asked for again. Rebuilding an 800-token prefix costs
        ~8x rebuilding a 100-token one, so length is not a tiebreaker — it is the
        numerator."""
        return len(e.ids) / (1.0 + (now - e.last_used))

    def _evict_to_fit(self) -> None:
        now = self._clock()
        total = sum(e.nbytes for e in self._entries)
        while self._entries and (total > self.max_bytes or len(self._entries) > self.max_entries):
            victim = min(self._entries, key=lambda e: self._value(e, now))
            self._entries.remove(victim)
            total -= victim.nbytes
            self.stats.evictions += 1
