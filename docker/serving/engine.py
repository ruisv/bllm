"""The serving engine: one worker thread owns the session, requests queue for it.

Why a worker thread rather than a lock
--------------------------------------
The previous design held an ``asyncio.Lock`` around the SSE response while the actual
generation ran on a per-request thread. When a client disconnected mid-stream the
async generator was closed, the lock was released — and the generation thread kept
driving the engine. The next request then acquired the lock and ran *concurrently*
against the same engine. With a prefix cache that is much worse than garbled output:
the snapshot taken by one request can capture a context another request was writing,
and a poisoned blob is then handed to every later request that matches it.

So the session is owned by exactly one thread and reached only through a queue. No
HTTP-side event can release it early, because the HTTP side never holds it.

Cancellation is cooperative and lands inside the engine (``request_cancel()``), which
breaks the decode loop at a token boundary the same way a stop string does — the
context stays consistent, so the cancelled session is immediately reusable.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import bllm

from .cache import (PrefixCache, auto_budget_bytes, read_mem_available_mb,
                    snapshot_point)


class QueueFull(Exception):
    """Admission control refused the request — the caller should return 429."""


class Cancelled(Exception):
    """The client went away; the generation was cut short."""


@dataclass
class Result:
    text: str = ""
    n_prompt: int = 0
    n_completion: int = 0
    n_cached: int = 0
    finish_reason: str = "stop"
    ttft_ms: float = 0.0
    decode_tps: float = 0.0
    logprobs: Optional[list] = None   # OpenAI logprobs.content[], only when requested


@dataclass
class _Job:
    prompt: str          # kept for the legacy path, which generates from text
    ids: list[int]
    max_new: int
    stop: list[str]
    sampling: dict
    grammar: str = ""    # GBNF constraining this reply; "" = unconstrained
    cacheable: int = -1  # tokens of `ids` that are stable history; -1 = all of them
    out: "queue.Queue[object]" = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[Result] = None
    error: Optional[BaseException] = None


_SENTINEL = object()

# Sampling knobs newer than some released runtimes: droppable if set_sampling rejects
# them, rather than failing the request. See Engine._apply_sampling.
_OPTIONAL_SAMPLING = frozenset({"logit_bias", "logprobs"})


class Engine:
    """Loads the model, owns it on a worker thread, serves jobs one at a time."""

    def __init__(self) -> None:
        model = os.environ.get("BLLM_MODEL")
        if not model:
            raise RuntimeError("BLLM_MODEL is not set (model dir with model.json, or a .hbm)")
        kwargs: dict[str, Any] = {"backend": os.environ.get("BLLM_BACKEND", "auto")}
        tok = os.environ.get("BLLM_TOKENIZER_DIR")
        if tok:
            kwargs["tokenizer_dir"] = tok
        self.sess = bllm.load(model, **kwargs)
        prio = os.environ.get("BLLM_BPU_PRIORITY")
        if prio:  # empty string (unset env in compose) counts as no priority
            self.sess.set_bpu_priority(int(prio))
        self.name = getattr(self.sess, "name", None) or os.path.basename(model.rstrip("/"))
        self.default_max_new = int(os.environ.get("BLLM_MAX_NEW", "1024"))

        # The prefix cache needs runtime primitives that landed after 0.1.4, and the
        # image installs bllm from the conda channel — so the server can easily be
        # newer than the runtime under it. Degrade instead of failing: without them we
        # still get the worker-thread fix, admission control, real usage and metrics,
        # and the cache switches itself on once a new enough bllm is installed.
        self._prefix_ops = ("snapshot", "restore", "feed_ids", "decode_stream", "cache_align")
        self.supports_prefix_cache = all(hasattr(self.sess, a) for a in self._prefix_ops)
        self.supports_cancel = hasattr(self.sess, "request_cancel")
        self.supports_grammar = hasattr(self.sess, "set_grammar")
        self._unsupported_sampling: set[str] = set()

        # Chunk alignment is a correctness requirement for the cache, not a tuning
        # knob — see cache.py. 0 means this engine does not chunk (hybrid), so any
        # prefix is safe.
        self.prefill_chunk = int(getattr(self.sess, "prefill_chunk", 0) or 0)

        # "auto" (the default) sizes the budget from the memory this board actually has
        # left after loading the model — a fixed number cannot suit both engines, and
        # over-committing here does not raise, it takes the board down.
        raw = os.environ.get("BLLM_CACHE_MAX_MB", "auto").strip().lower()
        if raw in ("auto", ""):
            avail = read_mem_available_mb()
            cache_bytes = auto_budget_bytes(avail)
            print(f"[bllm-serve] prefix cache budget {cache_bytes >> 20} MB "
                  f"(auto, from {avail} MB available; set BLLM_CACHE_MAX_MB to override, "
                  f"0 to disable)", flush=True)
        else:
            cache_bytes = int(raw) << 20
        self.cache = (
            PrefixCache(
                max_bytes=cache_bytes,
                ttl_s=float(os.environ.get("BLLM_CACHE_TTL_S", "1800")),
                max_entries=int(os.environ.get("BLLM_CACHE_MAX_ENTRIES", "64")),
            )
            if cache_bytes > 0 and self.supports_prefix_cache
            else None
        )
        if cache_bytes > 0 and not self.supports_prefix_cache:
            missing = [a for a in self._prefix_ops if not hasattr(self.sess, a)]
            print(f"[bllm-serve] prefix cache disabled: this bllm build lacks "
                  f"{', '.join(missing)} — every turn re-prefills its history. "
                  f"Install a newer bllm to enable it.", flush=True)

        self.max_queue = int(os.environ.get("BLLM_MAX_QUEUE", "8"))
        self._jobs: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self._depth = 0
        self._depth_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="bllm-engine", daemon=True)
        self._worker.start()
        self.started = time.time()
        self.n_requests = 0
        self.n_cancelled = 0

    # -- public API ------------------------------------------------------------
    @property
    def queue_depth(self) -> int:
        return self._depth

    def encode(self, text: str) -> list[int]:
        """Tokenize with the model's own tokenizer — the cache keys on these ids, and
        so does the reported usage (no 4-chars-per-token guessing)."""
        return self.sess.encode(text)

    def submit(self, prompt: str, ids: list[int], max_new: int, stop: list[str],
               sampling: dict, cacheable: int = -1, grammar: str = "") -> _Job:
        """`cacheable` bounds how much of `ids` may be snapshotted: the caller knows
        which tail is specific to this request (a chat opener) and therefore will not
        appear in the next turn's prompt. Snapshotting past it wastes the entry —
        it can never match again."""
        with self._depth_lock:
            if self._depth >= self.max_queue:
                raise QueueFull(
                    f"{self._depth} requests already queued (BLLM_MAX_QUEUE={self.max_queue}); "
                    "one BPU graph serves one generation at a time"
                )
            self._depth += 1
        job = _Job(prompt=prompt, ids=ids, max_new=max_new, stop=stop,
                   sampling=sampling, cacheable=cacheable, grammar=grammar)
        self._jobs.put(job)
        self.n_requests += 1
        return job

    def stream(self, job: _Job) -> Iterator[str]:
        """Drain a submitted job's deltas. Always run this to completion or call
        ``cancel(job)`` — the worker owns the session until the job finishes."""
        while True:
            piece = job.out.get()
            if piece is _SENTINEL:
                break
            yield piece  # type: ignore[misc]
        if job.error:
            raise job.error

    def cancel(self, job: _Job) -> None:
        """Client went away. Safe to call at any point in the job's life."""
        job.cancel.set()
        if self.supports_cancel:
            self.sess.request_cancel()  # no-op if this job has not started decoding yet
        # Without runtime cancel the generation runs to completion, but it still runs
        # ON THE WORKER, so the next request waits rather than racing it.

    def wait(self, job: _Job, timeout: float | None = None) -> Result:
        job.done.wait(timeout)
        if job.error:
            raise job.error
        assert job.result is not None
        return job.result

    def metrics(self) -> dict:
        m = {
            "model": self.name,
            "uptime_s": round(time.time() - self.started, 1),
            "requests": self.n_requests,
            "cancelled": self.n_cancelled,
            "queue_depth": self.queue_depth,
            "max_queue": self.max_queue,
            "prefill_chunk": self.prefill_chunk,
            "supports_prefix_cache": self.supports_prefix_cache,
            "supports_cancel": self.supports_cancel,
            "supports_grammar": self.supports_grammar,
            "context_window": getattr(self.sess, "context_left", None),
        }
        m["cache"] = (self.cache.stats.as_dict(self.cache) if self.cache is not None
                      else {"enabled": False})
        return m

    def shutdown(self) -> None:
        self._jobs.put(None)
        self._worker.join(timeout=5)

    # -- the worker ------------------------------------------------------------
    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                job.result = self._serve(job)
            except BaseException as exc:  # noqa: BLE001 — handed to the requester
                job.error = exc
            finally:
                with self._depth_lock:
                    self._depth -= 1
                job.out.put(_SENTINEL)
                job.done.set()

    def _serve(self, job: _Job) -> Result:
        ids = job.ids
        res = Result(n_prompt=len(ids))

        # TTFT starts HERE, not at the decode. Restoring a blob and prefilling the delta
        # is most of what a client waits through — on a dense turn that crosses a chunk
        # it is ~366 ms against ~0.3 ms of actual first-token sampling, because prefill
        # already produced the logits the first token is drawn from. Timing only the
        # decode reported ~0.3 ms for every request, hit or miss, which made the one
        # number that should show the prefix cache working completely blind to it.
        t0 = time.perf_counter()

        if job.cancel.is_set():  # disconnected while queued — never touch the engine
            raise Cancelled("client disconnected before the request ran")

        if not self.supports_prefix_cache:
            return self._serve_legacy(job, res)

        # 1. restore the longest cached prefix, or start cold
        hit = self.cache.match(ids) if self.cache is not None else None
        blob_in = None
        if hit:
            blob_in, covered = hit
            try:
                self.sess.restore(blob_in)
            except Exception:
                # A blob the engine will not take is worse than useless — drop it and
                # fall back rather than failing the request.
                self.cache.drop(blob_in)
                blob_in, covered = None, 0
                self.sess.reset()
        else:
            covered = 0
            self.sess.reset()
        res.n_cached = covered
        delta = ids[covered:]

        self._apply_sampling(job.sampling)
        self._apply_grammar(job.grammar)

        # 2. prefill only what is new, snapshotting at the aligned boundary so the
        #    cached state is one the un-cached path would also have produced
        try:
            self._feed_and_cache(ids, delta, covered, job.cacheable)
        except Exception:
            if blob_in is not None:
                self.cache.drop(blob_in)  # never reuse a state we may have half-fed
            raise

        # 3. decode
        first: list[float] = []

        def on_text(piece: str) -> None:
            if not first:
                first.append(time.perf_counter())
            job.out.put(piece)

        try:
            res.text = self.sess.decode_stream(job.max_new, on_text, job.stop)
        except Exception:
            if blob_in is not None:
                self.cache.drop(blob_in)
            raise

        res.n_completion = len(self.sess.encode(res.text)) if res.text else 0
        res.ttft_ms = round((first[0] - t0) * 1e3, 1) if first else 0.0
        res.decode_tps = round(float(getattr(self.sess, "last_decode_tps", 0.0) or 0.0), 2)
        res.logprobs = self._logprobs_payload()
        if self.sess.canceled:
            self.n_cancelled += 1
            res.finish_reason = "cancelled"
        elif res.n_completion >= job.max_new:
            res.finish_reason = "length"
        return res

    def _logprobs_payload(self) -> Optional[list]:
        """The last turn's logprobs in OpenAI's `logprobs.content[]` shape, or None.

        Token ids are turned back into text here rather than in C++ because that is where
        the tokenizer already lives on the Python side; the decodes are memoised because a
        request with top_logprobs=5 asks for six of them per generated token, and the same
        few thousand ids recur."""
        fn = getattr(self.sess, "last_logprobs", None)
        if fn is None:               # runtime older than logprobs support
            return None
        raw = fn()
        if not raw:                  # not requested, or nothing was generated
            return None
        memo: dict[int, str] = {}

        def tok(i: int) -> dict:
            if i not in memo:
                memo[i] = self.sess.decode([i])
            text = memo[i]
            return {"token": text, "bytes": list(text.encode("utf-8"))}

        return [
            {**tok(e["id"]), "logprob": e["logprob"],
             "top_logprobs": [{**tok(i), "logprob": lp} for i, lp in e["top"]]}
            for e in raw
        ]

    def _apply_grammar(self, gbnf: str) -> None:
        """Constrain (or unconstrain) this reply. Always called, so a grammar never leaks
        from one request into the next — the session keeps it until told otherwise.

        Unlike a sampling knob this is NOT degraded away on an old runtime: a grammar is a
        guarantee the client asked for, and answering unconstrained would look like success.
        The request path checks `supports_grammar` first, so this only fires on a race."""
        if not self.supports_grammar:
            if gbnf:
                raise RuntimeError("this bllm runtime has no grammar support; "
                                   "install a newer bllm to use response_format/grammar")
            return
        self.sess.set_grammar(gbnf)

    def _apply_sampling(self, sampling: dict) -> None:
        """set_sampling(), minus any knob this runtime is too old to know about.

        The image installs bllm from the conda channel, so the server can be newer than
        the runtime under it (same reason the prefix cache feature-detects). A knob added
        after that runtime raises TypeError; dropping it serves the request with the rest
        of the sampling honoured, which beats 500-ing on an optional field. The drop is
        remembered, so the retry happens once and not per request."""
        s = {k: v for k, v in sampling.items() if k not in self._unsupported_sampling}
        try:
            self.sess.set_sampling(**s)
        except TypeError:
            extra = [k for k in s if k in _OPTIONAL_SAMPLING]
            if not extra:
                raise            # a real argument error, not an old runtime
            self._unsupported_sampling.update(extra)
            print(f"[bllm-serve] this bllm runtime does not support {', '.join(extra)} — "
                  f"the option will be ignored. Install a newer bllm.", flush=True)
            self.sess.set_sampling(**{k: v for k, v in s.items() if k not in extra})

    def _serve_legacy(self, job: _Job, res: Result) -> Result:
        """Runtime without the snapshot primitives: reset and generate from text, the
        way the pre-rewrite server did. No cache, but still serialized on the worker."""
        self.sess.reset()
        self._apply_sampling(job.sampling)
        self._apply_grammar(job.grammar)
        t0 = time.perf_counter()
        first: list[float] = []

        def on_text(piece: str) -> None:
            if not first:
                first.append(time.perf_counter())
            job.out.put(piece)

        res.text = self.sess.generate(job.prompt, job.max_new, on_text, job.stop)
        res.n_completion = len(self.sess.encode(res.text)) if res.text else 0
        res.ttft_ms = round((first[0] - t0) * 1e3, 1) if first else 0.0
        res.decode_tps = round(float(getattr(self.sess, "last_decode_tps", 0.0) or 0.0), 2)
        if getattr(self.sess, "canceled", False):
            self.n_cancelled += 1
            res.finish_reason = "cancelled"
        elif res.n_completion >= job.max_new:
            res.finish_reason = "length"
        return res

    def _feed_and_cache(self, ids: list[int], delta: list[int], covered: int,
                        cacheable: int = -1) -> None:
        """Prefill `delta`, and cache the state at the largest safe prefix of `ids`.

        The snapshot point must be chunk-aligned, so the prompt is fed in two pieces
        when the aligned point falls inside the delta: up to the boundary, snapshot,
        then the rest. That split does not change the segmentation — feed()'s chunking
        depends only on the number of tokens remaining, and the boundary is exactly
        where a straight-through prefill would have broken anyway.
        """
        if self.cache is None:
            if delta:
                self.sess.feed_ids(delta)
            return

        limit = len(ids) if cacheable < 0 else min(cacheable, len(ids))
        at = snapshot_point(len(ids), covered, self.sess.cache_align(limit))
        if at is None:
            # No safe snapshot point past what we already have — feed it all and cache
            # nothing. A dense prompt shorter than one chunk lands here, and
            # re-prefilling such a prompt is cheap anyway.
            if delta:
                self.sess.feed_ids(delta)
            return

        self.sess.feed_ids(ids[covered:at])
        self.cache.put(ids[:at], self.sess.snapshot())
        if at < len(ids):
            self.sess.feed_ids(ids[at:])
