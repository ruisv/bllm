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
from .media import image_url_to_tempfile


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


@dataclass
class _VlmJob:
    """A VLM request: the OpenAI ``messages[]`` array reduced to role/text/images
    turns (see ``server.py:_vlm_turns_from``), not pre-rendered ids — there is no
    ids-based prefix cache for the VLM session (it has no snapshot/restore), so the
    engine matches history at turn granularity instead. See ``Engine._serve_vlm``.
    """
    turns: list       # [{"role","text","images"}, ...]; last is the user turn to answer
    max_new: int
    stop: list
    sampling: dict
    grammar: str = ""
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

        # A VLM session (NativeVlmSession) is a distinct object from the text session
        # (NativeSession) and speaks a different API — chat()/append_turn() on whole
        # turns, not encode()/feed_ids()/decode_stream() on token ids. `vision_tokens`
        # only exists on the former, so it doubles as the routing flag; see
        # `_serve_vlm` for how requests to a VLM-loaded model are actually served.
        self.is_vlm = hasattr(self.sess, "vision_tokens")
        self._vlm_committed: list = []  # [(user_turn, assistant_turn), ...] already fed live

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
        if cache_bytes > 0 and not self.supports_prefix_cache and not self.is_vlm:
            missing = [a for a in self._prefix_ops if not hasattr(self.sess, a)]
            print(f"[bllm-serve] prefix cache disabled: this bllm build lacks "
                  f"{', '.join(missing)} — every turn re-prefills its history. "
                  f"Install a newer bllm to enable it.", flush=True)
        elif self.is_vlm:
            print("[bllm-serve] VLM session: the ids-based prefix cache does not apply "
                  "(no snapshot/restore) — history is instead replayed turn-by-turn via "
                  "append_turn(), see Engine._serve_vlm.", flush=True)

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

    def submit_vlm(self, turns: list, max_new: int, stop: list, sampling: dict,
                   grammar: str = "") -> _VlmJob:
        """`turns` is the whole conversation (see `_VlmJob`), not just the new part —
        the engine itself works out how much of it is already live in the session
        (`_serve_vlm`) and only replays what changed."""
        with self._depth_lock:
            if self._depth >= self.max_queue:
                raise QueueFull(
                    f"{self._depth} requests already queued (BLLM_MAX_QUEUE={self.max_queue}); "
                    "one BPU graph serves one generation at a time"
                )
            self._depth += 1
        job = _VlmJob(turns=turns, max_new=max_new, stop=stop, sampling=sampling, grammar=grammar)
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
            "is_vlm": self.is_vlm,
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
                job.result = self._serve_vlm(job) if isinstance(job, _VlmJob) else self._serve(job)
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
        res.logprobs = self._logprobs_payload(job.sampling.get("logprobs", -1) >= 0)
        if self.sess.canceled:
            self.n_cancelled += 1
            res.finish_reason = "cancelled"
        elif res.n_completion >= job.max_new:
            res.finish_reason = "length"
        return res

    def _logprobs_payload(self, wanted: bool) -> Optional[list]:
        """The last turn's logprobs in OpenAI's `logprobs.content[]` shape, or None.

        `wanted` comes from the request, not from whether anything was produced: a reply
        that generated no tokens still asked for logprobs, and answering with the key
        missing rather than an empty array would make a client unwrap None.

        `bytes` is the token's REAL bytes, via token_bytes() — `token.encode()` would be
        wrong for a token holding half a multi-byte character, which decodes to U+FFFD.
        That is exactly the case OpenAI's `bytes` field exists to cover. Both lookups are
        memoised: top_logprobs=5 asks for six per generated token, and the ids recur."""
        fn = getattr(self.sess, "last_logprobs", None)
        if fn is None or not wanted:      # runtime older than logprobs support, or not asked
            return None
        raw = fn()
        memo: dict[int, dict] = {}
        raw_bytes = getattr(self.sess, "token_bytes", None)

        def tok(i: int) -> dict:
            if i not in memo:
                text = self.sess.decode([i])
                b = raw_bytes(i) if raw_bytes is not None else text.encode("utf-8")
                memo[i] = {"token": text, "bytes": list(b)}
            return memo[i]

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

    # -- VLM ---------------------------------------------------------------------
    #
    # There is no ids-based prefix cache here (NativeVlmSession has no
    # snapshot/restore — see SERVING_PLAN.md), so the engine keeps its own record
    # of which (user, assistant) turn pairs are already live in `self.sess`
    # (`_vlm_committed`) and diffs the incoming conversation against it:
    #
    #   * unchanged prefix, request extends it by K turns -> replay only those K
    #     via `append_turn()` (prefill only, no decode — see native_vlm.h)
    #   * anything else (first request, history edited, a different conversation
    #     interleaved on this single-session engine) -> reset() and replay all of it
    #
    # This is weaker than the token-exact ids cache: two different conversations
    # taking turns on one engine will each see the other as "history changed" and
    # pay a full replay every time. That degrades to "no cache" gracefully, the
    # same way the ids cache degrades on a miss — it cannot make an answer wrong.
    def _serve_vlm(self, job: _VlmJob) -> Result:
        res = Result()
        if job.cancel.is_set():
            raise Cancelled("client disconnected before the request ran")

        history, final = job.turns[:-1], job.turns[-1]
        pairs = self._pair_vlm_history(history)

        self._apply_sampling(job.sampling)
        self._apply_grammar(job.grammar)

        t0 = time.perf_counter()
        committed = self._vlm_committed
        reused = len(pairs) >= len(committed) and pairs[:len(committed)] == committed
        if not reused:
            self.sess.reset()
            committed = self._vlm_committed = []

        try:
            res.n_cached = self.sess.tokens_used  # whatever is already live, for free

            for user_turn, asst_turn in pairs[len(committed):]:
                paths = self._materialize_images(user_turn["images"])
                try:
                    self.sess.append_turn(user_turn["text"], images=paths, reply=asst_turn["text"])
                finally:
                    self._cleanup_images(paths)
                committed.append((user_turn, asst_turn))

            paths = self._materialize_images(final["images"])
            first: list[float] = []

            def on_text(piece: str) -> None:
                if not first:
                    first.append(time.perf_counter())
                job.out.put(piece)

            try:
                res.text = self.sess.chat(final["text"], images=paths, max_new=job.max_new,
                                          on_text=on_text, stop=job.stop)
            finally:
                self._cleanup_images(paths)
            # This reply is now live in `self.sess` exactly like a replayed turn would
            # be, so record it as committed too — otherwise the NEXT request (even an
            # unrelated one that also happens to arrive with empty history, e.g. two
            # back-to-back single-turn conversations) would see `committed` as still
            # empty, wrongly "match" a from-scratch conversation, and silently answer
            # against this turn's leftover context instead of resetting.
            committed.append((final, {"role": "assistant", "text": res.text, "images": []}))
        except BaseException:
            # A partial feed (an image failed mid-replay, a cancel landed mid-decode)
            # can leave `self.sess` holding rows `_vlm_committed` does not know about.
            # There is no restore() to fall back to for a VLM session (unlike the
            # ids-based cache's snapshot/drop), so the only safe recovery is to force
            # the NEXT request to reset() and replay from scratch rather than trust a
            # `committed` prefix the live session may no longer actually match.
            self._vlm_committed = []
            raise

        res.n_completion = len(self.sess.encode(res.text)) if res.text else 0
        # tokens_used is the WHOLE conversation so far including this reply — OpenAI's
        # prompt_tokens means "the context this reply was drawn from", so subtract the
        # reply back out rather than trying to total up media/text tokens ourselves.
        res.n_prompt = max(self.sess.tokens_used - res.n_completion, 0)
        res.ttft_ms = round((first[0] - t0) * 1e3, 1) if first else 0.0
        res.decode_tps = round(float(getattr(self.sess, "last_decode_tps", 0.0) or 0.0), 2)
        if res.n_completion >= job.max_new:
            res.finish_reason = "length"
        return res

    @staticmethod
    def _pair_vlm_history(history: list) -> list:
        """`history` (job.turns minus the final unanswered user turn) as [(user,
        assistant), ...] pairs — the unit `_vlm_committed` tracks and diffs against.
        `server.py:_vlm_turns_from` already enforces strict user/assistant
        alternation with no system turns, so a well-formed history is even length;
        this only re-checks it because `Engine` must not trust a malformed job into
        corrupting `self.sess` (a raise here drops the request, not the session)."""
        if len(history) % 2 != 0:
            raise ValueError("conversation history must alternate user/assistant turns")
        pairs = []
        for i in range(0, len(history), 2):
            u, a = history[i], history[i + 1]
            if u["role"] != "user" or a["role"] != "assistant":
                raise ValueError("conversation history must alternate user/assistant turns")
            pairs.append((u, a))
        return pairs

    @staticmethod
    def _materialize_images(urls: list) -> list:
        """image_url values -> temp file paths NativeVlmSession can load. Caller
        must `_cleanup_images` them once the chat()/append_turn() call is done —
        cleaned up eagerly rather than left for the OS temp dir to accumulate,
        since a busy server can serve thousands of images a day."""
        paths = []
        try:
            for u in urls:
                paths.append(image_url_to_tempfile(u))
        except BaseException:
            Engine._cleanup_images(paths)
            raise
        return paths

    @staticmethod
    def _cleanup_images(paths: list) -> None:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass

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
