#!/usr/bin/env python3
"""Board test: the serving primitives — snapshot/restore/feed_ids/decode_stream.

The correctness floor for the session-pool + prefix-cache serving layer.
Two separate properties, and the distinction matters:

1. **restore() is transparent.** Restoring a snapshot and appending is identical to
   having fed the same two runs without ever snapshotting. This is about the state
   blob being complete and correctly reloaded.

2. **A CHUNK-ALIGNED cache hit is identical to no cache at all.** The dense engine
   ingests a run as batch-prefill chunks of ``prefill_chunk`` and decode-steps the
   remainder (``native_engine.h`` feed()), and those two graphs are not numerically
   identical under int8. Segmentation is a pure function of the remaining count, so
   the baseline's chunk boundaries sit at multiples of ``prefill_chunk``; snapshot at
   one of those and the tail is segmented exactly as the un-cached run. Snapshot
   anywhere else and the reply can change — measured 1/18 exact on unaligned splits,
   including one that flipped "Venus is the hottest" to "Earth is the hottest".

Needs a model on the board; skips cleanly when one is absent.

    BLLM_DENSE_MODEL=~/models/qwen25-fc0 \
    BLLM_HYBRID_MODEL=~/models/qwen35-0.8b \
    pytest tests/test_prefix_reuse.py -v
"""
from __future__ import annotations

import os
import threading
import time

import pytest

bllm = pytest.importorskip("bllm")

if not getattr(bllm, "available_backends", lambda: ())():
    pytest.skip("no native backend built", allow_module_level=True)

# Long enough to span several prefill chunks, ending on a genuinely low-margin
# question — that is where segmentation differences actually surface.
FILLER = (
    "The following is a list of facts about the solar system. "
    "Mercury is the closest planet to the Sun. "
    "Venus is the second planet and the hottest. "
    "Earth is the third planet and the only one known to host life. "
    "Mars is the fourth planet and is called the red planet. "
)
PROMPT = FILLER * 9 + "Question: which planet is the hottest? Answer:"
MAX_NEW = 20


def _model(env: str, default: str):
    path = os.path.expanduser(os.environ.get(env, default))
    if not os.path.isdir(path):
        pytest.skip(f"{env} not on this board ({path})")
    return path


def _session(path):
    s = bllm.load(path)
    s.set_sampling(temp=0.0)  # greedy — the comparison needs determinism
    return s


# ONE session per model, module-scoped and reused. Model weights live in ION, not in
# process RSS, so a session per test exhausts the carveout — which does not raise, it
# takes the board down (learned the hard way: it rebooted). For the same reason, run
# the dense and hybrid sets in separate processes when the carveout is tight:
#   pytest tests/test_prefix_reuse.py -k "not hybrid"
#   pytest tests/test_prefix_reuse.py -k hybrid
@pytest.fixture(scope="module")
def dense():
    return _session(_model("BLLM_DENSE_MODEL", "~/models/qwen25-fc0"))


@pytest.fixture(scope="module")
def hybrid():
    return _session(_model("BLLM_HYBRID_MODEL", "~/models/qwen35-0.8b"))


def _baseline(sess, ids):
    """No cache: one straight-through prefill, then decode."""
    sess.reset()
    sess.feed_ids(ids)
    return sess.decode_stream(MAX_NEW)


def _split_feed(sess, ids, at):
    """Two feeds, no snapshot — isolates segmentation from state handling."""
    sess.reset()
    sess.feed_ids(ids[:at])
    sess.feed_ids(ids[at:])
    return sess.decode_stream(MAX_NEW)


def _via_snapshot(sess, ids, at):
    """Snapshot the head, wipe the engine, restore, append the tail."""
    sess.reset()
    sess.feed_ids(ids[:at])
    blob = sess.snapshot()
    sess.reset()  # prove restore() rebuilds the state rather than it surviving in place
    sess.restore(blob)
    sess.feed_ids(ids[at:])
    return sess.decode_stream(MAX_NEW), blob


def test_restore_is_transparent(dense):
    """Property 1: snapshot/restore adds nothing and loses nothing — it reproduces a
    plain split feed exactly, at any split point, aligned or not."""
    sess = dense
    ids = sess.encode(PROMPT)
    for at in (37, 128, 256, 300, len(ids) - 5):
        if not 0 < at < len(ids):
            continue
        assert _via_snapshot(sess, ids, at)[0] == _split_feed(sess, ids, at), (
            f"snapshot/restore diverged from a plain split feed at {at}"
        )


def test_aligned_reuse_is_exact(dense):
    """Property 2, the one the cache depends on: a cache hit at a chunk-aligned prefix
    returns exactly what the un-cached request would have."""
    sess = dense
    ids = sess.encode(PROMPT)
    chunk = sess.prefill_chunk
    assert chunk > 0, "dense engine is expected to chunk its prefill"
    assert len(ids) > 2 * chunk, "prompt must span several chunks to test alignment"

    base = _baseline(sess, ids)
    assert base.strip(), "model produced nothing — check the model dir"

    for at in range(chunk, len(ids), chunk):
        got, blob = _via_snapshot(sess, ids, at)
        assert got == base, (
            f"aligned split {at} changed the reply — the prefix cache is not safe\n"
            f"  uncached: {base!r}\n"
            f"  restored: {got!r}"
        )
        assert len(blob) > 0


def test_cache_align_only_returns_safe_points(dense):
    sess = dense
    chunk = sess.prefill_chunk
    for n in (0, 1, chunk - 1, chunk, chunk + 1, 3 * chunk + 7):
        a = sess.cache_align(n)
        assert a <= n
        assert a % chunk == 0


def test_hybrid_reuse_is_exact_at_any_split(hybrid):
    """The hybrid engine has no chunked prefill (prefill_chunk == 0), so it is not
    segmentation-sensitive: every split is exact. The SSM state is a recurrent summary
    that cannot be truncated, but restoring and appending must still be transparent."""
    sess = hybrid
    assert sess.prefill_chunk == 0
    ids = sess.encode(PROMPT)
    base = _baseline(sess, ids)
    assert base.strip()

    for at in (37, len(ids) // 3, len(ids) // 2, len(ids) - 5):
        assert sess.cache_align(at) == at
        got, _ = _via_snapshot(sess, ids, at)
        assert got == base, f"hybrid split {at} diverged"


def test_snapshot_survives_later_work(dense):
    """A cached blob must stay valid after the engine has moved on — a server holds many
    snapshots and restores them out of order."""
    sess = dense
    ids = sess.encode(PROMPT)
    at = sess.cache_align(len(ids) // 2)
    base = _baseline(sess, ids)

    sess.reset()
    sess.feed_ids(ids[:at])
    blob = sess.snapshot()

    # Take the engine somewhere else entirely before coming back to the blob.
    sess.reset()
    sess.feed_ids(sess.encode("Something completely unrelated about cooking pasta."))
    sess.decode_stream(8)

    sess.restore(blob)
    sess.feed_ids(ids[at:])
    assert sess.decode_stream(MAX_NEW) == base


def test_restore_rejects_a_corrupt_blob(dense):
    sess = dense
    sess.reset()
    sess.feed_ids(sess.encode(FILLER))
    blob = sess.snapshot()
    with pytest.raises(Exception):
        sess.restore(blob[: len(blob) // 2])  # truncated → must not be silently accepted


def test_request_cancel_stops_and_leaves_the_session_usable(dense):
    """P0 depends on this: a disconnected client must not leave the engine running, and
    the next request must still get a clean context."""
    sess = dense
    ids = sess.encode(PROMPT)
    sess.reset()
    sess.feed_ids(ids)

    seen: list[str] = []

    def cancel_soon():
        deadline = time.time() + 30
        while len(seen) < 3 and time.time() < deadline:
            time.sleep(0.01)
        sess.request_cancel()

    t = threading.Thread(target=cancel_soon, daemon=True)
    t.start()
    out = sess.decode_stream(400, on_text=seen.append)
    t.join(timeout=5)

    assert sess.canceled, "generation did not end on the cancel"
    assert out == "".join(seen), "cancelled stream and return value disagree"
    assert len(seen) < 400, "cancel did not actually cut the generation short"

    # A cancel stops at a token boundary; it must not corrupt the context.
    assert _baseline(sess, ids).strip()
    assert not sess.canceled


def test_cancel_between_turns_is_dropped(dense):
    """A cancel that lands after a generation ended must not kill the next one."""
    sess = dense
    ids = sess.encode(PROMPT)
    sess.request_cancel()  # nothing is running
    out = _baseline(sess, ids)
    assert not sess.canceled
    assert out.strip()


def test_hybrid_snapshot_grows_with_context_not_with_the_window(hybrid):
    """The hybrid cache set is a MIX: fixed-size GDN recurrent state plus an attention
    K/V window. Dumping the whole window made every snapshot a flat ~72-122 MB no
    matter how short the conversation, which bounded a serving prefix cache by the
    window rather than by the conversation. Only the occupied K/V is saved now, so the
    blob must GROW with context — a constant size means the trimming regressed.
    """
    sess = hybrid
    unit = sess.encode("Mercury is the closest planet to the Sun and Venus is the hottest. ")

    sizes = []
    for mult in (1, 4, 16):
        ids = (unit * mult)[: sess.context_left - 64]
        sess.reset()
        sess.feed_ids(ids)
        sizes.append((len(ids), len(sess.snapshot())))

    (n0, b0), (n1, b1), (n2, b2) = sizes
    assert b0 < b1 < b2, f"snapshot size did not grow with context: {sizes}"

    # It must also be well under a full-window dump. The fixed GDN part dominates at
    # short contexts, so assert against the per-token slope rather than a magic number:
    # extrapolating to the full window must be much larger than what we just saw.
    per_tok = (b2 - b0) / (n2 - n0)
    full_window = b0 + per_tok * sess.context_left
    assert b2 < full_window / 2, (
        f"a {n2}-token snapshot ({b2/1e6:.1f} MB) is not meaningfully smaller than a "
        f"full-window dump (~{full_window/1e6:.1f} MB)")
