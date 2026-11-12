#!/usr/bin/env python3
"""Board test: a Qwen3.5 prompt tail shorter than the prefill chunk is ONE padded prefill.

Without it the tail is decode-stepped one token at a time, so time to first token was a
cliff, not a slope: 31 tokens took longer than 32. A package whose prefill graph takes
pads (``prefill_pads`` in its model.json) sends the tail through one chunk instead — the graph lets the
recurrent state pass the pad rows untouched, and the engine takes their K/V back out
of the attention window. The properties checked here:

* the context advances by the real tokens only, never by the pads;
* the padded tail and the stepped one answer the same (they are different graphs under
  int8, so this is checked on prompts with one clear answer);
* a snapshot taken right after a padded chunk restores exactly — the window fix-up
  leaves the state right-aligned, which the snapshot trimming relies on;
* it is actually faster.

Needs a pads-capable Qwen3.5 package on the board; skips otherwise:

    BLLM_HYBRID_MODEL=~/models/<qwen3.5 dir> pytest tests/test_prefill_pads.py -v
"""
from __future__ import annotations

import os
import time

import pytest

bllm = pytest.importorskip("bllm")

if not getattr(bllm, "available_backends", lambda: ())():
    pytest.skip("no native backend built", allow_module_level=True)

OFF = 10 ** 6       # prefill_min_tail above any chunk: the tail is stepped


@pytest.fixture(scope="module")
def sess():
    path = os.path.expanduser(os.environ.get("BLLM_HYBRID_MODEL", "~/models/qwen35-0.8b"))
    if not os.path.isdir(path):
        pytest.skip(f"BLLM_HYBRID_MODEL not on this board ({path})")
    s = bllm.NativeSession(path)
    if s.arch != "hybrid" or not s.prefill_pads:
        pytest.skip("this package's prefill graph does not take pads")
    s.set_sampling(temp=0.0)
    yield s
    s.reset()


@pytest.fixture
def default_tail(sess):
    n = sess.prefill_min_tail
    yield n
    sess.prefill_min_tail = n


def _chat(s, q):
    return s.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


TEXT = ("The history of the printing press begins in the fifteenth century, when Johannes "
        "Gutenberg combined movable metal type, oil-based ink and a wooden press adapted "
        "from those used for wine. Within fifty years printers were working in more than "
        "two hundred cities across Europe.")


def test_context_advances_by_real_tokens_only(sess, default_tail):
    ids = sess.encode(TEXT * 3)
    assert len(ids) >= 2 * sess.prefill_chunk
    sess.reset()
    room = sess.context_left
    n_chunk = sess.prefill_chunk
    for n in (default_tail, 20, n_chunk - 1, n_chunk + 7, 2 * n_chunk - 1):
        sess.reset()
        sess.feed_ids(ids[:n])
        assert room - sess.context_left == n, f"{n} tokens fed, context moved {room - sess.context_left}"


@pytest.mark.parametrize("q", ["What is the capital of France?", "1+1等于几？",
                               "Name the largest planet in the solar system."])
def test_padded_and_stepped_tails_answer_the_same(sess, default_tail, q):
    ids = _chat(sess, q)
    assert len(ids) % sess.prefill_chunk >= default_tail, "prompt must end in a padded tail"
    replies = []
    for tail in (default_tail, OFF):
        sess.prefill_min_tail = tail
        sess.reset()
        sess.feed_ids(ids)
        replies.append(sess.decode_stream(max_new=12))
    assert replies[0].split()[:2] == replies[1].split()[:2], replies


def test_snapshot_right_after_a_padded_chunk_is_exact(sess, default_tail):
    first = _chat(sess, "Remember the number 7.")
    second = sess.encode("<|im_start|>user\nWhat number did I ask you to remember?<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    sess.reset()
    sess.feed_ids(first)
    blob = sess.snapshot()
    sess.feed_ids(second)
    straight = sess.decode_stream(max_new=16)
    sess.restore(blob)
    sess.feed_ids(second)
    assert sess.decode_stream(max_new=16) == straight


def test_padded_tail_is_faster_than_stepping_it(sess, default_tail):
    ids = sess.encode(TEXT)[: sess.prefill_chunk - 5]
    times = []
    for tail in (default_tail, OFF):
        sess.prefill_min_tail = tail
        sess.reset()
        sess.feed_ids(ids[:4])                       # warm both graphs' buffers
        sess.reset()
        t0 = time.perf_counter()
        sess.feed_ids(ids)
        times.append(time.perf_counter() - t0)
    assert times[0] < 0.6 * times[1], f"padded {times[0]*1e3:.0f} ms vs stepped {times[1]*1e3:.0f} ms"
