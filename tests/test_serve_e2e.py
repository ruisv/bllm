#!/usr/bin/env python3
"""End-to-end tests against a running bllm-serve.

Skipped unless BLLM_SERVE_URL points at a live server, because they need the board
and a loaded model:

    BLLM_MODEL=~/models/qwen25-fc0 uvicorn server:app --port 8877 &
    BLLM_SERVE_URL=http://127.0.0.1:8877 pytest tests/test_serve_e2e.py -v

The disconnect test is the one that matters. The pre-rewrite server released its
lock when a client vanished mid-stream while the generation thread kept driving the
engine, so the next request ran concurrently against it. Without a cache that
garbles a reply; with one, a snapshot taken mid-corruption is stored and handed to
every later request that matches it — a persistent wrong answer. Hence: disconnect
repeatedly, and assert a deterministic question keeps its exact answer.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("BLLM_SERVE_URL")
if not BASE:
    pytest.skip("BLLM_SERVE_URL not set — needs a running server", allow_module_level=True)


def _post(body, timeout=600):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(messages, **kw):
    with _post({"messages": messages, "max_tokens": 24, "temperature": 0.0, **kw}) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


FACTS = "Please remember these facts. " + "The solar system has eight planets. " * 60
QUESTION = [{"role": "user", "content": "Reply with exactly the word: pineapple"}]

# The image installs bllm from the conda channel, so the server can be newer than the
# runtime beneath it. Without the snapshot primitives it degrades to full re-prefill —
# a supported mode, not a failure, so the cache tests skip rather than fail.
_CAPS = get("/metrics")
needs_cache = pytest.mark.skipif(
    not _CAPS.get("supports_prefix_cache"),
    reason="this bllm build lacks the snapshot primitives; server is in re-prefill mode")


def test_health():
    assert get("/health")["status"] == "ok"


def test_usage_is_real_tokens_not_an_estimate():
    r = chat([{"role": "user", "content": FACTS}])
    u = r["usage"]
    assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    # the old server guessed 4 chars per token; a real tokenizer will not land there
    assert u["prompt_tokens"] != len(FACTS) // 4
    # and it reported no cache accounting at all
    assert "cached_tokens" in u.get("prompt_tokens_details", {})


@needs_cache
def test_second_turn_reuses_the_prefix():
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": FACTS}]
    r1 = chat(msgs)
    msgs += [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
             {"role": "user", "content": "How many planets did I mention?"}]
    r2 = chat(msgs)
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


@needs_cache
def test_divergent_history_does_not_reuse_the_wrong_state():
    """Edit the opening turn and the cached prefix must no longer apply."""
    chat([{"role": "user", "content": FACTS + " Version A."}])
    b = chat([{"role": "user", "content": "Totally different opening. " + FACTS}])
    assert b["usage"]["prompt_tokens_details"]["cached_tokens"] < len(FACTS) // 8


def test_repeated_disconnects_do_not_corrupt_later_answers():
    """The P0 regression test."""
    baseline = chat(QUESTION)["choices"][0]["message"]["content"]

    for _ in range(5):
        resp = _post({"messages": [{"role": "user", "content": FACTS}],
                      "max_tokens": 200, "stream": True})
        n = 0
        for raw in resp:
            if raw.startswith(b"data: ") and b"[DONE]" not in raw:
                n += 1
                if n >= 2:
                    break
        resp.close()  # disconnect mid-stream

        again = chat(QUESTION)["choices"][0]["message"]["content"]
        assert again == baseline, (
            "a mid-stream disconnect changed a later answer — the engine was still "
            f"running when the next request started\n  before: {baseline!r}\n  after: {again!r}"
        )

    assert get("/health")["status"] == "ok"


def test_queue_rejects_with_429_instead_of_hanging():
    max_queue = get("/metrics")["max_queue"]
    codes: list[int] = []
    lock = threading.Lock()

    def hammer():
        try:
            with _post({"messages": [{"role": "user", "content": FACTS}],
                        "max_tokens": 64}) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception:
            code = -1
        with lock:
            codes.append(code)

    ts = [threading.Thread(target=hammer) for _ in range(max_queue * 4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert 429 in codes, "admission control never kicked in"
    assert 200 in codes, "everything was rejected"
    assert codes.count(200) <= max_queue + 1


def test_metrics_reports_capabilities():
    m = get("/metrics")
    assert "supports_prefix_cache" in m and "supports_cancel" in m
    if m["supports_prefix_cache"]:
        assert "hit_rate" in m["cache"]
        assert m["cache"]["bytes"] <= m["cache"]["max_bytes"]
    else:
        assert m["cache"] == {"enabled": False}
