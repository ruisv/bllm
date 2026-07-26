#!/usr/bin/env python3
"""End-to-end tests against a running bllm-serve loaded with a VLM (image+text) model.

Separate from test_serve_e2e.py because it needs a VLM model directory and its own
server instance — a text-only server 400s on image_url (covered here too, against a
second, plain-text server). Skipped unless BLLM_SERVE_VLM_URL points at a live VLM
server:

    BLLM_MODEL=~/models/qwen3.5-0.8b-vlm320-ctx2k-pf32f uvicorn server:app --port 8877 &
    BLLM_SERVE_VLM_URL=http://127.0.0.1:8877 \
    BLLM_TEST_IMAGE=~/qwen3_deploy/bears448.png \
    pytest tests/test_serve_e2e_vlm.py -v

Optionally BLLM_SERVE_TEXT_URL for the "image on a text-only model" 400 check:

    BLLM_MODEL=~/models/qwen35-0.8b uvicorn server:app --port 8878 &
    BLLM_SERVE_TEXT_URL=http://127.0.0.1:8878 pytest tests/test_serve_e2e_vlm.py -v

What matters here is not "does chat() work over HTTP" (that's the VLM session's own
job, covered by tests/test_hybrid_vlm.py) but the serving-layer-specific behavior:
image_url decoding (base64 + http), routing on is_vlm, and the turn-replay mechanism
in Engine._serve_vlm not leaking context between unrelated conversations — a real bug
caught by exactly this kind of test during development (see engine.py's _vlm_committed
comment). Because the server is a single live session, tests run in a fixed order and
each depends on the previous one's state — see the module-level note before the tests.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

import pytest

VLM_BASE = os.environ.get("BLLM_SERVE_VLM_URL")
TEXT_BASE = os.environ.get("BLLM_SERVE_TEXT_URL")
IMAGE = os.environ.get("BLLM_TEST_IMAGE")

if not VLM_BASE:
    pytest.skip("BLLM_SERVE_VLM_URL not set — needs a running VLM server", allow_module_level=True)
if not IMAGE:
    pytest.skip("BLLM_TEST_IMAGE not set", allow_module_level=True)


def _post(base, body, timeout=120):
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def chat(messages, base=VLM_BASE, **kw):
    return _post(base, {"messages": messages, "max_tokens": 48, "temperature": 0.0, **kw})


def get(path, base=VLM_BASE):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.load(r)


def _data_url() -> str:
    with open(IMAGE, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(IMAGE)[1].lstrip(".") or "png"
    return f"data:image/{ext};base64,{b64}"


IMG_PART = {"type": "image_url", "image_url": {"url": _data_url()}}


def test_health_reports_is_vlm():
    assert get("/health")["status"] == "ok"
    assert get("/metrics")["is_vlm"] is True


def test_plain_text_works_on_a_vlm_loaded_model():
    """Before this feature, a VLM-loaded server 500'd on ANY request — NativeVlmSession
    has no encode/feed_ids/generate, which is what the text-only legacy path needs."""
    status, body = chat([{"role": "user", "content": "法国的首都是哪里？只答城市名。"}])
    assert status == 200
    assert "巴黎" in body["choices"][0]["message"]["content"]


def test_image_url_base64_answers_correctly():
    status, body = chat([{"role": "user", "content": [
        {"type": "text", "text": "这张图里有什么？一句话回答。"}, IMG_PART]}])
    assert status == 200
    text = body["choices"][0]["message"]["content"]
    assert text.strip()
    u = body["usage"]
    assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_unrelated_single_turn_conversation_does_not_leak_context():
    """The bug this test exists to catch: two back-to-back single-turn requests (no
    history on either) both look like "history == []" to a naive turn-diff, so a cache
    that only tracks what to REPLAY (and forgets to record what it just ANSWERED) would
    let the second request's reply silently ride on the first request's leftover
    context. Both turns below have empty history; the second must start clean —
    checked the only way an HTTP client can: cached_tokens must be 0 (nothing carried
    over) rather than reflecting the previous unrelated turn."""
    status, body = chat([{"role": "user", "content": "1+1等于几？只答数字。"}])
    assert status == 200
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_multi_turn_follow_up_reuses_committed_history():
    """A real multi-turn conversation about the same image: the follow-up's history
    (the first Q + the FIRST reply this same server produced) must hit the turn-replay
    fast path — cached_tokens should equal the total tokens of the prior turn, meaning
    append_turn() was NOT needed and the vision tower did not re-run."""
    first_turn = [{"role": "user", "content": [
        {"type": "text", "text": "这张图里有什么？一句话回答。"}, IMG_PART]}]
    status, r1 = chat(first_turn)
    assert status == 200
    reply1 = r1["choices"][0]["message"]["content"]

    status, r2 = chat(first_turn + [
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": "图里是什么颜色？一个词回答。"},
    ])
    assert status == 200
    assert "棕" in r2["choices"][0]["message"]["content"] or "brown" in r2["choices"][0]["message"]["content"].lower()
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == r1["usage"]["total_tokens"]


def test_streaming_matches_non_streaming_shape():
    status, r1 = chat([{"role": "user", "content": [
        {"type": "text", "text": "这张图里有什么？一句话回答。"}, IMG_PART]}])
    assert status == 200

    req = urllib.request.Request(
        VLM_BASE + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "这张图里有什么？一句话回答。"}, IMG_PART]}],
            "max_tokens": 48, "temperature": 0.0, "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    pieces, finish_reason, usage = [], None, None
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = json.loads(line[len("data: "):])
            choice = obj["choices"][0]
            if "content" in choice["delta"]:
                pieces.append(choice["delta"]["content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
                usage = obj.get("usage")
    assert finish_reason == "stop"
    assert usage is not None and usage["completion_tokens"] > 0
    assert "".join(pieces) == r1["choices"][0]["message"]["content"]


def test_image_url_on_a_message_that_is_not_the_last_user_turn_is_rejected():
    status, body = chat([
        {"role": "user", "content": [{"type": "text", "text": "x"}, IMG_PART]},
        {"role": "assistant", "content": [{"type": "text", "text": "x"}, IMG_PART]},
    ])
    assert status == 400


def test_non_image_content_part_type_is_rejected():
    status, body = chat([{"role": "user", "content": [
        {"type": "text", "text": "x"}, {"type": "input_audio", "input_audio": {"data": "x"}}]}])
    assert status == 400


needs_text_server = pytest.mark.skipif(not TEXT_BASE, reason="set BLLM_SERVE_TEXT_URL")


@needs_text_server
def test_image_on_a_text_only_model_is_a_clear_400_not_a_500():
    status, body = chat([{"role": "user", "content": [
        {"type": "text", "text": "hi"}, IMG_PART]}], base=TEXT_BASE)
    assert status == 400
    assert "vision tower" in body["detail"]


@needs_text_server
def test_text_only_model_still_answers_plain_text():
    status, body = chat([{"role": "user", "content": "法国的首都是哪里？只答城市名。"}], base=TEXT_BASE)
    assert status == 200
    assert "巴黎" in body["choices"][0]["message"]["content"]
