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

# The server keeps its cache across tests, so a prompt sent by an earlier test would
# legitimately hit its own entry. Anything asserting on a MISS must be unique per run.
NONCE = os.urandom(6).hex()

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


def test_logprobs_come_back_in_openai_shape():
    r = chat(QUESTION, logprobs=True, top_logprobs=3)
    content = r["choices"][0]["logprobs"]["content"]
    assert content, "logprobs=true returned no content array"
    for e in content:
        assert isinstance(e["token"], str) and e["token"]
        assert e["bytes"] == list(e["token"].encode())
        assert e["logprob"] <= 0.0
        assert len(e["top_logprobs"]) == 3
    # The array describes exactly the reply that was returned — checked on the BYTES,
    # which is what `bytes` is for: a token can hold part of a multi-byte character, and
    # its `token` string is then U+FFFD while its bytes are still exact.
    rebuilt = bytes(b for e in content for b in e["bytes"])
    assert rebuilt.decode("utf-8") == r["choices"][0]["message"]["content"]
    # And it is absent, not null or empty, when nobody asked.
    assert "logprobs" not in chat(QUESTION)["choices"][0]


@pytest.mark.skipif(not _CAPS.get("supports_grammar"),
                    reason="token_bytes landed with the same runtime as grammars")
def test_logprobs_bytes_are_exact_for_chinese():
    """Chinese is where `token.encode()` and the token's real bytes diverge."""
    r = chat([{"role": "user", "content": "用一句话介绍北京。"}], logprobs=True, max_tokens=32)
    content = r["choices"][0]["logprobs"]["content"]
    rebuilt = bytes(b for e in content for b in e["bytes"])
    assert rebuilt.decode("utf-8") == r["choices"][0]["message"]["content"]


def test_bad_logprobs_request_is_a_400():
    """top_logprobs without logprobs=true would cost two full-vocab passes per token
    for a field the response never carries."""
    with pytest.raises(urllib.error.HTTPError) as e:
        chat(QUESTION, top_logprobs=3)
    assert e.value.code == 400


needs_grammar = pytest.mark.skipif(
    not _CAPS.get("supports_grammar"),
    reason="this bllm build has no grammar support")


@needs_grammar
def test_response_format_json_object_parses():
    r = chat([{"role": "user", "content": "Describe Paris briefly."}],
             response_format={"type": "json_object"}, max_tokens=64)
    json.loads(r["choices"][0]["message"]["content"])   # the whole guarantee, in one call


@needs_grammar
def test_response_format_json_schema_matches_the_schema():
    schema = {"type": "object",
              "properties": {"city": {"type": "string"}, "is_capital": {"type": "boolean"}},
              "required": ["city", "is_capital"]}
    r = chat([{"role": "user", "content": "Describe Paris."}], max_tokens=64,
             response_format={"type": "json_schema",
                              "json_schema": {"name": "city", "schema": schema}})
    obj = json.loads(r["choices"][0]["message"]["content"])
    assert set(obj) == {"city", "is_capital"}
    assert isinstance(obj["is_capital"], bool)


@needs_grammar
def test_raw_gbnf_grammar_field():
    r = chat([{"role": "user", "content": "Is the sky blue? Answer in English."}],
             grammar='root ::= "yes" | "no"\n', max_tokens=16)
    assert r["choices"][0]["message"]["content"] in ("yes", "no")


@needs_grammar
def test_a_grammar_does_not_leak_into_the_next_request():
    """The session keeps a grammar until told otherwise, and requests share one session."""
    chat(QUESTION, grammar='root ::= "yes" | "no"\n', max_tokens=8)
    assert "pineapple" in chat(QUESTION, max_tokens=16)["choices"][0]["message"]["content"].lower()


@needs_grammar
@pytest.mark.parametrize("grammar", [
    "root ::= root\n",                          # direct left recursion
    "root ::= a\na ::= root\n",                 # indirect
    'root ::= "a" | root\n',                    # via an alternate
    "root ::= x*\nx ::= \"a\"?\n",              # a nullable body under *
])
def test_a_left_recursive_grammar_cannot_kill_the_server(grammar):
    """This was a remote SIGSEGV: a rule that reaches itself without consuming input
    expanded forever inside the matcher and took the whole process — model included — down
    with it. `grammar` is a request field, so any client could send it."""
    with pytest.raises(urllib.error.HTTPError) as e:
        chat(QUESTION, grammar=grammar, max_tokens=8)
    assert e.value.code == 400
    assert get("/health")["status"] == "ok"     # still serving, which is the whole point


@needs_grammar
def test_a_self_referential_schema_is_a_400_not_a_crash():
    """The same hazard through the schema door: `A -> A` converts to a grammar rule that
    references only itself."""
    schema = {"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}}
    with pytest.raises(urllib.error.HTTPError) as e:
        chat(QUESTION, response_format={"type": "json_schema",
                                        "json_schema": {"name": "a", "schema": schema}})
    assert e.value.code == 400
    assert get("/health")["status"] == "ok"


@needs_grammar
@pytest.mark.parametrize("body,why", [
    ({"grammar": "root ::= undefined_thing\n"}, "a grammar that does not parse"),
    ({"response_format": {"type": "json_schema", "json_schema": {}}}, "no schema"),
    ({"response_format": {"type": "xml"}}, "an unsupported format"),
    ({"response_format": {"type": "json_object"}, "grammar": 'root ::= "a"\n'}, "both at once"),
])
def test_bad_structured_output_requests_are_400s(body, why):
    with pytest.raises(urllib.error.HTTPError) as e:
        chat(QUESTION, **body)
    assert e.value.code == 400, why


def test_logit_bias_is_validated_not_clamped():
    """A value off OpenAI's [-100, 100] means the client is working from a different
    scale; reinterpreting it silently would hide that. (Whether a bias actually moves the
    sampler is covered where token ids are reachable: tests/test_native_llm.py and
    tests/test_native_sampler.cc — the HTTP API publishes no token ids.)"""
    with pytest.raises(urllib.error.HTTPError) as e:
        chat(QUESTION, logit_bias={"1234": 1000})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        chat(QUESTION, logit_bias={"not-a-token": 1})
    assert e.value.code == 400


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
    """A prompt that diverges at the very start must not claim another one's prefix.

    Both prompts carry a per-run nonce so neither can match an entry left by an
    earlier test — otherwise a legitimate self-hit looks like a cross-conversation
    leak. The threshold is in TOKENS: only the few tokens of ChatML preamble the two
    genuinely share may be reused.
    """
    chat([{"role": "user", "content": f"[{NONCE}-a] " + FACTS}])
    b = chat([{"role": "user", "content": f"[{NONCE}-b] different opening. " + FACTS}])
    assert b["usage"]["prompt_tokens_details"]["cached_tokens"] < 16


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
