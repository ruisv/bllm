#!/usr/bin/env python3
"""End-to-end: tool calls and reasoning_content through a running bllm-serve.

Needs a server on a model whose manifest declares a chat dialect (an official Qwen3 /
Qwen2.5 package, or a Qwen3.5 package built with one); skipped otherwise:

    BLLM_MODEL=~/llm_s100/llm2/Qwen3-1.7B_w4 uvicorn server:app --port 8877 &
    BLLM_SERVE_URL=http://127.0.0.1:8877 pytest tests/test_serve_tools_e2e.py -v

Greedy throughout, and the prompts leave the model one sensible move, so a failure is
the serving layer's, not a coin flip — but these are still a model following a tool
schema, so a very small model may need a nudge in the prompt.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("BLLM_SERVE_URL")
if not BASE:
    pytest.skip("BLLM_SERVE_URL not set — needs a running server", allow_module_level=True)


def _post(body, timeout=300):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _stream(body, timeout=300):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return chunks


def _metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        return json.load(r)


M = _metrics()
needs_tools = pytest.mark.skipif(not M.get("supports_tools"), reason="this model has no tool format")
needs_reasoning = pytest.mark.skipif(not M.get("supports_reasoning_split"),
                                     reason="this model does not think")

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}
ASK = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]


def chat(messages, **kw):
    return _post({"messages": messages, "max_tokens": kw.pop("max_tokens", 256), "temperature": 0.0,
                  **kw})


@needs_tools
def test_the_model_calls_the_offered_tool():
    status, body = chat(ASK, tools=[WEATHER])
    assert status == 200, body
    ch = body["choices"][0]
    assert ch["finish_reason"] == "tool_calls", ch
    call = ch["message"]["tool_calls"][0]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    assert "paris" in json.loads(call["function"]["arguments"])["city"].lower()
    assert "<tool_call>" not in (ch["message"]["content"] or "")


@needs_tools
def test_a_tool_result_comes_back_into_the_answer():
    status, body = chat(ASK, tools=[WEATHER])
    call = body["choices"][0]["message"]["tool_calls"][0]
    follow = ASK + [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"temp_c": 17, "sky": "rain"})},
    ]
    status, body = chat(follow, tools=[WEATHER])
    assert status == 200, body
    ch = body["choices"][0]
    assert ch["finish_reason"] in ("stop", "length") and "tool_calls" not in ch["message"]
    assert "17" in ch["message"]["content"]


@needs_tools
def test_streaming_delivers_the_same_call():
    _, whole = chat(ASK, tools=[WEATHER])
    want = whole["choices"][0]["message"]["tool_calls"][0]["function"]
    chunks = _stream({"messages": ASK, "max_tokens": 256, "temperature": 0.0, "tools": [WEATHER]})
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
    assert len(calls) == 1 and calls[0]["index"] == 0 and calls[0]["function"] == want
    assert not any("<tool_call>" in (d.get("content") or "") for d in deltas)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@needs_tools
def test_tool_choice_none_offers_no_tools():
    status, body = chat(ASK, tools=[WEATHER], tool_choice="none", max_tokens=48)
    assert status == 200 and "tool_calls" not in body["choices"][0]["message"]


@needs_tools
@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "get_weather"}}])
def test_forcing_a_call_is_refused_not_ignored(choice):
    status, body = chat(ASK, tools=[WEATHER], tool_choice=choice)
    assert status == 400 and "tool_choice" in json.dumps(body)


@needs_tools
def test_malformed_tools_are_a_400():
    status, _ = chat(ASK, tools=[{"type": "function"}])
    assert status == 400


@needs_reasoning
def test_thinking_comes_back_as_reasoning_content():
    status, body = chat([{"role": "user", "content": "What is 17 * 3? Answer with the number."}],
                        max_tokens=1024, chat_template_kwargs={"enable_thinking": True})
    assert status == 200, body
    msg = body["choices"][0]["message"]
    assert msg.get("reasoning_content"), msg
    assert "<think>" not in msg["content"] and "</think>" not in msg["content"]
    assert "51" in msg["content"]


@needs_reasoning
def test_streamed_reasoning_and_content_are_separate_deltas():
    chunks = _stream({"messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number."}],
                      "max_tokens": 512, "temperature": 0.0,
                      "chat_template_kwargs": {"enable_thinking": True}})
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert reasoning and "4" in content and "</think>" not in content + reasoning
    # reasoning first, then content: no reasoning delta after the first content one
    kinds = [("r" if "reasoning_content" in d else "c") for d in deltas
             if "reasoning_content" in d or d.get("content")]
    assert "r" not in kinds[kinds.index("c"):] if "c" in kinds else True


@pytest.mark.skipif(M.get("thinking") == "always", reason="a thinking-only model always reasons")
def test_thinking_off_has_no_reasoning_key():
    status, body = chat([{"role": "user", "content": "Say hi."}], max_tokens=16)
    assert status == 200 and "reasoning_content" not in body["choices"][0]["message"]
