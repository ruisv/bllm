"""bllm._chatfmt against the templates it stands in for (ADR 0003).

Every rendering is compared byte for byte with jinja2 rendering the model's own HF chat
template (tests/fixtures/chat_templates/, identical to what the official LLM 2.0 packages
embed), in an environment set up as transformers sets it up. The dialect and the think
tails come from the same template through bllm._modelmeta, so the detection is tested on
the way. Pure Python: runs anywhere jinja2 is installed.
"""
import json
import pathlib
import random

import pytest

jinja2 = pytest.importorskip("jinja2")
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402

from bllm import _chatfmt as cf  # noqa: E402
from bllm import _modelmeta as mm  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "chat_templates"
TEMPLATES = ["qwen3", "qwen3-thinking-2507", "qwen3-instruct-2507", "qwen2.5", "qwen3-vl", "qwen3.5"]
EXPECTED_DIALECT = {"qwen3": "qwen3", "qwen3-thinking-2507": "qwen3", "qwen3-instruct-2507": "qwen3-instruct",
                    "qwen2.5": "qwen2.5", "qwen3-vl": "qwen3-instruct", "qwen3.5": "qwen3.5"}


def _env():
    # transformers' chat-template environment: trim/lstrip blocks, loop controls, its own
    # `tojson` (ensure_ascii off) and raise_exception.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


ENV = _env()


def _template(name):
    return (FIXTURES / f"{name}.jinja").read_text(encoding="utf-8")


def _format(name):
    """The ChatFormat bllm would build for a package carrying this template."""
    t = _template(name)
    chat = dict(mm._desc_thinking(t))
    m = mm._CHATML_DEFAULT_SYSTEM.search(t)
    chat["system"] = m.group(1) if m else ""
    chat["dialect"] = mm._desc_dialect(t)
    return cf.ChatFormat.from_config(chat)


def _hf(name, messages, tools=None, enable_thinking=False):
    kw = {"messages": messages, "add_generation_prompt": True, "enable_thinking": enable_thinking}
    if tools is not None:
        kw["tools"] = tools
    return ENV.from_string(_template(name)).render(**kw)


def _ours(name, messages, tools=None, enable_thinking=False):
    history, opener = _format(name).render(messages, tools=tools, enable_thinking=enable_thinking)
    return history + opener


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "查询城市天气 (weather by city)",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名"},
        "days": {"type": "integer"}, "metric": {"type": "boolean"}},
        "required": ["city"]}}}
CLOCK = {"type": "function", "function": {"name": "now", "description": "Current time.",
                                          "parameters": {"type": "object", "properties": {}}}}

CASES = {
    "one user": ([{"role": "user", "content": "你好"}], None),
    "system + user": ([{"role": "system", "content": "Be brief."},
                       {"role": "user", "content": "Hi"}], None),
    "empty system": ([{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}], None),
    "multi-turn": ([{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"},
                    {"role": "user", "content": "And 2+2?"}], None),
    "thinking kept inline in history": (
        [{"role": "user", "content": "Q1"},
         {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nA1"},
         {"role": "user", "content": "Q2"}], None),
    "padded contents": ([{"role": "system", "content": "  sys \n"},
                         {"role": "user", "content": "\n hi  "}], None),
    "tools, no system": ([{"role": "user", "content": "北京天气？"}], [WEATHER]),
    "tools + system": ([{"role": "system", "content": "You are a weather bot."},
                        {"role": "user", "content": "Weather in Paris?"}], [WEATHER, CLOCK]),
    "tools + empty system": ([{"role": "system", "content": ""},
                              {"role": "user", "content": "Time?"}], [CLOCK]),
    "tool loop": ([{"role": "user", "content": "北京和上海天气？"},
                   {"role": "assistant", "content": "", "tool_calls": [
                       {"id": "c1", "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "北京", "days": 2}}},
                       {"id": "c2", "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "上海", "metric": True}}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "晴，25°C"},
                   {"role": "tool", "tool_call_id": "c2", "content": "{\"sky\": \"rain\"}"}],
                  [WEATHER]),
    "tool loop with text and reasoning": (
        [{"role": "user", "content": "What time is it?"},
         {"role": "assistant", "content": "Let me check.", "reasoning_content": "Need the clock.",
          "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "now", "arguments": {}}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "12:00"}], [CLOCK]),
    "finished tool turn, next question": (
        [{"role": "user", "content": "Time?"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"type": "function", "function": {"name": "now", "arguments": {}}}]},
         {"role": "tool", "content": "12:00"},
         {"role": "assistant", "content": "It is noon."},
         {"role": "user", "content": "Thanks! And tomorrow's weather in 杭州?"}], [CLOCK, WEATHER]),
    "nested argument values": (
        [{"role": "user", "content": "go"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"type": "function", "function": {"name": "get_weather", "arguments": {
                 "city": "Rome", "opts": {"units": "C", "hours": [1, 2]}, "n": 1.5}}}]},
         {"role": "tool", "content": "ok"}], [WEATHER]),
}


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_template_is_recognised(name):
    assert mm._desc_dialect(_template(name)) == EXPECTED_DIALECT[name]


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("name", TEMPLATES)
def test_rendering_matches_the_template(name, case, thinking):
    messages, tools = CASES[case]
    try:
        want = _hf(name, json.loads(json.dumps(messages)), tools, thinking)
    except jinja2.exceptions.TemplateError:
        with pytest.raises(cf.ChatError):
            _ours(name, messages, tools, thinking)
        return
    assert _ours(name, messages, tools, thinking) == want


def test_arguments_sent_as_a_json_string_render_as_the_object():
    # OpenAI clients send arguments as a JSON string; vLLM parses it before the template.
    as_obj = CASES["tool loop"][0]
    as_str = json.loads(json.dumps(as_obj))
    for tc in as_str[1]["tool_calls"]:
        tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"], ensure_ascii=False)
    for name in ("qwen3", "qwen3.5"):
        assert _ours(name, as_str, [WEATHER]) == _ours(name, as_obj, [WEATHER])


def test_text_parts_render_as_their_text():
    parts = [{"role": "user", "content": [{"type": "text", "text": "He"}, {"type": "text", "text": "llo"}]}]
    assert _ours("qwen3", parts) == _ours("qwen3", [{"role": "user", "content": "Hello"}])


def test_a_model_without_a_dialect_refuses_tools_and_renders_plain_chatml():
    f = cf.ChatFormat()
    with pytest.raises(cf.ChatError, match="tool"):
        f.render([{"role": "user", "content": "hi"}], tools=[CLOCK])
    h, o = f.render([{"role": "user", "content": "hi"}])
    assert h == "<|im_start|>user\nhi<|im_end|>\n" and o == "<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ── replies ──────────────────────────────────────────────────────────────────────────────

def _parse(name, text, tools=None, thinking=False):
    return _format(name).parse_reply(text, enable_thinking=thinking, tools=tools)


def test_thinking_is_split_out_of_the_content():
    m = _parse("qwen3", "<think>\nLet me add.\n</think>\n\n2", thinking=True)
    assert m["reasoning_content"] == "Let me add." and m["content"] == "2"


def test_a_prompt_opened_think_block_is_reasoning_up_to_its_close():
    # Thinking-2507 opens <think> in the prompt; the reply starts inside it.
    m = _parse("qwen3-thinking-2507", "step one\nstep two\n</think>\n\nDone.")
    assert m["reasoning_content"] == "step one\nstep two" and m["content"] == "Done."


def test_a_reply_cut_off_while_thinking_is_all_reasoning():
    m = _parse("qwen3", "<think>\nstill going", thinking=True)
    assert m["reasoning_content"] == "still going" and m["content"] == ""


def test_no_reasoning_key_when_the_model_did_not_think():
    assert "reasoning_content" not in _parse("qwen3", "Paris.")
    assert "reasoning_content" not in _parse("qwen2.5", "<think>x</think>y")   # cannot think


def test_hermes_tool_calls_are_parsed():
    text = ('I will check both.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}\n'
            '</tool_call>\n<tool_call>\n{"name": "now", "arguments": {}}\n</tool_call>')
    m = _parse("qwen3", text, [WEATHER, CLOCK])
    assert m["content"] == "I will check both."
    assert [c["function"]["name"] for c in m["tool_calls"]] == ["get_weather", "now"]
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"city": "北京"}
    assert all(c["id"].startswith("call_") and c["type"] == "function" for c in m["tool_calls"])


def test_a_tool_call_only_reply_has_null_content():
    m = _parse("qwen3-instruct-2507", '<tool_call>\n{"name": "now", "arguments": {}}\n</tool_call>', [CLOCK])
    assert m["content"] is None and len(m["tool_calls"]) == 1


def test_xml_tool_calls_are_parsed_with_the_schema_types():
    text = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nSan Francisco\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n"
            "</function>\n</tool_call>")
    m = _parse("qwen3.5", text, [WEATHER])
    args = json.loads(m["tool_calls"][0]["function"]["arguments"])
    assert args == {"city": "San Francisco", "days": 3, "metric": True} and m["content"] is None


def test_without_tools_on_offer_a_tool_call_is_just_text():
    text = '<tool_call>\n{"name": "now", "arguments": {}}\n</tool_call>'
    m = _parse("qwen3", text)
    assert "tool_calls" not in m and m["content"] == text


@pytest.mark.parametrize("text", [
    '<tool_call>\n{"name": "now", "arguments": {}',                      # cut off mid-call
    '<tool_call>\n{not json}\n</tool_call>',                            # malformed
])
def test_a_broken_tool_call_comes_back_as_content(text):
    m = _parse("qwen3", text, [CLOCK])
    assert "tool_calls" not in m and m["content"] == text


REPLIES = [
    ("qwen3", True, None, "<think>\nA\n\nB\n</think>\n\nThe answer\nis 4.\n"),
    ("qwen3-thinking-2507", False, None, "x</think>\n\ny"),
    ("qwen3", False, [WEATHER], 'Sure.\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "R"}}\n</tool_call>'),
    ("qwen3.5", True, [WEATHER], "plan\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nR\n"
                                 "</parameter>\n</function>\n</tool_call>"),
    ("qwen3", False, [CLOCK], "No tool needed: <tool_ca is not a tag"),
]


@pytest.mark.parametrize("name,thinking,tools,text", REPLIES)
def test_streaming_matches_the_whole_reply_at_any_split(name, thinking, tools, text):
    fmt = _format(name)
    whole = fmt.parse_reply(text, enable_thinking=thinking, tools=tools)
    rng = random.Random(len(text))
    for _ in range(40):
        cuts = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(1, 12))))
        pieces = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
        rs = fmt.reply_stream(enable_thinking=thinking, tools=tools)
        parts = [p for piece in pieces for p in rs.feed(piece)] + rs.finish()
        assert "".join(t for k, t in parts if k == "content") == (whole["content"] or "")
        assert "".join(t for k, t in parts if k == "reasoning") == whole.get("reasoning_content", "")
        assert [c["function"] for c in rs.tool_calls] == [c["function"] for c in whole.get("tool_calls", [])]
