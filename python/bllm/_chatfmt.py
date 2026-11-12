"""Conversations in, prompt text out; reply text in, OpenAI message out — per template family.

A server (or any caller driving a session with `feed_ids`) has to turn an OpenAI
``messages[]`` array into the exact prompt the model was trained on, and turn the reply
back into ``content`` / ``reasoning_content`` / ``tool_calls``. The checkpoints' Jinja
templates decide both, and they differ in more than their tool syntax (ADR 0003):

  * ``qwen2.5``         a default system prompt when none is given, tools or not;
  * ``qwen3``           (Qwen3, Qwen3-Thinking-2507) the assistant's thinking is kept in the
                        history only after the last user question, and dropped before it;
  * ``qwen3-instruct``  (Qwen3-Instruct-2507, Qwen3-VL) the same without any thinking;
  * ``qwen3.5``         XML tool calls, the system prompt after the tool instructions, every
                        message stripped, and a think block on every assistant turn after
                        the last question.

The first three call tools in the Hermes format (``<tool_call>{json}</tool_call>``).
Without a dialect a model gets plain ChatML: no tools, no reasoning split — what the
server always did. No Jinja at runtime: tests/test_chatfmt.py renders the original HF
templates with jinja2 and requires these renderings to match them byte for byte.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Optional

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
DIALECTS = ("qwen2.5", "qwen3", "qwen3-instruct", "qwen3.5")
_CLOSED_THINK = "<think>\n\n</think>\n\n"

_HERMES_HEAD = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>")
_HERMES_TAIL = ("\n</tools>\n\nFor each function call, return a json object with function name "
                "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
_XML_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_XML_TAIL = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is "
    "no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>")


class ChatError(ValueError):
    """A conversation this format cannot render — the caller's error (an HTTP 400)."""


def _tojson(x) -> str:
    # HF's `tojson` filter: json.dumps with ensure_ascii off and default separators.
    return json.dumps(x, ensure_ascii=False)


def _text(content) -> str:
    """OpenAI content (a string, None, or text parts) as one string. Images never reach
    here: the text path refuses them and the VLM path renders its own turns."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict) and p.get("type", "text") == "text":
                out.append(p.get("text", "") or "")
            else:
                raise ChatError("only text content parts can be rendered here")
        return "".join(out)
    raise ChatError("message content must be a string, null, or an array of parts")


def _calls(msg: dict) -> list:
    """An assistant message's tool calls as [(name, arguments-dict)]. OpenAI sends the
    arguments as a JSON string; the templates want the object (as vLLM passes it)."""
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc) if isinstance(tc, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise ChatError("a tool call needs function.name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as e:
                raise ChatError(f"tool call {fn['name']!r}: arguments are not JSON ({e})")
        if not isinstance(args, dict):
            raise ChatError(f"tool call {fn['name']!r}: arguments must be a JSON object")
        out.append((fn["name"], args))
    return out


def _split_think(content: str) -> tuple[str, str]:
    """The templates' own split of an assistant message that carries its thinking inline."""
    if THINK_CLOSE not in content:
        return "", content
    reasoning = content.split(THINK_CLOSE)[0].rstrip("\n").split(THINK_OPEN)[-1].lstrip("\n")
    return reasoning, content.split(THINK_CLOSE)[-1].lstrip("\n")


@dataclass
class ChatFormat:
    dialect: str = ""                 # "" = plain ChatML
    system: str = ""                  # default system prompt (the template's own)
    think_on: str = ""                # generation prompt tail with thinking on / off
    think_off: str = _CLOSED_THINK

    @classmethod
    def from_config(cls, chat: dict) -> "ChatFormat":
        """From a session's `chat_config` (model.json's chat block)."""
        d = chat.get("dialect") or ""
        if d and d not in DIALECTS:
            raise ChatError(f"unknown chat dialect {d!r}")
        return cls(dialect=d, system=chat.get("system", "") or "",
                   think_on=chat.get("think_on", ""), think_off=chat.get("think_off", _CLOSED_THINK))

    # ── what the format can do ────────────────────────────────────────────────────────
    @property
    def tool_format(self) -> str:
        """"hermes", "xml", or "" when this model has no tool-calling format."""
        return "xml" if self.dialect == "qwen3.5" else ("hermes" if self.dialect else "")

    @property
    def thinks(self) -> bool:
        """Whether replies may carry a <think> block to split into reasoning_content."""
        return self.dialect in ("qwen3", "qwen3.5")

    def opener(self, enable_thinking: bool) -> str:
        """The generation prompt: the assistant marker plus the template's think tail."""
        if self.dialect == "qwen3.5":          # the template's own, whatever the manifest says
            tail = "<think>\n" if enable_thinking else _CLOSED_THINK
        else:
            tail = self.think_on if enable_thinking else self.think_off
        return IM_START + "assistant\n" + tail

    # ── rendering ─────────────────────────────────────────────────────────────────────
    def render(self, messages: list, *, tools: Optional[list] = None,
               enable_thinking: bool = False) -> tuple[str, str]:
        """(history, opener). Only `history` is stable across turns — next turn the opener's
        position holds the reply — so only it is cacheable."""
        if not isinstance(messages, list) or not messages:
            raise ChatError("messages[] required")
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
                raise ChatError("each message needs a role of system, user, assistant or tool")
        # Content as one string up front, as vLLM hands it to a template: the templates
        # test `content is string`, and a text-parts array would otherwise render empty.
        messages = [dict(m, content=_text(m.get("content"))) for m in messages]
        if tools:
            if not self.tool_format:
                raise ChatError("this model declares no tool-calling format (chat.dialect), so "
                                "it cannot be given tools")
            if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
                raise ChatError("tools must be an array of objects")
        if not self.dialect:
            history = self._render_plain(messages)
        elif self.dialect == "qwen3.5":
            history = self._render_qwen35(messages, tools)
        else:
            history = self._render_hermes(messages, tools)
        return history, self.opener(enable_thinking)

    def _render_plain(self, messages: list) -> str:
        # The server's long-standing rendering, unchanged: no default system, roles as given.
        return "".join(f"{IM_START}{m['role']}\n{_text(m.get('content'))}{IM_END}\n" for m in messages)

    def _system(self, messages: list) -> tuple[Optional[str], int]:
        if messages[0]["role"] == "system":
            return _text(messages[0].get("content")), 1
        return (self.system or None), 0

    def _render_hermes(self, messages: list, tools: Optional[list]) -> str:
        d = self.dialect
        out = []
        sys_text, _ = self._system(messages)
        if tools:
            out.append(IM_START + "system\n")
            if messages[0]["role"] == "system" or sys_text:   # an empty one still gets "\n\n"
                out.append(sys_text + "\n\n")
            out.append(_HERMES_HEAD + "".join("\n" + _tojson(t) for t in tools) + _HERMES_TAIL
                       + IM_END + "\n")
        elif sys_text is not None and (messages[0]["role"] == "system" or sys_text):
            out.append(IM_START + "system\n" + sys_text + IM_END + "\n")

        last_query = len(messages) - 1
        if d == "qwen3":
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                c = m.get("content")
                if m["role"] == "user" and isinstance(c, str) and not (
                        c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                    last_query = i
                    break

        n = len(messages)
        for i, m in enumerate(messages):
            role = m["role"]
            content = _text(m.get("content"))
            if role == "user" or (role == "system" and i > 0):
                out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
            elif role == "assistant":
                calls = _calls(m)
                if d == "qwen2.5":
                    if not calls:
                        out.append(f"{IM_START}assistant\n{content}{IM_END}\n")
                        continue
                    out.append(IM_START + "assistant" + ("\n" + content if content else ""))
                    for name, args in calls:
                        out.append('\n<tool_call>\n{"name": "' + name + '", "arguments": '
                                   + _tojson(args) + "}\n</tool_call>")
                    out.append(IM_END + "\n")
                    continue
                if d == "qwen3":
                    rc = m.get("reasoning_content")
                    if isinstance(rc, str):
                        reasoning = rc
                    else:
                        reasoning, content = _split_think(content)
                    if i > last_query and (i == n - 1 or reasoning):
                        out.append(IM_START + "assistant\n<think>\n" + reasoning.strip("\n")
                                   + "\n</think>\n\n" + content.lstrip("\n"))
                    else:
                        out.append(IM_START + "assistant\n" + content)
                else:                                     # qwen3-instruct
                    out.append(IM_START + "assistant\n" + content)
                for j, (name, args) in enumerate(calls):
                    if (j == 0 and content) or j > 0:
                        out.append("\n")
                    out.append('<tool_call>\n{"name": "' + name + '", "arguments": '
                               + _tojson(args) + "}\n</tool_call>")
                out.append(IM_END + "\n")
            elif role == "tool":
                if i == 0 or messages[i - 1]["role"] != "tool":
                    out.append(IM_START + "user")
                out.append("\n<tool_response>\n" + content + "\n</tool_response>")
                if i == n - 1 or messages[i + 1]["role"] != "tool":
                    out.append(IM_END + "\n")
        return "".join(out)

    def _render_qwen35(self, messages: list, tools: Optional[list]) -> str:
        out = []
        first_system = messages[0]["role"] == "system"
        sys_text = _text(messages[0].get("content")).strip() if first_system else (self.system or "").strip()
        if tools:
            out.append(IM_START + "system\n" + _XML_HEAD
                       + "".join("\n" + _tojson(t) for t in tools) + _XML_TAIL)
            if sys_text:
                out.append("\n\n" + sys_text)
            out.append(IM_END + "\n")
        elif first_system or sys_text:
            out.append(IM_START + "system\n" + sys_text + IM_END + "\n")

        last_query = None
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m["role"] == "user":
                c = _text(m.get("content")).strip()
                if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                    last_query = i
                    break
        if last_query is None:
            raise ChatError("no user question in messages")

        n = len(messages)
        for i, m in enumerate(messages):
            role = m["role"]
            content = _text(m.get("content")).strip()
            if role == "system":
                if i > 0:
                    raise ChatError("a system message must come first")
            elif role == "user":
                out.append(f"{IM_START}user\n{content}{IM_END}\n")
            elif role == "assistant":
                rc = m.get("reasoning_content")
                if isinstance(rc, str):
                    reasoning = rc
                else:
                    reasoning, content = _split_think(content)
                reasoning = reasoning.strip()
                if i > last_query:
                    out.append(IM_START + "assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content)
                else:
                    out.append(IM_START + "assistant\n" + content)
                for j, (name, args) in enumerate(_calls(m)):
                    if j == 0:
                        out.append(("\n\n" if content.strip() else "") + "<tool_call>\n<function=" + name + ">\n")
                    else:
                        out.append("\n<tool_call>\n<function=" + name + ">\n")
                    for k, v in args.items():
                        val = _tojson(v) if isinstance(v, (dict, list, tuple)) else str(v)
                        out.append("<parameter=" + k + ">\n" + val + "\n</parameter>\n")
                    out.append("</function>\n</tool_call>")
                out.append(IM_END + "\n")
            elif role == "tool":
                if i > 0 and messages[i - 1]["role"] != "tool":
                    out.append(IM_START + "user")
                out.append("\n<tool_response>\n" + content + "\n</tool_response>")
                if i == n - 1 or messages[i + 1]["role"] != "tool":
                    out.append(IM_END + "\n")
        return "".join(out)

    # ── replies ───────────────────────────────────────────────────────────────────────
    def reply_stream(self, *, enable_thinking: bool = False,
                     tools: Optional[list] = None) -> "ReplyStream":
        """An incremental splitter for one reply (see ReplyStream). `tools` is what the
        prompt offered — tool calls are only looked for when there were some."""
        return ReplyStream(self, enable_thinking, tools)

    def parse_reply(self, text: str, *, enable_thinking: bool = False,
                    tools: Optional[list] = None) -> dict:
        """A whole reply as an OpenAI assistant message: content, and reasoning_content /
        tool_calls when present. The same state machine as streaming, fed in one piece."""
        rs = self.reply_stream(enable_thinking=enable_thinking, tools=tools)
        parts = rs.feed(text) + rs.finish()
        reasoning = "".join(t for k, t in parts if k == "reasoning")
        content = "".join(t for k, t in parts if k == "content")
        msg: dict = {"role": "assistant", "content": content}
        if rs.saw_reasoning:
            msg["reasoning_content"] = reasoning
        if rs.tool_calls:
            msg["tool_calls"] = rs.tool_calls
            if not content.strip():
                msg["content"] = None
        return msg


def _hold(buf: str, tag: str) -> int:
    """How many trailing characters of `buf` could be the start of `tag`."""
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if tag.startswith(buf[-k:]):
            return k
    return 0


def _keep(buf: str, tag: str, ws) -> int:
    """Trailing characters to hold back until the next piece decides: a possible start of
    `tag`, plus the whitespace before it — which is dropped if the tag does come."""
    k = _hold(buf, tag)
    rest = buf[:len(buf) - k]
    return k + len(rest) - len(rest.rstrip(ws))


class ReplyStream:
    """Splits a reply, piece by piece, into ("reasoning", text) / ("content", text) deltas
    and collects tool calls.

    Thinking is split only for dialects that think: a reply that opens with <think> (or
    whose prompt opened it) is reasoning up to </think>; the newlines the templates put
    around the block are dropped. With tools on offer, everything from the first
    <tool_call> on is held back and parsed at the end; if it does not parse, it comes out
    as content after all, so nothing the model wrote is ever lost.
    """

    def __init__(self, fmt: ChatFormat, enable_thinking: bool, tools: Optional[list]):
        self.fmt = fmt
        self.tools = tools or []
        self.tool_calls: list = []
        self.saw_reasoning = False
        self._buf = ""
        self._tool = ""                       # held tool-call text
        if fmt.thinks and fmt.opener(enable_thinking).endswith(THINK_OPEN + "\n"):
            self._mode = "reasoning"          # the prompt opened the block
            self.saw_reasoning = True
        elif fmt.thinks:
            self._mode = "pre"                # may open with <think>
        else:
            self._mode = "content"
        self._lead = True                     # drop leading newlines of the current segment
        self._content_out = False             # any content emitted yet

    def _emit(self, out: list, kind: str, text: str):
        if self._lead:
            text = text.lstrip("\n")
            if text:
                self._lead = False
        if text:
            out.append((kind, text))
            self._content_out = self._content_out or kind == "content"

    def feed(self, piece: str) -> list:
        out: list = []
        self._buf += piece
        while True:
            if self._mode == "pre":
                s = self._buf.lstrip()
                if s.startswith(THINK_OPEN):
                    self._buf = s[len(THINK_OPEN):]
                    self._mode, self._lead, self.saw_reasoning = "reasoning", True, True
                    continue
                if not s or THINK_OPEN.startswith(s):
                    return out                # undecided yet
                self._mode = "content"
                continue
            if self._mode == "reasoning":
                k = self._buf.find(THINK_CLOSE)
                if k >= 0:
                    self._emit(out, "reasoning", self._buf[:k].rstrip("\n"))
                    self._buf = self._buf[k + len(THINK_CLOSE):]
                    self._mode, self._lead = "content", True
                    continue
                keep = _keep(self._buf, THINK_CLOSE, "\n")
                self._emit(out, "reasoning", self._buf[:len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep:]
                return out
            if self._mode == "content":
                if self.tools and self.fmt.tool_format:
                    k = self._buf.find(TOOL_OPEN)
                    if k >= 0:
                        self._emit(out, "content", self._buf[:k].rstrip())
                        self._tool, self._buf, self._mode = self._buf[k:], "", "tool"
                        continue
                    keep = _keep(self._buf, TOOL_OPEN, None)
                else:
                    keep = 0
                self._emit(out, "content", self._buf[:len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep:]
                return out
            # tool: hold everything until the end
            self._tool += self._buf
            self._buf = ""
            return out

    def finish(self) -> list:
        """Flush the reply's end: held text, and the tool calls if they parse."""
        out: list = []
        if self._mode == "pre":
            self._mode = "content"
        if self._mode == "reasoning":
            self._emit(out, "reasoning", self._buf.rstrip("\n"))   # cut off while thinking
        elif self._mode == "content":
            self._emit(out, "content", self._buf)
        elif self._mode == "tool":
            calls, rest = (_parse_hermes if self.fmt.tool_format == "hermes" else _parse_xml)(
                self._tool + self._buf, self.tools)
            if calls:
                self.tool_calls = calls
                if rest.strip():                # text the model wrote around its calls
                    self._lead = False
                    self._emit(out, "content", ("\n\n" if self._content_out else "") + rest.strip())
            else:                               # not a well-formed call: it was content
                self._lead = False
                self._emit(out, "content", self._tool + self._buf)
        self._buf = self._tool = ""
        return out


def _call(name: str, args: dict) -> dict:
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


_HERMES_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_XML_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _parse_hermes(text: str, tools: list) -> tuple[list, str]:
    """(calls, text outside them) for every <tool_call>{"name", "arguments"}</tool_call>;
    no calls at all if any block does not parse — then the whole text is content."""
    blocks = list(_HERMES_RE.finditer(text))
    calls = []
    for b in blocks:
        try:
            obj = json.loads(b.group(1))
        except json.JSONDecodeError:
            return [], text
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return [], text
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return [], text
        if not isinstance(args, dict):
            return [], text
        calls.append(_call(obj["name"], args))
    rest = _HERMES_RE.sub("", text)
    if TOOL_OPEN in rest:                     # an unclosed block: cut off mid-call
        return [], text
    return calls, rest


def _param_types(tools: list, name: str) -> dict:
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if isinstance(fn, dict) and fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties") or {}
            return {k: (v or {}).get("type") for k, v in props.items() if isinstance(v, dict)}
    return {}


def _typed(value: str, kind) -> object:
    """An XML parameter's text as the schema's type says; text when it does not convert."""
    kinds = kind if isinstance(kind, list) else [kind]
    if "string" in kinds or kind is None:
        return value
    try:
        if "integer" in kinds:
            return int(value)
        if "number" in kinds:
            f = float(value)
            return int(f) if f.is_integer() and "." not in value and "e" not in value.lower() else f
        if "boolean" in kinds:
            v = value.strip().lower()
            if v in ("true", "false"):
                return v == "true"
            return value
        if "null" in kinds and value.strip().lower() in ("null", "none"):
            return None
        if "object" in kinds or "array" in kinds:
            return json.loads(value)
    except (ValueError, json.JSONDecodeError):
        return value
    return value


def _parse_xml(text: str, tools: list) -> tuple[list, str]:
    """Qwen3.5's <tool_call><function=n><parameter=k>v</parameter>…</function></tool_call>."""
    blocks = list(_XML_CALL_RE.finditer(text))
    rest = _XML_CALL_RE.sub("", text)
    if not blocks or TOOL_OPEN in rest:
        return [], text
    calls = []
    for b in blocks:
        name = b.group(1).strip()
        types = _param_types(tools, name)
        args = {m.group(1).strip(): _typed(m.group(2), types.get(m.group(1).strip()))
                for m in _XML_PARAM_RE.finditer(b.group(2))}
        calls.append(_call(name, args))
    return calls, rest
