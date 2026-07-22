"""JSON Schema -> GBNF, so structured output can be asked for as a schema.

The grammar engine (C++, `include/bllm/native_grammar.h`) speaks GBNF. Schemas are a
Python-shaped problem — nested dicts, `$ref` resolution, optional keys — so the conversion
lives here rather than in the runtime: it is pure text in, pure text out, testable without
a board, and a bug in it can never corrupt a generation, only produce a grammar that
refuses.

Supported: `type` (object / array / string / number / integer / boolean / null), `enum`,
`const`, `properties` + `required`, `items`, `anyOf` / `oneOf`, `$ref` into `$defs` or
`definitions`, and `additionalProperties: false`. Anything else is ignored rather than
rejected — an unconstrained value is a weaker guarantee, never a wrong one.

Two deliberate narrowings, both documented in the serving docs:

- Optional properties must appear in schema order. Allowing every permutation is
  factorial in the number of keys; JSON objects are order-insensitive to parsers, so this
  costs nothing real and keeps the grammar small.
- `additionalProperties` is treated as false throughout. A schema-constrained reply that
  invented extra keys would defeat the point of asking for the schema.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

__all__ = ["json_grammar", "gbnf_literal"]

# The primitives every generated grammar can lean on. `ws` allows (but never requires)
# whitespace, so the model may format the JSON either way.
_PREAMBLE = r'''
ws        ::= [ \t\n]*
string    ::= "\"" char* "\""
char      ::= [^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex       ::= [0-9a-fA-F]
integer   ::= "-"? ("0" | [1-9] [0-9]*)
number    ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean   ::= "true" | "false"
null      ::= "null"
value     ::= object | array | string | number | boolean | null
object    ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)* ws)? "}"
array     ::= "[" ws (value (ws "," ws value)* ws)? "]"
'''.lstrip()


def gbnf_literal(text: str) -> str:
    """A GBNF string literal for `text`, escaped."""
    out = []
    for ch in text:
        if ch == '"':
            out.append(r'\"')
        elif ch == "\\":
            out.append(r"\\")
        elif ch == "\n":
            out.append(r"\n")
        elif ch == "\r":
            out.append(r"\r")
        elif ch == "\t":
            out.append(r"\t")
        elif ord(ch) < 0x20:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def json_grammar(schema: Optional[dict] = None) -> str:
    """GBNF for JSON matching `schema`, or for any JSON value when `schema` is None.

    >>> json_grammar({"type": "object", "properties": {"n": {"type": "integer"}},
    ...               "required": ["n"]})            # doctest: +ELLIPSIS
    'root ::= ...'
    """
    if schema is None:
        return _PREAMBLE + "root ::= ws value ws\n"
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    return _Converter(schema).build()


class _Converter:
    def __init__(self, schema: dict) -> None:
        self.schema = schema
        self.defs = schema.get("$defs") or schema.get("definitions") or {}
        self.rules: list[str] = []
        self.n = 0
        self.by_ref: dict[str, str] = {}      # $ref path -> rule name, for recursive schemas

    def build(self) -> str:
        body = self.visit(self.schema, "root")
        return _PREAMBLE + "\n".join(self.rules + [f"root ::= ws {body} ws"]) + "\n"

    def name(self, hint: str) -> str:
        self.n += 1
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", hint) or "r"
        return f"{safe}_{self.n}"

    def rule(self, hint: str, body: str) -> str:
        """Define a named rule for `body` and return the name."""
        n = self.name(hint)
        self.rules.append(f"{n} ::= {body}")
        return n

    def visit(self, s: Any, hint: str) -> str:
        """Return a GBNF expression matching `s` (a rule name or an inline expression)."""
        if not isinstance(s, dict) or s == {}:
            return "value"                      # `true` / {} / anything unrecognised

        if "$ref" in s:
            return self.visit_ref(s["$ref"])
        if "const" in s:
            return self.rule(hint, gbnf_literal(json.dumps(s["const"], ensure_ascii=False)))
        if "enum" in s:
            alts = " | ".join(gbnf_literal(json.dumps(v, ensure_ascii=False)) for v in s["enum"])
            return self.rule(hint, alts)
        for key in ("anyOf", "oneOf"):
            if key in s:
                alts = " | ".join(self.visit(sub, f"{hint}_alt") for sub in s[key])
                return self.rule(hint, alts)

        t = s.get("type")
        if isinstance(t, list):                 # {"type": ["string", "null"]}
            alts = " | ".join(self.visit({**s, "type": one}, f"{hint}_{one}") for one in t)
            return self.rule(hint, alts)

        if t == "object":
            return self.visit_object(s, hint)
        if t == "array":
            return self.visit_array(s, hint)
        if t in ("string", "number", "integer", "boolean", "null"):
            return t                            # straight from the preamble
        return "value"                          # untyped: any JSON

    def visit_ref(self, ref: str) -> str:
        if ref in self.by_ref:
            return self.by_ref[ref]             # already being built: recursion, tie the knot
        prefix = ("#/$defs/", "#/definitions/")
        if not ref.startswith(prefix):
            raise ValueError(f"unsupported $ref {ref!r} (only #/$defs/... and "
                             f"#/definitions/... are resolved)")
        key = ref.split("/")[-1]
        if key not in self.defs:
            raise ValueError(f"$ref {ref!r} does not resolve")
        # Reserve the name BEFORE visiting, so a self-referential schema terminates.
        n = self.name(key)
        self.by_ref[ref] = n
        body = self.visit(self.defs[key], key)
        self.rules.append(f"{n} ::= {body}")
        return n

    def visit_object(self, s: dict, hint: str) -> str:
        props = s.get("properties")
        if not props:
            return "object"                     # any object
        required = list(s.get("required") or [])
        req = [k for k in props if k in required]
        opt = [k for k in props if k not in required]

        def pair(key: str) -> str:
            body = self.visit(props[key], f"{hint}_{key}")
            # json.dumps, not the bare key: the literal has to include the JSON quotes, or
            # the grammar would demand `{name: 1}`.
            return f'{gbnf_literal(json.dumps(key, ensure_ascii=False))} ws ":" ws {body}'

        parts: list[str] = ['"{" ws']
        if req:
            # Every comma has a pair in front of it, so each optional can carry its own.
            parts.append(pair(req[0]))
            for k in req[1:]:
                parts.append(f'ws "," ws {pair(k)}')
            for k in opt:
                parts.append(f'(ws "," ws {pair(k)})?')
        elif opt:
            # No required key means any of them could come first, and a leading comma would
            # be invalid JSON — so branch on which one opens the object, each branch trailed
            # by the ones that may still follow. Quadratic in the key count, linear in
            # practice (schemas have a handful of keys), and it never emits `{, "b": 1}`.
            branches = []
            for i, k in enumerate(opt):
                tail = "".join(f' (ws "," ws {pair(o)})?' for o in opt[i + 1:])
                branches.append(f"{pair(k)}{tail}")
            parts.append("(" + " | ".join(branches) + ")?")
        parts.append('ws "}"')
        return self.rule(hint, " ".join(parts))

    def visit_array(self, s: dict, hint: str) -> str:
        items = s.get("items")
        if items is None:
            return "array"
        body = self.visit(items, f"{hint}_item")
        return self.rule(hint, f'"[" ws ({body} (ws "," ws {body})* ws)? "]"')
