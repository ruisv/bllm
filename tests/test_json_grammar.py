#!/usr/bin/env python3
"""JSON Schema -> GBNF, checked against the real matcher.

The converter (``bllm._grammar``) is pure text, but asserting on the text it emits would
only test it against itself. So these tests compile ``tests/grammar_check.cc`` — the C++
grammar engine, header-only and hobot-free — and ask it whether a JSON document actually
matches the generated grammar. Runs anywhere with a C++ compiler, board or not.

The property that matters: a document is accepted **iff** it satisfies the schema. Both
directions are tested, because a grammar that accepts everything would pass a
"valid JSON is accepted" test perfectly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from bllm._grammar import gbnf_literal, json_grammar  # noqa: E402

MATCH, PREFIX, REJECT, BAD_GRAMMAR = 0, 1, 2, 3


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    """Compile the grammar CLI once for the module."""
    out = tmp_path_factory.mktemp("bin") / "grammar_check"
    try:
        subprocess.run(["c++", "-std=c++17", "-I", str(ROOT / "include"),
                        str(ROOT / "tests" / "grammar_check.cc"), "-o", str(out)],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"no usable c++ compiler: {e}")
    return out


def check(checker, grammar: str, text: str, tmp_path) -> int:
    g = tmp_path / "g.gbnf"
    g.write_text(grammar)
    return subprocess.run([str(checker), str(g), text]).returncode


def accepts(checker, schema, doc, tmp_path) -> bool:
    return check(checker, json_grammar(schema), json.dumps(doc), tmp_path) == MATCH


# --- the grammar the converter emits is at least parseable -------------------
def test_generated_grammars_parse(checker, tmp_path):
    for schema in [None, {}, {"type": "string"},
                   {"type": "object", "properties": {"a": {"type": "integer"}}}]:
        rc = check(checker, json_grammar(schema), '"x"', tmp_path)
        assert rc != BAD_GRAMMAR, f"schema {schema} produced a grammar that does not parse"


# --- any-JSON mode ------------------------------------------------------------
@pytest.mark.parametrize("doc", [
    {"a": 1}, [1, 2, 3], "text", 42, -0.5, 1.5e3, True, None, {}, [],
    {"nested": {"deep": [1, {"x": None}]}}, {"unicode": "北京"}, {"esc": 'a"b\\c\nd'},
])
def test_any_json_is_accepted(checker, tmp_path, doc):
    assert check(checker, json_grammar(), json.dumps(doc), tmp_path) == MATCH


@pytest.mark.parametrize("bad", ['{"a": }', "{a: 1}", "[1,]", "'single'", "nul", "01",
                                 "{'a': 1}", "[1 2]"])
def test_malformed_json_is_rejected(checker, tmp_path, bad):
    assert check(checker, json_grammar(), bad, tmp_path) in (PREFIX, REJECT)


# --- scalars ------------------------------------------------------------------
def test_scalar_types(checker, tmp_path):
    assert accepts(checker, {"type": "integer"}, 42, tmp_path)
    assert not accepts(checker, {"type": "integer"}, 4.2, tmp_path)
    assert accepts(checker, {"type": "number"}, 4.2, tmp_path)
    assert accepts(checker, {"type": "string"}, "hi", tmp_path)
    assert not accepts(checker, {"type": "string"}, 1, tmp_path)
    assert accepts(checker, {"type": "boolean"}, True, tmp_path)
    assert not accepts(checker, {"type": "boolean"}, "true", tmp_path)
    assert accepts(checker, {"type": "null"}, None, tmp_path)


def test_enum_and_const(checker, tmp_path):
    s = {"enum": ["red", "green", "blue"]}
    assert accepts(checker, s, "red", tmp_path)
    assert not accepts(checker, s, "purple", tmp_path)
    assert accepts(checker, {"const": 7}, 7, tmp_path)
    assert not accepts(checker, {"const": 7}, 8, tmp_path)


def test_union_type(checker, tmp_path):
    s = {"type": ["string", "null"]}
    assert accepts(checker, s, "x", tmp_path)
    assert accepts(checker, s, None, tmp_path)
    assert not accepts(checker, s, 1, tmp_path)


def test_any_of(checker, tmp_path):
    s = {"anyOf": [{"type": "integer"}, {"enum": ["auto"]}]}
    assert accepts(checker, s, 5, tmp_path)
    assert accepts(checker, s, "auto", tmp_path)
    assert not accepts(checker, s, "manual", tmp_path)


# --- objects ------------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age"],
}


@pytest.mark.parametrize("doc", [
    {"name": "ada", "age": 36},
    {"name": "ada", "age": 36, "tags": []},
    {"name": "ada", "age": 36, "tags": ["a", "b"]},
])
def test_object_accepts_what_the_schema_allows(checker, tmp_path, doc):
    assert accepts(checker, SCHEMA, doc, tmp_path)


@pytest.mark.parametrize("doc", [
    {"name": "ada"},                              # missing a required key
    {"age": 36},
    {"name": "ada", "age": "36"},                 # wrong type
    {"name": "ada", "age": 36, "extra": 1},       # additionalProperties are not allowed
    {"name": "ada", "age": 36, "tags": [1]},      # wrong item type
])
def test_object_rejects_what_it_should(checker, tmp_path, doc):
    assert not accepts(checker, SCHEMA, doc, tmp_path)


def test_all_optional_object_never_needs_a_leading_comma(checker, tmp_path):
    """With no required key, any property may come first — the naive "each optional carries
    its own comma" grammar emits `{, "b": 1}` for anything but the first."""
    s = {"type": "object", "properties": {"a": {"type": "integer"},
                                          "b": {"type": "integer"},
                                          "c": {"type": "integer"}}}
    for doc in [{}, {"a": 1}, {"b": 2}, {"c": 3}, {"a": 1, "c": 3}, {"a": 1, "b": 2, "c": 3}]:
        assert accepts(checker, s, doc, tmp_path), doc
    # order still has to follow the schema, which is the documented narrowing
    assert not accepts(checker, s, {"c": 3, "a": 1}, tmp_path)


def test_nested_objects_and_refs(checker, tmp_path):
    s = {
        "type": "object",
        "properties": {"user": {"$ref": "#/$defs/User"}},
        "required": ["user"],
        "$defs": {"User": {"type": "object",
                           "properties": {"id": {"type": "integer"}},
                           "required": ["id"]}},
    }
    assert accepts(checker, s, {"user": {"id": 1}}, tmp_path)
    assert not accepts(checker, s, {"user": {"id": "1"}}, tmp_path)


def test_recursive_ref_terminates(checker, tmp_path):
    """A self-referential schema must produce a finite grammar, not hang the converter."""
    s = {
        "$ref": "#/$defs/Node",
        "$defs": {"Node": {"type": "object",
                           "properties": {"v": {"type": "integer"},
                                          "next": {"$ref": "#/$defs/Node"}},
                           "required": ["v"]}},
    }
    assert accepts(checker, s, {"v": 1}, tmp_path)
    assert accepts(checker, s, {"v": 1, "next": {"v": 2, "next": {"v": 3}}}, tmp_path)


def test_whitespace_is_allowed_but_not_required(checker, tmp_path):
    g = json_grammar(SCHEMA)
    assert check(checker, g, '{"name":"ada","age":36}', tmp_path) == MATCH
    assert check(checker, g, '{ "name" : "ada" , "age" : 36 }', tmp_path) == MATCH
    assert check(checker, g, '{\n  "name": "ada",\n  "age": 36\n}', tmp_path) == MATCH


def test_partial_document_is_a_prefix_not_a_rejection(checker, tmp_path):
    """This is what makes constrained decoding work at all: an unfinished document must
    stay live so generation can continue into it."""
    assert check(checker, json_grammar(SCHEMA), '{"name": "a', tmp_path) == PREFIX


# --- the escaping helper ------------------------------------------------------
def test_gbnf_literal_escapes():
    assert gbnf_literal('a"b') == '"a\\"b"'
    assert gbnf_literal("a\\b") == '"a\\\\b"'
    assert gbnf_literal("a\nb") == '"a\\nb"'
    assert gbnf_literal("\x01") == '"\\x01"'


def test_unsupported_ref_is_an_error():
    with pytest.raises(ValueError):
        json_grammar({"$ref": "https://example.com/schema.json"})
    with pytest.raises(ValueError):
        json_grammar({"$ref": "#/$defs/missing"})
