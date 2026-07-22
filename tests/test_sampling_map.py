#!/usr/bin/env python3
"""Unit tests for the OpenAI -> engine sampling mapping.

Pure logic, but importing server.py pulls in fastapi and bllm, so this skips off-board.

The bugs these lock down were both silent: a client that set OpenAI's own
presence_penalty/frequency_penalty had them dropped on the floor, and `x or default`
folded an explicit 0 into the default, so `temperature: 0` — which clients send
deliberately to get greedy decoding — was indistinguishable from not sending it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))
os.environ.setdefault("BLLM_MODEL", "/nonexistent")   # server.py only annotates Engine

server = pytest.importorskip("server", reason="needs fastapi + bllm (board)")
_sampling_from = server._sampling_from


def test_defaults_when_nothing_is_given():
    s = _sampling_from({})
    assert s["temp"] == 0.0
    assert s["top_p"] == 1.0
    assert s["top_k"] == 0
    assert s["min_p"] == 0.0
    assert s["rep_pen"] == 1.0
    assert s["penalty_present"] == 0.0
    assert s["penalty_freq"] == 0.0


def test_openai_penalties_are_honoured():
    """These are OpenAI's own field names; they used to be ignored entirely."""
    s = _sampling_from({"presence_penalty": 0.7, "frequency_penalty": 0.4})
    assert s["penalty_present"] == pytest.approx(0.7)
    assert s["penalty_freq"] == pytest.approx(0.4)


def test_negative_penalties_pass_through():
    """OpenAI defines both on [-2, 2]; a negative value legitimately encourages repeats
    and must not be clamped away or read as "unset"."""
    s = _sampling_from({"presence_penalty": -1.5, "frequency_penalty": -2})
    assert s["penalty_present"] == pytest.approx(-1.5)
    assert s["penalty_freq"] == pytest.approx(-2.0)


@pytest.mark.parametrize("key,engine_key", [
    ("temperature", "temp"),
    ("top_p", "top_p"),
    ("repetition_penalty", "rep_pen"),
    ("seed", "seed"),
])
def test_explicit_zero_is_not_the_default(key, engine_key):
    s = _sampling_from({key: 0})
    assert s[engine_key] == 0, f"{key}=0 was folded back into its default"


def test_no_seed_means_vary_per_request():
    """A fixed default seed pinned the RNG on every request, so temperature did
    nothing: three identical temperature=0.8 requests returned byte-identical text."""
    import bllm
    assert _sampling_from({"temperature": 0.8})["seed"] == bllm.SEED_RANDOM


def test_an_explicit_seed_is_honoured_exactly():
    """OpenAI's `seed` promises repeated identical requests reproduce."""
    assert _sampling_from({"temperature": 0.8, "seed": 99})["seed"] == 99


def test_null_is_treated_as_absent():
    """Clients (and some SDKs) send explicit nulls for unset fields."""
    s = _sampling_from({"temperature": None, "top_p": None, "presence_penalty": None})
    assert s["temp"] == 0.0
    assert s["top_p"] == 1.0
    assert s["penalty_present"] == 0.0


def test_values_pass_through():
    s = _sampling_from({"temperature": 0.8, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
                        "repetition_penalty": 1.1, "seed": 7})
    assert (s["temp"], s["top_p"], s["top_k"]) == (0.8, 0.9, 40)
    assert (s["min_p"], s["rep_pen"], s["seed"]) == (0.05, 1.1, 7)
