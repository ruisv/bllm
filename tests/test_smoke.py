"""Smoke test for the bllm Python bindings.

Runs inside the board conda env (needs the built module on PYTHONPATH and the
OE-LLM libs on LD_LIBRARY_PATH). The model-loading test is skipped unless a
model is provided via env:

    export BLLM_TEST_HBM=<...>.hbm
    export BLLM_TEST_TOKENIZER=<...>_config/
    pytest tests/test_smoke.py
"""

import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

HBM = os.environ.get("BLLM_TEST_HBM")
TOK = os.environ.get("BLLM_TEST_TOKENIZER")
_need_model = pytest.mark.skipif(
    not (HBM and TOK), reason="set BLLM_TEST_HBM + BLLM_TEST_TOKENIZER to run"
)


def test_module_surface():
    assert isinstance(bllm.__version__, str)
    assert hasattr(bllm, "LlmSession")
    assert hasattr(bllm, "GenerationStats")


@_need_model
def test_generate_and_stream():
    llm = bllm.LlmSession(HBM, TOK)
    assert llm.model_type != "auto"  # inferred to a concrete family

    reply = llm.generate("你好")
    assert isinstance(reply, str) and reply

    chunks = list(llm.stream("用一句话介绍一下人工智能。"))
    assert chunks and "".join(chunks)

    s = llm.last_stats
    assert s.decode_tokens > 0
    # decode rate is defined over tokens after the first (TTFT); only meaningful
    # once more than one token was produced.
    if s.decode_tokens > 1:
        assert s.decode_tps > 0.0

    llm.reset()  # must not raise
