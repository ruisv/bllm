"""Surface tests, and the libxlm-backed sessions.

Two runtimes, either of which may be absent from an install, so each is gated rather
than assumed. Model tests are skipped unless a model is pointed at via env — see
`scripts/board_test.sh`, which is how this suite is meant to run.

**One big model per process.** A session pins its weights in the BPU's ION carveout
until it is destroyed. A hybrid session plus a 448 px omni session at once exhausted
the carveout and reset the board, so the native LLM and VLM tests live in separate
modules: pytest finalises a module-scoped fixture before the next module starts.
"""

import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

HBM = os.environ.get("BLLM_TEST_HBM")
TOK = os.environ.get("BLLM_TEST_TOKENIZER")

_need_libxlm = pytest.mark.skipif(not bllm._HAVE_LIBXLM, reason="OE-LLM SDK not importable")
_need_libxlm_model = pytest.mark.skipif(
    not (bllm._HAVE_LIBXLM and HBM and TOK), reason="set BLLM_TEST_HBM + BLLM_TEST_TOKENIZER"
)
_need_native = pytest.mark.skipif(not bllm._HAVE_NATIVE, reason="native runtime not built")


def test_version():
    assert isinstance(bllm.__version__, str) and bllm.__version__


def test_at_least_one_runtime_imported():
    assert bllm._HAVE_LIBXLM or bllm._HAVE_NATIVE, "neither runtime imported"


def test_all_only_advertises_names_that_exist():
    """`from bllm import *` must work on a host with no runtime at all."""
    assert all(hasattr(bllm, n) for n in bllm.__all__), bllm.__all__
    assert set(bllm.available_backends()) <= {"libxlm", "native"}


@_need_libxlm
def test_libxlm_surface():
    for name in ("LlmSession", "OmniSession", "VlmSession", "Content",
                 "SamplingParams", "GenerationStats"):
        assert hasattr(bllm, name), name


@_need_native
def test_native_surface():
    for name in ("NativeSession", "NativeVlmSession", "load_video"):
        assert hasattr(bllm, name), name


@_need_libxlm_model
def test_libxlm_generate_and_stream():
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

    # Stop strings, on the SAME session — a second libxlm init in one process exhausts the
    # ION carveout ("one big model per process"). libxlm can't cancel, but the reply and
    # stream are still trimmed at the stop, its prefix held back. Stop on the first newline
    # (appears early, so the uncancellable runtime stays short of its token cap); if a
    # sampled answer never contains it, the runtime runs to that cap and raises — which IS
    # the documented no-cancellation limitation, so tolerate that.
    llm.set_stop(["\n"])
    chunks = []
    try:
        reply = llm.generate("列出数字 1、2、3，每个数字单独占一行。", lambda c: chunks.append(c))
    except RuntimeError:
        return  # hit the token cap before the stop appeared — expected without cancellation
    assert "\n" not in reply, reply           # the stop string is gone from the reply
    assert "".join(chunks) == reply           # stream trimmed the same way, no leak
