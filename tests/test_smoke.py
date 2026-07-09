"""Smoke tests for the bllm Python bindings.

Runs inside the board conda env. The two runtimes are independent and either may be
absent from a given install, so each is gated rather than assumed:

  * libxlm-backed sessions need the OE-LLM SDK on LD_LIBRARY_PATH,
  * the native runtime needs only the hobot runtime + the C++ tokenizer.

Model-loading tests are skipped unless a model is provided via env:

    export BLLM_TEST_HBM=<...>.hbm           # libxlm path
    export BLLM_TEST_TOKENIZER=<...>_config/
    export BLLM_TEST_NATIVE_DIR=<model dir>  # native path (model.json)
    export BLLM_TEST_OMNI_DIR=<model dir>    # native multimodal (arch="omni")
    export BLLM_TEST_IMAGE=<...>.jpg
    pytest tests/
"""

import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

HBM = os.environ.get("BLLM_TEST_HBM")
TOK = os.environ.get("BLLM_TEST_TOKENIZER")
NATIVE_DIR = os.environ.get("BLLM_TEST_NATIVE_DIR")
OMNI_DIR = os.environ.get("BLLM_TEST_OMNI_DIR")
IMAGE = os.environ.get("BLLM_TEST_IMAGE")

_need_libxlm = pytest.mark.skipif(not bllm._HAVE_LIBXLM, reason="OE-LLM SDK not importable")
_need_libxlm_model = pytest.mark.skipif(
    not (bllm._HAVE_LIBXLM and HBM and TOK), reason="set BLLM_TEST_HBM + BLLM_TEST_TOKENIZER"
)
_need_native = pytest.mark.skipif(not bllm._HAVE_NATIVE, reason="native runtime not built")
_need_native_model = pytest.mark.skipif(
    not (bllm._HAVE_NATIVE and NATIVE_DIR), reason="set BLLM_TEST_NATIVE_DIR"
)
_need_omni = pytest.mark.skipif(
    not (bllm._HAVE_NATIVE and OMNI_DIR), reason="set BLLM_TEST_OMNI_DIR"
)


def test_version():
    assert isinstance(bllm.__version__, str) and bllm.__version__


def test_at_least_one_runtime_imported():
    assert bllm._HAVE_LIBXLM or bllm._HAVE_NATIVE, "neither runtime imported"


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


@_need_native_model
def test_native_chat_and_stream():
    llm = bllm.NativeSession(NATIVE_DIR)
    assert llm.arch in ("dense", "hybrid")
    assert "巴黎" in llm.chat("法国的首都是哪里？只答城市名。", max_new=16)
    llm.reset()
    chunks = list(llm.stream_chat("用一句话介绍北京。", max_new=32))
    assert chunks and "".join(chunks)
    assert llm.last_decode_tps >= 0


@_need_native_model
def test_native_multi_turn_recalls_context():
    llm = bllm.NativeSession(NATIVE_DIR)
    llm.chat("请记住数字 4271。", max_new=24)
    assert "4271" in llm.chat("刚才让你记的数字是多少？", max_new=24)


@_need_native_model
def test_native_context_overflow_raises_rather_than_degrading():
    """Rolling past cache_len evicts the earliest tokens — which act as attention
    sinks — and turns these full-attention models into gibberish. So the engine must
    refuse. This guards behaviour, not performance."""
    llm = bllm.NativeSession(NATIVE_DIR)
    huge = "北京是中国的首都，上海是最大的城市。" * 400
    with pytest.raises(RuntimeError, match="context overflow"):
        llm.chat(huge, max_new=4)


@_need_native
def test_native_session_rejects_an_omni_model_dir():
    if not OMNI_DIR:
        pytest.skip("set BLLM_TEST_OMNI_DIR")
    with pytest.raises(RuntimeError, match="omni"):
        bllm.NativeSession(OMNI_DIR)


@_need_omni
def test_native_vlm_text_and_image():
    vlm = bllm.NativeVlmSession(OMNI_DIR)
    assert vlm.vision_tokens > 0
    assert vlm.vision_image_size % 28 == 0          # patch 14 x merge 2
    assert "巴黎" in vlm.chat("法国的首都是哪里？只答城市名。", max_new=16)
    if IMAGE:
        vlm.reset()
        reply = vlm.chat("这张图里有什么？一句话。", images=[IMAGE], max_new=48)
        assert isinstance(reply, str) and reply
        assert vlm.last_ttft_ms > 0


@_need_omni
def test_native_vlm_stream_budget_is_enforced():
    """The KV window is the whole context: pushing more frames than it holds must
    raise instead of silently evicting."""
    import numpy as np

    vlm = bllm.NativeVlmSession(OMNI_DIR)
    side = vlm.vision_image_size
    frame = np.zeros((side, side, 3), dtype=np.uint8)
    vlm.stream_begin(fps=2.0, with_audio=False)
    before = vlm.context_left
    with pytest.raises(RuntimeError, match="context overflow"):
        for _ in range(4096 // vlm.vision_tokens + 8):
            vlm.stream_frame(frame)
    assert vlm.context_left < before        # frames were ingested before it refused
