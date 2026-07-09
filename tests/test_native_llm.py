"""The native text runtime: the dense and hybrid engines, no libxlm.

Point BLLM_TEST_NATIVE_DIR at a model directory (`model.json`); everything skips
otherwise. Run via `scripts/board_test.sh`.

**One big model per process.** A session pins its weights in the BPU's ION carveout
until it is destroyed — a hybrid session plus a 448 px omni session at once exhausted
the carveout and reset the board. Hence the module-scoped fixture, and hence the VLM
tests living in their own module: pytest finalises this one before that one starts.
"""

import gc
import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

NATIVE_DIR = os.environ.get("BLLM_TEST_NATIVE_DIR")
OMNI_DIR = os.environ.get("BLLM_TEST_OMNI_DIR")


@pytest.fixture(scope="module")
def llm():
    if not (bllm._HAVE_NATIVE and NATIVE_DIR):
        pytest.skip("set BLLM_TEST_NATIVE_DIR")
    session = bllm.NativeSession(NATIVE_DIR)
    yield session
    del session
    gc.collect()          # release the ION carveout before the next module loads a model


def test_chat_and_stream(llm):
    llm.reset()
    assert llm.arch in ("dense", "hybrid")
    assert "巴黎" in llm.chat("法国的首都是哪里？只答城市名。", max_new=16)
    llm.reset()
    chunks = list(llm.stream_chat("用一句话介绍北京。", max_new=32))
    assert chunks and "".join(chunks)
    assert llm.last_decode_tps >= 0


def test_multi_turn_recalls_context(llm):
    llm.reset()
    llm.chat("请记住数字 4271。", max_new=24)
    assert "4271" in llm.chat("刚才让你记的数字是多少？", max_new=24)


def test_context_overflow_raises_rather_than_degrading(llm):
    """Rolling past cache_len evicts the earliest tokens — which act as attention
    sinks — and turns these full-attention models into gibberish. So the engine
    refuses. This pins behaviour, not performance."""
    llm.reset()
    huge = "北京是中国的首都，上海是最大的城市。" * 400
    with pytest.raises(RuntimeError, match="context overflow"):
        llm.chat(huge, max_new=4)
    llm.reset()


@pytest.mark.skipif(not bllm._HAVE_NATIVE, reason="native runtime not built")
def test_native_session_rejects_an_omni_model_dir():
    """Loading a multimodal model dir with the text session must say so, not fail
    deep inside hbDNN with "Model not exists: prefill"."""
    if not OMNI_DIR:
        pytest.skip("set BLLM_TEST_OMNI_DIR")
    with pytest.raises(RuntimeError, match="omni"):
        bllm.NativeSession(OMNI_DIR)
