"""The native multimodal runtime (arch="omni"): text + image, and the live-stream budget.

Point BLLM_TEST_OMNI_DIR at an omni model directory (and optionally BLLM_TEST_IMAGE);
everything skips otherwise. Run via `scripts/board_test.sh`.

Kept in its own module because a VLM session pins the text tower, the vision tower and
the 1.2 GB embedding table in the BPU's ION carveout for as long as it lives — see the
note in `test_native_llm.py`.
"""

import gc
import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

OMNI_DIR = os.environ.get("BLLM_TEST_OMNI_DIR")
IMAGE = os.environ.get("BLLM_TEST_IMAGE")


@pytest.fixture(scope="module")
def vlm():
    if not (bllm._HAVE_NATIVE and OMNI_DIR):
        pytest.skip("set BLLM_TEST_OMNI_DIR")
    session = bllm.NativeVlmSession(OMNI_DIR)
    yield session
    del session
    gc.collect()


def test_text_and_image(vlm):
    vlm.reset()
    assert vlm.vision_tokens > 0
    assert vlm.vision_image_size % 28 == 0          # patch 14 x merge 2
    assert "巴黎" in vlm.chat("法国的首都是哪里？只答城市名。", max_new=16)
    if IMAGE:
        vlm.reset()
        reply = vlm.chat("这张图里有什么？一句话。", images=[IMAGE], max_new=48)
        assert isinstance(reply, str) and reply
        assert vlm.last_ttft_ms > 0


def test_stream_budget_is_enforced(vlm):
    """The KV window is the whole context: pushing more frames than it holds must
    raise instead of silently evicting."""
    import numpy as np

    vlm.reset()
    side = vlm.vision_image_size
    frame = np.zeros((side, side, 3), dtype=np.uint8)
    vlm.stream_begin(fps=2.0, with_audio=False)
    before = vlm.context_left
    with pytest.raises(RuntimeError, match="context overflow"):
        for _ in range(4096 // vlm.vision_tokens + 8):
            vlm.stream_frame(frame)
    assert vlm.context_left < before        # frames were ingested before it refused
    vlm.reset()
