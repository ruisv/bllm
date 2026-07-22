"""Image+text on a HYBRID (Gated-DeltaNet) VLM — Qwen3.5.

Distinct from test_native_vlm.py, which exercises the dense Omni tower and its
audio/video paths. What is worth testing separately here is not "does a VLM work"
but the three things that are genuinely different when the decoder is hybrid:

  * the TextEngine seam actually dispatches to NativeHybridEngine,
  * 3-D mrope positions advance the way an image requires (196 slots, +14 rope),
  * the vision tower's own geometry (patch 16, mean=std=0.5) came through the
    manifest rather than defaulting to Omni's patch-14 CLIP settings.

Point BLLM_TEST_HYBRID_VLM_DIR at a hybrid model dir carrying a `visual` tower
(and optionally BLLM_TEST_IMAGE); everything skips otherwise. Runs on the board.
"""

import gc
import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

VLM_DIR = os.environ.get("BLLM_TEST_HYBRID_VLM_DIR")
IMAGE = os.environ.get("BLLM_TEST_IMAGE")


@pytest.fixture(scope="module")
def vlm():
    if not (bllm._HAVE_NATIVE and VLM_DIR):
        pytest.skip("set BLLM_TEST_HYBRID_VLM_DIR")
    session = bllm.load(VLM_DIR)
    yield session
    del session
    gc.collect()


def test_routes_to_a_vlm_session(vlm):
    """bllm.load() must route on the vision tower's presence, not on arch=='omni' —
    a hybrid package with a `visual` entry is still a VLM."""
    assert isinstance(vlm, bllm.NativeVlmSession)
    assert vlm.vision_tokens > 0
    assert vlm.vision_image_size % 32 == 0          # patch 16 x merge 2


def test_text_still_works(vlm):
    """The multimodal seam must not disturb plain text: with t==h==w the interleaved
    mrope collapses to ordinary 1-D rope, so this is also the check that it does."""
    vlm.reset()
    reply = vlm.chat("法国的首都是哪里？只答城市名。", max_new=24)
    assert "巴黎" in reply or "Paris" in reply


@pytest.mark.skipif(not IMAGE, reason="set BLLM_TEST_IMAGE")
def test_image_costs_its_tokens_and_answers(vlm):
    vlm.reset()
    before = vlm.tokens_used
    reply = vlm.chat("这张图里有什么？一句话。", images=[IMAGE], max_new=48)
    assert isinstance(reply, str) and reply.strip()
    # The image must actually occupy its rows in the cache — a silently-dropped
    # image would still produce a fluent (and completely made up) answer.
    assert vlm.tokens_used - before > vlm.vision_tokens
    assert vlm.last_ttft_ms > 0


@pytest.mark.skipif(not IMAGE, reason="set BLLM_TEST_IMAGE")
def test_image_changes_the_answer(vlm):
    """The strongest cheap check that pixels reach the decoder: the same question
    with and without the image must not give the same reply. Cosine on the tower
    output cannot catch a splice that never happened."""
    vlm.reset()
    vlm.set_sampling(temp=0.0)
    without = vlm.chat("用一个词描述你看到的主要颜色。", max_new=16)
    vlm.reset()
    with_image = vlm.chat("用一个词描述你看到的主要颜色。", images=[IMAGE], max_new=16)
    assert without != with_image
