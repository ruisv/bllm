"""An official D-Robotics LLM 2.0 VLM package directory, loaded as-is, image chat included.

Point BLLM_TEST_OFFICIAL_VLM_DIR at an unpacked VLM package (e.g. `Qwen3-VL-2B_w4/`: a
`*_language_*.hbm`, a `*_vision_*.hbm` and a `*_embed_tokens.bin`); everything skips
otherwise. The image is drawn here, so the test needs Pillow but no asset files.

One model per process, as in test_native_llm.py: the module-scoped fixture is released
before the next module loads anything.
"""

import gc
import json
import os

import pytest

np = pytest.importorskip("numpy")
bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

VLM_DIR = os.environ.get("BLLM_TEST_OFFICIAL_VLM_DIR")


@pytest.fixture(scope="module")
def vlm(tmp_path_factory):
    if not (bllm._HAVE_NATIVE and VLM_DIR):
        pytest.skip("set BLLM_TEST_OFFICIAL_VLM_DIR")
    cache = tmp_path_factory.mktemp("bllm_cache")
    old = os.environ.get("BLLM_CACHE")
    os.environ["BLLM_CACHE"] = str(cache)
    session = bllm.load(VLM_DIR)
    yield session, cache
    del session
    gc.collect()
    if old is None:
        os.environ.pop("BLLM_CACHE", None)
    else:
        os.environ["BLLM_CACHE"] = old


def _flag(color, size=(640, 360)):
    """A plain image of one colour with a white disc — easy for any VLM to name."""
    h, w = size[1], size[0]
    img = np.empty((h, w, 3), np.uint8)
    img[:] = color
    y, x = np.ogrid[:h, :w]
    img[(y - h // 2) ** 2 + (x - w // 2) ** 2 < (h // 4) ** 2] = 255
    return img


def test_loads_as_an_image_session(vlm):
    s, cache = vlm
    assert type(s).__name__ == "NativeVlmSession"
    assert "official LLM 2.0" in s.name and s.vision_tokens > 0
    cfg = json.loads(next(cache.glob("native/*/model.json")).read_text())
    assert cfg["visual"] == "visual.hbm" and cfg["vision"]["resize"] == "cv_cubic"
    assert len(cfg["mrope_section"]) == 3


def test_names_the_colour_of_an_image(vlm):
    s, _ = vlm
    s.reset()
    reply = s.chat("What is the background colour of this image? Answer in one word.",
                   images=[_flag((200, 20, 20))], max_new=16)
    assert "red" in reply.lower(), reply


def test_an_image_path_and_its_pixels_give_the_same_reply(vlm, tmp_path):
    # A path is decoded with Pillow (libjpeg, as the official runtime decodes); handing the
    # same decoded pixels in must therefore be the same turn exactly.
    PIL = pytest.importorskip("PIL.Image")
    s, _ = vlm
    p = tmp_path / "flag.png"
    PIL.fromarray(_flag((20, 40, 200))).save(p)
    s.reset()
    a = s.chat("Describe this image in one sentence.", images=[str(p)], max_new=24)
    s.reset()
    b = s.chat("Describe this image in one sentence.", images=[_flag((20, 40, 200))], max_new=24)
    assert a == b


def test_a_second_turn_follows_the_first(vlm):
    s, _ = vlm
    s.reset()
    s.chat("What colour is the background? One word.", images=[_flag((20, 160, 20))], max_new=8)
    used = s.tokens_used
    reply = s.chat("And what colour is the circle in the middle? One word.", max_new=8)
    assert s.tokens_used > used and "white" in reply.lower(), reply


def test_counts_generated_tokens_and_stops_at_max_new(vlm):
    s, _ = vlm
    s.reset()
    s.chat("Describe this image in detail.", images=[_flag((200, 20, 20))], max_new=6)
    assert s.last_ntok == 6

