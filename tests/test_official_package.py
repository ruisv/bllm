"""An official D-Robotics LLM 2.0 package directory, loaded as-is.

Point BLLM_TEST_OFFICIAL_DIR at an unpacked text package (e.g. `Qwen3-0.6B_w8/`:
one `*_language_*.hbm` + one `*_embed_tokens.bin`, nothing else); everything skips
otherwise. The synthesized manifest goes to a private cache root so the test never
reads a stale one.

One model per process, as in test_native_llm.py: the module-scoped fixture is
released before the next module loads anything.
"""

import gc
import json
import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

OFFICIAL_DIR = os.environ.get("BLLM_TEST_OFFICIAL_DIR")


@pytest.fixture(scope="module")
def official(tmp_path_factory):
    if not (bllm._HAVE_NATIVE and OFFICIAL_DIR):
        pytest.skip("set BLLM_TEST_OFFICIAL_DIR")
    cache = tmp_path_factory.mktemp("bllm_cache")
    old = os.environ.get("BLLM_CACHE")
    os.environ["BLLM_CACHE"] = str(cache)
    session = bllm.load(OFFICIAL_DIR)
    yield session, cache
    del session
    gc.collect()
    if old is None:
        os.environ.pop("BLLM_CACHE", None)
    else:
        os.environ["BLLM_CACHE"] = old


def test_loads_from_the_package_dir_alone(official):
    s, cache = official
    # ctx / quant / board are in the name, read off the package rather than guessed
    assert "official LLM 2.0" in s.name and "ctx" in s.name
    manifests = list(cache.glob("native/*/model.json"))
    assert len(manifests) == 1
    cfg = json.loads(manifests[0].read_text())
    assert cfg["arch"] == "dense" and cfg["embed"] == "embed_tokens.bin"
    assert cfg["eos"], "stop tokens must come from the package description"


def test_chat_answers(official):
    s, _ = official
    s.reset()
    if hasattr(s, "set_thinking"):
        s.set_thinking(False)
    reply = s.chat("What is the capital of France? Answer in one word.", max_new=24)
    assert "Paris" in reply


def test_second_load_reads_the_cache_not_the_hbm(official, monkeypatch):
    _, cache = official
    from bllm import _modelmeta

    def boom(path):
        raise AssertionError("the description was read again on a cache hit")

    monkeypatch.setattr(_modelmeta, "_read_desc_native", boom)
    d = _modelmeta.official_dir_for(OFFICIAL_DIR, cache_root=cache / "native")
    assert (d / "model.json").is_file()
