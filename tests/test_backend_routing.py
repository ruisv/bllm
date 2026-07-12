"""bllm.load() routing — native-only now (the libxlm backend was removed; the design notes).
The decisions and error paths, no model needed. A test that actually loads a model and
confirms it runs lives in test_native_llm.py (hardware)."""

import json

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")


def _pkg(tmp_path, arch, backend="auto"):
    d = tmp_path / f"{arch}_{backend}"
    d.mkdir()
    (d / "model.json").write_text(json.dumps(
        {"name": "t", "arch": arch, "backend": backend, "hbm": "model.hbm",
         "tokenizer": "tokenizer.json"}))
    return str(d)


def test_read_manifest_distinguishes_package_from_bare_hbm(tmp_path):
    native = _pkg(tmp_path, "dense")
    assert bllm._read_manifest(native)["arch"] == "dense"
    hbm = tmp_path / "bare.hbm"
    hbm.write_bytes(b"\x00")
    assert bllm._read_manifest(str(hbm)) is None
    assert bllm._read_manifest(str(tmp_path / "does_not_exist")) is None


def test_libxlm_is_a_hard_error(tmp_path):
    # The backend is gone; a leftover pin (call or manifest) must fail loudly, not be
    # silently reinterpreted as native.
    with pytest.raises(ValueError, match="libxlm.*removed|removed.*libxlm"):
        bllm.load(_pkg(tmp_path, "dense"), backend="libxlm")
    if bllm._HAVE_NATIVE:
        with pytest.raises(ValueError, match="libxlm"):
            bllm.load(_pkg(tmp_path, "dense", "libxlm"))


def test_load_rejects_impossible_requests(tmp_path):
    if not bllm._HAVE_NATIVE:
        pytest.skip("native backend not built")
    # embed is an encoder, not a chat session.
    with pytest.raises(ValueError, match="encoder"):
        bllm.load(_pkg(tmp_path, "embed"))
    # A bare .hbm needs a tokenizer dir to synthesize its manifest (Blocker A).
    hbm = tmp_path / "bare.hbm"
    hbm.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="tokenizer_dir"):
        bllm.load(str(hbm))


def test_available_backends_is_native_only():
    assert bllm.available_backends() in ((), ("native",))
    assert "libxlm" not in bllm.available_backends()
