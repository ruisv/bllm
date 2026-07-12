"""bllm.load() backend routing — the pure decision and the error paths, no model needed.

The rule itself is mirrored from C++ bllm::chooseBackend (tests/test_choose_backend.cc);
this checks the Python side stays in step and that load() dispatches / rejects correctly.
A test that actually loads a model and confirms the chosen runtime runs lives in
test_native_llm.py (hardware)."""

import json
import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")


def test_choose_backend_rule():
    ch = bllm._choose_backend
    # Capability boundary: these can only run native.
    for arch in ("hybrid", "omni", "embed"):
        assert ch(arch, "auto") == "native"
        assert ch(arch, "native") == "native"
        with pytest.raises(ValueError):
            ch(arch, "libxlm")            # impossible request → error, not silent override
    # Dense runs on either; the field decides, default native.
    assert ch("dense", "auto") == "native"
    assert ch("dense", "native") == "native"
    assert ch("dense", "libxlm") == "libxlm"
    assert ch("", "auto") == "native"     # empty arch treated as dense


def _pkg(tmp_path, arch, backend="auto"):
    d = tmp_path / f"{arch}_{backend}"
    d.mkdir()
    (d / "model.json").write_text(json.dumps(
        {"name": "t", "arch": arch, "backend": backend, "hbm": "model.hbm",
         "tokenizer": "tokenizer.json"}))
    return str(d)


def test_read_manifest_distinguishes_native_from_libxlm(tmp_path):
    native = _pkg(tmp_path, "dense")
    assert bllm._read_manifest(native)["arch"] == "dense"
    # A bare .hbm path is not a native package.
    hbm = tmp_path / "bare.hbm"
    hbm.write_bytes(b"\x00")
    assert bllm._read_manifest(str(hbm)) is None
    assert bllm._read_manifest(str(tmp_path / "does_not_exist")) is None


def test_load_rejects_impossible_requests(tmp_path):
    # Forcing libxlm on a native-only arch must fail before any model load.
    with pytest.raises(ValueError, match="native only|only runs on the native"):
        bllm.load(_pkg(tmp_path, "omni", "auto"), backend="libxlm")
    # embed is an encoder, not a chat session.
    if bllm._HAVE_NATIVE:
        with pytest.raises(ValueError, match="encoder"):
            bllm.load(_pkg(tmp_path, "embed", "auto"))
    # Native on a bare .hbm now synthesizes a manifest (Blocker A) but needs a tokenizer dir.
    hbm = tmp_path / "bare.hbm"
    hbm.write_bytes(b"\x00")
    if bllm._HAVE_NATIVE:
        with pytest.raises(ValueError, match="tokenizer_dir"):
            bllm.load(str(hbm), backend="native")


def test_load_reports_a_missing_backend(tmp_path):
    if not bllm._HAVE_LIBXLM:
        with pytest.raises(RuntimeError, match="no libxlm backend"):
            bllm.load(_pkg(tmp_path, "dense", "libxlm"))
