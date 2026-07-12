"""bllm._modelmeta — the Blocker-A synthesizer that turns a bare .hbm + tokenizer dir into a
native model package. Pure metadata work, so this runs off-board (no model load)."""

import json

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")
mm = pytest.importorskip("bllm._modelmeta")


def _tokenizer(tmp_path, *, chatml=True, added=True, gen_eos=None):
    """Write a minimal tokenizer.json (+ optional generation_config.json) and return its dir."""
    d = tmp_path / "tok"
    d.mkdir()
    at = []
    if added:
        at = [{"id": 151643, "content": "<|endoftext|>", "special": True}]
        if chatml:
            at += [{"id": 151644, "content": "<|im_start|>", "special": True},
                   {"id": 151645, "content": "<|im_end|>", "special": True}]
    vocab = {"a": 0, "b": 1}
    if not added:                         # legacy sentencepiece style
        vocab["</s>"] = 2
    (d / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": at}))
    if gen_eos is not None:
        (d / "generation_config.json").write_text(json.dumps({"eos_token_id": gen_eos}))
    return d


def test_dense_chatml_from_tokenizer_specials(tmp_path):
    tok = _tokenizer(tmp_path, chatml=True)
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x00")
    cfg = mm.dense_model_json(hbm, tok)
    assert cfg["arch"] == "dense" and cfg["backend"] == "native"
    assert cfg["chat"] == {"format": "chatml", "im_start": 151644, "im_end": 151645,
                           "system": "You are a helpful assistant."}
    # No generation_config → declared specials in EOS_CANDIDATES order: im_end (the turn
    # stop) first, then endoftext.
    assert cfg["eos"] == [151645, 151643]


def test_generation_config_eos_wins(tmp_path):
    tok = _tokenizer(tmp_path, chatml=True, gen_eos=[151645, 151643])
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x00")
    cfg = mm.dense_model_json(hbm, tok)
    assert cfg["eos"] == [151645, 151643]   # HF's list, verbatim


def test_no_chatml_markers_falls_back_to_raw(tmp_path):
    tok = _tokenizer(tmp_path, chatml=False)
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x00")
    cfg = mm.dense_model_json(hbm, tok)
    assert cfg["chat"] == {"format": "none"}


def test_sentencepiece_eos_from_vocab(tmp_path):
    tok = _tokenizer(tmp_path, chatml=False, added=False)   # no added_tokens (InternLM2-like)
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x00")
    cfg = mm.dense_model_json(hbm, tok)
    assert cfg["eos"] == [2]              # </s> from the vocabulary


def test_native_dir_for_materializes_symlinks(tmp_path):
    tok = _tokenizer(tmp_path, chatml=True)
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x01\x02")
    out = mm.native_dir_for(hbm, tok, cache_root=tmp_path / "cache")
    assert (out / "model.json").is_file()
    assert (out / "model.hbm").resolve() == hbm.resolve()
    assert (out / "tokenizer.json").resolve() == (tok / "tokenizer.json").resolve()
    cfg = json.loads((out / "model.json").read_text())
    assert cfg["arch"] == "dense"
    # Loading the same pair again is idempotent (same cache key).
    assert mm.native_dir_for(hbm, tok, cache_root=tmp_path / "cache") == out


def test_native_dir_for_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        mm.native_dir_for(tmp_path / "nope.hbm", tmp_path)
