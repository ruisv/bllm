"""Tests for `bllm.make_model_dir` — the only writer of `model.json`.

Pure Python: no board, no BPU. What is worth pinning down is the eos resolution,
because getting it wrong produces a model that never stops, and the failure looks
like a runtime bug rather than a manifest bug.

The module is loaded by path rather than imported as `bllm.make_model_dir`, so these
tests stay independent of the package's extension module (absent on a dev host).
`scripts/make_model_dir.py` is a thin wrapper around this same module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "python" / "bllm" / "make_model_dir.py"
_spec = importlib.util.spec_from_file_location("make_model_dir", _SCRIPT)
mmd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mmd)


def write_tokenizer(dir_: Path, *, added=None, vocab=None) -> Path:
    p = dir_ / "tokenizer.json"
    p.write_text(json.dumps({
        "added_tokens": added or [],
        "model": {"type": "BPE", "vocab": vocab or {"a": 0, "b": 1}},
    }), encoding="utf-8")
    return p


QWEN_ADDED = [
    {"content": "<|endoftext|>", "id": 151643, "special": True},
    {"content": "<|im_start|>", "id": 151644, "special": True},
    {"content": "<|im_end|>", "id": 151645, "special": True},
]
# The trap: modern tokenizers carry a LITERAL "</s>" as an ordinary merge token.
QWEN_VOCAB = {"a": 0, "</s>": 128247}


def resolve(tok_path: Path):
    declared, vocab, has_added = mmd.load_tokenizer(tok_path)
    return mmd.resolve_eos(tok_path, declared, vocab, has_added)


def test_generation_config_wins(tmp_path):
    tok = write_tokenizer(tmp_path, added=QWEN_ADDED, vocab=QWEN_VOCAB)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [151645, 151643]}')
    eos, src = resolve(tok)
    assert eos == [151645, 151643]
    assert "generation_config" in src


def test_generation_config_scalar_eos(tmp_path):
    tok = write_tokenizer(tmp_path, added=QWEN_ADDED, vocab=QWEN_VOCAB)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 151643}')
    assert resolve(tok)[0] == [151643]


def test_falls_back_to_special_tokens_map_and_declared_specials(tmp_path):
    tok = write_tokenizer(tmp_path, added=QWEN_ADDED, vocab=QWEN_VOCAB)
    (tmp_path / "special_tokens_map.json").write_text('{"eos_token": "<|im_end|>"}')
    eos, src = resolve(tok)
    assert eos[0] == 151645                     # the declared eos comes first
    assert 151643 in eos                        # and <|endoftext|> is still a stop token
    assert "special_tokens_map" in src


def test_never_matches_a_literal_bpe_token(tmp_path):
    """A "</s>" sitting in the BPE vocabulary is not a stop token."""
    tok = write_tokenizer(tmp_path, added=QWEN_ADDED, vocab=QWEN_VOCAB)
    eos, _ = resolve(tok)
    assert 128247 not in eos


def test_legacy_sentencepiece_uses_the_vocabulary(tmp_path):
    """InternLM2 declares no added_tokens at all; there </s> really is the eos."""
    tok = write_tokenizer(tmp_path, added=[], vocab={"<unk>": 0, "<s>": 1, "</s>": 2})
    eos, src = resolve(tok)
    assert eos == [2]
    assert "sentencepiece" in src


def test_unresolvable_eos_fails_loudly(tmp_path):
    tok = write_tokenizer(tmp_path, added=[{"content": "<|weird|>", "id": 7, "special": True}])
    with pytest.raises(SystemExit, match="stop token"):
        resolve(tok)


def test_not_a_tokenizer_json(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_text('{"nope": 1}')
    with pytest.raises(SystemExit, match="not a tokenizer.json"):
        mmd.load_tokenizer(p)


def _run(argv):
    old = sys.argv
    sys.argv = ["make_model_dir.py", *argv]
    try:
        return mmd.main()
    finally:
        sys.argv = old


def test_dense_manifest_end_to_end(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    tok = write_tokenizer(src, added=QWEN_ADDED, vocab=QWEN_VOCAB)
    (src / "generation_config.json").write_text('{"eos_token_id": [151645, 151643]}')
    hbm = src / "m.hbm"
    hbm.write_bytes(b"\0")

    out = tmp_path / "out"
    assert _run(["dense", str(out), "--hbm", str(hbm), "--tokenizer", str(tok),
                 "--cache-len", "1024"]) == 0
    cfg = json.loads((out / "model.json").read_text())
    assert cfg["arch"] == "dense" and cfg["backend"] == "native"
    assert cfg["eos"] == [151645, 151643]
    assert cfg["cache_len"] == 1024
    assert cfg["chat"] == {"format": "chatml", "im_start": 151644, "im_end": 151645,
                           "system": "You are a helpful assistant."}
    # payload is linked next to the manifest, under the names the manifest uses
    assert (out / "model.hbm").resolve() == hbm.resolve()
    assert (out / "tokenizer.json").resolve() == tok.resolve()


def test_chat_format_none_without_chatml_markers(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    tok = write_tokenizer(src, added=[{"content": "<|endoftext|>", "id": 9, "special": True}])
    hbm = src / "m.hbm"
    hbm.write_bytes(b"\0")
    assert _run(["dense", str(tmp_path / "o"), "--hbm", str(hbm), "--tokenizer", str(tok)]) == 0
    cfg = json.loads((tmp_path / "o" / "model.json").read_text())
    assert cfg["chat"] == {"format": "none"}


PHI_ADDED = [
    {"content": "<|endoftext|>", "id": 199999, "special": True},
    {"content": "<|end|>", "id": 200020, "special": True},
    {"content": "<|user|>", "id": 200021, "special": True},
    {"content": "<|assistant|>", "id": 200019, "special": True},
    {"content": "<|system|>", "id": 200022, "special": True},
]


def test_chat_format_phi_from_role_markers(tmp_path):
    """Phi-4-mini declares <|user|>/<|assistant|>/<|end|> instead of ChatML markers."""
    src = tmp_path / "src"
    src.mkdir()
    tok = write_tokenizer(src, added=PHI_ADDED)
    (src / "generation_config.json").write_text('{"eos_token_id": [200020, 199999]}')
    hbm = src / "m.hbm"
    hbm.write_bytes(b"\0")
    assert _run(["dense", str(tmp_path / "o"), "--hbm", str(hbm), "--tokenizer", str(tok)]) == 0
    cfg = json.loads((tmp_path / "o" / "model.json").read_text())
    assert cfg["chat"] == {"format": "phi", "r_user": 200021, "r_assistant": 200019,
                           "r_end": 200020, "r_system": 200022,
                           "system": "You are a helpful assistant."}
    # <|end|> must be an eos, or a phi turn never stops.
    assert 200020 in cfg["eos"]


def test_omni_requires_its_towers(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    tok = write_tokenizer(src, added=QWEN_ADDED)
    (src / "generation_config.json").write_text('{"eos_token_id": 151645}')
    hbm = src / "m.hbm"
    hbm.write_bytes(b"\0")
    with pytest.raises(SystemExit):          # missing --embed and --visual
        _run(["omni", str(tmp_path / "o"), "--hbm", str(hbm), "--tokenizer", str(tok)])


def test_omni_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    tok = write_tokenizer(src, added=QWEN_ADDED)
    (src / "generation_config.json").write_text('{"eos_token_id": 151645}')
    for n in ("text.hbm", "visual.hbm", "audio.hbm", "embed.bin", "mel.txt"):
        (src / n).write_bytes(b"\0")
    out = tmp_path / "o"
    assert _run(["omni", str(out), "--hbm", str(src / "text.hbm"),
                 "--visual", str(src / "visual.hbm"), "--audio", str(src / "audio.hbm"),
                 "--mel-filters", str(src / "mel.txt"), "--embed", str(src / "embed.bin"),
                 "--tokenizer", str(tok)]) == 0
    cfg = json.loads((out / "model.json").read_text())
    assert cfg["arch"] == "omni"
    assert cfg["visual"] == "visual.hbm" and cfg["audio"] == "audio.hbm"
    assert cfg["mel_filters"] == "mel_filters_t.txt"
    assert cfg["rope_theta"] == 1e6 and cfg["mrope_section"] == [16, 24, 24]


def test_audio_needs_mel_filters(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    tok = write_tokenizer(src, added=QWEN_ADDED)
    (src / "generation_config.json").write_text('{"eos_token_id": 151645}')
    for n in ("text.hbm", "visual.hbm", "audio.hbm", "embed.bin"):
        (src / n).write_bytes(b"\0")
    with pytest.raises(SystemExit):
        _run(["omni", str(tmp_path / "o"), "--hbm", str(src / "text.hbm"),
              "--visual", str(src / "visual.hbm"), "--audio", str(src / "audio.hbm"),
              "--embed", str(src / "embed.bin"), "--tokenizer", str(tok)])
