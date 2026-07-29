"""Synthesize a native model package from a bare `.hbm` + a tokenizer dir.

This is the libxlm retirement: native needs a `model.json`
directory, but an OE-LLM package is a bare `.hbm` + a tokenizer dir + a `config.json`. This
module bridges the gap at load time — it reads the tokenizer, resolves the stop tokens and
the ChatML markers exactly as the offline `scripts/make_model_dir.py` does, writes a
`model.json` into a small cache dir of symlinks, and hands that dir to `NativeSession`. So
`bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="...")` runs native with no hand-authored
manifest, the same call that used to route to libxlm.

Only the **dense** arch is synthesized here: for a dense text model the embedding table and
rope live inside the `.hbm`, so a bare `.hbm` + tokenizer is complete. Hybrid (Qwen3.5) and
omni packages additionally need the host embed table (and towers), which the OE-LLM package
ships as separate files — those still go through `make_model_dir.py`, which knows the arch.

Pure stdlib so it imports on any host. The eos/ChatML resolution is intentionally identical
to `scripts/make_model_dir.py` (its sibling); the two will be unified once the loader is
board-proven.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Stop-token candidates, most-specific first — only ever matched against tokens the tokenizer
# DECLARES as special, never plain BPE entries (Qwen2.5's vocab has a literal "</s>" at 128247
# that would make the model never stop). Mirrors make_model_dir.EOS_CANDIDATES.
EOS_CANDIDATES = ["<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<|end|>", "<|eot_id|>"]
CHATML_START, CHATML_END = "<|im_start|>", "<|im_end|>"
SENTENCEPIECE_EOS = ["</s>", "<eos>"]


def _load_tokenizer(tokenizer_json: Path):
    """-> (declared_specials, bpe_vocab, has_added_tokens)."""
    with tokenizer_json.open(encoding="utf-8") as f:
        tk = json.load(f)
    if "model" not in tk:
        raise ValueError(f"{tokenizer_json}: not a tokenizer.json (no `model` key)")
    added = tk.get("added_tokens", [])
    flagged = {t["content"]: t["id"] for t in added if t.get("special")}
    declared = flagged or {t["content"]: t["id"] for t in added}
    return declared, dict(tk.get("model", {}).get("vocab", {})), bool(added)


def _resolve_eos(tokenizer_json: Path, declared: dict, vocab: dict, has_added: bool):
    """Stop-token ids from the most authoritative source. HF generates with
    generation_config.json:eos_token_id, so that wins outright."""
    cfg_dir = tokenizer_json.parent
    gen = cfg_dir / "generation_config.json"
    if gen.is_file():
        val = json.loads(gen.read_text(encoding="utf-8")).get("eos_token_id")
        if isinstance(val, int):
            return [val]
        if isinstance(val, list) and val:
            return list(val)

    ids: list = []
    stm = cfg_dir / "special_tokens_map.json"
    if stm.is_file():
        tok = json.loads(stm.read_text(encoding="utf-8")).get("eos_token")
        content = tok if isinstance(tok, str) else (tok or {}).get("content")
        if content and content in declared:
            ids.append(declared[content])
    for c in EOS_CANDIDATES:
        if c in declared and declared[c] not in ids:
            ids.append(declared[c])
    if ids:
        return ids

    if not has_added:                       # legacy sentencepiece (InternLM2 etc.)
        ids = [vocab[c] for c in SENTENCEPIECE_EOS if c in vocab]
        if ids:
            return ids

    raise ValueError(
        f"{tokenizer_json}: could not determine a stop token — pass eos explicitly "
        "(no generation_config.json, no declared special among "
        f"{', '.join(EOS_CANDIDATES)}).")


def dense_model_json(hbm_path: Path, tokenizer_dir: Path, *, name: "str | None" = None,
                     system: str = "You are a helpful assistant.",
                     eos: "list[int] | None" = None) -> dict:
    """Synthesize the `model.json` dict for a dense text model from its `.hbm` + tokenizer dir.

    Paths in the dict are the fixed names materialize() links to (`model.hbm`,
    `tokenizer.json`), not the sources — so the dict is location-independent.
    """
    tok = tokenizer_dir / "tokenizer.json"
    if not tok.is_file():
        raise FileNotFoundError(f"no tokenizer.json in {tokenizer_dir}")
    specials, vocab, has_added = _load_tokenizer(tok)
    if eos is None:
        eos = _resolve_eos(tok, specials, vocab, has_added)
    cfg: dict = {
        "name": name or hbm_path.stem,
        "arch": "dense",
        "backend": "native",
        "hbm": "model.hbm",
        "tokenizer": "tokenizer.json",
        "eos": list(eos),
    }
    # ChatML only when the tokenizer actually declares the markers; else raw completion.
    if CHATML_START in specials and CHATML_END in specials:
        cfg["chat"] = {"format": "chatml", "im_start": specials[CHATML_START],
                       "im_end": specials[CHATML_END], "system": system}
    else:
        cfg["chat"] = {"format": "none"}
    return cfg


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _cache_key(hbm_path: Path, tokenizer_dir: Path) -> str:
    """A stable key over the resolved paths + the .hbm mtime, so a rebuilt .hbm re-synthesizes."""
    try:
        mtime = int(hbm_path.stat().st_mtime)
    except OSError:
        mtime = 0
    raw = f"{hbm_path.resolve()}::{tokenizer_dir.resolve()}::{mtime}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def text_only_dir_for(vlm_dir: "str | Path", *,
                      cache_root: "str | Path | None" = None) -> Path:
    """Return a text-only view of a VLM package: same model.json with `visual` (and any
    audio tower) dropped, so `load()` routes to NativeSession instead of NativeVlmSession —
    the vision/audio tower is never instantiated, saving its BPU/ION footprint for a caller
    that only wants text. All other files (model.hbm, model_prefill.hbm, tokenizer.json,
    embed_tokens.bin) are symlinked, nothing is copied. This is `bllm.load(path, vision=False)`."""
    vlm_dir = Path(vlm_dir)
    manifest = vlm_dir / "model.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"no model.json in {vlm_dir}")
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    if cfg.get("arch") != "omni" and not (cfg.get("visual") or cfg.get("audio")):
        return vlm_dir   # already text-only, nothing to strip

    try:
        mtime = int(manifest.stat().st_mtime)
    except OSError:
        mtime = 0
    key = hashlib.sha1(f"{vlm_dir.resolve()}::{mtime}::textonly".encode()).hexdigest()[:16]
    root = Path(cache_root) if cache_root else Path(
        os.environ.get("BLLM_CACHE", Path.home() / ".cache" / "bllm")) / "native"
    out = root / key
    out.mkdir(parents=True, exist_ok=True)

    cfg.pop("visual", None)
    cfg.pop("audio", None)
    cfg.pop("mel_filters", None)
    if cfg.get("arch") == "omni":
        # NativeLlm hard-rejects arch="omni" outright (it's defined as "multimodal, load
        # via NativeVlm"), even with visual/audio absent — an omni package's text tower is
        # a plain dense decoder underneath, so that's what a stripped copy actually is.
        cfg["arch"] = "dense"
    for field in ("hbm", "hbm_prefill", "tokenizer", "embed"):
        rel = cfg.get(field)
        if rel:
            _link(vlm_dir / rel, out / rel)
    (out / "model.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    return out


def native_dir_for(hbm_path: "str | Path", tokenizer_dir: "str | Path", *,
                   name: "str | None" = None, system: str = "You are a helpful assistant.",
                   eos: "list[int] | None" = None,
                   cache_root: "str | Path | None" = None) -> Path:
    """Return a ready-to-load native model dir for a bare dense `.hbm` + tokenizer dir,
    synthesizing (and caching) it on first use. The dir holds `model.json` plus symlinks
    `model.hbm` and `tokenizer.json` pointing at the originals — nothing is copied."""
    hbm_path = Path(hbm_path)
    tokenizer_dir = Path(tokenizer_dir)
    if not hbm_path.is_file():
        raise FileNotFoundError(f"no such .hbm: {hbm_path}")
    if not tokenizer_dir.is_dir():
        raise NotADirectoryError(f"tokenizer_dir is not a directory: {tokenizer_dir}")

    root = Path(cache_root) if cache_root else Path(
        os.environ.get("BLLM_CACHE", Path.home() / ".cache" / "bllm")) / "native"
    out = root / _cache_key(hbm_path, tokenizer_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = dense_model_json(hbm_path, tokenizer_dir, name=name, system=system, eos=eos)
    _link(hbm_path, out / "model.hbm")
    _link(tokenizer_dir / "tokenizer.json", out / "tokenizer.json")
    (out / "model.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    return out
