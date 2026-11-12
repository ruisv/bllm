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

The official **LLM 2.0** packages are the other case handled here (see the section at the end):
they carry their tokenizer and configs inside the `.hbm`, so a package directory alone is
complete.

Pure stdlib so it imports on any host. The eos/ChatML resolution is intentionally identical
to `scripts/make_model_dir.py` (its sibling); the two will be unified once the loader is
board-proven.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# Stop-token candidates, most-specific first — only ever matched against tokens the tokenizer
# DECLARES as special, never plain BPE entries (Qwen2.5's vocab has a literal "</s>" at 128247
# that would make the model never stop). Mirrors make_model_dir.EOS_CANDIDATES.
EOS_CANDIDATES = ["<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<|end|>", "<|eot_id|>"]
CHATML_START, CHATML_END = "<|im_start|>", "<|im_end|>"
PHI_USER, PHI_ASSISTANT, PHI_SYSTEM, PHI_END = "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"
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


# ── official D-Robotics LLM 2.0 packages ────────────────────────────────────────────────
#
# An official 2.0 package is a directory of `<Model>_language_*.hbm` + `<Model>_embed_tokens.bin`
# (+ `<Model>_vision_*.hbm` for a VLM) and nothing else: no tokenizer, no config. Everything a
# runtime needs is a description JSON embedded in the language `.hbm` — every HF config file of
# the source model under its own key (`config`, `generation_config`, `tokenizer`,
# `tokenizer_config`, …) plus a `horizon` section the toolchain derives (chunk, cache length,
# vocab). So the synthesis below reads that once, writes `model.json` + `tokenizer.json` into a
# cache dir of symlinks, and the native engine takes it from there like any dense model.

_LANG_RE = re.compile(
    r"^(?P<model>.+?)_language_chunk_(?P<chunk>\d+)_cache_(?P<cache>\d+)_(?P<quant>w\d+)_"
    r"(?P<march>nash-[a-z])_corenum_(?P<cores>[\d_]+)\.hbm$")
_BOARD = {"nash-e": "RDK S100", "nash-m": "RDK S100P", "nash-p": "RDK S600"}


def find_official_package(pkg_dir: "str | Path") -> "dict | None":
    """`{'language', 'vision', 'embed'}` paths if `pkg_dir` is an official LLM 2.0 package
    (vision is None for a text model), else None. A dir with a model.json is ours, not theirs."""
    d = Path(pkg_dir)
    if not d.is_dir() or (d / "model.json").exists():
        return None
    lang = sorted(d.glob("*_language_*.hbm"))
    emb = sorted(d.glob("*_embed_tokens.bin"))
    if not lang or not emb:
        return None
    if len(lang) > 1 or len(emb) > 1:
        raise ValueError(
            f"{d}: {len(lang)} language .hbm and {len(emb)} embed tables — a package with more "
            "than one model (e.g. a speculative-decoding base + draft pair) is not supported")
    vis = sorted(d.glob("*_vision_*.hbm"))
    return {"language": lang[0], "vision": vis[0] if vis else None, "embed": emb[0]}


def _desc_chat_template(desc: dict) -> str:
    tc = desc.get("tokenizer_config") or {}
    t = tc.get("chat_template") or desc.get("chat_template") or ""
    if isinstance(t, dict):                      # chat_template.json: {"chat_template": "..."}
        t = t.get("chat_template", "")
    if isinstance(t, list):                      # HF named templates: [{"name", "template"}]
        named = {x.get("name"): x.get("template", "") for x in t if isinstance(x, dict)}
        t = named.get("default") or next(iter(named.values()), "")
    return t if isinstance(t, str) else ""


def _desc_specials(desc: dict) -> dict:
    added = (desc.get("tokenizer") or {}).get("added_tokens", [])
    flagged = {t["content"]: t["id"] for t in added if t.get("special")}
    return flagged or {t["content"]: t["id"] for t in added}


def _desc_eos(desc: dict, specials: dict) -> list:
    """HF generates with generation_config's eos_token_id, so that wins; then config (the text
    sub-config for a VLM); then the tokenizer's declared eos token."""
    cfg = desc.get("config") or {}
    for sec in (desc.get("generation_config") or {}, cfg, cfg.get("text_config") or {}):
        v = sec.get("eos_token_id")
        if isinstance(v, int):
            return [v]
        if isinstance(v, list) and v:
            return [int(x) for x in v]
    tok = (desc.get("tokenizer_config") or {}).get("eos_token")
    content = tok if isinstance(tok, str) else (tok or {}).get("content")
    ids = [specials[content]] if content in specials else []
    ids += [specials[c] for c in EOS_CANDIDATES if c in specials and specials[c] not in ids]
    if not ids:
        raise ValueError("could not determine a stop token from the package description")
    return ids


# Literal jinja source fragments. A template's string literals keep `\n` as the two characters
# backslash + n, so these are matched against the source text, not a rendering of it.
_THINK_CLOSED = "<think>\\n\\n</think>\\n\\n"
_THINK_OPEN = "<think>\\n"
_BOS_FIRST = re.compile(r"^\s*\{\{-?\s*bos_token\s*-?\}\}")
# The default system turn may sit inside a string literal (`\n` as two characters) or in the
# template's literal text (a real newline) — Qwen2.5 does the first, Qwen2.5-VL the second.
_CHATML_DEFAULT_SYSTEM = re.compile(r"<\|im_start\|>system(?:\\n|\n)([^'\"<{]+?)<\|im_end\|>")


def _desc_thinking(t: str) -> dict:
    """The generation-prompt tail after the assistant marker, with thinking on and off, as the
    template itself writes it. Defaults match what the official runtime does: thinking off
    unless the checkpoint only ever thinks."""
    if "enable_thinking" in t:
        if _THINK_CLOSED not in t:
            raise ValueError("the template has an enable_thinking switch BLLM does not "
                             "recognise (no closed empty <think> block for the off case)")
        on = "<think>\n" if re.search(r"enable_thinking is true\s*%\}\s*\{\{-?\s*'"
                                         + re.escape(_THINK_OPEN) + "'", t) else ""
        return {"think_on": on, "think_off": "<think>\n\n</think>\n\n", "thinking_default": False}
    if "assistant\\n" + _THINK_OPEN in t:              # a thinking-only checkpoint
        return {"think_on": "<think>\n", "think_off": "<think>\n", "thinking_default": True}
    return {"think_on": "", "think_off": "", "thinking_default": False}


def _desc_chat(desc: dict, specials: dict, system: "str | None") -> dict:
    """The template FAMILY, mapped onto a built-in chat format. The template itself is not
    rendered: BLLM's formats are what the prefix cache, stop matching and thinking control
    are built on, and every official package so far is ChatML or Phi-style. What does vary
    inside a family is read off the template — the think prefill, a BOS it always emits, the
    system prompt it falls back to — so the default rendering matches the official runtime's.

    `system=None` means the template's own default: none for most, a fixed one for Qwen2.5."""
    t = _desc_chat_template(desc)
    tc = desc.get("tokenizer_config") or {}
    if CHATML_START in t and CHATML_START in specials and CHATML_END in specials:
        if system is None:
            m = _CHATML_DEFAULT_SYSTEM.search(t)
            system = m.group(1) if m else ""
        chat = {"format": "chatml", "im_start": specials[CHATML_START],
                "im_end": specials[CHATML_END], "system": system}
        chat.update(_desc_thinking(t))
    elif (PHI_END in t and PHI_ASSISTANT in t and
          all(m in specials for m in (PHI_USER, PHI_ASSISTANT, PHI_END))):
        # Phi builds its role markers by concatenation ('<|' + role + '|>'), so only the
        # generation prompt and the turn end appear literally.
        chat = {"format": "phi", "r_user": specials[PHI_USER],
                "r_assistant": specials[PHI_ASSISTANT], "r_end": specials[PHI_END],
                "system": system or ""}
        if PHI_SYSTEM in specials:
            chat["r_system"] = specials[PHI_SYSTEM]
        chat.update({"think_on": "", "think_off": "", "thinking_default": False})
    else:
        raise ValueError(
            "this package's chat template is neither ChatML nor Phi-style, and BLLM renders "
            "chats with built-in formats only — write a model.json by hand (chat.format "
            "\"none\" gives raw completion)")
    if tc.get("add_bos_token") or _BOS_FIRST.match(t):
        bos = tc.get("bos_token") or ((desc.get("special_tokens_map") or {}).get("bos_token"))
        content = bos if isinstance(bos, str) else (bos or {}).get("content")
        if content in specials:
            chat["bos"] = specials[content]
    return chat


def _desc_vision(desc: dict) -> dict:
    """model.json's `vision` block for an official VLM package. The tower's geometry is the
    package's own preprocessor_config — which for Qwen2.5-VL is rewritten to
    temporal_patch_size 1, matching how its tower was compiled — and the official runtime
    stretches the image to the tower's side with OpenCV INTER_CUBIC."""
    pp = desc.get("preprocessor_config") or {}
    h = desc.get("horizon") or {}
    w, hh = h.get("image_width"), h.get("image_height")
    if w is not None and hh is not None and w != hh:
        raise ValueError(f"a {w}x{hh} vision tower: only square towers are supported")
    v = {"patch": int(pp.get("patch_size", 14)), "merge": int(pp.get("merge_size", 2)),
         "temporal": int(pp.get("temporal_patch_size", 2)), "resize": "cv_cubic"}
    # Qwen2.5-VL's ViT runs windowed attention, and the official export returns its tokens
    # in that window order — HF's closing reverse permutation is not in the graph (checked
    # on the official runtime's dump: it reorders on the host). Window side in merged tokens.
    vc = (desc.get("config") or {}).get("vision_config") or {}
    if vc.get("window_size"):
        v["window"] = int(vc["window_size"]) // (v["patch"] * v["merge"])
    for k, src in (("mean", "image_mean"), ("std", "image_std")):
        if src in pp:
            vals = [float(x) for x in pp[src]]
            if len(vals) != 3:
                raise ValueError(f"preprocessor_config.{src} must have 3 entries")
            v[k] = vals
    return v


def _desc_mrope(desc: dict) -> "tuple[list[int], bool]":
    """(mrope_section, interleaved) from the checkpoint config, wherever its transformers
    version put it (rope_parameters or rope_scaling, top level or text_config)."""
    cfg = desc.get("config") or {}
    for sec in (cfg.get("text_config") or {}, cfg):
        for key in ("rope_parameters", "rope_scaling"):
            r = sec.get(key) or {}
            if isinstance(r, dict) and r.get("mrope_section"):
                return [int(x) for x in r["mrope_section"]], bool(r.get("mrope_interleaved"))
    raise ValueError("a VLM package whose config names no mrope_section")


def official_model_json(desc: dict, hbm_name: str, *, system: "str | None" = None,
                        name: "str | None" = None, eos: "list[int] | None" = None,
                        vision: bool = False) -> dict:
    """The `model.json` dict for an official LLM 2.0 language graph, from its embedded description
    and file name. Pure — no file or BPU access — so it is testable anywhere. `vision` adds the
    package's vision tower (linked as `visual.hbm`)."""
    h = desc.get("horizon")
    if not isinstance(h, dict) or not h:
        raise ValueError(f"{hbm_name}: the description has no `horizon` section — not an "
                         "LLM 2.0 language graph")
    if not isinstance(desc.get("tokenizer"), dict):
        raise ValueError(f"{hbm_name}: the description carries no tokenizer")
    specials = _desc_specials(desc)
    m = _LANG_RE.match(hbm_name)
    ctx = h.get("decode_cache_len") or (int(m["cache"]) if m else 0)
    if name is None:
        # Context, quantization and board in the name: two packages of one model differ in
        # exactly those, and a bare "Qwen3-1.7B" says nothing about which one is loaded.
        parts = [m["model"] if m else Path(hbm_name).stem,
                 f"ctx{ctx}" if ctx else None,
                 m["quant"] if m else None,
                 _BOARD.get(m["march"]) if m else None]
        name = " · ".join(p for p in parts if p) + " (official LLM 2.0)"
    cfg = {
        "name": name,
        "arch": "dense",
        "backend": "native",
        "hbm": "model.hbm",
        "embed": "embed_tokens.bin",
        "tokenizer": "tokenizer.json",
        "eos": list(eos) if eos else _desc_eos(desc, specials),
        "chat": _desc_chat(desc, specials, system),
    }
    if ctx:
        cfg["cache_len"] = int(ctx)
    if vision:
        cfg["visual"] = "visual.hbm"
        cfg["vision"] = _desc_vision(desc)
        # The graph applies M-RoPE itself; recorded so the manifest says what it is.
        sec, inter = _desc_mrope(desc)
        cfg["mrope_section"] = sec
        cfg["mrope_interleaved"] = inter
    return cfg


def _read_desc_native(path: str) -> str:
    from ._bllm_native import read_hbm_desc      # board only; imported lazily
    return read_hbm_desc(path)


def write_official_dir(pkg: dict, out: Path, *, system: "str | None" = None,
                       name: "str | None" = None, eos: "list[int] | None" = None,
                       read_desc=None, copy: bool = False, vision: bool = True) -> Path:
    """Materialise a native model dir for an official package (`find_official_package`'s dict)
    in `out`. model.json is written LAST, so a dir that has one is complete. A VLM package
    gets its vision tower unless `vision=False` (then it is its text tower alone)."""
    lang = pkg["language"]
    raw = (read_desc or _read_desc_native)(str(lang))
    if not raw:
        raise ValueError(f"{lang.name}: no embedded description — not an LLM 2.0 graph")
    desc = json.loads(raw)
    with_vision = vision and pkg.get("vision") is not None
    cfg = official_model_json(desc, lang.name, system=system, name=name, eos=eos,
                              vision=with_vision)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(desc["tokenizer"], ensure_ascii=False),
                                        encoding="utf-8")
    for key in ("tokenizer_config", "generation_config"):   # kept for reference, never read
        if isinstance(desc.get(key), dict):
            (out / f"{key}.json").write_text(json.dumps(desc[key], indent=2, ensure_ascii=False)
                                             + "\n", encoding="utf-8")
    files = [(lang, out / "model.hbm"), (pkg["embed"], out / "embed_tokens.bin")]
    if with_vision:
        files.append((pkg["vision"], out / "visual.hbm"))
    for src, dst in files:
        if copy:
            import shutil
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.copy2(src, dst)
        else:
            _link(src, dst)
    tmp = out / "model.json.tmp"
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out / "model.json")
    return out


def official_dir_for(pkg_dir: "str | Path", *, vision: bool = True,
                     system: "str | None" = None, name: "str | None" = None,
                     eos: "list[int] | None" = None, cache_root: "str | Path | None" = None,
                     read_desc=None) -> Path:
    """A ready-to-load native model dir for an official LLM 2.0 package, synthesized on first
    use and cached: reading the description loads the GB-scale .hbm, a cache hit reads nothing.

    A VLM package loads with its vision tower; vision=False gives its text tower alone."""
    pkg = find_official_package(pkg_dir)
    if pkg is None:
        raise ValueError(f"{pkg_dir}: not an official LLM 2.0 package (expected one "
                         "*_language_*.hbm and one *_embed_tokens.bin)")
    lang = pkg["language"]
    m = _LANG_RE.match(lang.name)
    if m and m["march"] == "nash-p":
        # Checked before the .hbm is opened: on an S100P the load would fail deep in hbDNN
        # with no hint why, and on an S600 the engine would refuse the fp16 I/O anyway.
        raise NotImplementedError(
            f"{lang.name} is an S600 (nash-p) package; its graphs take fp16 embeddings and "
            "return fp16 logits, which BLLM does not drive yet — the S100P (nash-m) packages "
            "load")
    st = lang.stat()
    vis = pkg["vision"] if vision else None
    raw = "::".join([str(lang.resolve()), str(st.st_size), str(int(st.st_mtime)),
                     str(pkg["embed"].resolve()), str(vis.resolve()) if vis else "",
                     "\x00" if system is None else system, name or "",
                     ",".join(map(str, eos or [])), "official2.2"])
    root = Path(cache_root) if cache_root else Path(
        os.environ.get("BLLM_CACHE", Path.home() / ".cache" / "bllm")) / "native"
    out = root / hashlib.sha1(raw.encode()).hexdigest()[:16]
    if (out / "model.json").is_file():
        return out
    return write_official_dir(pkg, out, system=system, name=name, eos=eos, read_desc=read_desc,
                              vision=vision)
