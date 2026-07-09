#!/usr/bin/env python3
"""Assemble a BLLM model directory: symlinks + a validated `model.json` manifest.

`model.json` is the seam between conversion (offline, on an x86 host) and the
runtime (which must never guess). This script is the only thing that should ever
write one — hand-editing manifests is how a model dir ends up with an eos id that
belongs to a different tokenizer.

    # dense text model (Qwen2.5 / DeepSeek / Qwen3 / GLM / Phi …)
    make_model_dir.py dense  out_dir --hbm qwen2.5-1.5b.hbm --tokenizer tokenizer.json

    # hybrid Gated-DeltaNet / SSM (Qwen3.5)
    make_model_dir.py hybrid out_dir --hbm qwen3.5-0.8b.hbm --tokenizer tokenizer.json \
        --embed embed_tokens_fp16.bin --graph qwen35 --rope-theta 1e7

    # multimodal (Qwen2.5-Omni): text + vision (+ audio) towers
    make_model_dir.py omni   out_dir --hbm Text.hbm --visual Visual.hbm \
        --audio Audio.hbm --mel-filters mel_filters_t.txt \
        --embed embed_tokens.bin --tokenizer tokenizer.json

`--eos` is resolved from the most authoritative source present, and the tool prints
which one it used: `generation_config.json:eos_token_id` (what HF generates with),
then `special_tokens_map.json` + the tokenizer's declared specials, then — only for
legacy sentencepiece tokenizers, which declare no added_tokens — the vocabulary.
Plain BPE entries are never matched: Qwen2.5's vocabulary contains a literal "</s>"
at id 128247, and using it as eos yields a model that never stops.

Pass `--copy` to copy the payload instead of symlinking it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Fallback stop tokens, most-specific first. Only ever matched against tokens the
# tokenizer DECLARES as added/special — never plain BPE entries. Qwen2.5's vocabulary
# contains a literal "</s>" at id 128247; using it as eos yields a model that never stops.
EOS_CANDIDATES = ["<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<|end|>", "<|eot_id|>"]
CHATML_START, CHATML_END = "<|im_start|>", "<|im_end|>"
# Legacy sentencepiece tokenizers carry no added_tokens at all (InternLM2); there the
# stop token genuinely does live in the vocabulary.
SENTENCEPIECE_EOS = ["</s>", "<eos>"]


def load_tokenizer(tokenizer_json: Path) -> tuple[dict, dict, bool]:
    """-> (declared_tokens, bpe_vocab, has_added_tokens)."""
    with tokenizer_json.open(encoding="utf-8") as f:
        tk = json.load(f)
    if "model" not in tk:
        raise SystemExit(f"{tokenizer_json}: not a tokenizer.json (no `model` key)")
    added = tk.get("added_tokens", [])
    flagged = {t["content"]: t["id"] for t in added if t.get("special")}
    declared = flagged or {t["content"]: t["id"] for t in added}   # older files omit `special`
    return declared, dict(tk.get("model", {}).get("vocab", {})), bool(added)


def resolve_eos(tokenizer_json: Path, declared: dict, vocab: dict, has_added: bool):
    """Stop-token ids, from the most authoritative source available.

    HF generates with `generation_config.json:eos_token_id`, so that wins outright.
    """
    cfg_dir = tokenizer_json.parent

    gen = cfg_dir / "generation_config.json"
    if gen.is_file():
        val = json.loads(gen.read_text(encoding="utf-8")).get("eos_token_id")
        if isinstance(val, int):
            return [val], "generation_config.json"
        if isinstance(val, list) and val:
            return list(val), "generation_config.json"

    ids: list[int] = []
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
        return ids, "special_tokens_map.json + tokenizer specials" if stm.is_file() else "tokenizer specials"

    if not has_added:                       # legacy sentencepiece
        ids = [vocab[c] for c in SENTENCEPIECE_EOS if c in vocab]
        if ids:
            return ids, "sentencepiece vocabulary"

    raise SystemExit(
        f"{tokenizer_json}: could not determine a stop token "
        f"(no generation_config.json, no declared special among {', '.join(EOS_CANDIDATES)}). "
        "Pass --eos explicitly.")


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if not src.exists():
        raise SystemExit(f"missing: {src}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arch", choices=["dense", "hybrid", "omni"])
    ap.add_argument("out_dir")
    ap.add_argument("--hbm", required=True, help="text .hbm (prefill+decode)")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--embed", help="host embedding table (required: hybrid, omni)")
    ap.add_argument("--visual", help="vision tower .hbm (omni)")
    ap.add_argument("--audio", help="audio tower .hbm (omni)")
    ap.add_argument("--mel-filters", help="mel_filters_t.txt (omni audio)")
    ap.add_argument("--name", help="display name (default: out_dir basename)")
    ap.add_argument("--graph", help="graph name inside the .hbm (hybrid; default qwen35)")
    ap.add_argument("--rope-theta", type=float, help="rope base (default: 1e7 hybrid, 1e6 omni)")
    ap.add_argument("--cache-len", type=int, default=0, help="informational")
    ap.add_argument("--mrope-section", type=int, nargs=3, metavar=("T", "H", "W"),
                    help="omni rope dim split (default 16 24 24)")
    ap.add_argument("--eos", type=int, nargs="+", help="override the stop-token ids")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--copy", action="store_true", help="copy payload instead of symlinking")
    args = ap.parse_args()

    if args.arch in ("hybrid", "omni") and not args.embed:
        ap.error(f"{args.arch} needs --embed (the host embedding table)")
    if args.arch == "omni" and not args.visual:
        ap.error("omni needs --visual (the vision tower .hbm)")
    if bool(args.audio) != bool(args.mel_filters):
        ap.error("--audio and --mel-filters go together")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_src = Path(args.tokenizer)
    specials, vocab, has_added = load_tokenizer(tokenizer_src)

    if args.eos:
        eos, eos_src = args.eos, "--eos"
    else:
        eos, eos_src = resolve_eos(tokenizer_src, specials, vocab, has_added)

    cfg: dict = {
        "name": args.name or out.name,
        "arch": args.arch,
        "backend": "native",
        "hbm": "model.hbm",
        "tokenizer": "tokenizer.json",
        "eos": eos,
    }
    payload = {"model.hbm": Path(args.hbm), "tokenizer.json": tokenizer_src}

    if args.cache_len:
        cfg["cache_len"] = args.cache_len

    # ChatML only when the tokenizer actually has the markers.
    if CHATML_START in specials and CHATML_END in specials:
        cfg["chat"] = {"format": "chatml", "im_start": specials[CHATML_START],
                       "im_end": specials[CHATML_END], "system": args.system}
    else:
        cfg["chat"] = {"format": "none"}
        print(f"[warn] {tokenizer_src.name} has no ChatML markers — chat.format=none "
              "(raw completion only)", file=sys.stderr)

    if args.arch in ("hybrid", "omni"):
        cfg["embed"] = "embed_tokens.bin"
        payload["embed_tokens.bin"] = Path(args.embed)
    if args.arch == "hybrid":
        cfg["graph"] = args.graph or "qwen35"
        cfg["rope_theta"] = args.rope_theta or 1e7
    if args.arch == "omni":
        cfg["visual"] = "visual.hbm"
        payload["visual.hbm"] = Path(args.visual)
        cfg["rope_theta"] = args.rope_theta or 1e6
        cfg["mrope_section"] = list(args.mrope_section or (16, 24, 24))
        if args.audio:
            cfg["audio"] = "audio.hbm"
            cfg["mel_filters"] = "mel_filters_t.txt"
            payload["audio.hbm"] = Path(args.audio)
            payload["mel_filters_t.txt"] = Path(args.mel_filters)

    for dst_name, src in payload.items():
        link_or_copy(src, out / dst_name, args.copy)

    (out / "model.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    print(f"[ok] {out}/model.json   (eos from {eos_src})")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"\nload it:  bllm.NativeSession(\"{out}\")" if args.arch != "omni"
          else f"\nload it:  bllm.NativeVlmSession(\"{out}\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
