#!/usr/bin/env python3
"""Assemble a BLLM model directory: symlinks + a validated `model.json` manifest.

`model.json` is the seam between conversion (offline, on an x86 host) and the
runtime (which must never guess). This script is the only thing that should ever
write one — hand-editing manifests is how a model dir ends up with an eos id that
belongs to a different tokenizer.

Ships with the package, so `conda install bllm` puts it on PATH:

    bllm-make-model-dir ...          # or: python -m bllm.make_model_dir ...

    # dense text model (Qwen2.5 / DeepSeek / InternLM2 / GLM / Phi …)
    bllm-make-model-dir dense  out_dir --hbm qwen2.5-1.5b.hbm --tokenizer tokenizer.json

    # hybrid Gated-DeltaNet / SSM (Qwen3.5)
    bllm-make-model-dir hybrid out_dir --hbm qwen3.5-0.8b.hbm --tokenizer tokenizer.json \
        --embed embed_tokens_fp16.bin --graph qwen35 --rope-theta 1e7

    # multimodal (Qwen2.5-Omni): text + vision (+ audio) towers
    bllm-make-model-dir omni   out_dir --hbm Text.hbm --visual Visual.hbm \
        --audio Audio.hbm --mel-filters mel_filters_t.txt \
        --embed embed_tokens.bin --tokenizer tokenizer.json

    # multimodal hybrid (Qwen3.5): the SAME hybrid text .hbm plus a vision tower.
    # Qwen3.5 ships as image-text-to-text; a text-only package is just a build that
    # dropped the vision half. Patch geometry, normalization and the interleaved
    # mrope layout are NOT discoverable from the .hbm, so they are all required.
    bllm-make-model-dir hybrid out_dir --hbm qwen3.5-0.8b.hbm --embed embed_tokens_fp16.bin \
        --tokenizer tokenizer.json --visual qwen35_visual_448.hbm \
        --mrope-section 11 11 10 --mrope-interleaved \
        --vision-patch 16 --vision-mean 0.5 --vision-std 0.5

For the official D-Robotics Omni release, every input already exists in the SDK —
note that `embed_tokens.bin` sits in `model/` under a name that does not mention
Omni, and is a separate download in `resolve_model_nash-m.txt`:

    R=<sdk>/oellm_runtime
    bllm-make-model-dir omni ~/models/qwen2.5-omni-3b \
        --hbm         $R/model/Qwen2.5_Omni_3B_Text.hbm \
        --visual      $R/model/Qwen2.5_Omni_3B_Visual.hbm \
        --audio       $R/model/Qwen2.5_Omni_3B_Audio.hbm \
        --embed       $R/model/embed_tokens.bin \
        --mel-filters $R/config/Qwen2.5_Omni_3B_config/mel_filters_t.txt \
        --tokenizer   $R/config/Qwen2.5_Omni_3B_config/tokenizer.json \
        --cache-len 2048

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
# Phi-style role markers: each turn is `<role>content<|end|>`, no role text (the marker
# is the role). Detected when the tokenizer declares <|user|>/<|assistant|>/<|end|>.
PHI_USER, PHI_ASSISTANT, PHI_SYSTEM, PHI_END = "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"
# Legacy sentencepiece tokenizers carry no added_tokens at all (InternLM2); there the
# stop token genuinely does live in the vocabulary.
SENTENCEPIECE_EOS = ["</s>", "<eos>"]

# The official Omni release splits its inputs across model/ and config/, and the host
# embedding table is named without any hint that it belongs to Omni — so a user who
# downloads the three `*_Omni_3B_*.hbm` files reasonably believes the set is complete.
# Say where each missing piece comes from rather than just naming the flag.
OMNI_SOURCES = {
    "--embed": "model/embed_tokens.bin (a SEPARATE 1.2 GB download in the official "
               "resolve_model_nash-m.txt — its name does not mention Omni, so it is "
               "easy to miss alongside the three *_Omni_3B_*.hbm files)",
    "--visual": "model/Qwen2.5_Omni_3B_Visual.hbm",
    "--audio": "model/Qwen2.5_Omni_3B_Audio.hbm",
    "--mel-filters": "config/Qwen2.5_Omni_3B_config/mel_filters_t.txt",
}


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bllm-make-model-dir",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arch", choices=["dense", "hybrid", "omni", "pi05"])
    ap.add_argument("out_dir")
    ap.add_argument("--hbm", required=True,
                    help="text .hbm (prefill+decode); for pi05, the PaliGemma prefill")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    # --- pi05 (openpi VLA) ---------------------------------------------------
    ap.add_argument("--expert", help="pi05: action-expert .hbm")
    ap.add_argument("--cond-table", help="pi05: cond_table.f32")
    ap.add_argument("--norm-stats", help="pi05: norm_stats.json from export_pi05_package.py")
    ap.add_argument("--cameras", type=int, default=2, help="pi05: camera slots in the prefix")
    ap.add_argument("--prompt-len", type=int, default=32, help="pi05: prompt slots in the prefix")
    ap.add_argument("--horizon", type=int, default=10, help="pi05: action chunk length")
    ap.add_argument("--steps", type=int, default=10, help="pi05: denoising steps")
    ap.add_argument("--embed", help="host embedding table (required: hybrid, omni)")
    ap.add_argument("--visual", help="vision tower .hbm (omni, or a hybrid VLM like Qwen3.5)")
    ap.add_argument("--vision-patch", type=int, help="vision patch size (14 omni, 16 Qwen3.5)")
    ap.add_argument("--vision-merge", type=int, help="spatial merge size (default 2)")
    ap.add_argument("--vision-mean", type=float, nargs="+",
                    help="pixel mean, 1 or 3 values (CLIP for omni, 0.5 for Qwen3.5)")
    ap.add_argument("--vision-std", type=float, nargs="+", help="pixel std, 1 or 3 values")
    ap.add_argument("--mrope-interleaved", action="store_true",
                    help="rope layout is [THWTHW...] (Qwen3.5) rather than contiguous runs (omni)")
    ap.add_argument("--audio", help="audio tower .hbm (omni)")
    ap.add_argument("--mel-filters", help="mel_filters_t.txt (omni audio)")
    ap.add_argument("--name", help="display name (default: out_dir basename)")
    ap.add_argument("--graph", help="graph name inside the .hbm (hybrid; default qwen35)")
    ap.add_argument("--hbm-prefill", help="optional seq_len=N prefill graph .hbm (hybrid). "
                                          "Amortizes the weight stream over N tokens; without "
                                          "it prefill runs one token at a time.")
    ap.add_argument("--rope-theta", type=float, help="rope base (default: 1e7 hybrid, 1e6 omni)")
    ap.add_argument("--cache-len", type=int, default=0, help="informational")
    ap.add_argument("--bpu-cores", type=int, choices=[1, 2, 4],
                    help="BPU core count the .hbm was compiled for (S600/nash-p; a "
                         "core_num=N hbm must be bound to N specific cores at runtime)")
    ap.add_argument("--mrope-section", type=int, nargs=3, metavar=("T", "H", "W"),
                    help="rope dim split (omni default 16 24 24; Qwen3.5 is 11 11 10)")
    ap.add_argument("--eos", type=int, nargs="+", help="override the stop-token ids")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--copy", action="store_true", help="copy payload instead of symlinking")
    args = ap.parse_args(argv)

    if args.arch == "pi05":
        # Each of these is separately fatal at load, so fail here where the
        # message can name the flag rather than the manifest field.
        for flag, val in (("--visual (SigLIP tower .hbm)", args.visual),
                          ("--expert (action expert .hbm)", args.expert),
                          ("--embed (embedding table, pre-scaled)", args.embed),
                          ("--cond-table (cond_table.f32)", args.cond_table),
                          ("--norm-stats (norm_stats.json)", args.norm_stats)):
            if not val:
                ap.error(f"pi05 needs {flag} — all of these come out of "
                         f"host_toolchain/convert/pi05/export_pi05_package.py")
    if args.arch in ("hybrid", "omni") and not args.embed:
        hint = f"  For the official Omni release it is {OMNI_SOURCES['--embed']}." \
            if args.arch == "omni" else ""
        ap.error(f"{args.arch} needs --embed (the host embedding table).{hint}")
    if args.arch == "omni" and not args.visual:
        ap.error(f"omni needs --visual (the vision tower .hbm) — {OMNI_SOURCES['--visual']}.")
    if args.visual and args.arch == "hybrid":
        # Qwen3.5 is natively image-text-to-text; the vision half is only absent from
        # a build because the compile dropped it. Nothing about the tower .hbm reveals
        # its patch size or normalization, and mrope layout differs from omni's, so
        # both have to be stated or the images silently decode as noise.
        if not args.mrope_section:
            ap.error("a hybrid VLM needs --mrope-section (Qwen3.5: 11 11 10)")
        if not args.mrope_interleaved:
            ap.error("a hybrid VLM needs --mrope-interleaved (Qwen3.5 lays rope out "
                     "[THWTHW...]; without it every image position is wrong)")
        if not args.vision_patch:
            ap.error("a hybrid VLM needs --vision-patch (Qwen3.5: 16)")
        if not args.vision_mean or not args.vision_std:
            ap.error("a hybrid VLM needs --vision-mean and --vision-std (Qwen3.5: 0.5 0.5)")
    if bool(args.audio) != bool(args.mel_filters):
        ap.error("--audio and --mel-filters go together: the audio tower needs its mel "
                 f"filter bank ({OMNI_SOURCES['--mel-filters']}). Pass both, or neither "
                 "for a vision-only Omni.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_src = Path(args.tokenizer)
    specials, vocab, has_added = load_tokenizer(tokenizer_src)

    if args.eos:
        eos, eos_src = args.eos, "--eos"
    elif args.arch == "pi05":
        # A policy emits a fixed-length action chunk; there is nothing to stop on,
        # and the PaliGemma tokenizer declares no chat markers to guess from.
        eos, eos_src = [], "n/a (policy)"
    else:
        eos, eos_src = resolve_eos(tokenizer_src, specials, vocab, has_added)

    cfg: dict = {
        "name": args.name or out.name,
        "arch": args.arch,
        "backend": "native",
        **({"bpu_cores": args.bpu_cores} if args.bpu_cores else {}),
        "hbm": "model.hbm",
        "tokenizer": "tokenizer.json",
        "eos": eos,
    }
    payload = {"model.hbm": Path(args.hbm), "tokenizer.json": tokenizer_src}

    if args.cache_len:
        cfg["cache_len"] = args.cache_len

    # Pick the chat template from the markers the tokenizer actually declares.
    if CHATML_START in specials and CHATML_END in specials:
        cfg["chat"] = {"format": "chatml", "im_start": specials[CHATML_START],
                       "im_end": specials[CHATML_END], "system": args.system}
    elif PHI_USER in specials and PHI_ASSISTANT in specials and PHI_END in specials:
        cfg["chat"] = {"format": "phi", "r_user": specials[PHI_USER],
                       "r_assistant": specials[PHI_ASSISTANT], "r_end": specials[PHI_END],
                       "system": args.system}
        if PHI_SYSTEM in specials:
            cfg["chat"]["r_system"] = specials[PHI_SYSTEM]
    elif args.arch == "pi05":
        cfg["chat"] = {"format": "none"}          # a policy has no chat template
    else:
        cfg["chat"] = {"format": "none"}
        print(f"[warn] {tokenizer_src.name} has no ChatML markers — chat.format=none "
              "(raw completion only)", file=sys.stderr)

    if args.arch in ("hybrid", "omni", "pi05"):
        cfg["embed"] = "embed_tokens.bin"
        payload["embed_tokens.bin"] = Path(args.embed)
    if args.arch == "pi05":
        stats = json.loads(Path(args.norm_stats).read_text())
        if not stats.get("quantile"):
            raise SystemExit("[pi05] norm_stats.json is not quantile-normalised; π0.5 is")
        cfg["visual"] = "visual.hbm"
        cfg["chat"] = {"format": "none"}      # a policy has no chat template
        cfg["pi05"] = {
            "expert": "expert.hbm",
            "cond_table": "cond_table.f32",
            "cameras": args.cameras,
            "prompt_len": args.prompt_len,
            "horizon": args.horizon,
            "steps": args.steps,
            # Row count of the embedding table, not the BPE map: `vocab` here is
            # the {token: id} dict, and PaliGemma's declared specials (<seg*>,
            # <loc*>) run past the BPE entries.
            "vocab": max([*vocab.values(), *specials.values()], default=0) + 1,
            "action_dim_real": stats["action_dim_real"],
            "action_q01": stats["action_q01"],
            "action_q99": stats["action_q99"],
            # The runtime refuses a table without this. `embed_prefix` scales the
            # tied row by sqrt(hidden) outside `embed_language_tokens`, and a
            # package that ships the raw row makes the prompt 45x too small —
            # invisible to any cosine taken against a golden built the same way,
            # and worth 24 points of LIBERO success rate.
            "embed_prescaled": True,
            "rope_theta": 1e4,
        }
        payload["visual.hbm"] = Path(args.visual)
        payload["expert.hbm"] = Path(args.expert)
        payload["cond_table.f32"] = Path(args.cond_table)
    if args.arch == "hybrid":
        cfg["graph"] = args.graph or "qwen35"
        cfg["rope_theta"] = args.rope_theta or 1e7
        if args.hbm_prefill:
            cfg["hbm_prefill"] = "model_prefill.hbm"
            payload["model_prefill.hbm"] = Path(args.hbm_prefill)
        if args.visual:
            cfg["visual"] = "visual.hbm"
            payload["visual.hbm"] = Path(args.visual)
            cfg["mrope_section"] = list(args.mrope_section)
            cfg["mrope_interleaved"] = True
    if args.arch == "omni":
        cfg["visual"] = "visual.hbm"
        payload["visual.hbm"] = Path(args.visual)
        cfg["rope_theta"] = args.rope_theta or 1e6
        cfg["mrope_section"] = list(args.mrope_section or (16, 24, 24))
        if args.mrope_interleaved:
            cfg["mrope_interleaved"] = True
        if args.audio:
            cfg["audio"] = "audio.hbm"
            cfg["mel_filters"] = "mel_filters_t.txt"
            payload["audio.hbm"] = Path(args.audio)
            payload["mel_filters_t.txt"] = Path(args.mel_filters)

    if args.visual:
        def _triple(v):
            return [float(v[0])] * 3 if len(v) == 1 else [float(x) for x in v]
        vis = {}
        if args.vision_patch: vis["patch"] = args.vision_patch
        if args.vision_merge: vis["merge"] = args.vision_merge
        if args.vision_mean: vis["mean"] = _triple(args.vision_mean)
        if args.vision_std: vis["std"] = _triple(args.vision_std)
        if vis:
            cfg["vision"] = vis

    for dst_name, src in payload.items():
        link_or_copy(src, out / dst_name, args.copy)

    (out / "model.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    print(f"[ok] {out}/model.json   (eos from {eos_src})")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    if cfg["arch"] == "pi05":
        print(f"\nload it:  bllm.load_policy(\"{out}\")"
              f"\n     or:  bllm_pi05_policy --model {out} --image a.jpg --wrist b.jpg "
              f"--prompt \"...\"")
    else:
        print(f"\nload it:  bllm.NativeVlmSession(\"{out}\")" if cfg.get("visual")
              else f"\nload it:  bllm.NativeSession(\"{out}\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
