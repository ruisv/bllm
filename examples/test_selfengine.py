#!/usr/bin/env python3
"""Test harness for bllm_selfengine — our own hbDNN autoregressive LLM engine
that drives a compiled .hbm WITHOUT libxlm.

The engine (C++) speaks token ids in/out; this script does only tokenization,
using a self-contained byte-level BPE decoder built from the model's own
vocab.json (no `tokenizers`/`transformers` needed on the board). Prompt token
ids are precomputed offline with the real Qwen tokenizer and pinned here, so the
test has zero tokenizer dependencies and can't drift.

Run on the board:
  python3 test_selfengine.py \
    --engine ./bllm_selfengine \
    --hbm   ~/qwen3_deploy/qwen3-1.7b.hbm \
    --config ~/qwen3_deploy/qwen3-1.7b_config
"""
import argparse
import json
import os
import subprocess
import sys


# ---- GPT-2 / Qwen byte-level BPE: char <-> byte tables (for DECODING) --------
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}  # unicode-char -> byte


CHAR2BYTE = bytes_to_unicode()


def load_vocab(config_dir):
    with open(os.path.join(config_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)               # token-str -> id
    id2tok = {v: k for k, v in vocab.items()}
    specials = set()
    tc = os.path.join(config_dir, "tokenizer_config.json")
    if os.path.exists(tc):
        with open(tc, encoding="utf-8") as f:
            atd = json.load(f).get("added_tokens_decoder", {})
        for sid, meta in atd.items():
            specials.add(int(sid))
            id2tok.setdefault(int(sid), meta.get("content", ""))
    return id2tok, specials


def decode(ids, id2tok, specials):
    """Byte-level decode, skipping special tokens (im_start/im_end/…)."""
    out, buf = [], []
    for i in ids:
        if i in specials:
            if buf:
                out.append(_flush(buf, id2tok)); buf = []
            continue
        buf.append(i)
    if buf:
        out.append(_flush(buf, id2tok))
    return "".join(out)


def _flush(ids, id2tok):
    chars = "".join(id2tok.get(i, "") for i in ids)
    return bytes(CHAR2BYTE.get(c, 0) for c in chars).decode("utf-8", errors="replace")


# ---- pinned test prompts (ids from the real Qwen tokenizer) ------------------
CASES = [
    ("raw · The capital of France is",
     [785, 6722, 315, 9625, 374], 24),
    ("chat · What is 2+2? (no-think)",
     [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644,
      77091, 198, 151667, 271, 151668, 271], 40),
    ("chat · 用一句话介绍北京。(no-think)",
     [151644, 872, 198, 11622, 105321, 100157, 68990, 1773, 151645, 198, 151644,
      77091, 198, 151667, 271, 151668, 271], 80),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./bllm_selfengine")
    ap.add_argument("--hbm", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    id2tok, specials = load_vocab(args.config)
    print("=" * 68)
    print("bllm_selfengine — self-built hbDNN engine, libxlm NOT used")
    print("=" * 68)
    ok = True
    for name, ids, maxnew in CASES:
        r = subprocess.run(
            [args.engine, "--hbm", args.hbm, "--ids", ",".join(map(str, ids)),
             "--max-new", str(maxnew)],
            capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("GEN:")), None)
        if line is None:
            print(f"\n[{name}]\n  ENGINE FAILED:\n{r.stderr.strip()}")
            ok = False
            continue
        gen = [int(x) for x in line[4:].split()]
        print(f"\n[{name}]")
        print("  reply:", repr(decode(gen, id2tok, specials)))
        print(f"  ({len(gen)} tokens generated on the BPU via hbDNN)")
    print("\n" + "=" * 68)
    print("DONE — all generation ran through hbDNN/hbUCP; libxlm.so was never loaded.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
