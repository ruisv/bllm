#!/usr/bin/env python3
"""Micro-benchmark bllm across one or more models, using the library itself.

Reports wall-clock TTFT and decode tok/s (bllm's own measured stats). Run on the
board inside the conda env:

    export LD_LIBRARY_PATH=<sdk>/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=<repo>/python:$PYTHONPATH
    python scripts/bench.py \
        --model qwen2.5:<hbm>:<tokenizer_dir> \
        --model deepseek:<hbm>:<tokenizer_dir>

Each --model is NAME:HBM:TOKENIZER_DIR. --prompt sets the shared prompt.
"""

import argparse
import time

import bllm

DEFAULT_PROMPT = "请用中文写一段关于人工智能如何改变日常生活的短文，大约一百字。"


def bench_one(name: str, hbm: str, tok: str, prompt: str, warmup: bool) -> dict:
    t0 = time.perf_counter()
    llm = bllm.load(hbm, tokenizer_dir=tok)     # native (a bare .hbm + tokenizer dir)
    load_s = time.perf_counter() - t0

    if warmup:
        llm.chat("你好")
        llm.reset()

    t1 = time.perf_counter()
    reply = llm.chat(prompt)
    wall_s = time.perf_counter() - t1
    return {
        "name": name,
        "model_type": llm.arch,
        "load_s": load_s,
        "wall_s": wall_s,
        "ttft_ms": 0.0,                          # native surfaces only decode tok/s
        "decode_tps": llm.last_decode_tps,
        "tokens": 0,
        "chars": len(reply),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="bllm benchmark")
    p.add_argument("--model", action="append", default=[], metavar="NAME:HBM:TOK",
                   help="repeatable; NAME:hbm_path:tokenizer_dir")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()
    if not args.model:
        p.error("give at least one --model NAME:HBM:TOKENIZER_DIR")

    rows = []
    for spec in args.model:
        name, hbm, tok = spec.split(":", 2)
        print(f"# running {name} ...", flush=True)
        rows.append(bench_one(name, hbm, tok, args.prompt, not args.no_warmup))

    hdr = f"{'model':<26}{'type':<11}{'load s':>7}{'ttft ms':>9}{'decode t/s':>12}{'tokens':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<26}{r['model_type']:<11}{r['load_s']:>7.1f}"
              f"{r['ttft_ms']:>9.1f}{r['decode_tps']:>12.2f}{r['tokens']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
