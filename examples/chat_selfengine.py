#!/usr/bin/env python3
"""Interactive chat on bllm_selfengine — our own hbDNN LLM engine, no libxlm.

Loads the model ONCE (engine runs in --serve mode), then you type questions and
watch tokens stream back. Tokenization uses the model's real tokenizer
(`tokenizers`); the BPU generation runs entirely through hbDNN/hbUCP — libxlm is
never loaded (check with `ldd bllm_selfengine`).

Board setup (one-time):  pip install tokenizers
Run:
  python3 chat_selfengine.py \
    --engine ./bllm_selfengine \
    --hbm    ~/qwen3_deploy/qwen3-1.7b.hbm \
    --config ~/qwen3_deploy/qwen3-1.7b_config

Commands inside the chat:  /think on|off   /exit
Each question is independent (single-turn; prompt must fit one 256-token chunk).
"""
import argparse
import os
import subprocess
import sys

from tokenizers import Tokenizer

EOS = {151645, 151643}
MAX_PROMPT = 250  # prefill is a single 256-token chunk


def build_prompt(q, think):
    body = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    if not think:
        body += "<think>\n\n</think>\n\n"   # Qwen3 /no_think -> crisp answers
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./bllm_selfengine")
    ap.add_argument("--hbm", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--think", action="store_true", help="let the model think (verbose)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.config, "tokenizer.json"))
    think = args.think

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/hobot/lib:" + env.get("LD_LIBRARY_PATH", "")
    eng = subprocess.Popen(
        [args.engine, "--hbm", args.hbm, "--serve", "--max-new", str(args.max_new)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
        text=True, bufsize=1)

    # wait for READY (engine prints model-load logs to stderr)
    for line in eng.stdout:
        if line.strip() == "READY":
            break
    print("=" * 64)
    print("bllm_selfengine chat — running on hbDNN, libxlm NOT loaded")
    print("commands: /think on|off   /exit     (thinking is %s)" % ("on" if think else "off"))
    print("=" * 64)

    while True:
        try:
            q = input("\n\033[1myou ›\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q.startswith("/think"):
            think = q.endswith("on")
            print("  [thinking %s]" % ("on" if think else "off"))
            continue

        ids = tok.encode(build_prompt(q, think)).ids
        if len(ids) > MAX_PROMPT:
            print("  [prompt too long: %d tokens > %d]" % (len(ids), MAX_PROMPT))
            continue

        eng.stdin.write(",".join(map(str, ids)) + "\n")
        eng.stdin.flush()

        print("\033[36mbot ›\033[0m ", end="", flush=True)
        out_ids, shown = [], ""
        for line in eng.stdout:
            line = line.strip()
            if line == "END":
                break
            if line.startswith("TOK "):
                out_ids.append(int(line[4:]))
                text = tok.decode([i for i in out_ids if i not in EOS])
                sys.stdout.write(text[len(shown):])
                sys.stdout.flush()
                shown = text
        print()

    eng.stdin.close()
    eng.wait()
    print("\nbye — that whole conversation ran on our own engine, never libxlm.")


if __name__ == "__main__":
    main()
