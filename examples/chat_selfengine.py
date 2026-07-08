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


def build_delta(q, think, first):
    # Only the NEW tokens for this turn; the engine's KV cache holds the history.
    # Turn >= 2 opens with <|im_end|> to close the previous assistant turn.
    head = "" if first else "<|im_end|>\n"
    body = f"{head}<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
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
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.config, "tokenizer.json"))
    think = args.think

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/hobot/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [args.engine, "--hbm", args.hbm, "--serve", "--max-new", str(args.max_new),
           "--temp", str(args.temp), "--top-k", str(args.top_k),
           "--top-p", str(args.top_p), "--seed", str(args.seed)]
    eng = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           env=env, text=True, bufsize=1)

    def wait_for(marker):
        for line in eng.stdout:
            if line.strip() == marker:
                return

    def reset():
        eng.stdin.write("R\n"); eng.stdin.flush()
        wait_for("OK")

    wait_for("READY")
    reset()
    first = True
    print("=" * 64)
    print("bllm_selfengine chat — multi-turn on hbDNN, libxlm NOT loaded")
    print("commands: /think on|off   /reset   /exit     (thinking is %s)" % ("on" if think else "off"))
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
        if q == "/reset":
            reset(); first = True
            print("  [context cleared]")
            continue
        if q.startswith("/think"):
            think = q.endswith("on")
            print("  [thinking %s]" % ("on" if think else "off"))
            continue

        ids = tok.encode(build_delta(q, think, first)).ids
        first = False
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
