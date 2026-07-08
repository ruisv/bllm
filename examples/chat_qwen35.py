#!/usr/bin/env python3
# Interactive chat REPL for the strict-100%-BPU Qwen3.5-0.8B, driving the resident
# C++ engine (bllm_qwen35 --serve) over a tiny stdin/stdout protocol. The C++ side
# holds the model + KV/SSM caches (loaded once); this front-end does tokenization,
# the ChatML template, streaming detokenization, and multi-turn.
#
# Run on the board:
#   source ~/conda/etc/profile.d/conda.sh && conda activate rdkpy312
#   cd ~/qwen3_deploy
#   python chat_qwen35.py            # chat mode (ChatML, multi-turn)
#   python chat_qwen35.py --raw      # raw completion (type text, model continues)
import os, sys, subprocess, argparse
from tokenizers import Tokenizer

HOME = os.path.expanduser("~/qwen3_deploy")
ENGINE = os.path.expanduser("~/projects/bllm/build/examples/bllm_qwen35")
IM_START, IM_END, EOT = 248045, 248046, 248044   # <|im_start|>, <|im_end|>, <|endoftext|>

ap = argparse.ArgumentParser()
ap.add_argument("--hbm", default=f"{HOME}/qwen35_0.8b_bpu.hbm")
ap.add_argument("--embed", default=f"{HOME}/embed_fp16.bin")
ap.add_argument("--tok", default=f"{HOME}/qwen35_tok/tokenizer.json")
ap.add_argument("--max-new", type=int, default=200)
ap.add_argument("--raw", action="store_true", help="raw completion instead of ChatML chat")
ap.add_argument("--system", default="You are a helpful assistant.")
args = ap.parse_args()

tk = Tokenizer.from_file(args.tok)
env = dict(os.environ); env["LD_LIBRARY_PATH"] = "/usr/hobot/lib:" + env.get("LD_LIBRARY_PATH", "")
# chat stops on <|im_end|>; raw stops on <|endoftext|>
eos = f"{IM_END},{EOT}" if not args.raw else str(EOT)
proc = subprocess.Popen(
    [ENGINE, "--hbm", args.hbm, "--embed", args.embed, "--serve", "--max-new", str(args.max_new), "--eos", eos],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, bufsize=1)

def wait_ready():
    for line in proc.stdout:
        if line.strip() == "READY":
            return
wait_ready()

def send(ids):
    proc.stdin.write(",".join(map(str, ids)) + "\n"); proc.stdin.flush()
    out_ids = []
    for line in proc.stdout:
        line = line.strip()
        if line == "E":
            break
        if line.startswith("T "):
            tid = int(line[2:]); out_ids.append(tid)
            piece = tk.decode([tid])
            print(piece, end="", flush=True)
    return out_ids

print(f"Qwen3.5-0.8B  [{'raw completion' if args.raw else 'chat'}]  "
      f"(commands: /reset  /exit)   model on BPU\n")
first = True
try:
    while True:
        try:
            msg = input("\033[1mYou:\033[0m ")
        except EOFError:
            break
        if msg.strip() in ("/exit", "/quit"):
            break
        if msg.strip() == "/reset":
            proc.stdin.write("R\n"); proc.stdin.flush(); proc.stdout.readline()
            first = True; print("[context reset]\n"); continue
        if not msg.strip():
            continue
        if args.raw:
            ids = tk.encode(msg).ids
        else:
            # ChatML incremental: on turn 1 open with the system message; on later
            # turns prepend <|im_end|> to close the previous assistant reply in the
            # resident cache (generate() stops before feeding the closing token).
            parts = []
            if first:
                parts += [IM_START] + tk.encode("system\n" + args.system).ids + [IM_END]
            else:
                parts += [IM_END]
            parts += [IM_START] + tk.encode("user\n" + msg).ids + [IM_END]
            parts += [IM_START] + tk.encode("assistant\n").ids
            ids = parts
        first = False
        print("\033[1mBot:\033[0m ", end="", flush=True)
        send(ids)
        print("\n")
finally:
    try:
        proc.stdin.close(); proc.terminate()
    except Exception:
        pass
