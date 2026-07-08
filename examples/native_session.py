#!/usr/bin/env python3
"""bllm.native — a reusable Python API over the self-built hbDNN engine
(bllm_selfengine), the BLLM native backend that runs a `.hbm` WITHOUT libxlm.

    from native_session import NativeSession
    s = NativeSession("qwen3-1.7b.hbm", "qwen3-1.7b_config")
    print(s.generate("The capital of France is", max_new=24))   # raw completion
    print(s.chat("What is 2+2?"))                                # multi-turn chat
    print(s.chat("And times 3?"))                                # remembers context
    for piece in s.stream_chat("Tell me a short joke."):         # streaming
        print(piece, end="", flush=True)
    print(s.last_stats)   # {'ttft_ms': .., 'decode_tps': .., 'ntok': ..}

Tokenization uses the model's real tokenizer (`tokenizers`); generation runs on
hbDNN/hbUCP. This mirrors bllm.LlmSession but on the native backend. The C++
tokenizer + folding into the compiled bllm.LlmSession (backend="native") is the
remaining SE3 work; this class is the usable API today.
"""
import json
import os
import subprocess

from tokenizers import Tokenizer


class NativeSession:
    def __init__(self, hbm, config_dir, engine="./bllm_selfengine",
                 hobot_lib="/usr/hobot/lib", max_new=200,
                 temp=0.0, top_k=0, top_p=1.0, rep_pen=1.0, seed=1234):
        self.tok = Tokenizer.from_file(os.path.join(config_dir, "tokenizer.json"))
        cfg = os.path.join(config_dir, "config.json")
        eos = []
        if os.path.exists(cfg):
            with open(cfg) as f:
                e = json.load(f).get("eos_token_id")
            eos = e if isinstance(e, list) else ([e] if e is not None else [])
        self.eos = set(eos) or {151645, 151643}
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = hobot_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = [engine, "--hbm", hbm, "--serve", "--max-new", str(max_new),
               "--temp", str(temp), "--top-k", str(top_k), "--top-p", str(top_p),
               "--rep-pen", str(rep_pen), "--seed", str(seed),
               "--eos", ",".join(map(str, self.eos))]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env=env, text=True, bufsize=1)
        self._wait("READY")
        self.last_stats = {}
        self._first = True   # multi-turn: True until the first chat turn is sent

    def _wait(self, marker):
        for line in self.p.stdout:
            if line.strip() == marker:
                return

    def reset(self):
        """Clear the conversation KV cache."""
        self.p.stdin.write("R\n"); self.p.stdin.flush()
        self._wait("OK")
        self._first = True

    def _run(self, ids, on_token=None, max_new=None):
        prefix = f"{int(max_new)}:" if max_new else ""
        self.p.stdin.write(prefix + ",".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        out, shown = [], ""
        for line in self.p.stdout:
            line = line.rstrip("\n")
            if line == "END":
                break
            if line.startswith("STATS "):
                for kv in line[6:].split():
                    k, v = kv.split("=")
                    self.last_stats[k] = float(v) if "." in v else int(v)
            elif line.startswith("TOK "):
                out.append(int(line[4:]))
                if on_token is not None:
                    text = self.tok.decode([i for i in out if i not in self.eos])
                    on_token(text[len(shown):]); shown = text
        return self.tok.decode([i for i in out if i not in self.eos])

    def _chat_delta(self, msg, think):
        head = "" if self._first else "<|im_end|>\n"
        self._first = False
        body = f"{head}<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        if not think:
            body += "<think>\n\n</think>\n\n"
        return self.tok.encode(body).ids

    # --- public API -----------------------------------------------------------
    def generate(self, prompt, max_new=None, on_token=None):
        """Raw completion of `prompt` from a fresh context."""
        self.reset()
        return self._run(self.tok.encode(prompt).ids, on_token, max_new)

    def chat(self, msg, think=False, on_token=None, max_new=None):
        """One multi-turn chat turn; context persists across calls until reset()."""
        return self._run(self._chat_delta(msg, think), on_token, max_new)

    def stream_chat(self, msg, think=False):
        """Generator yielding text pieces for a chat turn."""
        pieces = []
        self.chat(msg, think=think, on_token=pieces.append)
        yield from pieces

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./bllm_selfengine")
    ap.add_argument("--hbm", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    s = NativeSession(a.hbm, a.config, engine=a.engine)
    print("generate:", repr(s.generate("The capital of France is", max_new=20)))
    print("chat 1  :", repr(s.chat("My favorite number is 7. Remember it.")))
    print("chat 2  :", repr(s.chat("What is my favorite number, doubled?")))
    print("stats   :", s.last_stats)
    s.close()
