#!/usr/bin/env python3
"""bllm.native — a reusable Python API over the native hbDNN engine, running a
`.hbm` WITHOUT libxlm. Uses the in-process nanobind binding
(`bllm._bllm_native.NativeEngine`); tokenization is Python-side (`tokenizers`).

    from native_session import NativeSession
    s = NativeSession("qwen3-1.7b.hbm", "qwen3-1.7b_config")
    print(s.generate("The capital of France is", max_new=24))   # raw completion
    print(s.chat("What is 2+2?"))                                # multi-turn
    print(s.chat("And times 3?"))                                # remembers context
    for piece in s.stream_chat("Tell me a joke."): print(piece, end="")
    print(s.last_stats)   # {'ttft_ms':.., 'decode_tps':.., 'ntok':..}

Build the binding first: `cmake -B build -DBLLM_BUILD_PYTHON=ON && ninja -C build
_bllm_native`, then run with PYTHONPATH=<repo>/python and LD_LIBRARY_PATH=/usr/hobot/lib.
"""
import json
import os

from tokenizers import Tokenizer

# Import the native module directly (not via the `bllm` package, whose __init__
# pulls in the libxlm-backed _bllm and its opencv deps we don't need here).
try:
    import _bllm_native as _N
except ImportError:
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in (os.path.join(_here, "..", "python", "bllm"),
               os.path.join(_here, "bllm")):
        if os.path.exists(_p):
            sys.path.insert(0, _p)
    import _bllm_native as _N


class NativeSession:
    def __init__(self, hbm, config_dir, max_new=200,
                 temp=0.0, top_k=0, top_p=1.0, rep_pen=1.0, seed=1234):
        self.tok = Tokenizer.from_file(os.path.join(config_dir, "tokenizer.json"))
        eos = []
        cfg = os.path.join(config_dir, "config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                e = json.load(f).get("eos_token_id")
            eos = e if isinstance(e, list) else ([e] if e is not None else [])
        self.eos = set(eos) or {151645, 151643}
        self.eng = _N.NativeEngine(hbm)
        self.sp = _N.SamplingParams()
        self.sp.temp, self.sp.top_k, self.sp.top_p = temp, top_k, top_p
        self.sp.rep_pen, self.sp.max_new, self.sp.seed = rep_pen, max_new, seed
        self.sp.eos = list(self.eos)
        self.last_stats = {}
        self._first = True

    def reset(self):
        self.eng.reset()
        self._first = True

    def _run(self, ids, on_token=None, max_new=None):
        self.eng.feed(ids)
        self.sp.max_new = int(max_new) if max_new else self.sp.max_new
        state = {"ids": [], "shown": ""}
        def cb(t):
            state["ids"].append(t)
            text = self.tok.decode([i for i in state["ids"] if i not in self.eos])
            on_token(text[len(state["shown"]):]); state["shown"] = text
        gen = self.eng.generate(self.sp, cb if on_token else None)
        s = self.eng.last_stats()
        self.last_stats = {"ttft_ms": s.ttft_ms, "decode_tps": s.decode_tps, "ntok": s.ntok}
        return self.tok.decode([i for i in gen if i not in self.eos])

    def _chat_delta(self, msg, think):
        head = "" if self._first else "<|im_end|>\n"
        self._first = False
        body = f"{head}<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        if not think:
            body += "<think>\n\n</think>\n\n"
        return self.tok.encode(body).ids

    # --- public API -----------------------------------------------------------
    def generate(self, prompt, max_new=None, on_token=None):
        self.reset()
        return self._run(self.tok.encode(prompt).ids, on_token, max_new)

    def chat(self, msg, think=False, on_token=None, max_new=None):
        return self._run(self._chat_delta(msg, think), on_token, max_new)

    def stream_chat(self, msg, think=False):
        pieces = []
        self.chat(msg, think=think, on_token=pieces.append)
        yield from pieces


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbm", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    s = NativeSession(a.hbm, a.config)
    print("engine: layers=%d kv=%d head_dim=%d vocab=%d" %
          (s.eng.n_layers, s.eng.n_kv, s.eng.head_dim, s.eng.vocab))
    print("generate:", repr(s.generate("The capital of France is", max_new=20)))
    print("chat 1  :", repr(s.chat("My favorite number is 7. Remember it.")))
    print("chat 2  :", repr(s.chat("What is my favorite number, doubled?")))
    print("stats   :", s.last_stats)
