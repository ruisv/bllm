#!/usr/bin/env python3
"""bllm chat REPL (Python) — the streaming-generator equivalent of bllm_chat.

    export LD_LIBRARY_PATH=<sdk>/oellm_runtime/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=<repo>/python:$PYTHONPATH
    python examples/chat.py --hbm <model.hbm> --tokenizer <config_dir>

Type 'reset' for a fresh conversation, 'exit' to quit.
"""

import argparse
import sys

import bllm


def main() -> int:
    p = argparse.ArgumentParser(description="bllm chat REPL")
    p.add_argument("--hbm", required=True, help="compiled .hbm model")
    p.add_argument("--tokenizer", required=True, help="tokenizer config dir")
    p.add_argument("--model-type", default="auto",
                   help="qwen2.5|deepseek|internlm2|... (default: auto)")
    p.add_argument("--template", default="", help=".jinja (auto-found if omitted)")
    p.add_argument("--system", default="", help="system prompt")
    p.add_argument("--stats", action="store_true", help="print timing per reply")
    args = p.parse_args()

    llm = bllm.LlmSession(
        args.hbm,
        args.tokenizer,
        model_type=args.model_type,
        chat_template_path=args.template,
        system_prompt=args.system,
    )
    print(f"bllm chat ready (model_type={llm.model_type}). "
          "Type 'reset' to clear history, 'exit' to quit.\n")

    try:
        while True:
            try:
                line = input("[User] <<< ").strip()
            except EOFError:
                break
            if line == "exit":
                break
            if line == "reset":
                llm.reset()
                print("[system] history cleared")
                continue
            if not line:
                continue

            print("[Assistant] >>> ", end="", flush=True)
            for chunk in llm.stream(line):
                print(chunk, end="", flush=True)
            print()
            if args.stats:
                s = llm.last_stats
                print(f"  (ttft {s.ttft_ms:.1f} ms, decode {s.decode_tps:.1f} "
                      f"tok/s, {s.decode_tokens} tokens)")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
