#!/usr/bin/env python3
"""Does the prefill graph agree with one-token-at-a-time? (CHUNKED_PREFILL.md gate 2)

Runs the SAME model twice — once with its prefill graph, once without — and
compares what the two paths produce. The unrolled prefill graph computes the
identical recurrence to decode, so in exact arithmetic the last prompt token's
logits are the same; what this measures is whether the compiler's scheduling and
the two graphs' quantization keep them the same in practice.

Compared, in decreasing order of strictness:
  1. the first sampled token's top-k logprobs (this IS the gate: those come from
     the last prompt position, i.e. the handoff from prefill to decode)
  2. the full greedy continuation (drift shows up here even if token 1 matched)
  3. perplexity, as a control — it steps one token at a time in BOTH runs, so it
     must match regardless, and a mismatch means something other than prefill moved

    python scripts/check_prefill_parity.py --with-prefill DIR_A --without DIR_B
"""
import argparse

# MUST exceed the prefill chunk, or no chunk runs and the test is vacuous — the
# whole point is to exercise the chunk. Built long and asserted against the reported
# prefill_chunk below.
PROMPT = ("The history of computing spans many decades and countless people. " * 24
          + " In a single word, name the field just described.")
CONTROL_TEXT = "The quick brown fox jumps over the lazy dog. " * 4


def run(path, prompt, max_new, topk):
    import bllm
    s = bllm.load(path)
    chunk = getattr(s, "prefill_chunk", None)
    s.set_sampling(temp=0.0, logprobs=topk)
    s.reset()
    text = s.chat(prompt, max_new=max_new)
    lps = s.last_logprobs()
    ppl = s.perplexity(CONTROL_TEXT)
    ntok = len(s.encode(prompt)) if hasattr(s, "encode") else 0
    return {"chunk": chunk, "text": text, "lps": lps, "ppl": ppl, "prompt_tokens": ntok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-prefill", required=True)
    ap.add_argument("--without", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    a = run(args.with_prefill, PROMPT, args.max_new, args.topk)
    b = run(args.without, PROMPT, args.max_new, args.topk)
    print(f"prefill_chunk:  with={a['chunk']}   without={b['chunk']}")
    if not a["chunk"]:
        print("!! the 'with' package reports no prefill graph — nothing was actually compared")
        return 2
    # The prompt MUST be longer than the chunk, or the whole prompt falls to decode
    # and no prefill chunk runs — a green result that proves nothing. This hole made
    # an earlier run report PARITY OK on a graph that in fact produced garbage.
    ntok = a.get("prompt_tokens", 0)
    if ntok <= a["chunk"]:
        print(f"!! prompt is {ntok} tokens, not > chunk {a['chunk']}: NO chunk ran, "
              "the comparison is vacuous. Lengthen PROMPT.")
        return 2
    print(f"prompt {ntok} tokens > chunk {a['chunk']}: at least one prefill chunk ran")

    ok = True

    # 1. the handoff itself
    if a["lps"] and b["lps"]:
        ta, tb = a["lps"][0], b["lps"][0]
        same_id = ta["id"] == tb["id"]
        da = {t[0]: t[1] for t in ta.get("top", [])}
        db = {t[0]: t[1] for t in tb.get("top", [])}
        shared = set(da) & set(db)
        worst = max((abs(da[i] - db[i]) for i in shared), default=float("nan"))
        print(f"\n[1] first token (the prefill->decode handoff)")
        print(f"    id           {ta['id']} vs {tb['id']}   {'MATCH' if same_id else 'DIFFER'}")
        print(f"    top-{args.topk} ids  {'identical' if list(da) == list(db) else 'differ'}"
              f"   ({len(shared)}/{len(da)} shared)")
        print(f"    max |dlogprob| over shared ids = {worst:.6f}")
        ok &= same_id
    else:
        print("\n[1] no logprobs recorded — set_sampling(logprobs=) did not take")
        ok = False

    # 2. drift over a full continuation
    print(f"\n[2] greedy continuation ({args.max_new} tokens)")
    if a["text"] == b["text"]:
        print("    IDENTICAL")
    else:
        n = next((i for i, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y),
                 min(len(a["text"]), len(b["text"])))
        print(f"    DIVERGES at char {n}")
        print(f"      with:    ...{a['text'][max(0,n-40):n+60]!r}")
        print(f"      without: ...{b['text'][max(0,n-40):n+60]!r}")
        ok = False

    # 3. control
    print(f"\n[3] control — perplexity (steps one token at a time in BOTH; must match)")
    print(f"    with={a['ppl']:.6f}  without={b['ppl']:.6f}  "
          f"{'MATCH' if abs(a['ppl']-b['ppl']) < 1e-4 else 'DIFFER <- not a prefill issue'}")

    print("\n=>", "PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
