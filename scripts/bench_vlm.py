#!/usr/bin/env python3
"""Compare VLM packages on one image, on the board.

The point of this script is not "which model is better" — it is to separate the
two costs that a single TTFT number hides:

    TTFT  =  vision encode  +  ingesting (vision rows + prompt) into the decoder

The official Qwen2.5-Omni package runs on a dense KV engine that HAS a prefill
graph; our Qwen3.5 hybrid engine does not yet, and feeds one token at a time. So
comparing them mostly measures the prefill graph, not the models — which is
exactly what we want to know. Reporting ms-per-ingested-token makes that legible
across models of different sizes and different vision-token counts.

    python scripts/bench_vlm.py --image bears.png \\
        --model ~/models/qwen2.5-omni-3b \\
        --model ~/models/qwen3.5-0.8b-vlm448-ctx4096-int8-s100p
"""
import argparse
import time


def bench(path, image, prompt, max_new, repeat):
    import bllm
    v = bllm.load(path)
    rows = []
    for _ in range(repeat):
        v.reset()
        t0 = time.perf_counter()
        text = v.chat(prompt, images=[image], max_new=max_new)
        wall = time.perf_counter() - t0
        rows.append({
            "ttft": v.last_ttft_ms / 1000.0,
            "tps": v.last_decode_tps,
            "wall": wall,
            "tokens": v.tokens_used,
            "text": text,
        })
    best = min(rows, key=lambda r: r["ttft"])
    # Time the vision tower alone by asking the same question with no image: the
    # difference in ingested tokens is the image's rows, so the residual isolates
    # what the tower itself costs.
    v.reset()
    t0 = time.perf_counter()
    v.chat(prompt, max_new=1)
    text_only_ttft = v.last_ttft_ms / 1000.0
    text_only_tokens = v.tokens_used
    return {
        "name": getattr(v, "name", "") or path.rstrip("/").split("/")[-1],
        "vision_tokens": v.vision_tokens,
        "image_size": v.vision_image_size,
        **best,
        "text_only_ttft": text_only_ttft,
        "text_only_tokens": text_only_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe this image in one sentence.")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    out = []
    for m in args.model:
        r = bench(m, args.image, args.prompt, args.max_new, args.repeat)
        out.append(r)
        print(f"\n=== {r['name']} ===")
        print(f"  vision {r['vision_tokens']} tok @ {r['image_size']}px")
        print(f"  TTFT {r['ttft']:.2f}s   decode {r['tps']:.1f} tok/s   "
              f"wall {r['wall']:.1f}s   prompt {r['tokens']} tok")
        print(f"  ANSWER: {r['text'].replace(chr(10), ' ')[:160]}")

    print("\n" + "=" * 78)
    print(f"{'model':<34}{'vis tok':>8}{'TTFT s':>9}{'ms/tok in':>11}{'dec tok/s':>11}")
    print("-" * 78)
    for r in out:
        # ms per ingested token: the honest cross-model number, since TTFT scales
        # with how many tokens had to go in before the first one came out.
        per = r["ttft"] * 1000.0 / max(r["tokens"], 1)
        print(f"{r['name'][:33]:<34}{r['vision_tokens']:>8}{r['ttft']:>9.2f}"
              f"{per:>11.1f}{r['tps']:>11.1f}")
    print("=" * 78)
    print("ms/tok in = TTFT / prompt tokens. A model with a prefill graph amortizes")
    print("the weight stream across a chunk; one without it pays it per token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
