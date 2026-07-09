#!/usr/bin/env python3
"""Image + text chat on the BLLM native runtime (no libxlm).

    export PYTHONPATH=<repo>/python
    python examples/vlm_native.py --model /models/qwen2.5-omni-3b --image bus.jpg \
        --prompt "描述这张图片。"

An image may be a path (decoded by the built-in stb loader) or, with --numpy, a
raw HxWx3 uint8 array — the shape a camera pipeline already hands you.
"""

from __future__ import annotations

import argparse
import sys

import bllm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="omni model directory (model.json)")
    ap.add_argument("--image", action="append", default=[], help="image path (repeatable)")
    ap.add_argument("--prompt", default="描述一下这张图片。")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--numpy", action="store_true",
                    help="decode images with PIL and pass raw arrays, as a camera would")
    args = ap.parse_args()

    vlm = bllm.NativeVlmSession(args.model)
    vlm.set_sampling(temp=args.temp)

    images = args.image
    if args.numpy and images:
        import numpy as np
        from PIL import Image
        images = [np.asarray(Image.open(p).convert("RGB")) for p in images]

    print(f"[{vlm.name}] {len(images)} image(s) x {vlm.vision_tokens} vision tokens")
    print(f"> {args.prompt}")
    for chunk in vlm.stream_chat(args.prompt, images, max_new=args.max_new):
        print(chunk, end="", flush=True)
    print(f"\n[ttft {vlm.last_ttft_ms:.0f} ms, decode {vlm.last_decode_tps:.1f} tok/s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
