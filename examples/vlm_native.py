#!/usr/bin/env python3
"""Multimodal chat on the BLLM native runtime (no libxlm).

    export PYTHONPATH=<repo>/python
    python examples/vlm_native.py --model /models/qwen2.5-omni-3b --image bus.jpg \
        --prompt "描述这张图片。"
    python examples/vlm_native.py --model ... --audio clip.wav --prompt "听到了什么？"
    python examples/vlm_native.py --model ... --video clip.mp4 --prompt "描述这段视频。"

Images may be paths or HxWx3 uint8 arrays, audio a .wav path or 16 kHz mono
float32 — the shapes a camera/microphone pipeline already has. --video uses
bllm.load_video (ffmpeg) and feeds the soundtrack too unless --mute.
"""

from __future__ import annotations

import argparse
import sys

import bllm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="omni model directory (model.json)")
    ap.add_argument("--image", action="append", default=[], help="image path (repeatable)")
    ap.add_argument("--audio", action="append", default=[], help="wav path (repeatable)")
    ap.add_argument("--video", action="append", default=[], help="video path (repeatable)")
    ap.add_argument("--prompt", default="描述一下这张图片。")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=2.0, help="video frame sampling rate")
    ap.add_argument("--max-frames", type=int, default=10, help="cap on sampled frames")
    ap.add_argument("--mute", action="store_true", help="drop the video soundtrack")
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

    videos = [bllm.load_video(p, fps=args.fps, max_frames=args.max_frames,
                              with_audio=not args.mute) for p in args.video]

    print(f"[{vlm.name}] {len(images)} image(s), {len(args.audio)} audio, {len(videos)} video")
    for v in videos:
        pairs = (len(v["frames"]) + 1) // 2
        print(f"  video: {len(v['frames'])} frames @ {v['fps']} fps"
              f" -> {pairs * vlm.vision_tokens} vision tokens"
              f"{' + soundtrack' if v['audio'] is not None else ''}")
    print(f"> {args.prompt}")
    for chunk in vlm.stream_chat(args.prompt, images, args.audio, videos, max_new=args.max_new):
        print(chunk, end="", flush=True)
    print(f"\n[ttft {vlm.last_ttft_ms:.0f} ms, decode {vlm.last_decode_tps:.1f} tok/s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
