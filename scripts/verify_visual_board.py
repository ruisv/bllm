#!/usr/bin/env python3
"""P2 gate: score the on-board vision tower against the HF fp32 reference.

Runs on the BOARD. Drives bllm_vision_probe twice and reports both numbers,
because they answer different questions:

  graph-only     probe fed HF's own pixel_values. Isolates quantization.
  end-to-end     probe fed the image. Adds our resize/normalize/patchify.

A low graph-only score means the compile or the activation ranges are wrong. A
good graph-only score with a low end-to-end score means the host preprocessing
disagrees with HF (patch order, mean/std, resize kernel) — a completely different
fix, which is why they are not collapsed into one number.

Cosine is necessary but NOT sufficient: the real gate is end-to-end answer
quality. Qwen3-VL-Embedding passed cosine and still lost retrieval accuracy.

    python scripts/verify_visual_board.py --visual ~/models/.../visual.hbm \\
        --ref ~/boardref [--patch-size 16] [--mean 0.5] [--std 0.5]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

GATE = 0.99


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def run(probe, visual, args, extra, out):
    cmd = [probe, "--visual", visual, "--out", out,
           "--patch-size", str(args.patch_size), "--merge", str(args.merge),
           "--mean", str(args.mean), "--std", str(args.std)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit(f"probe failed: {' '.join(cmd)}")
    for line in r.stdout.splitlines():
        if line.startswith("[probe]"):
            print("   ", line)
    return np.fromfile(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", required=True)
    ap.add_argument("--ref", required=True, help="dir from export_ref_for_board.py")
    ap.add_argument("--probe", default="build/examples/bllm_vision_probe")
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--merge", type=int, default=2)
    ap.add_argument("--mean", type=float, default=0.5)
    ap.add_argument("--std", type=float, default=0.5)
    args = ap.parse_args()

    ref = np.fromfile(os.path.join(args.ref, "ref_visout.f32"), dtype=np.float32)
    pv = os.path.join(args.ref, "ref_pv.f32")
    img = os.path.join(args.ref, "ref_image.png")

    ok = True
    with tempfile.TemporaryDirectory() as td:
        print("== graph only (HF pixel_values in) ==")
        rows = run(args.probe, args.visual, args, ["--patches", pv], f"{td}/g.f32")
        c_graph = cosine(rows, ref)
        print(f"    cosine vs HF = {c_graph:.6f}   gate >= {GATE}"
              f"   {'PASS' if c_graph >= GATE else 'FAIL'}")
        ok &= c_graph >= GATE

        if os.path.exists(img):
            print("== end to end (image in, host preprocessing + graph) ==")
            rows2 = run(args.probe, args.visual, args, ["--image", img], f"{td}/e.f32")
            c_e2e = cosine(rows2, ref)
            c_pre = cosine(rows2, rows)
            print(f"    cosine vs HF = {c_e2e:.6f}   gate >= {GATE}"
                  f"   {'PASS' if c_e2e >= GATE else 'FAIL'}")
            print(f"    cosine vs graph-only = {c_pre:.6f}"
                  f"   (this is the preprocessing agreement, on its own)")
            ok &= c_e2e >= GATE
        else:
            print("== end to end: skipped, no ref_image.png ==")

    print("\nRemember: cosine is a smoke test. The shipping gate is answer quality.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
