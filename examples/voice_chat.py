#!/usr/bin/env python3
"""Talk to Qwen3.5: speech in (Qwen3-ASR), Qwen3.5's answer out, optionally spoken back.

Everything runs on the BPU of one board: the official Qwen3-ASR package transcribes,
Qwen3.5 answers, and — if bcdl is installed and you point --tts-dir at its Matcha-TTS
models — the answer is synthesised to a wav (Chinese voice).

    python examples/voice_chat.py --asr Qwen3-ASR-0.6B_w8/ \\
        --llm bllm-qwen3.5-2b/s100p/ctx512 question.wav
    python examples/voice_chat.py --asr ... --llm ... --record 5          # 5 s from the mic
    python examples/voice_chat.py --asr ... --llm ... q.wav --tts-dir ~/bcdl-models --out a.wav

The two models share the board: Qwen3-ASR-0.6B takes ~1 GB of ION and Qwen3.5-2B (text
only) ~3.8 GB, which fits the `balanced` carveout.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

import bllm

TTS_FILES = ["matcha_zh_en_enc_l128_int16_nashm.hbm", "matcha_zh_en_dec_t512_int16_nashm.hbm",
             "matcha_zh_en_embedding.npy", "vocos_16k_t256_int16_nashm.hbm",
             "matcha_zh_en_tokens.txt", "matcha_zh_en_lexicon.txt"]


def record(seconds: float, device: str) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    print(f"recording {seconds:g} s from {device} ...", flush=True)
    subprocess.run(["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1",
                    "-d", str(int(np.ceil(seconds))), path], check=True)
    return path


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="*", help="speech clips (wav; other formats need ffmpeg)")
    ap.add_argument("--asr", required=True, help="official Qwen3-ASR package dir")
    ap.add_argument("--llm", required=True, help="Qwen3.5 model dir (any bllm chat model works)")
    ap.add_argument("--language", default=None, help="force the spoken language (default: detect)")
    ap.add_argument("--record", type=float, default=0, help="record this many seconds instead")
    ap.add_argument("--device", default="default", help="arecord capture device")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tts-dir", default="", help="bcdl Matcha-TTS model dir: speak the answer")
    ap.add_argument("--out", default="reply.wav", help="where the spoken answer goes")
    args = ap.parse_args()

    clips = list(args.audio)
    if args.record:
        clips.append(record(args.record, args.device))
    if not clips:
        ap.error("give a speech clip or --record SECONDS")

    t = time.time()
    asr = bllm.load(args.asr)
    llm = bllm.load(args.llm, vision=False)             # the text tower only: less ION
    tts = None
    if args.tts_dir:
        import bcdl                                     # optional: only for the spoken reply
        tts = bcdl.TextToSpeech.from_files(*[os.path.join(args.tts_dir, f) for f in TTS_FILES])
    print(f"loaded {asr.name} + {llm.name} in {time.time() - t:.1f} s\n")

    for i, clip in enumerate(clips):
        r = asr.transcribe(clip, language=args.language)
        lang = f" [{r['language']}]" if r["language"] else ""
        print(f"you{lang}: {r['text']}   ({r['e2e_ms']:.0f} ms for {r['audio_seconds']:.1f} s)")
        print("qwen3.5: ", end="", flush=True)
        t = time.time()
        reply = "".join(piece for piece in _echo(llm.stream_chat(r["text"], max_new=args.max_new)))
        print(f"\n         ({time.time() - t:.1f} s, {llm.last_decode_tps:.1f} tok/s)")
        if tts is not None and reply.strip():
            t = time.time()
            pcm = tts.synthesize(speakable(reply)).pcm
            out = args.out if len(clips) == 1 else f"{os.path.splitext(args.out)[0]}_{i}.wav"
            write_wav(out, pcm, tts.sample_rate)
            print(f"         spoken -> {out} ({len(pcm) / tts.sample_rate:.1f} s of audio in "
                  f"{time.time() - t:.1f} s)")
        print()
    return 0


def speakable(text: str) -> str:
    """An answer as it should be read aloud: no Markdown emphasis, headings, bullets or code."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*(?:[-+]|\d+[.)])\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def _echo(pieces):
    for p in pieces:
        print(p, end="", flush=True)
        yield p


if __name__ == "__main__":
    sys.exit(main())
