"""Speech recognition for bllm-serve: one Qwen3-ASR session, one transcription at a time.

The chat Engine owns its model on a worker thread; transcription is simpler — a request is
one blocking call with no cache or turn state — so a lock serialises it, and the same
admission bound as chat (BLLM_MAX_QUEUE) turns an overload into a 429, not a pile-up.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable, Optional

import bllm

from .engine import QueueFull


class AsrEngine:
    def __init__(self, model: str) -> None:
        self.sess = bllm.load(model)
        if not hasattr(self.sess, "transcribe"):
            raise RuntimeError(f"BLLM_ASR_MODEL={model} is not a speech recognition package "
                               "(expected an official Qwen3-ASR package or an arch=\"asr\" dir)")
        self.name = self.sess.name
        self.max_queue = int(os.environ.get("BLLM_MAX_QUEUE", "8"))
        self._lock = threading.Lock()
        self._depth = 0
        self._depth_lock = threading.Lock()
        self.n_requests = 0
        self.audio_seconds = 0.0
        self.busy_seconds = 0.0
        self.ffmpeg = shutil.which("ffmpeg")

    def decode(self, data: bytes, filename: str):
        """Upload bytes -> what the session takes: a WAV goes to the runtime's own decoder,
        anything else through ffmpeg when the image has it."""
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.write(data)
            f.close()
            return f.name
        if not self.ffmpeg:
            raise ValueError(f"{filename or 'the upload'} is not a WAV file and this server "
                             "has no ffmpeg to decode it — send 16 kHz WAV, or install ffmpeg")
        import numpy as np
        p = subprocess.run([self.ffmpeg, "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar",
                            "16000", "-f", "f32le", "pipe:1"], input=data, capture_output=True)
        if p.returncode != 0:
            raise ValueError("could not decode the audio: " + p.stderr.decode(errors="replace")[-300:])
        return np.frombuffer(p.stdout, dtype=np.float32).copy()

    def transcribe(self, audio, language: str, context: str,
                   on_text: Optional[Callable[[str], None]] = None) -> dict:
        with self._depth_lock:
            if self._depth >= self.max_queue:
                raise QueueFull(f"{self._depth} transcriptions already queued "
                                f"(BLLM_MAX_QUEUE={self.max_queue}); retry later")
            self._depth += 1
        try:
            with self._lock:
                t0 = time.time()
                r = self.sess.transcribe(audio, language=language or None, context=context or "",
                                         on_text=on_text)
                self.busy_seconds += time.time() - t0
                self.n_requests += 1
                self.audio_seconds += r["audio_seconds"]
                return r
        finally:
            with self._depth_lock:
                self._depth -= 1
            if isinstance(audio, str) and audio.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(audio)
                except OSError:
                    pass

    def metrics(self) -> dict:
        return {"model": self.name, "requests": self.n_requests, "queue_depth": self._depth,
                "audio_seconds": round(self.audio_seconds, 1),
                "rtf": round(self.busy_seconds / self.audio_seconds, 3) if self.audio_seconds else None,
                "decodes_non_wav": bool(self.ffmpeg)}
