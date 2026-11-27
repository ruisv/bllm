#!/usr/bin/env python3
"""End-to-end tests of /v1/audio/transcriptions against a running bllm-serve.

Skipped unless BLLM_SERVE_ASR_URL points at a live server started with BLLM_ASR_MODEL,
and BLLM_TEST_ASR_AUDIO names a speech clip (wav):

    BLLM_ASR_MODEL=~/llm_s100/llm2/Qwen3-ASR-0.6B_w8 uvicorn server:app --port 8878 &
    BLLM_SERVE_ASR_URL=http://127.0.0.1:8878 BLLM_TEST_ASR_AUDIO=clip.wav \\
        pytest tests/test_serve_asr_e2e.py -v
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid

import pytest

BASE = os.environ.get("BLLM_SERVE_ASR_URL")
AUDIO = os.environ.get("BLLM_TEST_ASR_AUDIO", "")
if not BASE or not os.path.isfile(AUDIO):
    pytest.skip("BLLM_SERVE_ASR_URL / BLLM_TEST_ASR_AUDIO not set — needs a running ASR server",
                allow_module_level=True)


def _post(fields: dict, file_bytes: bytes | None = None, filename: str = "clip.wav",
          timeout: float = 300):
    """A multipart/form-data POST, standard library only."""
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    if file_bytes is not None:
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                     f'Content-Type: application/octet-stream\r\n\r\n'.encode() + file_bytes + b"\r\n")
    body = b"".join(parts) + f"--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + "/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return urllib.request.urlopen(req, timeout=timeout)


@pytest.fixture(scope="module")
def wav() -> bytes:
    return open(AUDIO, "rb").read()


def test_json(wav):
    with _post({"language": "zh"}, wav) as r:
        out = json.load(r)
    assert set(out) == {"text"} and out["text"]


def test_text_and_verbose(wav):
    with _post({"response_format": "text", "language": "Chinese"}, wav) as r:
        text = r.read().decode()
    with _post({"response_format": "verbose_json", "language": "Chinese"}, wav) as r:
        v = json.load(r)
    assert v["text"] == text and v["language"] == "Chinese" and v["duration"] > 0
    assert v["bllm"]["rtf"] > 0


def test_detected_language_is_reported(wav):
    with _post({"response_format": "verbose_json"}, wav) as r:
        v = json.load(r)
    assert v["text"] and v["language"] in ("", "Chinese", "English")


def test_stream_ends_with_the_whole_text(wav):
    with _post({"stream": "true", "language": "zh"}, wav) as r:
        events = [json.loads(l[6:]) for l in r.read().decode().splitlines()
                  if l.startswith("data: ") and l != "data: [DONE]"]
    deltas = "".join(e["delta"] for e in events if e["type"] == "transcript.text.delta")
    done = [e for e in events if e["type"] == "transcript.text.done"]
    assert len(done) == 1 and done[0]["text"] and deltas.strip() == done[0]["text"]


def test_bad_requests_are_4xx(wav):
    for fields, data, code in (({}, None, 400),                         # no file
                               ({"response_format": "srt"}, wav, 400),  # unsupported format
                               ({}, b"not audio at all", 415)):         # undecodable
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(fields, data)
        if code == 415 and e.value.code == 400:
            continue                  # a server with ffmpeg rejects it as undecodable instead
        assert e.value.code == code, (fields, e.value.code)


def test_models_lists_the_asr_model():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        ids = [m["id"] for m in json.load(r)["data"]]
    assert any("ASR" in i for i in ids), ids
