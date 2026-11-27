"""Speech recognition on an official D-Robotics Qwen3-ASR package (board only).

    BLLM_TEST_OFFICIAL_ASR_DIR   the unzipped package (Qwen3-ASR-0.6B_w8/ ...)
    BLLM_TEST_ASR_AUDIO          a speech clip (wav, 16 kHz mono preferred)
    BLLM_TEST_ASR_TEXT           its reference transcript (optional; enables the CER check)
    BLLM_TEST_ASR_LANGUAGE       the clip's language as the model names it (default Chinese)
    BLLM_TEST_ASR_EMPTY_AUDIO    optional: speech the model leaves empty unless rescued (with
                                 the 0.6B package, AISHELL-1 test a02993 forced to Chinese)

Transcripts are compared by character error rate, not for identity with the official runtime:
the package's audio encoder is so sensitive that a 1-LSB change in a single input value moves
nearly every output value (row cosine down to 0.81), so two front ends that agree to the last
bit but one give different — equally valid — punctuation and words.
"""

import os
import re
import unicodedata
import wave

import numpy as np
import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

PKG = os.environ.get("BLLM_TEST_OFFICIAL_ASR_DIR", "")
AUDIO = os.environ.get("BLLM_TEST_ASR_AUDIO", "")
REF = os.environ.get("BLLM_TEST_ASR_TEXT", "")
LANG = os.environ.get("BLLM_TEST_ASR_LANGUAGE", "Chinese")
EMPTY = os.environ.get("BLLM_TEST_ASR_EMPTY_AUDIO", "")

pytestmark = [
    pytest.mark.skipif(not getattr(bllm, "_HAVE_ASR", False), reason="native ASR not built"),
    pytest.mark.skipif(not PKG or not os.path.isdir(PKG), reason="set BLLM_TEST_OFFICIAL_ASR_DIR"),
    pytest.mark.skipif(not AUDIO or not os.path.isfile(AUDIO), reason="set BLLM_TEST_ASR_AUDIO"),
]


def _norm(s: str) -> str:
    """Characters only: punctuation, symbols and spaces dropped, Latin lowercased."""
    return "".join(c for c in s.lower() if not unicodedata.category(c)[0] in "PSZ")


def _cer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(h)] / max(1, len(r))


def _pcm(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getsampwidth() == 2 and w.getframerate() == 16000, "test clip must be 16 kHz int16"
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())
    return (a.mean(axis=1) / 32768.0).astype(np.float32)


@pytest.fixture(scope="module")
def asr():
    return bllm.load(PKG)


def test_load_routes_an_asr_package(asr):
    assert isinstance(asr, bllm.NativeAsrSession)
    assert "Qwen3-ASR" in asr.name
    assert 5 <= asr.default_chunk_seconds <= 40


def test_forced_language(asr):
    r = asr.transcribe(AUDIO, language=LANG)
    print(r)
    assert r["language"] == LANG and r["text"] and r["chunks"] == 1
    assert "<asr_text>" not in r["text"] and not r["raw"].startswith("language")
    assert 0 < r["rtf"] < 1
    if REF:
        assert _cer(REF, r["text"]) <= 0.15, (REF, r["text"])


def test_detected_language(asr):
    r = asr.transcribe(AUDIO)
    print(r)
    assert r["text"]
    assert r["raw"].startswith("language")
    # The model names the language, or (rarely) writes <asr_text> straight away: never junk.
    assert r["language"] in ("", LANG), r
    assert "<|im_start|>" not in r["text"]
    if REF:
        assert _cer(REF, r["text"]) <= 0.15, (REF, r["text"])


def test_short_codes_are_language_names(asr):
    code = {"Chinese": "zh", "English": "en"}.get(LANG)
    if not code:
        pytest.skip("no short code for " + LANG)
    assert asr.transcribe(AUDIO, language=code)["language"] == LANG


def test_stream_is_the_transcript(asr):
    pieces = list(asr.stream_transcribe(AUDIO, language=LANG))
    assert pieces
    assert "".join(pieces).strip() == asr.last_result["text"]


def test_array_input_is_the_same_as_the_file(asr):
    if not AUDIO.lower().endswith(".wav"):
        pytest.skip("needs a wav clip")
    a = asr.transcribe(AUDIO, language=LANG)["text"]
    b = asr.transcribe(_pcm(AUDIO), language=LANG)["text"]
    assert a == b
    # int16 and another sample rate go through the same conversion
    pcm = _pcm(AUDIO)
    c = asr.transcribe((pcm * 32767).astype(np.int16), language=LANG)["text"]
    assert _cer(a, c) <= 0.15
    d = asr.transcribe(np.repeat(pcm, 3), language=LANG, sample_rate=48000)["text"]
    assert _cer(a, d) <= 0.2


def test_long_audio_is_cut_and_joined(asr):
    """A clip past what the cache holds in one pass: cut at quiet points, every piece
    transcribed, the texts joined."""
    pcm = _pcm(AUDIO) if AUDIO.lower().endswith(".wav") else None
    if pcm is None:
        pytest.skip("needs a wav clip")
    gap = np.zeros(8000, np.float32)                      # 0.5 s of silence between copies
    reps = int(np.ceil(asr.default_chunk_seconds * 1.6 * 16000 / (len(pcm) + len(gap)))) + 1
    long = np.concatenate([np.concatenate([pcm, gap]) for _ in range(reps)])
    one = asr.transcribe(pcm, language=LANG)["text"]
    r = asr.transcribe(long, language=LANG)
    print(r["chunks"], r["audio_seconds"], r["text"][:120])
    assert r["chunks"] >= 2
    n = len(_norm(one))
    assert abs(len(_norm(r["text"])) - reps * n) <= max(4, 0.2 * reps * n)


def _noise(seconds: float, db: float, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=int(seconds * 16000))
    return (x / np.sqrt((x ** 2).mean()) * 10 ** (db / 20)).astype(np.float32)


@pytest.mark.parametrize("language", [LANG, None], ids=["forced", "detected"])
def test_no_speech_stays_empty(asr, language):
    """An empty reply is decoded once more past its stop. On audio with nothing said, what
    comes out of that is filler ("嗯。", "The.") and must be thrown away."""
    cases = {"silence 1 s": np.zeros(16000, np.float32), "silence 3 s": np.zeros(48000, np.float32),
             "white -55 dB": _noise(4, -55), "white -40 dB": _noise(4, -40),
             "white -28 dB": _noise(4, -28)}
    for name, pcm in cases.items():
        off = asr.transcribe(pcm, language=language, rescue_empty=False)
        on = asr.transcribe(pcm, language=language)
        print(name, repr(off["text"]), repr(on["text"]))
        assert on["text"] == off["text"], name
        if name.startswith("silence"):
            assert on["text"] == "" and on["ttft_ms"] == 0, (name, on)
            if language is None:
                assert on["language"] == "", (name, on)


def test_rescue_leaves_a_reply_alone(asr):
    a = asr.transcribe(AUDIO, language=LANG, rescue_empty=False)
    b = asr.transcribe(AUDIO, language=LANG)
    assert a["text"] == b["text"] and a["ntok"] == b["ntok"]


@pytest.mark.skipif(not EMPTY or not os.path.isfile(EMPTY), reason="set BLLM_TEST_ASR_EMPTY_AUDIO")
def test_speech_left_empty_is_rescued(asr):
    off = asr.transcribe(EMPTY, language=LANG, rescue_empty=False)
    assert off["text"] == "", "this clip is not left empty here; pick another"
    on = asr.transcribe(EMPTY, language=LANG)
    print(on)
    assert len(_norm(on["text"])) >= 5 and on["ttft_ms"] > 0
    pieces = list(asr.stream_transcribe(EMPTY, language=LANG))
    assert "".join(pieces).strip() == asr.last_result["text"] == on["text"]


def test_a_chat_session_refuses_it():
    from bllm import _modelmeta
    d = _modelmeta.official_dir_for(PKG)
    with pytest.raises(Exception, match="asr"):
        bllm.NativeSession(str(d))
