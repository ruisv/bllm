"""Decode an OpenAI ``image_url`` value into bytes on disk.

Two forms: a ``data:image/...;base64,...`` inline image, or an ``http(s)://``
reference. Both come back as a real file rather than an in-memory array because
the native VLM session's image argument already accepts a path (decoded with
stb_image on the C++ side) — writing one temp file avoids adding a Pillow/numpy
dependency to the server image just to re-decode what stb_image decodes anyway.

No bllm import: this is pure I/O and runs in CI on any host (mock urlopen for the
http branch — see tests/test_serve_media.py).
"""
from __future__ import annotations

import base64
import os
import re
import tempfile
import urllib.request
from urllib.parse import urlparse

_DATA_URL = re.compile(r"^data:([\w.+-]+/[\w.+-]+)?(;charset=[^;,]+)?;base64,(.*)$", re.S)

# A generic extension is fine: stb_image sniffs the format from the file's magic
# bytes, not its name.
_TMP_SUFFIX = ".bllm-image"


def fetch_image_bytes(url: str, *, timeout: float = 10.0, max_bytes: int = 32 << 20) -> bytes:
    """Decode ``url`` to raw image bytes. Raises ``ValueError`` on anything malformed
    or unsupported — the caller turns that into a 400, not a 500."""
    m = _DATA_URL.match(url)
    if m:
        try:
            data = base64.b64decode(m.group(3), validate=True)
        except Exception as exc:
            raise ValueError(f"image_url: bad base64 data URL: {exc}") from exc
        if not data:
            raise ValueError("image_url: empty base64 data URL")
        return data

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"image_url: unsupported scheme {parsed.scheme!r}; use a data: URI "
            f"or an http(s):// URL")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — scheme checked above
            data = resp.read(max_bytes + 1)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"image_url: fetching {url!r} failed: {exc}") from exc
    if len(data) > max_bytes:
        raise ValueError(f"image_url: {url!r} exceeds the {max_bytes} byte limit")
    if not data:
        raise ValueError(f"image_url: {url!r} returned no data")
    return data


def image_url_to_tempfile(url: str, **kwargs) -> str:
    """``fetch_image_bytes`` written to a fresh temp file; returns its path. The
    caller owns the file and must remove it (``os.unlink``) once done with it."""
    data = fetch_image_bytes(url, **kwargs)
    fd, path = tempfile.mkstemp(suffix=_TMP_SUFFIX)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        os.unlink(path)
        raise
    return path
