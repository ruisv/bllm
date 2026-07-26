#!/usr/bin/env python3
"""Unit tests for image_url decoding. Pure logic — runs on any host, no board."""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))

from serving.media import fetch_image_bytes, image_url_to_tempfile  # noqa: E402

PNG_1PX = base64.b64decode(  # a real 1x1 PNG, so this also exercises a plausible payload
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def test_data_url_roundtrips():
    assert fetch_image_bytes(data_url(PNG_1PX)) == PNG_1PX


def test_data_url_without_explicit_mime():
    assert fetch_image_bytes(f"data:;base64,{base64.b64encode(PNG_1PX).decode()}") == PNG_1PX


def test_data_url_bad_base64_is_a_value_error():
    with pytest.raises(ValueError, match="base64"):
        fetch_image_bytes("data:image/png;base64,not-valid-base64!!!")


def test_data_url_empty_is_a_value_error():
    with pytest.raises(ValueError, match="empty"):
        fetch_image_bytes("data:image/png;base64,")


def test_unsupported_scheme_is_a_value_error():
    with pytest.raises(ValueError, match="scheme"):
        fetch_image_bytes("ftp://example.com/x.png")
    with pytest.raises(ValueError, match="scheme"):
        fetch_image_bytes("not a url at all")


def test_http_url_fetched_and_size_capped():
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return PNG_1PX[:n]

    with patch("serving.media.urllib.request.urlopen", return_value=FakeResp()):
        assert fetch_image_bytes("https://example.com/x.png") == PNG_1PX

    with patch("serving.media.urllib.request.urlopen", return_value=FakeResp()):
        with pytest.raises(ValueError, match="byte limit"):
            fetch_image_bytes("https://example.com/x.png", max_bytes=len(PNG_1PX) - 1)


def test_http_url_failure_is_a_value_error_not_whatever_urllib_raised():
    with patch("serving.media.urllib.request.urlopen", side_effect=OSError("dns fail")):
        with pytest.raises(ValueError, match="fetching"):
            fetch_image_bytes("https://example.com/x.png")


def test_tempfile_written_and_caller_owns_cleanup():
    path = image_url_to_tempfile(data_url(PNG_1PX))
    try:
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == PNG_1PX
    finally:
        os.unlink(path)
