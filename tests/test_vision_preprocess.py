"""The official LLM 2.0 VLM preprocessing, checked without loading a model.

The official runtime stretches every image to its tower's side with OpenCV INTER_CUBIC; a
VLM reply is only token-identical to theirs when the pixels are identical too. BLLM does
not link OpenCV, so it carries its own copy of that resize — this holds it to cv2, bit
for bit, across shrink/enlarge/odd sizes (OpenCV's vertical pass takes a float path for
whole 8-lane groups and an integer one for the rest of a row, so widths that are not a
multiple of 8 channels-elements matter).

cv2 runs in a subprocess: a conda OpenCV needs conda's newer libstdc++, and once another
test module has imported the native module the process may already be bound to the
system one.
"""

import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

if not bllm._HAVE_NATIVE:
    pytest.skip("native module not built", allow_module_level=True)

from bllm import _bllm_native as nat  # noqa: E402

_CV2 = """
import sys, numpy as np, cv2
src, dst, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
np.save(dst, cv2.resize(np.load(src), (w, h), interpolation=cv2.INTER_CUBIC))
"""


@pytest.fixture(scope="module")
def cv2_resize(tmp_path_factory):
    d = tmp_path_factory.mktemp("cv2")
    if subprocess.run([sys.executable, "-c", "import cv2"], capture_output=True).returncode:
        pytest.skip("OpenCV (cv2) not importable")

    def run(img, w, h):
        src, dst = d / "src.npy", d / "dst.npy"
        np.save(src, img)
        subprocess.run([sys.executable, "-c", _CV2, str(src), str(dst), str(w), str(h)],
                       check=True, capture_output=True)
        return np.load(dst)
    return run


SIZES = [
    ((747, 1000), (448, 448)),   # the official demo image, shrunk to the Qwen3-VL tower
    ((300, 200), (448, 448)),    # enlarged
    ((1080, 1920), (448, 448)),  # wide
    ((2000, 600), (448, 448)),   # tall
    ((448, 448), (448, 448)),    # same size: a copy
    ((97, 131), (37, 53)),       # odd both ways, row width not a multiple of 8
    ((5, 3), (11, 7)),           # smaller than the 4-tap kernel
]


@pytest.mark.parametrize("src,dst", SIZES)
def test_resize_matches_opencv_inter_cubic(cv2_resize, src, dst):
    rng = np.random.default_rng(src[0] * 7919 + src[1])
    img = rng.integers(0, 256, size=(*src, 3), dtype=np.uint8)
    # Natural images mix flat areas with edges; noise alone never lands a sum near .5.
    img[: src[0] // 2] = (img[: src[0] // 2] // 32) * 32
    ref = cv2_resize(img, dst[1], dst[0])
    ours = nat.resize_cubic_cv(img, dst[1], dst[0])
    assert ours.shape == ref.shape
    diff = np.abs(ours.astype(int) - ref.astype(int))
    assert diff.max() == 0, f"{int((diff > 0).sum())} values differ, max {diff.max()}"


def test_resize_matches_on_a_smooth_gradient(cv2_resize):
    # Smooth ramps put many vertical sums exactly on a rounding tie — where the float
    # (half-to-even) and integer (half-up) paths would part.
    y, x = np.mgrid[0:333, 0:517]
    img = np.stack([(x * 255 // 516), (y * 255 // 332), ((x + y) % 256)], -1).astype(np.uint8)
    img = np.ascontiguousarray(img)
    assert np.array_equal(nat.resize_cubic_cv(img, 448, 448), cv2_resize(img, 448, 448))
