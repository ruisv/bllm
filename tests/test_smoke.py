"""Surface tests for the native runtime — the only runtime now (the libxlm backend was
removed; the design notes, tag v-libxlm-final).

The native engine may be absent from a dev-host checkout, so its tests are gated. Model
tests are skipped unless a model is pointed at via env — set the `BLLM_TEST_*` env vars (see CONTRIBUTING.md).

**One big model per process.** A session pins its weights in the BPU's ION carveout until
it is destroyed, so the native LLM and VLM tests live in separate modules (pytest finalises
a module-scoped fixture before the next module starts).
"""

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

_need_native = pytest.mark.skipif(not bllm._HAVE_NATIVE, reason="native runtime not built")


def test_version():
    assert isinstance(bllm.__version__, str) and bllm.__version__


def test_all_only_advertises_names_that_exist():
    """`from bllm import *` must work even on a host with no runtime built."""
    assert all(hasattr(bllm, n) for n in bllm.__all__), bllm.__all__
    assert set(bllm.available_backends()) <= {"native"}


def test_no_libxlm_symbols_survive():
    """The removed backend leaves no public surface behind."""
    for gone in ("LlmSession", "OmniSession", "VlmSession", "SamplingParams",
                 "GenerationStats", "Content"):
        assert not hasattr(bllm, gone), gone
    assert "libxlm" not in bllm.available_backends()


@_need_native
def test_native_surface():
    for name in ("NativeSession", "NativeVlmSession", "load_video"):
        assert hasattr(bllm, name), name
