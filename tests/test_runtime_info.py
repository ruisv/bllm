"""bllm.runtime_info(): which hobot runtime the process actually loaded.

BLLM links against the board's own hobot SDK instead of bundling one. A bundled copy
shadows the board's: up to 0.8.0 the conda packages pulled `hobot-dnn`, which put an S100
build of libdnn / libhbucp / libhbrt4 / libhbtl into the env, ahead of /usr/hobot/lib —
even on an S100P those were not the files the board ships. runtime_info() is how a user
(or a bug report) tells which copy was in play, so it has to name a real file and real
versions.

Runs on the board.
"""

import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

pytestmark = pytest.mark.skipif(not bllm._HAVE_NATIVE, reason="native runtime not built")


def test_runtime_info_names_the_loaded_runtime():
    info = bllm.runtime_info()
    print(info)
    assert set(info) == {"dnn_version", "ucp_version", "soc_name", "dnn_library"}
    assert info["dnn_version"] and info["ucp_version"]
    assert "RDK" in info["soc_name"]
    lib = info["dnn_library"]
    assert os.path.basename(lib).startswith("libdnn.so"), lib
    assert os.path.isfile(lib), lib


def test_libdnn_comes_from_the_board_not_a_bundled_copy():
    """No copy of libdnn inside a conda env may win over the board's. That is exactly
    what an SDK bundled into the package did, and it is invisible from the results."""
    lib = os.path.realpath(bllm.runtime_info()["dnn_library"])
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        assert not lib.startswith(os.path.realpath(prefix) + os.sep), (
            f"libdnn loaded from the conda env ({lib}), shadowing the board's SDK")
