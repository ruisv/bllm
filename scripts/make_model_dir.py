#!/usr/bin/env python3
"""In-tree entry point for the model-dir tool — see `bllm/make_model_dir.py`.

The tool itself lives in the Python package so that it ships with
`conda install bllm` as the `bllm-make-model-dir` command; users who only install
the package must not have to clone this repo to assemble a model directory.
This wrapper keeps `scripts/make_model_dir.py ...` working from a source tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from bllm.make_model_dir import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
