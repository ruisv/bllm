#!/usr/bin/env bash
# Build BLLM from source, run ON an RDK S100 / S100P / S600 board.
#
#   ./scripts/build.sh              # build the C++ library
#   ./scripts/build.sh --python     # + the Python bindings (needs nanobind)
#   ./scripts/build.sh --clean      # wipe build/ first
#   ./scripts/build.sh --python --install   # also install into the active env/prefix
#
# Prerequisites on the board — the build/runtime dependencies, e.g. from our conda channel:
#
#   conda install -c https://mirrors.ruis.ai/conda -c conda-forge \
#       cmake ninja cxx-compiler nlohmann_json hobot-dnn tokenizers-cpp nanobind numpy
#
# then activate that environment and run this from the repository root. (Most users can skip
# building and just `conda install bllm` — see the README. Build from source to hack on it.)
set -euo pipefail

CLEAN=0; PYTHON=OFF; INSTALL=0
for a in "$@"; do
  case "$a" in
    --clean)   CLEAN=1 ;;
    --python)  PYTHON=ON ;;
    --install) INSTALL=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

command -v cmake >/dev/null || { echo "error: cmake not found — install the prereqs above" >&2; exit 1; }
command -v ninja >/dev/null || { echo "error: ninja not found — install the prereqs above" >&2; exit 1; }

[ "$CLEAN" = 1 ] && rm -rf build

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBLLM_BUILD_PYTHON="$PYTHON" \
  -DBLLM_INSTALL_PYTHON=$( [ "$INSTALL" = 1 ] && echo ON || echo OFF )

cmake --build build -j"$(nproc 2>/dev/null || echo 4)"

if [ "$INSTALL" = 1 ]; then
  cmake --install build
  echo ">> installed into the active prefix."
else
  echo ">> build done. C++ library in build/."
  [ "$PYTHON" = ON ] && echo ">> Python: export PYTHONPATH=\"$PWD/python\" to import bllm in-tree."
fi
