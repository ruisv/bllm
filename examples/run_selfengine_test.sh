#!/usr/bin/env bash
# Build + run the self-built hbDNN LLM engine (bllm_selfengine) on the board and
# generate text from our own converted Qwen3-1.7B .hbm — WITHOUT libxlm.
#
# The engine links only the generic hobot BPU runtime (libdnn / libhbucp /
# libhbrt4). It never links or loads libxlm.so. `ldd` on the built binary
# confirms this.
#
# Usage (on the RDK S100P board):
#   bash run_selfengine_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HBM="${HBM:-$HOME/qwen3_deploy/qwen3-1.7b.hbm}"
CONFIG="${CONFIG:-$HOME/qwen3_deploy/qwen3-1.7b_config}"
BIN="$HERE/bllm_selfengine"

echo ">> building bllm_selfengine (hbDNN only, no libxlm) ..."
g++ -O2 -std=c++17 -I/usr/include/hobot "$HERE/selfengine.cc" \
    -L/usr/hobot/lib -ldnn -lhbucp -lhbrt4 -o "$BIN"

export LD_LIBRARY_PATH=/usr/hobot/lib:${LD_LIBRARY_PATH:-}

echo ">> proof it is not linked against libxlm:"
if ldd "$BIN" | grep -qi xlm; then
  echo "   !! libxlm present in ldd — unexpected"; ldd "$BIN" | grep -i xlm
else
  echo "   OK: libxlm.so absent from ldd. Runtime libs:"
  ldd "$BIN" | grep -iE "dnn|hbucp|hbrt" | sed 's/^/     /'
fi

# best-effort performance mode (ignore if not permitted)
sudo /usr/bin/busybox devmem 0x2b047000 32 0x99 2>/dev/null || true
sudo /usr/bin/busybox devmem 0x2b047004 32 0x99 2>/dev/null || true

echo ">> running generation test ..."
python3 "$HERE/test_selfengine.py" --engine "$BIN" --hbm "$HBM" --config "$CONFIG"
