#!/usr/bin/env bash
# Launch an interactive chat on bllm_selfengine — our own hbDNN LLM engine
# driving our converted Qwen3-1.7B .hbm, WITHOUT libxlm.
#
# Builds the engine if needed, makes sure `tokenizers` is present, then drops you
# into a REPL. Type questions; watch tokens stream from the BPU. `/think on|off`
# toggles Qwen3 reasoning, `/exit` quits.
#
# Run ON the board with a terminal:
#   bash chat_selfengine.sh
# or from your Mac (note the -t for an interactive TTY):
#   ssh -t rdk 'bash ~/qwen3_deploy/chat_selfengine.sh'
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HBM="${HBM:-$HOME/qwen3_deploy/qwen3-1.7b.hbm}"
CONFIG="${CONFIG:-$HOME/qwen3_deploy/qwen3-1.7b_config}"
PY="${PY:-$HOME/conda/envs/rdkpy312/bin/python}"
BIN="$HERE/bllm_selfengine"

if [ ! -x "$BIN" ] || [ "$HERE/selfengine.cc" -nt "$BIN" ]; then
  echo ">> building bllm_selfengine (hbDNN only, no libxlm) ..."
  g++ -O2 -std=c++17 -I/usr/include/hobot "$HERE/selfengine.cc" \
      -L/usr/hobot/lib -ldnn -lhbucp -lhbrt4 -o "$BIN"
fi

if ! "$PY" -c "import tokenizers" 2>/dev/null; then
  echo ">> installing tokenizers (one-time) ..."
  "$PY" -m pip install -q tokenizers
fi

export LD_LIBRARY_PATH=/usr/hobot/lib:${LD_LIBRARY_PATH:-}
exec "$PY" "$HERE/chat_selfengine.py" \
  --engine "$BIN" --hbm "$HBM" --config "$CONFIG" "$@"
