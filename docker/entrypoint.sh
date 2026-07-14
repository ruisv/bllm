#!/usr/bin/env bash
# bllm-serve entrypoint: optional BPU performance mode, then launch uvicorn.
set -euo pipefail

# Optional BPU performance mode (S100/S100P register pokes). Needs --privileged
# + a writable /dev/mem. Prefer doing this once on the HOST; this is a
# convenience for self-contained deploys. Register addresses are board-specific
# (these are S100/S100P); override via BLLM_PERF_REGS="addr0 addr1".
if [ "${BLLM_PERF_MODE:-0}" = "1" ]; then
    regs="${BLLM_PERF_REGS:-0x2b047000 0x2b047004}"
    if [ -w /dev/mem ]; then
        for r in $regs; do busybox devmem "$r" 32 0x99 || true; done
        echo "[entrypoint] BPU performance mode set ($regs)"
    else
        echo "[entrypoint] WARN: perf mode requested but /dev/mem not writable (need --privileged)"
    fi
fi

if [ -z "${BLLM_MODEL:-}" ]; then
    echo "[entrypoint] ERROR: BLLM_MODEL is not set (mount a model and point BLLM_MODEL at it)" >&2
    exit 1
fi

exec uvicorn server:app \
    --host "${BLLM_HOST:-0.0.0.0}" \
    --port "${BLLM_PORT:-8000}" \
    --workers 1
