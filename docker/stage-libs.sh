#!/usr/bin/env bash
# Stage the D-Robotics BPU userspace lib closure from the board's /usr/hobot/lib
# into _stage/ (gitignored) so the image can bundle it. Run once on the board,
# from this docker/ dir, before `docker compose build` (compose cannot copy
# host libs):
#
#   ./stage-libs.sh
#   MODEL_DIR=~/models/qwen3.5-2b docker compose up -d --build
set -euo pipefail

cd "$(dirname "$0")"             # docker/ (the build context)
HOBOT_LIB="${HOBOT_LIB:-/usr/hobot/lib}"
STAGE="_stage/hobot/lib"

# Runtime closure of the conda-packaged _bllm_native.so under /usr/hobot/lib.
LIBS=(
    libdnn.so libhbucp.so libhbrt4.so libhbtl.so libhb_arm_rpc.so
    libhlog_wrapper.so libbpu.so.2 libhbmem.so.1 libhbipcfhal.so.1
    libalog.so.1 libvdsp.so.1 libperfetto_sdk.so
)

echo "[stage] ${HOBOT_LIB} -> docker/${STAGE}"
rm -rf _stage
mkdir -p "$STAGE"
for l in "${LIBS[@]}"; do
    if [ -e "${HOBOT_LIB}/${l}" ]; then
        cp -aL "${HOBOT_LIB}/${l}" "$STAGE/"
    else
        echo "[stage] WARN: ${HOBOT_LIB}/${l} not found — image may fail to load" >&2
    fi
done
echo "[stage] done ($(ls "$STAGE" | wc -l) libs). Now: docker compose up -d --build"
