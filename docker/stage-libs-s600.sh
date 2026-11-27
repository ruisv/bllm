#!/usr/bin/env bash
# Stage the S600 (nash-p) board's hobot runtime + HAL closure into _stage_s600/ so
# Dockerfile.s600 can bundle it. UNLIKE stage-libs.sh (which stages from the
# LOCAL board's /usr/hobot/lib), this pulls from a remote S600 host over ssh, so
# the image can be built on any board. Point S600_HOST at your S600 ssh target:
#
#   S600_HOST=my-s600 ./stage-libs-s600.sh
#   docker build -f Dockerfile.s600 -t bllm-serve-s600:latest .
#
# The same closure stage-libs.sh takes on an S100, but the S600's own copy: the
# conda packages carry no hobot runtime (0.9.0+), and it has to match the board's
# firmware.
set -euo pipefail

cd "$(dirname "$0")"                 # docker/ (the build context)
S600_HOST="${S600_HOST:?set S600_HOST to your S600 board/sandbox ssh target}"
HOBOT_LIB="${S600_HOBOT_LIB:-/usr/hobot/lib}"
STAGE="_stage_s600/hobot/lib"

# Runtime closure of the conda-packaged _bllm_native.so under /usr/hobot/lib.
# These need GLIBC_2.38 -> Dockerfile.s600 uses an ubuntu:24.04 base.
LIBS=(
    libdnn.so libhbucp.so libhbrt4.so libhbtl.so libhb_arm_rpc.so
    libhlog_wrapper.so libbpu.so.2 libhbmem.so.1 libhbipcfhal.so.1
    libalog.so.1 libvdsp.so.1 libperfetto_sdk.so
)

echo "[stage-s600] ${S600_HOST}:${HOBOT_LIB} -> docker/${STAGE}"
rm -rf _stage_s600
mkdir -p "$STAGE"
# One ssh, deref symlinks (tar -h), so each SONAME arrives as a real file.
ssh "$S600_HOST" "cd '${HOBOT_LIB}' && tar chf - ${LIBS[*]}" | tar xf - -C "$STAGE"
echo "[stage-s600] done ($(ls "$STAGE" | wc -l) libs). Now: docker build -f Dockerfile.s600 -t bllm-serve-s600:latest ."
