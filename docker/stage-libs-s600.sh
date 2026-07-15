#!/usr/bin/env bash
# Stage the S600 (nash-p) board's device-facing HAL libs into _stage_s600/ so
# Dockerfile.s600 can bundle them. UNLIKE stage-libs.sh (which stages from the
# LOCAL board's /usr/hobot/lib), the S600 HAL lives on a remote S600 host, so this
# pulls over ssh. Point S600_HOST at your S600 board/sandbox ssh target:
#
#   S600_HOST=my-s600 ./stage-libs-s600.sh
#   docker build -f Dockerfile.s600 -t bllm-serve-s600:latest .
#
# Only the HAL is staged. The DNN runtime (libdnn/libhbrt4/libhbtl/libhbucp) is
# NOT — it comes from the hobot-dnn-s600 conda package (HBRT 4.10.1) baked by the
# Dockerfile. Staging the board's 4.8.1 libhbrt4 here would shadow it and break
# nash-p .hbm loading.
set -euo pipefail

cd "$(dirname "$0")"                 # docker/ (the build context)
S600_HOST="${S600_HOST:?set S600_HOST to your S600 board/sandbox ssh target}"
HOBOT_LIB="${S600_HOBOT_LIB:-/usr/hobot/lib}"
STAGE="_stage_s600/hobot/lib"

# Device-facing HAL closure (what the conda .so's dlopen from /usr/hobot/lib).
# These need GLIBC_2.38 -> Dockerfile.s600 uses an ubuntu:24.04 base.
LIBS=(
    libbpu.so.2 libhbmem.so.1 libhbipcfhal.so.1 libvdsp.so.1 libalog.so.1
    libhb_arm_rpc.so libhlog_wrapper.so libperfetto_sdk.so
)

echo "[stage-s600] ${S600_HOST}:${HOBOT_LIB} -> docker/${STAGE}"
rm -rf _stage_s600
mkdir -p "$STAGE"
# One ssh, deref symlinks (tar -h), so each SONAME arrives as a real file.
ssh "$S600_HOST" "cd '${HOBOT_LIB}' && tar chf - ${LIBS[*]}" | tar xf - -C "$STAGE"
echo "[stage-s600] done ($(ls "$STAGE" | wc -l) libs). Now: docker build -f Dockerfile.s600 -t bllm-serve-s600:latest ."
