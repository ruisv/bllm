#!/usr/bin/env bash
# Bake a model package into a self-contained, tagged bllm-serve image
# (`docker run` and chat — no volume mount). Run on the board, from docker/.
#
#   ./bake-model.sh ~/models/qwen3.5-2b bllm-serve:qwen3.5-2b-ctx512-int8-s100p
#
# Tag convention (delivery naming): <name>:<model>-<ctx>-<quant>-<board>, e.g.
# bllm-serve:qwen3.5-2b-ctx512-int8-s100p — read the pieces off model.json.
set -euo pipefail
cd "$(dirname "$0")"                       # docker/ (build context)

MODEL_SRC="${1:?usage: bake-model.sh <model_dir> <image:tag>}"
TAG="${2:?usage: bake-model.sh <model_dir> <image:tag>}"
BASE="${BASE:-bllm-serve:latest}"

# 1) BPU lib closure + model-agnostic base runtime image.
./stage-libs.sh
echo "[bake] building base runtime -> ${BASE}"
docker build --network host -f Dockerfile -t "${BASE}" .

# 2) stage the model into the build context and layer it on top of the base.
echo "[bake] staging model ${MODEL_SRC} -> _model/"
rm -rf _model && mkdir -p _model
cp -aL "${MODEL_SRC}"/. _model/
echo "[bake] baking model -> ${TAG}"
docker build --network host -f Dockerfile.model --build-arg BASE="${BASE}" -t "${TAG}" .
rm -rf _model

echo "[bake] done: ${TAG}  (check size with: docker image ls ${TAG%%:*})"
echo "[bake] run: docker run -d --name bllm-serve --network host \\"
echo "         --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \\"
echo "         --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu ${TAG}"
