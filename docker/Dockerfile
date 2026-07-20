# bllm-serve — OpenAI-compatible HTTP inference server for the BLLM native
# runtime on RDK S100/S100P/S600 (aarch64 only).
#
# Base: ubuntu:22.04 (glibc 2.35). The D-Robotics BPU userspace libs bundled
# below hard-require GLIBC_2.34 + GLIBCXX_3.4.30, so an older base (Debian 10 /
# glibc 2.28) will NOT load them — 22.04 also byte-matches the board OS.
#
# What is baked in : ubuntu22.04 + python3.12 + our stack (bllm/libbllm/
#                    tokenizers-cpp via conda) + the BPU userspace .so closure.
# What is NOT baked : the kernel BPU driver + /dev nodes (mapped from the board
#                     at run time) and the .hbm model (mounted as a volume).
#
# Build context is this docker/ dir. Run stage-libs.sh first (it stages the
# /usr/hobot lib closure into _stage/ — proprietary, gitignored), then
# `docker compose up -d --build` from here.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl bzip2 busybox-static \
        libcjson1 libjsoncpp25 \
    && rm -rf /var/lib/apt/lists/*

# --- python 3.12 + our stack from the conda channel -------------------------
ARG MAMBA_VERSION=1.5.8
ARG BLLM_CHANNEL=https://mirrors.ruis.ai/conda
# Pinned so this layer is not silently reused with a stale runtime: the
# install line is otherwise identical across releases, so Docker would cache
# it and the image would keep shipping whatever bllm it first resolved.
ARG BLLM_VERSION=0.1.5
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
# Everything in one layer: extract micromamba to /tmp (thrown away here, not
# persisted), create the env, then purge the pkg cache + micromamba + bytecode.
# clean -a leaves /opt/conda/pkgs (it is hardlinked into the env), so rm it too.
RUN curl -Ls "https://micro.mamba.pm/api/micromamba/linux-aarch64/${MAMBA_VERSION}" \
        | tar -xj -C /tmp bin/micromamba \
    && /tmp/bin/micromamba create -y -p /opt/conda \
        -c "${BLLM_CHANNEL}" -c conda-forge \
        python=3.12 bllm=${BLLM_VERSION} libbllm=${BLLM_VERSION} tokenizers-cpp \
    && /tmp/bin/micromamba clean -a -y \
    && rm -rf /opt/conda/pkgs /tmp/bin \
    && find /opt/conda -type f -name '*.pyc' -delete \
    && find /opt/conda -type d -name '__pycache__' -exec rm -rf {} +

# --- pure-python HTTP layer -------------------------------------------------
COPY requirements.txt /tmp/req.txt
RUN /opt/conda/bin/pip install --no-cache-dir -r /tmp/req.txt

# --- bundled BPU userspace libs (staged from board /usr/hobot/lib) ----------
# Mirror the board path so any hard-coded RPATH (=/usr/hobot/lib) resolves too.
COPY _stage/hobot/lib/ /usr/hobot/lib/
ENV LD_LIBRARY_PATH=/usr/hobot/lib:/opt/conda/lib

# --- server -----------------------------------------------------------------
COPY server.py /app/server.py
COPY serving/ /app/serving/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
WORKDIR /app
EXPOSE 8866
ENTRYPOINT ["/entrypoint.sh"]
