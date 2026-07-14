# bllm-serve — containerized deployment

An OpenAI-compatible HTTP inference server for the BLLM native runtime, packaged
as a Docker image for RDK **S100 / S100P / S600** (aarch64 only).

## Design

A *device-coupled* container (the NVIDIA model), not an isolated app:

| Layer | Where it lives | Why |
|-------|----------------|-----|
| ubuntu:22.04 + py3.12 + `bllm`/`libbllm`/`tokenizers-cpp` (conda) + our server | **baked in** | our stack, weakly coupled to board OS |
| BPU userspace `.so` closure (`libdnn`/`libhbucp`/… → `/usr/hobot/lib`) | **baked in** (staged from the board at build) | self-contained; mirrors the board path so RPATH resolves |
| kernel BPU driver + `/dev/{bpu,ion,…}` nodes | **mapped from the board** (`--device`) | driver is in the host kernel; must match the running board |
| `.hbm` model + tokenizer/`model.json` | **volume mounted** (`-v …:/models`) | GB-scale, per-deploy choice |

Base is `ubuntu:22.04` because the D-Robotics BPU libs hard-require
**GLIBC_2.34 + GLIBCXX_3.4.30** — a glibc-2.28 base (Debian 10) cannot load them.

## Build + run (on the board, via docker compose)

One-time prep: stage the proprietary BPU lib closure from the board's
`/usr/hobot/lib` into `docker/_stage/` (gitignored) — compose bundles it but
can't copy host files itself:

```bash
docker/stage-libs.sh
```

Then build + start (from the repo root, where `compose.yaml` lives):

```bash
MODEL_DIR=~/models/qwen3.5-2b docker compose up -d --build
docker compose logs -f          # follow startup / model load
docker compose down             # stop + remove
```

`compose.yaml` sets `network_mode: host` **and** `build.network: host` — this
board's kernel lacks the iptables `raw` table, so Docker's default bridge fails
both at build and run. The BPU/ION device nodes and model volume are declared in
the compose file; per-deploy knobs go through env vars (`MODEL_DIR`, `BLLM_PORT`,
`BLLM_API_KEY`, …) or a `.env` file next to `compose.yaml`.

### API

```bash
curl localhost:8000/health
curl localhost:8000/v1/models
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"数到5"}],"stream":true}'
```

OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming),
`/v1/models`, `/health`. The engine is a **singleton** — one generation in
flight at a time (BPU = single prefill/decode graph); requests serialize. Run a
**single** uvicorn worker.

### Env knobs

| Var | Default | Meaning |
|-----|---------|---------|
| `BLLM_MODEL` | — (required) | model dir (with `model.json`) or a bare `.hbm`, path inside the container |
| `BLLM_TOKENIZER_DIR` | — | tokenizer dir, only when `BLLM_MODEL` is a bare `.hbm` |
| `BLLM_BACKEND` | `auto` | `bllm.load` backend |
| `BLLM_MAX_NEW` | `400` | default `max_tokens` when a request omits it |
| `BLLM_API_KEY` | — | if set, require `Authorization: Bearer <key>` |
| `BLLM_BPU_PRIORITY` | — | `set_bpu_priority()` at startup |
| `BLLM_PERF_MODE` | `0` | `1` = poke BPU perf registers (needs `--privileged`; prefer doing it on the host) |
| `BLLM_PORT` / `BLLM_HOST` | `8000` / `0.0.0.0` | bind |

## Host prerequisites (once, outside the container)

- **ION carveout**: `sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot`
  (`bpu_first` for 7B). A boot-time kernel carveout — cannot be set from a container.
- **Docker**: if the daemon fails on `iptables (nf_tables)`, switch to the legacy
  backend: `sudo update-alternatives --set iptables /usr/sbin/iptables-legacy`
  (this kernel only ships the legacy xtables path).

## Rebuild triggers

Re-run `docker/stage-libs.sh && docker compose up -d --build` when the board's
**BSP / BPU driver** is upgraded — the bundled `/usr/hobot/lib` closure must
match the running kernel driver ABI.
