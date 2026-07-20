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

Everything lives in this `docker/` dir — run the commands from here. One-time
prep stages the proprietary BPU lib closure from the board's `/usr/hobot/lib`
into `_stage/` (gitignored) — compose bundles it but can't copy host files:

```bash
cd docker
./stage-libs.sh
```

Then build + start:

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

### Self-contained image (model baked in)

For a distributable "just `docker run` and chat" deliverable, bake a model into
the image with `bake-model.sh` (builds the agnostic base, then layers the model
on top). Tag it by the delivery convention `<name>:<model>-<ctx>-<quant>-<board>`
(read the pieces off `model.json`):

```bash
cd docker
./bake-model.sh ~/models/qwen3.5-2b bllm-serve:qwen3.5-2b-ctx512-int8-s100p
docker run -d --name bllm-serve --network host \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu \
  bllm-serve:qwen3.5-2b-ctx512-int8-s100p     # no volume, no MODEL_DIR
```

The baked image is base (~760 MB) + the model package (Qwen3.5-2B ≈ 2.8 GB, incl.
`embed_tokens.bin`), so ~3.5 GB to push/`docker save`. On-disk it reads larger in
`docker image ls` because the containerd store keeps both the compressed blob and
the unpacked snapshot. Still needs the BPU/ION device passthrough at run time.

### Deliverable images

| Tag | Contents | ~push/`save` size |
|-----|----------|-------------------|
| `bllm-serve:0.1.2` (+ `latest`) | general runtime — model via volume | ~760 MB |
| `bllm-serve:qwen3.5-0.8b-ctx2k-int8-s100p` | + Qwen3.5-0.8B baked in | ~2.1 GB |
| `bllm-serve:qwen3.5-2b-ctx512-int8-s100p` | + Qwen3.5-2B baked in | ~3.5 GB |

The general image is the base the baked ones layer on (`bake-model.sh` builds it
as `bllm-serve:latest`; tag it by the bllm version, e.g. `bllm-serve:0.1.2`).

### API

```bash
curl localhost:8866/health
curl localhost:8866/v1/models
curl localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
curl -N localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
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
| `BLLM_MAX_NEW` | `1024` | default `max_tokens` when a request omits it |
| `BLLM_API_KEY` | — | if set, require `Authorization: Bearer <key>` |
| `BLLM_BPU_PRIORITY` | — | `set_bpu_priority()` at startup |
| `BLLM_MAX_QUEUE` | `8` | queued requests before the server returns 429 (one BPU graph = one generation at a time) |
| `BLLM_CACHE_MAX_MB` | `auto` | prefix-cache budget in MB, or `auto` (40% of `MemAvailable`, 256 MB–4 GB); `0` disables |
| `BLLM_CACHE_TTL_S` | `1800` | drop a cached prefix unused for this long |
| `BLLM_CACHE_MAX_ENTRIES` | `64` | entry cap alongside the byte budget |
| `BLLM_PERF_MODE` | `0` | `1` = poke BPU perf registers (needs `--privileged`; prefer doing it on the host) |
| `BLLM_CORS_ORIGINS` | `*` | allowed CORS origins (comma-separated); default open for browser clients |
| `BLLM_PORT` / `BLLM_HOST` | `8866` / `0.0.0.0` | bind |


### Prefix cache (multi-turn without re-prefilling)

The OpenAI protocol is stateless — the client re-sends the whole `messages[]` every
turn — so a naive server re-prefills the entire history each time. This one caches
engine snapshots keyed by **the token sequence they cover** and prefills only what is
new. Per-request hits show up in `usage.prompt_tokens_details.cached_tokens`;
aggregates are on `GET /metrics`.

Matching is exact token-prefix comparison, so a **wrong reuse is not possible**: an
edited history, a changed system prompt or different tokenization simply matches a
shorter entry or misses entirely and falls back to a full prefill.

**The win depends on the engine** (measured on an S100P, client re-sending history):

| Engine | Model | miss → hit | Speedup |
|---|---|---|---|
| dense | Qwen2.5-1.5B | 271ms → 138ms | ~2x |
| hybrid | Qwen3.5-0.8B | 31.6s → 1.3s | ~25x |

The hybrid engine has no batch prefill graph — it ingests ~68ms/token, so a
512-token prompt costs 35 s. **For hybrid (Qwen3.5) models this is not an
optimization but the difference between usable and not; do not disable it.**

**When it does not help** (normal degradation, not a fault):

- one-shot questions on a fresh topic each time — there is no prefix to reuse
- clients that trim or summarize history (some web UIs do) — the prefix changes and
  every entry misses
- prompts shorter than one prefill chunk (typically 256 tokens on dense) — there is
  no safe snapshot point, though re-prefilling such a prompt is cheap anyway


> **Note:** the image installs `bllm` from the conda channel. If that build predates
> the snapshot primitives the server **degrades to full re-prefill** (it says so at
> startup): no prefix cache, but the concurrency fix, admission control, real token
> usage and `/metrics` all still apply. Check `supports_prefix_cache` on `GET /metrics`.

**Cache budget:** measured in bytes, not entries — a snapshot's size differs by an order
of magnitude between the two engines and grows with the conversation. Dense is ~22 KB/token;
hybrid carries a ~22 MB floor of GDN recurrent state plus ~24.6 KB/token on top (measured on
Qwen3.5-2B ctx4k: ~25 MB for a 136-token conversation, ~30 MB for a typical two-turn one).

**`auto` is the default and usually needs no tuning:** with `BLLM_CACHE_MAX_MB` unset the
server takes a 40% share of `MemAvailable` at startup (floored at 256 MB, capped at 4 GB) and
logs what it chose. On an 8 GB board that came out around 2.5 GB, holding **80+**
conversations.

This matters most when several servers or containers share a board — each picking a
plausible-looking fixed number is how you exceed total RAM (two processes set to 2 GB and
3 GB drove `MemAvailable` down to 2.4 GB in testing). Override with an explicit MB value, or
`0` to disable the cache:

```bash
docker run -d --name bllm-serve --network host --restart unless-stopped \
  -e BLLM_CACHE_MAX_MB=1024 \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu <image>
```

A steadily rising `evictions` on `GET /metrics` means the budget is short; `max_bytes` shows
what is actually in effect.

### Chat from a browser / existing client

No UI is bundled — the API is OpenAI-compatible and CORS is enabled, so point
any existing client at `http://<board-ip>:8866/v1`:

- **[chatbox-lite](https://github.com/lfbear/chatbox-lite)** — a single HTML
  file, fully client-side; add an "OpenAI-compatible" provider with base URL
  `http://<board-ip>:8866/v1` (any API key unless `BLLM_API_KEY` is set).
- Desktop **Chatbox**, **Open WebUI**, **LibreChat**, or the OpenAI SDK
  (`base_url="http://<board-ip>:8866/v1"`) all work the same way.

For a locked-down deployment, set `BLLM_API_KEY` and narrow `BLLM_CORS_ORIGINS`.

## Host prerequisites (once, outside the container)

- **ION carveout**: `sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot`
  (`bpu_first` for 7B). A boot-time kernel carveout — cannot be set from a container.
- **Docker**: if the daemon fails on `iptables (nf_tables)`, switch to the legacy
  backend: `sudo update-alternatives --set iptables /usr/sbin/iptables-legacy`
  (this kernel only ships the legacy xtables path).

## Rebuild triggers

Re-run `./stage-libs.sh && docker compose up -d --build` (from `docker/`) when
the board's **BSP / BPU driver** is upgraded — the bundled `/usr/hobot/lib`
closure must match the running kernel driver ABI.
