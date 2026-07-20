# bllm-serve container image · Usage guide

`bllm-serve` packages the BLLM native runtime as an **OpenAI-compatible HTTP
server** in a container image for D-Robotics RDK **S100 / S100P / S600**
(aarch64 only). `docker run` it and you get a local LLM service any OpenAI client
can talk to.

> This is the **usage** guide. For how the image is built and the design
> trade-offs, see [`README.md`](README.md).

---

## 0. First: this is a device-coupled image

The image **self-contains** our runtime stack + the BPU userspace `.so` closure,
but the kernel BPU driver, device nodes, and `.hbm` model are **not** in the
image — they come from the board at run time:

| Thing | Where |
|---|---|
| runtime stack (python + `bllm`/`libbllm` + server), BPU userspace libs | ✅ baked into the image |
| kernel BPU driver + `/dev/{bpu,ion,…}` | board kernel → mapped via `--device` |
| `.hbm` model + tokenizer + `model.json` | baked in for delivery images; volume-mounted (`-v`) for the general image |

So the image **only runs on the matching board** (its BSP/driver must match) — it
is not a portable, run-anywhere application image.

---

## 1. One-time host prep (outside the container, once)

Run these on the **board**, not inside the container:

```bash
# ① Enlarge the ION carveout (required for LLMs, or the runtime segfaults on alloc), then reboot
sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot     # balanced for ≤3B; bpu_first for 7B

# ② If the Docker daemon errors on iptables (nf_tables), switch to legacy
#    (this kernel ships only the legacy xtables path)
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy

# ③ (optional) Enable BPU performance mode for faster decode — do it once on the host
sudo /usr/bin/busybox devmem 0x2b047000 32 0x99
sudo /usr/bin/busybox devmem 0x2b047004 32 0x99
```

---

## 2. Load the image

A delivery is usually a `docker save` tarball:

```bash
docker load -i bllm-serve-qwen3.5-2b-ctx4k-int8-s100p.tar
docker image ls bllm-serve         # confirm the tag
```

Image tags follow the delivery convention `bllm-serve:<model>-<ctx>-<quant>-<board>`,
e.g. `bllm-serve:qwen3.5-2b-ctx4k-int8-s100p` — model / context length / quant /
target board at a glance.

---

## 3. Run

### A. Delivery image (model baked in) — just run, no model mount

```bash
docker run -d --name bllm-serve --network host \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu \
  bllm-serve:qwen3.5-2b-ctx4k-int8-s100p
```

- `--network host` is **required** (this board's kernel lacks the iptables `raw`
  table, so the default bridge fails).
- The five `--device` flags map the board's BPU/ION nodes into the container —
  none are optional.

### B. General image (model via volume)

The image carries no model; mount a model dir from the board at `/models`:

```bash
docker run -d --name bllm-serve --network host \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu \
  -v ~/models/qwen3.5-2b:/models:ro \
  -e BLLM_MODEL=/models \
  bllm-serve:latest
```

`~/models/qwen3.5-2b` is a **model package dir** (with `model.json` + `.hbm` +
tokenizer + `embed_tokens.bin`). You can also point at a bare `.hbm`, in which
case add `-e BLLM_TOKENIZER_DIR=...`.

### C. docker compose (from the `docker/` dir)

```bash
cd docker
MODEL_DIR=~/models/qwen3.5-2b docker compose up -d
docker compose logs -f          # follow startup / model load
docker compose down             # stop + remove
```

Device nodes, volume, and healthcheck live in `compose.yaml`; per-deploy knobs go
through env vars or a `.env` file.

---

## 4. Verify it's up

Model load takes seconds to tens of seconds (depending on size); it's ready once
`/health` responds:

```bash
curl localhost:8866/health          # {"status":"ok",...}
curl localhost:8866/v1/models       # lists the current model id
```

If it's not up immediately, `docker logs -f bllm-serve` to watch the load.

---

## 5. Chat (OpenAI-compatible API)

### Command line

```bash
# non-streaming
curl localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello, introduce yourself"}]}'

# streaming (SSE, token by token)
curl -N localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 5"}],"stream":true}'
```

Supports `/v1/chat/completions` (streaming + non-streaming), `/v1/models`,
`/health`.

### OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://<board-ip>:8866/v1", api_key="anything")  # key is ignored unless BLLM_API_KEY is set
r = client.chat.completions.create(
    model="bllm", messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

### GUI clients (browser / desktop)

No UI is bundled, but the API is OpenAI-compatible with CORS enabled — point any
existing client at `http://<board-ip>:8866/v1`:

- **[chatbox-lite](https://github.com/lfbear/chatbox-lite)** — a single
  client-side HTML file; add an "OpenAI-compatible" provider with that base URL.
- Desktop **Chatbox** / **Open WebUI** / **LibreChat** work the same way.

> ⚠️ The engine is a **singleton**: the BPU has one prefill/decode graph at a
> time, so it handles **one generation at a time** — concurrent requests
> serialize. This is a hardware property; don't expect concurrent throughput.

---

## 6. Environment variables

| Var | Default | Meaning |
|---|---|---|
| `BLLM_MODEL` | preset in delivery images | model dir (with `model.json`) or a bare `.hbm`, path inside the container |
| `BLLM_TOKENIZER_DIR` | — | only when `BLLM_MODEL` is a bare `.hbm` |
| `BLLM_MAX_QUEUE` | `8` | queued requests before the server returns 429 (one BPU graph = one generation at a time) |
| `BLLM_CACHE_MAX_MB` | `512` | prefix-cache budget in MB; `0` disables the cache |
| `BLLM_CACHE_TTL_S` | `1800` | drop a cached prefix unused for this long |
| `BLLM_CACHE_MAX_ENTRIES` | `64` | entry cap alongside the byte budget |
| `BLLM_MAX_NEW` | `1024` | default generation cap when a request omits `max_tokens` |
| `BLLM_API_KEY` | — | if set, require `Authorization: Bearer <key>` |
| `BLLM_CORS_ORIGINS` | `*` | allowed CORS origins (comma-separated); open by default for browser clients |
| `BLLM_BPU_PRIORITY` | — | set BPU priority at startup |
| `BLLM_PERF_MODE` | `0` | `1` = poke BPU perf registers inside the container (needs `--privileged`; prefer doing it on the host) |
| `BLLM_PORT` / `BLLM_HOST` | `8866` / `0.0.0.0` | bind |

**Locked-down deploy**: set `BLLM_API_KEY` and narrow `BLLM_CORS_ORIGINS` to your
frontend origin.

---

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

**Memory** is budgeted in bytes, not entries: a dense snapshot is ~22 KB/token and
grows with the context, while a hybrid one is a flat **~72 MB** regardless of length.
So 512 MB holds dozens of dense entries but only 7 hybrid ones. Lower
`BLLM_CACHE_MAX_MB` if memory is tight, or set it to `0` to disable.

**Sizing it:** the budget is in bytes, not entries. A hybrid snapshot is a *constant*
(measured 122 MB for Qwen3.5-2B ctx4k), so the 512 MB default holds only **4 conversations**
— past that they evict each other. Raising `BLLM_CACHE_MAX_MB` to 2048 held **16** in
testing, using 1957 MB and still leaving 4.4 GB free on an 8 GB board:

```bash
docker run -d --name bllm-serve --network host --restart unless-stopped \
  -e BLLM_CACHE_MAX_MB=2048 \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu <image>
```

Dense models need far less: their snapshots are ~22 KB/token and grow with the context, so
the same budget holds an order of magnitude more. A steadily rising `evictions` on
`GET /metrics` is the signal that the budget is too small.

> **Hybrid snapshot size:** since 0.1.6 a snapshot saves only the occupied attention K/V,
> so it grows with the conversation (~22 MB plus ~24.6 KB/token) instead of being fixed by
> the context window. Qwen3.5-2B ctx4k was a flat 122 MB per entry; a 136-token
> conversation is now 25 MB — **the same 2 GB budget holds ~66 instead of 16**.

## 7. Operations

```bash
docker logs -f bllm-serve           # logs
docker restart bllm-serve           # restart
docker stop bllm-serve && docker rm bllm-serve   # stop + remove
docker stats bllm-serve             # resource usage
```

Switch models: rerun with the corresponding delivery image, or (general image)
change the `-v` mount + `BLLM_MODEL` and rerun.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Exits on startup, log shows alloc / segfault | ION carveout not enlarged → run step 1 ① `hb_switch_ion.sh balanced` and **reboot** |
| `docker run` fails on iptables / networking | must use `--network host`; at the daemon level switch to `iptables-legacy` (step 1 ②) |
| `/dev/bpu` etc. not found | BPU driver not loaded / node names differ; check with `ls /dev/ | grep -E 'bpu|ion'` |
| Load fails on missing `.so` symbol / GLIBC | board BSP/driver was upgraded and the image's `/usr/hobot/lib` closure no longer matches the ABI → rebuild the image |
| Decode is slow | enable BPU performance mode on the host (step 1 ③); note longer context = slower decode, a BPU static-graph property |
| Context-overflow error | full-attention models **raise** rather than drop tokens past the KV window; use a larger-ctx delivery image (e.g. ctx4k) |
| Reply cut off mid-sentence | hit the generation cap (default `BLLM_MAX_NEW=1024`); raise the request's `max_tokens` or launch with `-e BLLM_MAX_NEW=2048`. `finish_reason:"length"` in the response means it was length-truncated (`"stop"` = natural end). Prompt + generation must fit the ctx window together |
