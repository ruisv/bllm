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

**Sampling fields**: `temperature` (default `0`, i.e. greedy), `top_p`,
`presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs` / `top_logprobs`,
`seed`, `max_tokens` / `max_completion_tokens`, `stop`, `stream`. Plus a few common
non-OpenAI extensions: `top_k`, `min_p`, `repetition_penalty`, and `enable_thinking`
(the Qwen3.5 reasoning toggle, default `false`). Omitted fields take their default, and
an explicit `0` is distinct from omitting one (`temperature: 0` means greedy).

**Structured output**: `response_format` or `grammar`, not both (sending both is a 400 —
better an error than a precedence rule nobody remembers).

```jsonc
{"response_format": {"type": "json_object"}}                    // any valid JSON
{"response_format": {"type": "json_schema",
                     "json_schema": {"name": "city", "schema": {...}}}}  // that exact shape
{"grammar": "root ::= \"yes\" | \"no\"\n"}                    // raw GBNF (llama.cpp's field)
```

This is not "asking the model nicely for JSON" — it is a decode-time constraint: at every
step only tokens that keep the grammar satisfiable can be sampled, so `json.loads()` cannot
fail. Measured cost is ~6% (14.3 -> 13.4 tok/s) plus ~315 ms to build the table on first
use. A grammar or schema that does not parse is a 400; it never degrades silently to free
generation, because the guarantee is the reason the client asked. See `json_grammar` in the
API docs for the supported schema subset.

`logit_bias` takes `{"token id": bias}` on OpenAI's [-100, 100]; a value outside that
range is a 400 rather than a silent clamp. `logprobs: true` returns each token's log
probability, and `top_logprobs: N` (0..20) adds N alternatives. These are the **raw**
model distribution (an exact full-vocab log-softmax), unmoved by temperature, the
penalties or `logit_bias` — so the generated token need not be `top_logprobs[0]`, and
the numbers stay comparable across requests with different sampling settings. When
**streaming**, the logprobs arrive **all at once on the final chunk**: a delta is a
decoded text slice (a stop-string holdback can merge or defer tokens), so there is no
honest per-chunk alignment to publish, and a client that concatenates every chunk's
`logprobs.content` — the normal way to consume them — still gets exactly the right
array.

With no `seed`, each request draws a fresh one, so `temperature` actually varies the
reply; with an explicit `seed`, identical requests reproduce byte for byte (OpenAI's
`seed` semantics).

### Image + text (VLM)

When `BLLM_MODEL` points at a directory carrying a `visual` tower (see the hybrid
packaging section of `docs/MODELS.en.md`), `/v1/chat/completions` accepts OpenAI's
`image_url` content part — either an inline base64 data URI or an `http(s)://` link:

```bash
curl localhost:8866/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K..."}}
  ]}]
}'
```

`/metrics`'s `is_vlm` field says whether the loaded model is an image+text package.
Pointing a text-only model at `image_url` is a clean 400, not a silently dropped image
followed by a confident, made-up answer.

A follow-up question about the same image does not re-run the vision tower: the
engine remembers what it has already fed the live session and only replays the new
tail. If the history was edited, the image changed, or two unrelated conversations
interleave on this single-session engine, it replays from scratch instead — the
answer is always correct, it just misses the savings in that case (an image
conversation has no ids-based prefix cache the way plain text does; there is no
snapshot/restore for it, only turn-by-turn replay). Video is not supported (Qwen3.5
separates frames with timestamp tokens, not Omni's TMRoPE).

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
| `BLLM_CACHE_MAX_MB` | `auto` | prefix-cache budget in MB, or `auto` (40% of `MemAvailable`, 256 MB–4 GB); `0` disables |
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
