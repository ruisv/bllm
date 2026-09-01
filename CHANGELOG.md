# Changelog

All notable changes to BLLM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] — 2026-09-01

### Added
- **SmolVLA as a first-class arch (`arch: "smolvla"`), on both S100P and S600.**
  LeRobot's vision-language-action policy as three graphs — the SmolVLM vision
  tower, the first 16 SmolLM2 layers as a prefix pass, and the interleaved action
  expert — packaged so nothing at run time needs lerobot or the checkpoint.
  `bllm.load_policy(dir).act(scene, wrist, prompt, state=...)` returns a 50 × 7
  action chunk in **741 ms** on S100P and **285 ms** on S600.

  The expert is the structurally new piece: even layers self-attend over
  `[prefix KV; their own KV]` with a causal action block at absolute RoPE
  positions, odd layers cross-attend the prefix at 0-based positions against K/V
  the prefix graph already projected through the expert's own k/v. Writing either
  the reasonable way costs cosine 0.32 and 0.58 respectively. The timestep never
  reaches the BPU — `action_time_mlp_in` is linear in its concatenation, so the
  per-step term is a constant table.

  Unlike `pi05_libero`, SmolVLA **reads proprioception**: the state is projected
  into a prefix token, so omitting it is a different observation (2.0 in action
  space, a full gripper unit).

  On `libero_spatial`, scored through the same harness with only the side
  computing the chunk changed: **70/100** (S100P), **71/100** (S600), against
  **73/100** for the same checkpoint on an RTX 4090. Indistinguishable at n = 100,
  and the per-task profile matches — the tasks the boards fail are the tasks the
  reference fails.

- `bllm_smolvla_policy`, `bllm_smolvla_probe` and `bllm_smolvla_server`, and the
  `smolvla` mode of `bllm-make-model-dir`.
- **`BLLM_BPU_CORES`** pins which BPU cores a multi-core graph runs on.

### Fixed
- **`bllm.load()` routed a policy package into the chat-VLM branch.** That branch
  tests for a vision tower and a policy package has one, so the user got an error
  from deep inside a chat session rather than "this is a policy, use
  `load_policy`". Policies are now routed first.
- **`load_policy()` rejected everything that was not `pi05`** — the entry point
  every model card advertises — so a new policy arch could not be loaded through
  the documented API at all.
- **Multi-core `.hbm` could not be submitted.** A graph compiled for several BPU
  cores is *rejected* under `HB_UCP_CORE_ANY` ("the backend must be specified to
  specific cores") rather than falling back, so every caller would have had to
  know its core count. `Graph::init` now reads it from the model
  (`hbDNNGetCompileBpuCoreNum`) and binds accordingly; a single-core and a
  multi-core artifact behave identically from the outside.

## [0.4.2] — 2026-08-20

### Fixed
- **The attention mask was written for one head only, so generation depended on
  what else the board had been running.** The decode mask tensor is
  `[n_heads, cache_len]` and the graph reads every head row; the code wrote the
  first and assumed the graph broadcast it. On the shipped Qwen3.5-0.8B that left
  7 of 8 rows holding whatever `hbUCPMallocCached` returned — and freed BPU memory
  comes back dirty, at the same physical address. Clean memory therefore read as
  an all-zero mask (no causal masking at all on those heads); memory dirtied by an
  unrelated workload read as garbage and the model degenerated into repetition.
  The same binary and the same greedy prompt could answer correctly or ramble for
  400 bytes depending on history, which is what made it look like flakiness rather
  than a bug. Affects every `arch: "hybrid"` model — the whole Qwen3.5 family.
  On the board suite this moves 4 failed / 151 passed to 1 failed / 157 passed,
  and output is now bit-identical across a 2 GB memory poison in three patterns.
  The chunked-prefill mask had the same defect and is fixed identically, though it
  cannot be verified here: no available model carries the optional `_prefill`
  graph, so that path never executes.

### Changed
- **BPU allocations are zeroed.** This does not make a partial write correct, but
  it makes one deterministic — the difference between a bug that can be bisected
  and one that reproduces only until you look at it. It costs a single pass over
  buffers allocated once at load; the large caches were already being zeroed by
  their owners. Decode throughput is unchanged (17.3 tok/s measured before and
  after).
- Mask construction moved to `bllm/attn_mask.h` as two named functions, so it has
  a test seam at all — previously the logic was unreachable from a test without
  exposing engine internals. `tests/test_attn_mask.cc` covers it and needs no
  board.

## [0.4.1] — 2026-08-19

### Fixed
- **A rank-3 vision tower read hundreds of megabytes past its output buffer.**
  `VisionTower` derived `n_token`/`hidden` from the last two dimensions — so a
  `[1, n_token, hidden]` graph was explicitly supported — but took its row pitch
  from the stride of dimension 0, which for that rank is the size of the whole
  tensor. The dense fast path therefore missed and each row was read one whole
  tensor apart. Now uses the rank-aware `outRowPitch`, as every other reader in
  the runtime already did.
- **Output tensor strides are resolved at load, like input strides.** The runtime
  reports `-1` for "dense, compute it yourself"; only the inputs were being
  expanded, so any reader trusting an output stride got a negative pitch and read
  *before* the buffer.
- **A failed model load leaked the entire `.hbm` mapping.** The packed handle was
  released in an explicit destructor, which never runs when the constructor throws
  — and constructors here keep validating after loading (graph shapes, tensor
  types, a KV window that can `bad_alloc`). Every load that failed leaked
  gigabytes, in the Python binding and `bllm-serve` alike. `Pi05Stack` never
  released its handles at all. All owners now hold an RAII `PackedHbm`, which also
  orders the release after the buffers it backs.
- **`bllm_vision_probe` silently ignored an unknown `--patch-order`** and used the
  merged layout. Both layouts yield a correctly shaped tensor differing only in
  its numbers, so a typo presented as a quantization fault in the one tool meant
  to tell those two apart. Unknown values are now rejected.
- Three headers used `std::memset`/`std::memcpy` while relying on `<cstring>`
  arriving transitively.

### Changed
- **The hbDNN/hbUCP graph layer moved to `bllm/native_graph.h` +
  `src/native_graph.cc`.** It is a thin, stable wrapper over the vendor runtime
  with nothing worth inlining, and was being recompiled into every translation
  unit. The model classes above it stay header-only, so the CLI, the Python module
  and `bllm-serve` keep sharing one implementation with no ABI to maintain. Build
  time is unchanged; the gain is one definition per behaviour and a narrower
  include graph — the vision and audio towers, π0.5 and the hybrid engine no
  longer pull in the whole autoregressive engine and sampler just to get a
  `Graph`. `native_engine.h` drops from 963 to 617 lines.

### Added
- **Being-H0.5 replayed as four ordinary shared-KV passes.** The checkpoint looks
  like two entangled transformers, but at inference its understanding and
  generation streams occupy disjoint, contiguous position ranges under a plain
  causal mask, so a chunk decomposes into four single-stream calls into two
  compiled graphs over one KV window per layer — verified exact in fp32 against
  the packed reference. `bllm::BeingHStack` drives the window, mask and cache
  upload memoization; `bllm::BeingHHost` is the flow-matching action head, kept in
  fp32 on the CPU because it is a continuous velocity regression with no softmax
  to hide error. On S100P a 15-action chunk takes 798 ms with two cameras
  (MPG off), final chunk cosine 0.994892 against the reference.

## [0.4.0] — 2026-08-14

### Added
- **π0.5 (openpi) as a first-class model package.** Two camera frames and an
  English instruction in, a chunk of end-effector deltas out, at ~1 s per chunk
  with all three graphs (4.6 GB) resident on the board:

      policy = bllm.load_policy("pi05-libero-224-s100p")
      actions = policy.act(scene_rgb, wrist_rgb, "pick up the black bowl ...")

  On `libero_spatial` the package scores **99/100** through openpi's own
  evaluation harness, matching the x86 reference's 99/100; `libero_goal` 97/100
  and `libero_object` 95/100. The package carries its own tokenizer, embedding
  table, action quantiles and flow-matching conditioning, so nothing at run time
  needs openpi or its 14 GB checkpoint. New: `arch: "pi05"` manifests built by
  `bllm-make-model-dir pi05`, `bllm.load_policy()`, the `bllm_pi05_policy` CLI,
  `bllm::Pi05Policy` in C++, and `docs/PI05.md`. Weights are published at
  [ruisv/bllm-pi05-libero-224-s100p](https://huggingface.co/ruisv/bllm-pi05-libero-224-s100p).

### Fixed
- **A streaming reply could abort mid-sentence** with `'utf-8' codec can't
  decode byte 0x8a in position 0: invalid start byte`. Byte-level BPE spreads a
  multi-byte character over several tokens, and decoding a partial prefix yields
  U+FFFD — three bytes where the finished character is often four. The
  placeholder is itself well-formed UTF-8, so it was emitted, leaving the stream
  offset inside a four-byte character; the next delta then began on a
  continuation byte and failed to decode on its own. Any reply mixing emoji with
  text could hit it, which made it look intermittent. Emit offsets are now
  trimmed back off both incomplete sequences and a trailing U+FFFD, so every
  delta is independently decodable — at the cost of one token of latency on the
  character being assembled, and self-correcting (a genuine U+FFFD still arrives
  on the end-of-stream flush). Affects both `NativeSession` and
  `NativeVlmSession`.

## [0.3.0] — 2026-07-29

### Added
- **VLM thinking toggle.** `NativeVlmSession.set_thinking(False)` / `.thinking`,
  mirroring the text-only `NativeSession` — the VLM session had no way to force
  a direct answer, so a reasoning-heavy reply could exhaust a caller's token
  budget before ever reaching the answer.
- **`bllm.load(path, vision=False)`.** Forces a VLM/omni package through
  `NativeSession` instead of `NativeVlmSession`, so the vision/audio tower's
  weights never touch BPU/ION at all — for a caller that only wants text and
  wants that memory budget back. `chat(images=[...])` already skipped the
  tower's *compute* per call when no images were passed; this is for skipping
  the *load*, not just a given call.
- **C++ API reference, in both languages.** `docs/CPP_API.zh.md` and
  `docs/CPP_API.en.md` document the surface that previously existed only as
  headers: `NativeLlm` (generation, sampling, grammars, stop strings, thinking,
  state and reusable prefixes, the serving primitives), `NativeVlm` (media types,
  live streaming, the KV budget, the Omni-only video path), `NativeEmbedder`
  (which had no documentation at all in either language), `ModelConfig`,
  `Tokenizer`, the sampling parameters, and the engine layer beneath the
  sessions. Includes a C++ ⇄ Python mapping table; the Python references and both
  READMEs now link to it.
- **Image+text now served over HTTP.** `/v1/chat/completions` accepts OpenAI's
  `image_url` content part (inline base64 or an `http(s)://` link) when
  `BLLM_MODEL` points at a VLM package; `/metrics` exposes `is_vlm` so a client
  can tell. A multi-turn conversation about the same image does not re-run the
  vision tower: the engine tracks which `(user, assistant)` turn pairs are
  already live in the session and replays only new turns via the new
  `NativeVlm::append_turn` / `NativeVlmSession.append_turn` (prefill only, no
  decode — the caller already has the reply, so regenerating it would waste BPU
  time and, sampling being stochastic, might not even reproduce the same text).
  History that was edited, or two unrelated conversations interleaving on this
  single-session engine, falls back to a full reset+replay rather than risk
  splicing the wrong context. Also adds `NativeVlm::encode` /
  `NativeVlmSession.encode` for usage-accounting token counts.

### Changed
- **Causal mask deduplication in the hybrid decode/prefill graphs.** The mask
  is head-independent (the causal window doesn't vary by attention head), so
  the fixed mask buffer now carries one row and the graph broadcasts it
  internally, instead of the host writing `nHead_` identical copies every
  step/chunk.

### Fixed
- **S600 (`nash-p`) image chat() crashed on the first request.** `NativeVlm`
  never bound the text decoder to explicit BPU cores the way `NativeLlm`
  (text-only) already did, so every multi-core `.hbm` rejected the default
  `HB_UCP_CORE_ANY` with `hbUCPSubmitTask`: "the backend must be specified to
  specific cores". Fixed — the vision/audio towers are left on the default
  schedule (they are not multi-core-compiled and forcing them onto the
  decoder's mask does not help). **Known remaining issue, not yet understood**:
  with the crash fixed, S600 image `chat()` is intermittently wrong — roughly
  40-50% of calls return empty output instead of a real reply, temp=0, fresh
  process each time. The isolated vision tower alone and text-only chat are
  both 100% deterministic; only the combined pipeline is flaky. Do not treat
  S600 VLM `chat()` as reliable yet — see the constructor comment in
  `include/bllm/native_vlm.h`.

## [0.2.0] — 2026-07-24

The **Qwen3.5 image+text (VLM)** release. A self-built vision tower plus a
chunked-prefill graph bring on-board image understanding to the hybrid Qwen3.5
line, decode gets **1.4–1.7× faster on the same int8 weights**, and structured
output / sampling gain grammar constraints, `logit_bias`, `logprobs` and a batch
of correctness fixes.

### Added

- **Qwen3.5 image+text (VLM) — the headline of this release.** Qwen3.5 is natively
  an image-text-to-text model; BLLM now compiles its vision tower (patch-16 /
  LayerNorm, self-built — not the Omni patch-14 tower) and splices the vision rows
  into the hybrid decoder's embedding stream with interleaved 3-D mrope.
  `bllm.load(<vlm_dir>)` routes to `NativeVlmSession`; `chat(text, images=[...])`
  takes file paths or raw `HxWx3` uint8 arrays. Strictly 100% BPU vision tower.
  On S100P / 0.8B, image TTFT **1.23 s (320px, 100 tok) / 2.30 s (448px, 196 tok)**
  with the prefill graph; the tower output matches HF fp32 at cosine 0.999 / 0.997.
  The four vision settings a hybrid VLM cannot default (patch, mean/std, mrope
  section, interleaved) are forced by `bllm-make-model-dir` — a wrong one yields
  shape-correct garbage only after an image is fed. Video is refused on this family
  (Qwen3.5 separates frames with timestamp tokens, not Omni's TMRoPE).
- **Chunked prefill graph — prompt/image ingestion amortized ~76×.** The hybrid
  engine previously fed a prompt one token at a time (weight-bandwidth-bound,
  ~69 ms/token — a 2048-token document took ~5 minutes). It now drives an optional
  seq_len=N prefill graph carried in its own `.hbm` (`--hbm-prefill`), turning
  ingestion into one weight stream per N-token chunk. The GDN part uses the parallel
  chunked delta-rule (WY/UT transform) verified bit-exact against the sequential
  recurrence. A model without the prefill graph gracefully keeps the old per-token
  path. Prefill↔decode parity is a shipped gate (`scripts/check_prefill_parity.py`),
  asserted non-vacuous (prompt > chunk).
- **GBNF grammar-constrained decoding — structured output that is guaranteed, not
  requested.** `set_grammar("root ::= ...")` on the Python session, and over HTTP
  `response_format: {"type": "json_object"}` / `{"type": "json_schema", ...}` or a raw
  `grammar` field (llama.cpp's). At every step only the tokens that keep the grammar
  satisfiable can be sampled, so a reply that claims to be JSON parses — `json.loads()`
  cannot fail. `bllm.json_grammar(schema)` converts a JSON Schema (`type`, `enum`, `const`,
  `properties`/`required`, `items`, `anyOf`/`oneOf`, `$ref` into `$defs`); optional
  properties must appear in schema order and `additionalProperties` is treated as false.
  Cost on S100P / Qwen3.5-0.8B: ~315 ms once per session to build the token→bytes table,
  then 14.3 → 13.4 tok/s (~6%). A grammar that does not parse is a 400, and an old runtime
  without grammar support is refused rather than answered unconstrained — silently
  downgrading a guarantee to "the prompt says please" is the failure mode this feature
  exists to remove.

- **`logit_bias`** — force or ban specific tokens, on OpenAI's `[-100, 100]` scale.
  `set_sampling(logit_bias={id: bias})` in Python, `logit_bias: {"<id>": bias}` over HTTP
  (out-of-range values are a 400, not a silent clamp). The subtlety is in the sampler: it
  bounds its candidate set to the top 512 logits, and that cut is safe only because the
  penalties can *lower* a token but never raise one. A bias breaks exactly that assumption,
  so biased ids are unioned into the candidate set after the cut — otherwise a `+100` on a
  token buried in the tail would be silently thrown away before it could win.
- **`logprobs` / `top_logprobs`** — per-token log probabilities, in OpenAI's
  `choices[].logprobs.content[]` shape, plus `NativeSession.last_logprobs()` on the Python
  API. The values are an exact full-vocab log-softmax of the **raw** logits (double
  accumulation, no candidate truncation): they describe what the model believes, so
  temperature, the penalties and `logit_bias` do not move them and they stay comparable
  across requests — which does mean the generated token need not be `top_logprobs[0]`.
  Opt-in, because it costs two extra full-vocab passes per token (measured on S100P /
  Qwen3.5-0.8B: 14.3 → 13.6 tok/s with 5 alternatives, 3-5%). When streaming, the array
  arrives whole on the final chunk: a delta is a decoded text slice, not a token, so no
  honest per-chunk alignment exists — a client that concatenates each chunk's
  `logprobs.content` still ends up with exactly the right array. The report is also trimmed
  to the returned text, so a stop string that cuts the reply does not leave logprobs
  describing tokens the client never received.

### Changed

- **Decode 1.4–1.7× faster on the same weights — attention by grouped matmul, not
  `expand_win`.** The old form expanded the KV window from 2 to 8 heads and
  materialised four `[cache_len, n_heads, head_dim]` tensors per layer per token
  (403 MB/token at ctx2k, more than the model's own weights at ctx4k). It now groups
  the queries instead — `q[NKV, REP, HDa]` against `Kᵀ`, one batched matmul, K/V read
  exactly once (201 MB → 25 MB per 1024 slots). Pure memory-traffic reduction, the
  int8 weights are unchanged; re-download the latest `.hbm` to get it. Measured on
  S100P (decode graph latency, tok/s ceiling), same weights:

  | model | cache_len | before (expand_win) | after (matmul) | speedup |
  |---|---|---|---|---|
  | Qwen3.5-0.8B | ctx2k | 15.7 tok/s | **22.5 tok/s** | 1.43× |
  | Qwen3.5-0.8B | ctx4k | 11.3 tok/s | **19.5 tok/s** | 1.73× |
  | Qwen3.5-2B | ctx4k | 9.1 tok/s | **13.7 tok/s** | 1.51× |

  The speedup grows with `cache_len` because the old form's overhead scaled with the
  KV window while the matmul form does not — the longer the context, the bigger the win.
- **Sampler top-k by sequential scan + a k-sized min-heap, not `nth_element`.** The
  candidate-set build over a 248k-token vocabulary was a random-access trap (every
  comparison a cache miss): 4.25 ms/token, 9% of a sampled step. A single ordered pass
  with a heap touches each logit once → 1.00 ms, sampled decode +7%. Greedy is unchanged.
- **Prefill emits only the last row of logits.** For an N-row prefill the previous graph
  materialised `[N, 248320]` f32 (127 MB of ION at N=128) and ran the 254 MB lm_head N
  times, though only the last row seeds the first sampled token.

### Fixed

- **A hybrid `.hbm` fed to the dense engine now says what to use** instead of a cryptic
  `Model not exists: prefill`. `NativeEngine` / `bllm_selfengine` drives the dense
  prefill/decode format; a hybrid Qwen3.5 (Gated-DeltaNet/SSM) `.hbm` (graph named
  `qwen35`, needs a host embed table) is now detected up front and the error names the
  hybrid path (`bllm_qwen35 --hbm .. --embed ..`, or `bllm.load(<dir>)`), exiting cleanly
  rather than aborting.
- **A quantized vision-tower output was being read as float** — the tower can emit
  scaled int16, which as float is shape-correct noise; the VLM path now de-quantizes by
  the graph's declared dtype.
- **Image parts were silently dropped by the server.** An OpenAI `image_url` content part
  on a text-only model was ignored and the model answered from the text alone; it is now
  an explicit error rather than a confident hallucination.
- **A left-recursive grammar crashed the whole server.** `{"grammar": "root ::= root\n"}` —
  or any rule that can reach itself without consuming input, including indirectly, through
  an alternate, or past a nullable prefix — expanded forever inside the matcher and took the
  process down with SIGSEGV, model and all. `grammar` is a request field, so any client
  could send it; the same shape was reachable through `response_format` with a
  self-referential `$ref`. Grammars are now checked for left recursion at construction
  (nullability fixed point + a cycle search over the "can begin with" graph) and rejected
  with a 400, with a depth cap in the matcher as a backstop. Right recursion — what `*`,
  `+` and `?` expand into — is unaffected.
- **`logprobs` reported the wrong `bytes` for split characters.** The payload derived them
  from `decode([id]).encode()`, but a token holding only part of a multi-byte character
  decodes to U+FFFD: token 94 of the Qwen vocabulary contributes the single byte `0xA1`
  while the response claimed `[239, 191, 189]`. A client reassembling text from per-token
  reports corrupted any CJK the tokenizer had split — which is precisely what OpenAI's
  `bytes` field exists to prevent. It now comes from the token's real bytes
  (`NativeSession.token_bytes()`), and `logprobs` is present as an empty array rather than
  absent when a request asked for it but generated nothing.
- **`bllm.Grammar` was copyable but must not be.** Its positions are pointers into its own
  rule storage, so a copy kept pointing at the source and dangled when the source died. The
  copy constructor is now deleted (moving, which is how the session takes ownership, is
  unaffected).

- **`temperature` had no effect.** `seed` defaulted to a fixed `1234`, and the sampler is
  constructed fresh for every generation — so every request replayed the identical random
  stream. Three identical `temperature=0.8` requests came back byte-identical; the knob was
  decorative. The default is now `SEED_RANDOM` (`bllm.SEED_RANDOM`, the same sentinel and
  value as llama.cpp's `LLAMA_DEFAULT_SEED`), which draws a fresh seed per generation. An
  explicit `seed` still reproduces exactly, which is what OpenAI's `seed` promises — so the
  reproducible path is unchanged, only the *unspecified* path stopped being pinned.
- **`presence_penalty` and `frequency_penalty` were ignored over HTTP.** The server read
  only the non-standard `repetition_penalty`, so OpenAI's own penalty fields were dropped
  on the floor while the engine's `penalty_present`/`penalty_freq` sat unused. Both are now
  passed through unscaled (OpenAI and the engine agree on `[-2, 2]`, 0 = off). `min_p` is
  exposed too.
- **An explicit `0` was indistinguishable from an omitted field.** `body.get(k, d) or d`
  folded falsy values back into the default, so `temperature: 0` — which clients send
  deliberately to force greedy decoding — read as "unset", and `top_p: 0` silently became
  `1.0`. Only `null`/absent counts as unset now.
- **Reported TTFT excluded the prefill.** The clock started after the prompt was fed, but
  prefill has already produced the logits the first token is drawn from, so the measurement
  was ~0.3 ms for every request whether it hit the prefix cache or not — the one metric
  that should show the cache working was blind to it. It now starts when the request begins
  being served: a 428-token turn reusing 256 cached tokens reports 138 ms against 274 ms
  for the same prompt uncached, matching the documented 2x.

## [0.1.6]

### Added

- **`BLLM_CACHE_MAX_MB` defaults to `auto`.** A fixed default cannot suit both engines —
  snapshot size varies by an order of magnitude between them and with context length — and
  the right value also depends on how much RAM the board has left after loading the model,
  which only the running process can see. The server now sizes the budget from
  `MemAvailable` (a 40% share, floored at 256 MB and capped at 4 GB) and logs the number it
  picked. An explicit value still wins; `0` still disables the cache. Over-committing here
  does not raise an exception on an 8 GB board — it takes the board down — hence the
  deliberately conservative share.

### Changed

- **Hybrid snapshots are ~5x smaller.** The hybrid cache set is a mix: the GDN layers hold a
  fixed-size recurrent state that summarises all history, while the attention layers hold a
  K/V window whose leading dimension *is* the window. Saving the whole window made every
  snapshot a flat size regardless of how short the conversation was — 72 MB for
  Qwen3.5-0.8B (ctx2048), 122 MB for Qwen3.5-2B (ctx4096) — which bounded a serving prefix
  cache by the context window rather than by the conversation. Only the occupied K/V is
  saved now (it is right-aligned in the window, as the attention mask shows), so a blob
  grows with the conversation: **22 MB at 17 tokens, +24.6 KB/token**. A 136-token snapshot
  drops from 122 MB to 25 MB — 80% smaller — so a 2 GB prefix cache holds ~66 conversations
  instead of 16. The GDN state is fixed-size by nature and is still saved whole; dense
  models are unaffected.

  The state format changed, so the magic is bumped `BLKH` → `BLKI` and older hybrid
  `save_state()` files are rejected with a clear error rather than misread. In-memory
  snapshots do not cross versions, so a running server is unaffected.

## [0.1.5]

### Added

- **Session-aware serving layer.** `docker/server.py` is now a thin OpenAI-protocol facade
  over `docker/serving/`: an engine that owns the model on one worker thread, and a prefix
  cache of engine snapshots keyed by the token ids they cover. The OpenAI protocol is
  stateless — a client re-sends the whole history every turn — so the server used to
  re-prefill all of it; now a turn prefills only what is new. Matching is exact token-prefix
  comparison, so a wrong reuse is not possible: an edited history or different tokenization
  simply matches a shorter entry or misses and falls back to a full prefill.
  Measured on an S100P — hybrid (Qwen3.5-0.8B) 31.6 s → 1.3 s (~25x), dense
  (Qwen2.5-1.5B) TTFT 274 → 141 ms on turn 10. New knobs: `BLLM_CACHE_MAX_MB`,
  `BLLM_CACHE_TTL_S`, `BLLM_CACHE_MAX_ENTRIES`, `BLLM_MAX_QUEUE`; new `GET /metrics`.
- **Serving primitives on the session** — `snapshot()` / `restore()` (in-memory, no temp
  file), `feed_ids()` / `decode_stream()` (generate() split so a caller can snapshot exactly
  at the prefill/decode boundary), `request_cancel()` / `canceled`, plus `context_left`,
  `prefill_chunk` and `cache_align()`. `set_prefix()`/`ask()` serve ONE fixed prefix; these
  let a server hold many states and pick one per request.
- **`cache_align()` — snapshots must be chunk-aligned.** The dense engine batch-prefills runs
  of `prefill_chunk` tokens and decode-steps the remainder, and the two graphs are not
  numerically identical under int8, so a snapshot taken off a chunk boundary changes the
  reply. Measured: aligned splits reproduced the un-cached answer 5/5, unaligned 1/18 — one
  turning *"Venus is the hottest"* into *"Earth is the hottest"*. The rule lives in the
  library so callers cannot get it wrong.
- **Real token accounting.** `usage` now comes from the model's tokenizer instead of a
  4-characters-per-token estimate, and reports OpenAI's `prompt_tokens_details.cached_tokens`.

### Fixed

- **A mid-stream disconnect could corrupt the next request's answer.** The server held an
  `asyncio.Lock` around the SSE response while generation ran on a per-request thread; a
  client vanishing mid-stream closed the async generator and released the lock, leaving the
  generation thread still driving the engine, so the next request ran concurrently against
  it. Demonstrated against the old server: after a disconnect, *"reply with exactly the word:
  pineapple"* returned `"pineapple1. pineapple1.1. pineapple1. ..."`. The session is now
  owned by a single worker thread and reached only through a queue, so no HTTP-side event
  can release it early.
- **Requests queued silently and without bound** — clients just saw a timeout. Admission
  control now returns 429 past `BLLM_MAX_QUEUE` (default 8).
- **`kPrefillMinTokens` was 16, roughly 5x too high.** A prefill pass costs a padded chunk
  regardless of how few tokens it carries, so the crossover is (prefill pass)/(decode step)
  — measured 2.93 on Qwen2.5-1.5B and 3.31 on Phi-4-mini, stable because both are a single
  BPU graph invocation. It is now 4. A 15-token turn (a short follow-up like *"go on"* —
  exactly the common case the old value punished) drops from **676 ms to 135 ms**. Note this
  changes segmentation, and therefore replies, for runs of 4..15 tokens.

### Note

Two serving-layer bugs were found after the 0.1.5 tag and fixed on `main` (they affect the
container image only — the conda packages do not carry the serving layer, so 0.1.5 as
published is unaffected): the hybrid engine never cached at all (it does not chunk, so
`cache_align(n) == n`, which an over-eager guard read as "nothing to do"), and the snapshot
was taken past the per-request assistant opener, which can never be a prefix of the next
turn's prompt. Only the conversation history is cacheable now. Fixed hybrid A/B on an S100P
(Qwen3.5-2B, 6 turns): TTFT goes from growing 6.3 s → 31.0 s to flat at ~5.9 s.

The Docker image installs `bllm` from the conda channel, so the server can be newer than the
runtime beneath it. When the installed build predates the snapshot primitives the server
degrades to full re-prefill and says so at startup, rather than failing — the concurrency fix
applies either way, and the cache switches itself on under a new enough runtime.
`GET /metrics` reports `supports_prefix_cache`.

## [0.1.4]

### Added

- **`bllm-make-model-dir` ships with the package** — the model-dir tool moved into
  `bllm.make_model_dir` and installs to `$PREFIX/bin`, so `conda install bllm` is enough to
  assemble a model directory. It was previously only `scripts/make_model_dir.py` in a source
  tree, which left an installed-from-conda user with no way to turn an official release into
  something `bllm.load()` accepts. `scripts/make_model_dir.py` still works (thin wrapper), and
  `python -m bllm.make_model_dir` is equivalent.
- **[`docs/MODELS.zh.md`](docs/MODELS.zh.md) / [`docs/MODELS.en.md`](docs/MODELS.en.md)** —
  deploying an official release end-to-end: ION sizing, what the package actually contains,
  assembling dense and Omni model directories, the Omni video budget, and the failure modes
  worth knowing. Verified command-by-command on an S100P.

### Fixed

- **The ION carveout prerequisite was missing from the public docs entirely.** On-board LLMs
  need an enlarged carveout (`hb_switch_ion.sh balanced` + reboot) or the runtime segfaults on
  alloc — a first-run failure with no hint that it is a setup step, not a bug. Now in the
  README's install section, where it is a prerequisite rather than a footnote.
- **Documented the official Omni path end-to-end.** Getting from a downloaded Qwen2.5-Omni
  release to a loadable model directory was undocumented: the README's multimodal section began
  at an already-assembled directory, and the only `omni` example lived in a script docstring.
  The README (zh/en) and the new deployment docs now carry the verified command, and "making a
  model directory" is a top-level section rather than a subsection of "build from source" — a
  conda user never reached it, and it is not a build-from-source concern.
- **`embed_tokens.bin` is called out as required for Omni.** It is a separate 1.2 GB download
  in the official list, sits in `model/`, and its name does not mention Omni — so the three
  `*_Omni_3B_*.hbm` files look like a complete set. Missing it now produces an error that says
  where the file comes from, rather than just naming the flag.
- **The Omni video budget is stated instead of implied.** The README said a model directory
  "just points `visual` at the tower you want", but the official release ships only the 448px
  tower: 256 tokens per frame-pair against a 2048-slot window is ~8 s of video at 2 fps. Other
  resolutions require an offline re-export, which is outside this repo.
- **Dropped the obsolete "downgrade `tokenizer.json` to 0.19.1" advice** — that was a libxlm
  limitation. The native runtime reads `tokenizers` 0.20+ pair-format merges (with
  `ignore_merges`) as-is; verified on-board.
- **`__version__` was stuck at 0.1.2** — so the published 0.1.3 package reported the wrong
  version from `bllm.__version__`. It now tracks `CMakeLists.txt`.

## [0.1.3]

### Fixed

- **Multi-core `.hbm` binding (nash-p / S600).** An `.hbm` compiled with `core_num > 1` was
  rejected by the runtime when submitted with `HB_UCP_CORE_ANY` (*"When infer multi-core model
  task, the backend must be specified to specific cores"*). `model.json` already carried
  `bpu_cores`, but `ModelConfig` never parsed it, so the engine always submitted `CORE_ANY`.
  It is now parsed, and at load the sched core mask is bound to cores `0..(N-1)`;
  `set_bpu_priority` preserves the mask. Validated on S600: Qwen3.5-4B nash-p 4-core,
  22.6 tok/s.

## [0.1.2] — 2026-07-14

### Added

- **Reusable-prefix path** — `NativeSession.set_prefix()` / `ask()`: prefill a shared prefix
  (a document, a fixed system + tools context, few-shot exemplars) **once**, then answer many
  queries against it without re-prefilling — each `ask()` restores the prefix and is independent.
  ~3–4× per query on repeated asks over a shared context. Backed by in-memory `snapshot()` /
  `restore()` on the engines (the file-based `save_state`/`load_state` share the same core).
- **Thinking toggle** — `NativeSession.set_thinking(False)` prefills an empty `<think></think>`
  (the official template's `enable_thinking=False`) so Qwen3.5 reasoning models answer directly
  — faster, fewer tokens. Default reasons and shows the block. (The in-message `/no_think` soft
  switch is unreliable on the current checkpoints; this prefill is.)
- **`encode()` / `decode()`** on the session (the model's C++ tokenizer).
- **Qwen3.5-2B / 4B** (hybrid Gated-DeltaNet/SSM) validated on-board, 100% on-BPU int8
  (~14.5 / 6.9 tok/s), joining the 0.8B.
- **S600 multi-core BPU** — an `.hbm` compiled for N BPU cores (nash-p, N ∈ {1, 2, 4}) is
  recorded via `make_model_dir.py --bpu-cores N` (`model.json` `bpu_cores`). A large model
  saturates 4 cores (Qwen3.5-4B, 27 tok/s on S600); small models stay single-core.

## [0.1.1]

### Added

- **Phi chat template** — a third chat format (`phi`, alongside `chatml` and raw `none`)
  for models that mark roles with `<|user|>` / `<|assistant|>` / `<|system|>` / `<|end|>`
  tokens instead of ChatML. `make_model_dir.py` auto-detects the markers and emits it, so
  **Phi-4-mini** now ships as a first-class multi-turn chat package (previously it only ran
  as raw completion). Existing ChatML models are unaffected.

First public release — an on-board LLM / VLM runtime for D-Robotics RDK S100 / S100P / S600,
driving compiled `.hbm` graphs natively on the BPU (hbDNN / hbUCP), no extra LLM SDK.

### Added

- **Native text runtime** — dense models (Qwen2.5-1.5B/7B, DeepSeek-R1-Distill, InternLM2,
  GLM-Edge, Phi-4-mini) and the **Qwen3.5-0.8B hybrid Gated-DeltaNet/SSM** (100% on-BPU,
  ~21 tok/s), with a built-in C++ tokenizer, ChatML templating, streaming and multi-turn.
- **Multimodal** — `NativeVlm` / `NativeVlmSession` for Qwen2.5-Omni: text + image + audio +
  video, from file paths or raw arrays, with live streaming capture.
- **`bllm.load()`** — one entry point that loads a `model.json` package **or** a bare `.hbm`
  + tokenizer dir (manifest synthesized on the fly).
- **Sampling** — greedy / temperature / top-k / top-p / min-p / typical-p, plus repeat,
  frequency and presence penalties; reproducible with a seed.
- **Production features** — perplexity, async submit, prompt-cache save/restore (bit-identical
  resume), stop strings (stops early), and BPU priority / time-slice control to share the BPU
  with a vision pipeline.
- **Packaging** — conda packages (`bllm`, `libbllm`, `tokenizers-cpp`) for linux-aarch64;
  `find_package(bllm)` for C++ consumers.

[Unreleased]: https://github.com/ruisv/bllm/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/ruisv/bllm/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/ruisv/bllm/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/ruisv/bllm/compare/v0.4.0...v0.4.1
[0.1.2]: https://github.com/ruisv/bllm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ruisv/bllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ruisv/bllm/releases/tag/v0.1.0
