<!-- English; 中文（默认）见 MODELS.zh.md -->

# Running an official `.hbm` model on the board

How to run a downloaded D-Robotics release. **An official LLM 2.0 package (Qwen3, Qwen3-VL,
Qwen2.5-VL, MiniCPM5, Phi-4-mini, …) loads as soon as it is unzipped** — section 2. A 1.0 SDK
release (Qwen2.5, DeepSeek-R1-Distill, InternLM2, **Qwen2.5-Omni**) and the community models we
publish the same way are assembled into a model directory first — section 3 on. You download a
finished package; nothing is compiled here.

中文: [MODELS.zh.md](MODELS.zh.md) · API reference: [API.en.md](API.en.md)

## 1. One-time board setup (skip it and you get a segfault)

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bllm

# Enlarge the ION carveout, then reboot. Model weights live in ION rather than process RSS,
# and the runtime segfaults on alloc if the carveout is too small — that is not a BLLM bug.
sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot   # <=3B; bpu_first for Qwen3.5-4B and 7B
```

`balanced` reserves ~7.5 GiB, so a 24 GB board reports ~15.5 GiB to the OS — that is expected.
Reclaim it by switching the carveout back and rebooting.

Performance mode does **not** survive a reboot; set it once per boot:

```bash
sudo /usr/bin/busybox devmem 0x2b047000 32 0x99
sudo /usr/bin/busybox devmem 0x2b047004 32 0x99
```

## 2. Official LLM 2.0 packages: unzip and load (S100P)

The LLM 2.0.0 SDK D-Robotics released in October 2026 ships each model as its own zip: one
`*_language_*.hbm`, one `*_embed_tokens.bin`, and for an image model a `*_vision_*.hbm`. There
are **no** tokenizer or config files — they are embedded in the language `.hbm`'s description,
which BLLM reads, so the unzipped directory loads as it is:

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s100_models/Qwen3-1.7B_w4.zip
unzip Qwen3-1.7B_w4.zip
```

```python
import bllm

llm = bllm.load("Qwen3-1.7B_w4/")                  # text package  -> NativeSession
print(llm.chat("Hello"))

vlm = bllm.load("Qwen3-VL-2B_w4/")                 # image package -> NativeVlmSession
print(vlm.chat("Describe this image.", images=["a.jpg"]))

txt = bllm.load("Qwen3-VL-2B_w4/", vision=False)   # its text tower alone
```

The first load reads the description (1–3 s) and writes `model.json`, `tokenizer.json` and
symlinks to the package files under `$BLLM_CACHE` (default `~/.cache/bllm/native/`); later
loads use that. C++ users get the same directory from the same tool (run it on the board — it
reads the `.hbm`):

```bash
bllm-make-model-dir official ~/models/qwen3-1.7b Qwen3-1.7B_w4/
```

The full package list is the SDK's `oellm_runtime_s100/model/resolve_model_s100.md`; downloads
follow `https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s100_models/<package>.zip`.
BLLM neither redistributes nor mirrors these packages — download them from D-Robotics.

**Verified one by one on S100P**: greedy output token-identical to the official `chat_demo`,
decode at 97% or more of the official speed.

| kind | packages |
|---|---|
| text | Qwen3-0.6B w8 · Qwen3-1.7B w4 · Qwen3-4B w4 · Qwen3-4B-Instruct-2507 w4 · Qwen3-4B-Thinking-2507 w4 · Qwen3-8B w4 · Qwen2.5-3B w4 · MiniCPM5-2B w4 · Phi-4-mini w8 |
| image + text | Qwen3-VL-2B / 4B / 8B w4 · Qwen2.5-VL-3B / 7B w4 |

All of this matches the official runtime, but is worth knowing:

- **Thinking is off by default.** `llm.set_thinking(True)` turns it on (Qwen3, MiniCPM5); a
  thinking-only model (Thinking-2507) always thinks; models without a think mode (Qwen2.5,
  Phi-4, Instruct-2507) ignore the switch.
- **The default system prompt is the template's own** — none for most models, a fixed one for
  Qwen2.5 and Qwen2.5-VL. Override it with `bllm.load(..., system="...")`.
- **Images are stretched** to the vision tower's side (448×448, aspect ratio not kept). Image
  paths are decoded with Pillow when it is installed, pixel-identical to the official runtime;
  HxWx3 uint8 arrays work too.
- **ION `balanced` is enough up to 8B w4** — Qwen3-8B, Qwen3-VL-8B and Qwen2.5-VL-7B were all
  checked on it.
- **Not supported yet**: S600 packages (`nash-p`, fp16 I/O — refused with a clear error),
  speculative-decoding packages (`*_eagle3`, `*_dflash`), speech-recognition packages, and video
  input to the official image models. There are no official S100 (nash-e) packages.

BLLM's own features work on official packages as on any other: GBNF grammars, logprobs,
prompt-cache snapshots, the OpenAI-compatible server.

## 3. What a 1.0 SDK release contains

The OE-LLM SDK splits a model across `model/` and `config/`:

```
oellm_runtime/
  model/
    Qwen2.5_1.5B_Instruct_1024.hbm     # a dense model is a single .hbm
    Qwen2.5_Omni_3B_Text.hbm           # Omni is THREE towers …
    Qwen2.5_Omni_3B_Visual.hbm
    Qwen2.5_Omni_3B_Audio.hbm
    embed_tokens.bin                   # … PLUS this. See the warning below.
  config/
    Qwen2.5_Omni_3B_config/
      tokenizer.json  mel_filters_t.txt  generation_config.json  special_tokens_map.json
```

> ⚠️ **`embed_tokens.bin` is required by Omni and easy to miss.** It is a separate 1.2 GB
> `wget` line in the official `resolve_model_nash-m.txt`, it sits in `model/`, and **its name
> does not mention Omni** — so downloading the three `*_Omni_3B_*.hbm` files looks like a
> complete set, and is not. Omni keeps its token-embedding table on the host (the BPU graph
> consumes embeddings, not token ids), so without this file the model cannot run at all.

## 4. Assemble a model directory

BLLM loads a **directory**: the payload plus a `model.json` manifest. Build it with
`bllm-make-model-dir` (ships with the `bllm` package; also `python -m bllm.make_model_dir`, or
`scripts/make_model_dir.py` in a source tree).

**Never hand-write `model.json`**: the tool resolves the stop tokens from the most
authoritative source in the config dir and prints which one it used. A guessed eos gives you a
model that never stops — Qwen2.5's vocabulary contains a literal `</s>` at id 128247 that is
*not* a stop token.

**Dense** (Qwen2.5 / DeepSeek-R1-Distill / InternLM2 / GLM-Edge / Phi-4-mini):

```bash
R=~/D-Robotics_LLM_S100_1.0.0_SDK/oellm_runtime    # wherever you unpacked the SDK

bllm-make-model-dir dense ~/models/qwen2.5-1.5b \
    --hbm       $R/model/Qwen2.5_1.5B_Instruct_1024.hbm \
    --tokenizer $R/config/Qwen2.5_1.5B_Instruct_config/tokenizer.json \
    --cache-len 1024
```

**Omni** (multimodal — every input below exists in the official release):

```bash
bllm-make-model-dir omni ~/models/qwen2.5-omni-3b \
    --hbm         $R/model/Qwen2.5_Omni_3B_Text.hbm \
    --visual      $R/model/Qwen2.5_Omni_3B_Visual.hbm \
    --audio       $R/model/Qwen2.5_Omni_3B_Audio.hbm \
    --embed       $R/model/embed_tokens.bin \
    --mel-filters $R/config/Qwen2.5_Omni_3B_config/mel_filters_t.txt \
    --tokenizer   $R/config/Qwen2.5_Omni_3B_config/tokenizer.json \
    --cache-len 2048
```

Drop `--audio` + `--mel-filters` for a vision-only Omni (they go together — the audio tower
needs its mel filter bank). The tool symlinks the payload by default, so the directory costs
nothing; pass `--copy` to make it self-contained.

## 5. Run it

```python
import bllm

llm = bllm.load("~/models/qwen2.5-1.5b")
print(llm.chat("Hello"))

vlm = bllm.load("~/models/qwen2.5-omni-3b")          # arch="omni" -> NativeVlmSession
for chunk in vlm.stream_chat("Describe this image.", ["bus.jpg"]):
    print(chunk, end="", flush=True)
```

A bare `.hbm` works too, with the manifest synthesized on the fly — dense models only:

```python
llm = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="Qwen2.5_1.5B_config/")
```

See [`examples/vlm_native.py`](../examples/vlm_native.py) for image / audio / video and live
camera-microphone streaming, and the [API reference](API.en.md).

## 6. Omni: the vision tower fixes your video budget

The official `Visual.hbm` is frozen at **448×448**, which costs **256 decoder tokens per
frame-pair**. Against Omni's 2048-slot KV window that is about **8 seconds of video at 2 fps** —
before the prompt, and before audio (another ~25 tokens/s). The KV window *is* the context:
overrunning it raises rather than silently dropping tokens, because evicting the oldest tokens
makes a full-attention model emit garbage.

`vlm.context_left` tracks the remaining budget. **Still images are unaffected** — one image is a
single 256-token charge.

A lower-resolution tower buys proportionally more video (`(size/14/2)²` tokens per frame-pair:
336→144, 224→64), but the resolution is baked in at export time, so changing it means
re-exporting the tower offline on an x86 host with the `leap_llm` toolchain — outside this
repo's runtime path, and the official release ships no alternative.

## 7. Notes that still bite

1. **eos is per-model.** `bllm-make-model-dir` resolves it (`generation_config.json` wins, then
   the tokenizer's declared specials), but if a model never stops and fills the context, the eos
   is what to check first. Phi-4 ends on `<|end|>`, GLM-Edge on `<|user|>`, Qwen on `<|im_end|>`.
2. **Reasoning models (Qwen3 / Qwen3.5) are verbose.** A single creative prompt can fill a
   1024-slot cache with `<think>…</think>`. Use a 4096 build, or `llm.set_thinking(False)` to
   prefill an empty `<think></think>` and answer directly.
3. **The chat template comes from `model.json`, not a `.jinja`.** The native runtime builds
   ChatML/Phi turns itself from the ids the tokenizer declares; the SDK's `.jinja` files are for
   the old OE-LLM runtime and are ignored here.
4. **Tokenizer format is not a constraint.** The native runtime reads modern `tokenizers` 0.20+
   `tokenizer.json` (pair-format merges + `ignore_merges`) as-is. The old requirement to
   downgrade one to 0.19.1 was a libxlm limitation and no longer applies.
