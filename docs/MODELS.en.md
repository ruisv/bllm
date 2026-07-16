<!-- English; 中文（默认）见 MODELS.zh.md -->

# Running an official `.hbm` model on the board

How to turn a downloaded D-Robotics OE-LLM release into a BLLM model directory and run it —
covering the official pre-compiled Qwen2.5, DeepSeek-R1-Distill, InternLM2, **Qwen2.5-Omni**,
and the community models we publish the same way. You download a finished package; nothing is
compiled here.

中文: [MODELS.zh.md](MODELS.zh.md) · API reference: [API.en.md](API.en.md)

## 1. One-time board setup (skip it and you get a segfault)

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bllm

# Enlarge the ION carveout, then reboot. Model weights live in ION rather than process RSS,
# and the runtime segfaults on alloc if the carveout is too small — that is not a BLLM bug.
sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot   # <=3B; bpu_first for 7B
```

`balanced` reserves ~7.5 GiB, so a 24 GB board reports ~15.5 GiB to the OS — that is expected.
Reclaim it by switching the carveout back and rebooting.

Performance mode does **not** survive a reboot; set it once per boot:

```bash
sudo /usr/bin/busybox devmem 0x2b047000 32 0x99
sudo /usr/bin/busybox devmem 0x2b047004 32 0x99
```

## 2. What an official release contains

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

## 3. Assemble a model directory

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

## 4. Run it

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

## 5. Omni: the vision tower fixes your video budget

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

## 6. Notes that still bite

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
