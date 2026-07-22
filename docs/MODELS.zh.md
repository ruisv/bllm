<!-- 中文；English: MODELS.en.md -->

# 在板上跑官方 `.hbm` 模型

把下载来的 D-Robotics OE-LLM 官方模型变成一个 BLLM 模型目录并跑起来 —— 覆盖官方预编译的
Qwen2.5、DeepSeek-R1-Distill、InternLM2、**Qwen2.5-Omni**，以及我们同样方式发布的社区模型。
你下载的是成品包，这里不编译任何东西。

English: [MODELS.en.md](MODELS.en.md) · API 文档：[API.zh.md](API.zh.md)

## 一、一次性板上设置（不做会段错误）

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bllm

# 扩大 ION carveout，然后重启。模型权重活在 ION 里而不是进程 RSS,
# carveout 不够时运行时会在 alloc 处直接段错误 —— 这不是 BLLM 的 bug。
sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot   # ≤3B；7B 用 bpu_first
```

`balanced` 会保留约 7.5 GiB，所以 24 GB 的板子对 OS 只显示约 15.5 GiB —— 这是正常的。
把 carveout 切回去再重启即可回收。

性能模式**不跨重启保持**，每次开机后设一遍：

```bash
sudo /usr/bin/busybox devmem 0x2b047000 32 0x99
sudo /usr/bin/busybox devmem 0x2b047004 32 0x99
```

## 二、官方包里有什么

OE-LLM SDK 把一个模型拆在 `model/` 和 `config/` 两处：

```
oellm_runtime/
  model/
    Qwen2.5_1.5B_Instruct_1024.hbm     # 稠密模型就是单个 .hbm
    Qwen2.5_Omni_3B_Text.hbm           # Omni 是三个塔 ……
    Qwen2.5_Omni_3B_Visual.hbm
    Qwen2.5_Omni_3B_Audio.hbm
    embed_tokens.bin                   # …… 外加这个。见下方警告。
  config/
    Qwen2.5_Omni_3B_config/
      tokenizer.json  mel_filters_t.txt  generation_config.json  special_tokens_map.json
```

> ⚠️ **`embed_tokens.bin` 是 Omni 必需的，且极易漏下。** 它在官方 `resolve_model_nash-m.txt`
> 里是单独一行 1.2 GB 的 `wget`，躺在 `model/` 下，而且**文件名不带 Omni 字样** —— 只下三个
> `*_Omni_3B_*.hbm` 看着像下全了，其实并没有。Omni 把词嵌入表放在宿主侧（BPU 图吃的是
> embedding 而不是 token id），缺了它模型根本跑不起来。

## 三、拼出模型目录

BLLM 加载的是一个**目录**：模型文件 + 一份 `model.json` 清单。用 `bllm-make-model-dir` 生成
（随 `bllm` 包装好，也可以 `python -m bllm.make_model_dir`；源码树里是 `scripts/make_model_dir.py`）。

**别手写 `model.json`**：该工具会按权威性从 config 目录里解析停止符并打印用了哪个来源。
猜错 eos 的下场是模型永不停 —— Qwen2.5 的词表里就有一个 id 128247 的字面量 `</s>`，它**不是**停止符。

**稠密模型**（Qwen2.5 / DeepSeek-R1-Distill / InternLM2 / GLM-Edge / Phi-4-mini）：

```bash
R=~/D-Robotics_LLM_S100_1.0.0_SDK/oellm_runtime    # 你解压 SDK 的位置

bllm-make-model-dir dense ~/models/qwen2.5-1.5b \
    --hbm       $R/model/Qwen2.5_1.5B_Instruct_1024.hbm \
    --tokenizer $R/config/Qwen2.5_1.5B_Instruct_config/tokenizer.json \
    --cache-len 1024
```

**Omni**（多模态 —— 下面每个输入官方包里都有）：

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

只要视觉不要音频，就同时去掉 `--audio` 和 `--mel-filters`（这俩必须成对 —— 音频塔需要它的
mel 滤波器组）。工具默认用符号链接，所以目录几乎不占空间；想要自包含就加 `--copy`。

**Qwen3.5 图文**（hybrid 文本塔 + 视觉塔）：Qwen3.5 官方就是 image-text-to-text，
纯文本的包只是编译时把视觉半边丢掉了。同一个文本 `.hbm` 配上视觉塔就是图文包：

```bash
bllm-make-model-dir hybrid ~/models/qwen3.5-0.8b-vlm     --hbm       qwen35_0.8b_ctx4k.hbm     --embed     embed_fp16.bin     --tokenizer qwen35_tok/tokenizer.json     --visual    qwen35_visual_448.hbm     --mrope-section 11 11 10 --mrope-interleaved     --vision-patch 16 --vision-mean 0.5 --vision-std 0.5     --cache-len 4096
```

后四个参数**必填、且不能照抄 Omni**，因为编译好的 `.hbm` 里看不出这些，配错只会得到
**形状正确的垃圾**，没有任何检查能拦住：

| 参数 | Qwen3.5 | Omni | 配错的后果 |
|---|---|---|---|
| `--vision-patch` | **16** | 14 | 切块几何全错 |
| `--vision-mean/std` | **0.5** | CLIP 常数 | 像素归一化偏移 |
| `--mrope-section` | **11 11 10** | 16 24 24 | rope 维度分配错 |
| `--mrope-interleaved` | **必须加** | 不加 | 频率布局是 `[THWTHW…]` 而非连续分段 |

最后一条最阴：`t==h==w` 时两种布局**完全等价**，所以纯文本一切正常，只有喂进图像之后
位置才开始错。工具因此对 hybrid + `--visual` 强制要求这四个参数，不给默认值。

## 四、跑起来

```python
import bllm

llm = bllm.load("~/models/qwen2.5-1.5b")
print(llm.chat("你好"))

# load() 按包里有没有 visual 塔路由，omni 与 hybrid 图文包都走 NativeVlmSession
vlm = bllm.load("~/models/qwen2.5-omni-3b")
for chunk in vlm.stream_chat("描述这张图片。", ["bus.jpg"]):
    print(chunk, end="", flush=True)
```

裸 `.hbm` 也能直接吃（清单自动合成），但仅限稠密模型：

```python
llm = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="Qwen2.5_1.5B_config/")
```

图像 / 音频 / 视频与相机-麦克风实时流的完整用法见
[`examples/vlm_native.py`](../examples/vlm_native.py) 与 [API 文档](API.zh.md)。

## 五、Omni：视觉塔决定了你的视频预算

官方 `Visual.hbm` 固定在 **448×448**，代价是**每 frame-pair 256 个解码 token**。对上 Omni
2048 的 KV 窗口，也就是 **2 fps 下约 8 秒视频** —— 这还没算 prompt，也没算音频（另收约 25 token/s）。
KV 窗口**就是**上下文：越界会抛异常而不是静默丢弃，因为丢掉最早的 token 会让这种全注意力模型输出乱码。

`vlm.context_left` 跟踪剩余额度。**单张图片不受影响** —— 一张图就是一次 256 token 的开销。

更低分辨率的塔能换来成比例的视频时长（每 frame-pair `(size/14/2)²` 个 token：336→144、224→64），
但分辨率在导出时就已烤死，换塔意味着要在 x86 上用 `leap_llm` 工具链离线重新导出 —— 不在本仓运行时
范围内，官方也没有发布别的塔。

## 六、仍然会咬人的点

1. **eos 是每个模型不同的。** `bllm-make-model-dir` 会解析它（`generation_config.json` 优先，
   其次是 tokenizer 声明的 special token），但如果某个模型永不停、把上下文填满，第一个该查的就是 eos。
   Phi-4 停在 `<|end|>`，GLM-Edge 停在 `<|user|>`，Qwen 停在 `<|im_end|>`。
2. **推理型模型（Qwen3 / Qwen3.5）很啰嗦。** 一条创意 prompt 就能用 `<think>…</think>` 填满
   1024 的缓存。用 4096 的构建，或 `llm.set_thinking(False)` 预填空 `<think></think>` 直接作答。
3. **对话模板来自 `model.json`，不是 `.jinja`。** 原生运行时按 tokenizer 声明的 id 自己拼
   ChatML / Phi 轮次；SDK 里的 `.jinja` 是给旧 OE-LLM 运行时用的，这里会被忽略。
4. **tokenizer 格式不是约束。** 原生运行时直接读现代 `tokenizers` 0.20+ 的 `tokenizer.json`
   （pair-format merges + `ignore_merges`）。过去那条"必须降级到 0.19.1"是 libxlm 的限制，已不适用。
