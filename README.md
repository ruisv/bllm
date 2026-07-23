<!-- 中文为默认文档；English: README.en.md -->

# BLLM — RDK BPU 端侧 LLM / VLM 运行时

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg)](python/CMakeLists.txt)
[![Platform](https://img.shields.io/badge/platform-RDK%20S100%20%2F%20S100P%20%2F%20S600%20(aarch64)-0A7BBB.svg)](#安装)
[![Version](https://img.shields.io/badge/version-0.1.6-informational.svg)](CHANGELOG.md)

**简体中文** | [English](README.en.md)

**BLLM**（BPU LLM）是面向 D-Robotics **RDK S100 / S100P / S600** 的 C++17 端侧
**大语言模型 / 多模态**运行时库。它把已编译的 `.hbm` 模型图直接跑在板载 BPU
（hbDNN / hbUCP）上——自建的原生推理引擎，自带 KV / SSM 缓存、采样器与解码循环——
并给出一套简洁的 C++ / Python 任务式 API。**无需任何额外的大模型 SDK**。

它是视觉库 [bcdl](https://github.com/ruisv/bcdl)（RDK BPU 视觉推理）的大模型姊妹库。

<p align="center">
  <img src="docs/demo.gif" width="760" alt="Qwen3.5-0.8B 混合 SSM 在 RDK S100P BPU 上流式对话"><br>
  <em>Qwen3.5-0.8B（混合 SSM）在 RDK S100P BPU 上原生流式对话 · ~14 tok/s</em>
</p>

> 📖 API 文档：[中文](docs/API.zh.md) · [English](docs/API.en.md)

## 目录

- [特性](#特性) · [安装](#安装) · [快速上手](#快速上手) · [制作模型目录](#制作模型目录)
- [服务化部署](#服务化部署openai-兼容-http) · [多模态](#多模态) · [与视觉流水线共享 BPU](#与视觉流水线共享-bpu)
- [支持的模型](#支持的模型) · [从源码构建](#从源码构建)
- [社区交流](#社区交流) · [参与贡献](#参与贡献) · [致谢](#致谢) · [许可证](#许可证)

## 特性

- **原生 BPU 推理**：直接驱动 `.hbm`，跑在通用 hbDNN / hbUCP 栈上（和视觉共用同一 BPU 驱动）。
- **稠密 + 混合架构**：Qwen2.5 / DeepSeek-R1-Distill / InternLM2 / GLM-Edge / Phi-4-mini 等稠密文本模型，
  以及 **Qwen3.5-0.8B 混合 Gated-DeltaNet/SSM**（严格 100% 落 BPU，~21 tok/s）。
- **多模态**：Qwen2.5-Omni 的 文本 + 图像 + 音频 + 视频，以及 **Qwen3.5 图文**（自编视觉塔）；支持文件路径或原始数组（相机/麦克风零拷贝）。
- **一行式会话**：`bllm.load(...)` / `NativeSession`，内置 C++ tokenizer、ChatML 模板、流式、多轮、采样。
- **完整采样**：贪心 / 温度 / top-k / top-p / min-p / typical-p / 重复·频率·存在惩罚，可复现；
  另有 `logit_bias`（强推/封禁 token）与 `logprobs`（全词表精确对数概率）。
- **结构化输出**：GBNF 语法约束解码 + JSON Schema → 语法。**解码期**只允许仍能满足语法的
  token，所以 `json.loads()` 不会失败——是保证，不是提示词里的请求。
- **生产特性**：困惑度、异步提交、提示缓存（KV+SSM 状态存取，逐位一致）、停止串（命中即停）、
  与视觉流水线共享 BPU 的优先级/时间片控制。

## 安装

板上（aarch64，conda 环境）直接从我们的 conda 源安装：

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bllm
```

装上 `bllm`（Python 绑定）、`libbllm`（C++ 库 + 头文件 + cmake 配置）及其运行时依赖
（`hobot-dnn`、`tokenizers-cpp`）。RDK 平台运行库随板系统提供。

还需要**一次性**把 ION carveout 调大，然后重启 —— 模型权重活在 ION 里而不是进程 RSS，
carveout 不够时运行时会在 alloc 处直接**段错误**：

```bash
sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot   # ≤3B；7B 用 bpu_first
```

性能模式不跨重启保持，每次开机设一遍（见 [`docs/MODELS.zh.md`](docs/MODELS.zh.md)）。

## 快速上手

一个模型是一个**目录**（`.hbm` + `tokenizer.json` + `model.json` 清单），一行加载：

```python
import bllm

llm = bllm.load("/path/to/qwen2.5-1.5b")        # 目录，或 裸 .hbm + tokenizer_dir=
llm.set_sampling(temp=0.7, top_p=0.9)           # 不设即贪心
for chunk in llm.stream_chat("用一句话介绍北京"):  # 流式、多轮
    print(chunk, end="", flush=True)
print(llm.last_decode_tps)
```

裸 `.hbm`（OE 包）也能直接吃，运行时自动合成清单：

```python
llm = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="Qwen2.5_1.5B_config/")
print(llm.chat("你好"))
```

C++：

```cpp
#include "bllm/native_llm.h"
bllm::NativeLlm llm("/path/to/qwen2.5-1.5b");
std::string reply = llm.chat("你好", 400, [](const std::string& s){ std::cout << s << std::flush; });
```

REPL：`bllm_chat_native --model <dir>`（见 [`examples/chat_native.cc`](examples/chat_native.cc)）。

## 制作模型目录

下载的官方模型（`.hbm` 散在 `model/`、tokenizer 散在 `config/`）用 `bllm-make-model-dir`
拼成上面那个目录 —— 它随 `bllm` 包一起装好，不必 clone 本仓
（等价形式：`python -m bllm.make_model_dir`；源码树里是 `scripts/make_model_dir.py`）：

```bash
R=<sdk>/oellm_runtime
bllm-make-model-dir dense ~/models/qwen2.5-1.5b \
    --hbm       $R/model/Qwen2.5_1.5B_Instruct_1024.hbm \
    --tokenizer $R/config/Qwen2.5_1.5B_Instruct_config/tokenizer.json \
    --cache-len 1024
```

**别手写 `model.json`** —— 该工具会按权威性依次从 `generation_config.json`、
tokenizer 声明的 special token 解析停止符并打印用了哪个来源。猜错 eos 的下场是模型永不停：
Qwen2.5 的词表里就有一个 id 128247 的字面量 `</s>`，它并不是停止符。

三种 `arch`：`dense`（上面）、`hybrid`（Qwen3.5 SSM，另需 `--embed`）、
`omni`（[多模态](#多模态)，另需塔与嵌入表）。官方包结构、ION 设置等完整部署步骤见
[`docs/MODELS.zh.md`](docs/MODELS.zh.md)。

## 服务化部署（OpenAI 兼容 HTTP）

把运行时打包成 OpenAI 兼容的推理服务（`/v1/chat/completions`，流式 + 非流式），容器化部署在板上。
完整说明见 [`docker/README.md`](docker/README.md)。

**自包含镜像（模型烤进去，开箱即用）** —— `docker run` 直接对话，无需挂载模型：

```bash
cd docker
./bake-model.sh ~/models/qwen3.5-2b bllm-serve:qwen3.5-2b-ctx512-int8-s100p
docker run -d --network host \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu \
  bllm-serve:qwen3.5-2b-ctx512-int8-s100p
```

**模型无关镜像（模型走 volume，一个镜像跑任意模型）** —— 用 docker compose：

```bash
cd docker && ./stage-libs.sh
MODEL_DIR=~/models/qwen3.5-2b docker compose up -d --build
```

服务默认监听 `:8866`，已开 CORS。用任意 OpenAI 兼容客户端指向 `http://<板子IP>:8866/v1` 即可
（[chatbox-lite](https://github.com/lfbear/chatbox-lite)、桌面 Chatbox、Open WebUI、OpenAI SDK…）：

```bash
curl localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```


服务端会**缓存对话前缀**：OpenAI 协议无状态、客户端每轮重发全部历史，服务端只 prefill
新增部分。命中量在 `usage.prompt_tokens_details.cached_tokens`，整体统计看 `GET /metrics`。
匹配是逐 token 精确前缀比对，改了历史只会不命中并退化为全量 prefill，不会给出错答案。
板上实测：hybrid（Qwen3.5）31.6s → 1.3s（~25×），dense TTFT 274 → 141 ms。
一次只跑一个生成（单 BPU 图），排队超过 `BLLM_MAX_QUEUE` 返回 429。
详见 [`docker/README.md`](docker/README.md)。

## 多模态

两条路：官方 Qwen2.5-Omni（文/图/音/视频），以及 **Qwen3.5 图文**（我们自己编的视觉塔）。

### Qwen3.5 图文

Qwen3.5 官方就是 image-text-to-text —— 纯文本的包只是编译时把视觉半边丢掉了。
同一个 hybrid 文本 `.hbm` 配上视觉塔即可（视觉塔离线转换，成品包直接可用）：

```bash
bllm-make-model-dir hybrid ~/models/qwen3.5-0.8b-vlm448-ctx2k \
    --hbm qwen35_0.8b_ctx2k.hbm --embed embed_fp16.bin --tokenizer tokenizer.json \
    --visual qwen35_visual_448.hbm --hbm-prefill qwen35_prefill_n32.hbm \
    --mrope-section 11 11 10 --mrope-interleaved \
    --vision-patch 16 --vision-mean 0.5 --vision-std 0.5 --cache-len 2048
```

```python
vlm = bllm.load("~/models/qwen3.5-0.8b-vlm448-ctx4096-int8-s100p")
print(vlm.chat("Describe this image in one sentence.", images=["bears.jpg"]))
# -> Two brown bears walk across a dusty, arid landscape.
```

S100P 实测（0.8B，含分块 prefill 图）：

| 桶 | token/图 | 塔输出 cosine vs HF fp32 | 首字延迟 |
|---|---|---|---|
| 320×320（更快，推荐） | 100 | 0.9987 | **1.23 s** |
| 448×448 | 196 | 0.99654 | **2.30 s** |

> ⚠️ 后四个参数**必填且不能照抄 Omni**（patch 16 vs 14、mean/std 0.5 vs CLIP、
> rope 分段 `11 11 10` vs `16 24 24`、频率布局 interleaved vs 连续）。编译好的 `.hbm`
> 里看不出这些，配错只会得到**形状正确的垃圾**。尤其 `--mrope-interleaved`：
> `t==h==w` 时两种布局完全等价，所以纯文本一切正常，**只有喂进图像之后位置才开始错**。
>
> ⚠️ 首字延迟的大头是把视觉行喂进文本模型，**不是视觉塔**（448 塔 encode 仅 0.36 s）。
> 分块 delta-rule 的 prefill 图（`--hbm-prefill`）把逐 token 摄入摊薄到秒级，分辨率越低越快。
>
> 目前只支持**静止图像**：视频路径实现的是 Omni 的 TMRoPE，而 Qwen3.5 用时间戳 token
> 分隔帧，传 `videos=` 会明确报错而不是悄悄编出顺序错乱的答案。

### Qwen2.5-Omni（文/图/音/视频）

官方发布的 Omni 是**三个塔 + 一张宿主嵌入表**，先拼成一个模型目录
（`bllm-make-model-dir` 随 conda 包一起装好，不必 clone 本仓）：

```bash
R=<sdk>/oellm_runtime
bllm-make-model-dir omni ~/models/qwen2.5-omni-3b \
    --hbm         $R/model/Qwen2.5_Omni_3B_Text.hbm \
    --visual      $R/model/Qwen2.5_Omni_3B_Visual.hbm \
    --audio       $R/model/Qwen2.5_Omni_3B_Audio.hbm \
    --embed       $R/model/embed_tokens.bin \
    --mel-filters $R/config/Qwen2.5_Omni_3B_config/mel_filters_t.txt \
    --tokenizer   $R/config/Qwen2.5_Omni_3B_config/tokenizer.json \
    --cache-len 2048
```

> ⚠️ **`embed_tokens.bin` 是 Omni 必需的，且极易漏下**：它在官方下载清单里是单独一行
> 1.2 GB 的 `wget`，躺在 `model/` 下，而且**文件名不带 Omni 字样** —— 只下三个
> `*_Omni_3B_*.hbm` 看着像下全了，其实缺它根本跑不起来（Omni 的嵌入表放在宿主侧，
> BPU 图吃的是 embedding 而非 token id）。

然后文本 + 图像 + 音频 + 视频，媒体可传文件路径**或**原始数组（`HxWx3` uint8 帧、16 kHz 单声道 float PCM），
相机/麦克风流水线无需落盘：

```python
vlm = bllm.load("/models/qwen2.5-omni-3b")       # model.json 里 arch="omni"
for chunk in vlm.stream_chat("描述这张图片。", ["bus.jpg"]):
    print(chunk, end="", flush=True)

vlm.reset(); vlm.chat("这段音频里说了什么？", audios=["clip.wav"])
vlm.reset(); vlm.chat("描述这段视频。", videos=[bllm.load_video("clip.mp4")])
```

**实时输入**：边采集边把每 2 秒的分块编入 KV 缓存，提问时的首字延迟远低于离线：

```python
vlm.stream_begin(fps=2.0)
while capturing:
    vlm.stream_frame(frame_hwc_uint8)     # 相机帧
    vlm.stream_audio(pcm_float32)         # 麦克风 16 kHz 单声道
print(vlm.stream_ask("刚才发生了什么？"))
```

KV 窗口**就是**上下文，也是硬约束（`vlm.context_left` 跟踪剩余额度，越界会报错而非静默丢弃）。
**官方 `Visual.hbm` 固定 448px = 256 token/frame-pair，配 2048 的 KV 窗口 ⇒ 视频只有约 8 秒 @2fps**，
且这还没扣 prompt 和音频（音频另收 ~25 token/s）；单张图片不受影响，一张图就是一次 256 token。
更低分辨率的塔能换来成比例的时长（`(size/14/2)²`：336→144、224→64 token），但分辨率在离线导出时
就已定死，换塔要在 x86 上用 `leap_llm` 重新导出，不在本仓范围，官方也未发布别的塔。

完整部署步骤（dense 模型、ION 设置、官方包结构）见
[`docs/MODELS.zh.md`](docs/MODELS.zh.md)，更多用法见 [API 文档](docs/API.zh.md)。

## 与视觉流水线共享 BPU

S100/S100P 单 BPU 核，LLM 与视觉（bcdl）会争用。一次解码是一个大图，默认会挡住同驻的检测器。
让检测器插队需要**两个**条件一起：`.hbm` 编译时设 `max_time_per_fc`（让调度器能在图内切换）
+ 运行时把 LLM 优先级压低：

```python
llm.set_bpu_priority(0)      # 最低,让视觉抢占 BPU 队列
```

实测（yolo26s vs Qwen2.5-1.5B 解码，检测器 p99）：两个旋钮都开时检测器 p99 从 41 ms 降到 **2.95 ms**、
383 FPS，LLM 24→17 tok/s；空载时几乎不损耗。

## 支持的模型

下列模型走同一条免配置路径（自动识别类型、发现对话模板、处理各家差异如 int8 KV），
在 q8/q4 × 上下文 1024/4096 等变体上验证。

**D-Robotics 官方发布的 `.hbm`**：

- **Qwen2.5** 1.5B / 7B（Base + Instruct）
- **DeepSeek-R1-Distill-Qwen** 1.5B / 7B
- **InternLM2-1.8B**
- **Qwen2.5-Omni-3B**（多模态）

**本仓自行转换的 `.hbm`**（官方工具链未覆盖的架构，由我们离线转换后在板上验证）：

- **Qwen3.5-0.8B / 2B / 4B**（混合 SSM，**BLLM 原生独有**；100% 落 BPU int8）
- **GLM-Edge** · **Phi-4-mini**

模型转换（`.hbm` 如何产出）是离线流程，不在本仓范围；本仓消费成品 `.hbm` + tokenizer 配置。

> **S600 多核**：大模型可绑定多个 BPU 核并行 decode——Qwen3.5-4B 用满 4 核 **27 tok/s**、
> 0.8B 单核 **42.7 tok/s**（板端实测）。

### 预编译模型下载

我们把一批**编译好、可直接 `bllm.load()` 的模型包**放在网盘（每个是一个目录：`model.hbm` +
`tokenizer.json` + `model.json`，混合模型另含 `embed_tokens.bin`）。文件名已标注**上下文长度 /
目标板子 / 量化位宽**，一看即知：

**📦 [Google Drive — 模型下载](https://drive.google.com/drive/folders/1tR3MtP0iriptqpeHOSzdpRMT_ckdqDQ8)**

下载后校验、解压即可加载（以 Qwen3.5-4B 为例）：

```bash
sha256sum -c qwen3.5-4b-ctx512-int8-s100.tar.gz.sha256   # 完整性校验
tar xzf qwen3.5-4b-ctx512-int8-s100.tar.gz
python -c "import bllm; s=bllm.load('qwen3.5-4b-ctx512-int8-s100'); print(s.chat('你好'))"
```

## 从源码构建

多数用户直接 `conda install bllm` 即可（见[安装](#安装)）。想改源码/自行构建，则**在板上**
克隆并构建（运行时只在板上）：

```bash
# 在 RDK S100 / S100P / S600 板上
git clone https://github.com/ruisv/bllm.git && cd bllm

# 构建依赖（示例，从我们的 conda 源）
conda install -c https://mirrors.ruis.ai/conda -c conda-forge \
    cmake ninja cxx-compiler nlohmann_json hobot-dnn tokenizers-cpp nanobind numpy

scripts/build.sh --python          # cmake + ninja；产物在 build/
export PYTHONPATH="$PWD/python"     # 就地 import；或 scripts/build.sh --python --install
```

一个 CMake 目标 `bllm::native`，只依赖板载通用 hobot 运行时（+ `tokenizers-cpp` 做字符串 I/O）：

```cmake
find_package(bllm REQUIRED)
target_link_libraries(app PRIVATE bllm::native)
```

Python 里 `import bllm` 在有无扩展时都能导入；`bllm.available_backends()` 返回 `("native",)` 或 `()`。

## 社区交流

欢迎加入 **BPU 技术交流群**，一起讨论 RDK / BPU 端侧部署与本项目的使用（与
[bcdl](https://github.com/ruisv/bcdl) 共用同一社区群）。

<img src="docs/assets/bllm-group-qr.jpg" alt="BPU 技术交流群二维码" width="240">

> 微信群二维码有有效期，若已过期请在 [Issues](../../issues) 留言，我们会更新；
> 也欢迎直接在 [Issues](../../issues) 交流。

## 参与贡献

欢迎贡献 —— issue 与 pull request 皆可。完整指南见 [`CONTRIBUTING.md`](CONTRIBUTING.md)；要点：

- **在任意机器开发，在板上构建与运行。** hobot 运行时只存在于 RDK 硬件上，所以 C++
  库、Python 扩展和板端测试必须在 S100 / S100P / S600 上构建执行；纯元数据的单测
  （`tests/test_modelmeta.py` 等）可在任意主机上跑。
- **约定** —— 头文件 `.h`、实现 `.cc`，命名空间 `bllm`，错误经 `BLLM_CHECK(...)`；与周围
  代码风格保持一致。
- **测试** —— 行为变更需补测试：能离线跑的用元数据/路由单测钉死逻辑，涉及模型的补一个
  板端端到端测试且模型缺失时干净跳过。
- **提交** —— 保持聚焦，遵循日志里已有的 [Conventional Commits](https://www.conventionalcommits.org/) 风格。

## 致谢

- **D-Robotics** —— RDK S100 / S100P / S600 平台与 hobot 运行时（`hbDNN` / `hbUCP` /
  `HBRT`），BLLM 构建于其上。
- **[nanobind](https://github.com/wjakob/nanobind)** —— Python 绑定层。
- **[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)** +
  **[HuggingFace tokenizers](https://github.com/huggingface/tokenizers)** —— C++ 分词。
- 所运行模型的上游作者 —— **Qwen**（Qwen2.5 / Qwen3.5 / Qwen2.5-Omni，阿里巴巴）、
  **DeepSeek**、**InternLM**（上海人工智能实验室）、**GLM**（智谱）、**Phi**（微软）。
  各权重受其各自上游许可证约束。

## 许可证

BLLM 以 **Apache License 2.0** 授权 —— 见 [`LICENSE`](LICENSE)。仅覆盖 BLLM 自身源码；
`.hbm` 模型权重受其各自上游许可证约束。
