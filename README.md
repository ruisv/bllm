<!-- 中文为默认文档；English: README.en.md -->

# BLLM — RDK BPU 端侧 LLM / VLM 运行时

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg)](python/CMakeLists.txt)
[![Platform](https://img.shields.io/badge/platform-RDK%20S100%20%2F%20S100P%20%2F%20S600%20(aarch64)-0A7BBB.svg)](#安装)
[![Version](https://img.shields.io/badge/version-0.1.0-informational.svg)](CHANGELOG.md)

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

- [特性](#特性) · [安装](#安装) · [快速上手](#快速上手)
- [多模态](#多模态qwen25-omni) · [与视觉流水线共享 BPU](#与视觉流水线共享-bpu)
- [支持的模型](#支持的模型) · [从源码构建](#从源码构建)
- [社区交流](#社区交流) · [参与贡献](#参与贡献) · [致谢](#致谢) · [许可证](#许可证)

## 特性

- **原生 BPU 推理**：直接驱动 `.hbm`，跑在通用 hbDNN / hbUCP 栈上（和视觉共用同一 BPU 驱动）。
- **稠密 + 混合架构**：Qwen2.5 / DeepSeek-R1-Distill / InternLM2 / GLM-Edge / Phi-4-mini 等稠密文本模型，
  以及 **Qwen3.5-0.8B 混合 Gated-DeltaNet/SSM**（严格 100% 落 BPU，~21 tok/s）。
- **多模态**：Qwen2.5-Omni 的 文本 + 图像 + 音频 + 视频，支持文件路径或原始数组（相机/麦克风零拷贝）。
- **一行式会话**：`bllm.load(...)` / `NativeSession`，内置 C++ tokenizer、ChatML 模板、流式、多轮、采样。
- **完整采样**：贪心 / 温度 / top-k / top-p / min-p / typical-p / 重复·频率·存在惩罚，可复现。
- **生产特性**：困惑度、异步提交、提示缓存（KV+SSM 状态存取，逐位一致）、停止串（命中即停）、
  与视觉流水线共享 BPU 的优先级/时间片控制。

## 安装

板上（aarch64，conda 环境）直接从我们的 conda 源安装：

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bllm
```

装上 `bllm`（Python 绑定）、`libbllm`（C++ 库 + 头文件 + cmake 配置）及其运行时依赖
（`hobot-dnn`、`tokenizers-cpp`）。RDK 平台运行库随板系统提供。

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

## 多模态（Qwen2.5-Omni）

文本 + 图像 + 音频 + 视频，媒体可传文件路径**或**原始数组（`HxWx3` uint8 帧、16 kHz 单声道 float PCM），
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
视觉塔的分辨率在离线导出时定死，决定视频上下文预算；模型目录只需把 `visual` 指向想用的塔。
更多用法见 [API 文档](docs/API.zh.md)。

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

官方预编译 **文本** LLM 走同一条免配置路径（自动识别类型、发现对话模板、处理各家差异如 int8 KV）：

- **Qwen2.5** 1.5B / 7B（Base + Instruct）
- **DeepSeek-R1-Distill-Qwen** 1.5B / 7B
- **InternLM2-1.8B** · **GLM-Edge** · **Phi-4-mini**
- **Qwen3.5-0.8B / 2B / 4B**（混合 SSM，原生独有；100% 落 BPU int8）
- **Qwen2.5-Omni-3B**（多模态）

在 q8/q4 × 上下文 1024/4096 等变体上验证。模型转换（`.hbm` 如何产出）是离线流程，不在本仓范围；
本仓消费成品 `.hbm` + tokenizer 配置。

> **S600 多核**：大模型可绑定多个 BPU 核并行 decode——Qwen3.5-4B 用满 4 核 **27 tok/s**、
> 0.8B 单核 **42.7 tok/s**。板端实测见 [`docs/S600_RESULTS.md`](docs/S600_RESULTS.md)。

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

### 制作模型目录

原生模型是带 `model.json` 清单的目录。用 `scripts/make_model_dir.py` 生成（别手写——它会从
tokenizer 的 `generation_config.json` 解析停止符，避免猜错）：

```bash
scripts/make_model_dir.py dense ~/models/qwen2.5-1.5b \
    --hbm Qwen2.5_1.5B_Instruct_1024.hbm \
    --tokenizer Qwen2.5_1.5B_Instruct_config/tokenizer.json --cache-len 1024
```

`hybrid`（Qwen3.5 SSM）与 `omni`（多模态）通过额外参数传入各自的塔/嵌入表。

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
