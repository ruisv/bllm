# 更新日志

<!-- 中文；English: CHANGELOG.md -->

BLLM 的所有重要变更都记录在这里。格式遵循 [Keep a Changelog](https://keepachangelog.com/)，
版本号遵循[语义化版本](https://semver.org/)。

## [Unreleased]

## [0.10.1] — 2026-11-27

语音：模型跳过的句子不再返回空；静音仍然返回空。

### 修复
- **语音：对清楚的话回答空的情况能找回了。** Qwen3-ASR 偶尔会在一句说得很清楚的话的第一个字之前就停下，
  什么也不返回（AISHELL-1 的 2400 次解码里有 5 次；官方运行时也会这样）。模型对这个停止有多确定，并不能把它
  与静音区分开，但停止之后的内容可以：越过停止符接着解码，被跳过的句子会完整、有把握地出来，而静音、噪声、
  哼声和单音只会给出一个填充词（"嗯。"、"The."）。所以空回答会再解码一次，只有读起来是一句有把握的话时才保留
  （6 个以上不同的 token、平均对数概率不低于 −1、并由它自己停止）。5 句全部找回；其余 2395 句转写不变；
  对静音、噪声、环境底噪、哼声和单音的 96 次解码，没有一次多出文字。AISHELL-1 字错率：强制中文时干净
  4.11% → 3.92%、带噪 7.48% → 7.41%，自动识别语言时带噪 7.76% → 7.32%。默认开启；
  `rescue_empty=False`（C++ `AsrOptions::rescue_empty`）按原样返回空回答。
  在 AISHELL-1 上看到的其他提前停止——一句话说了几个字就断掉，官方运行时和 1.7B 包上也一样——查下来是测试
  音频的问题：这些片段里本来就只有这么多语音（Paraformer 停在同一处；其余是环境底噪或低频哼声），所以没有
  可以找回的内容。

## [0.10.0] — 2026-11-26

语音接入：官方 Qwen3-ASR 包解压即用，准确率与官方运行时持平、速度略快，录音可以直接交给 Qwen3.5。

### 新增
- **语音识别：官方 Qwen3-ASR 包解压即用。** 对带 `*_audio_*.hbm` 的官方 LLM 2.0 包（Qwen3-ASR-0.6B / 1.7B，
  S100P）调用 `bllm.load()`，返回一个 `NativeAsrSession`——`transcribe(audio, language=None, context="")` 与
  `stream_transcribe(...)`；C++ 为 `bllm::NativeAsr`（`bllm/native_asr.h`）。音频可以是文件（WAV 在进程内解码，
  其他格式经 ffmpeg）或数组（float32 / int16，任意采样率）。语言可以强制（名字或代码），也可以由模型说出；
  `context` 用来传入人名和术语。比 1024-token 缓存一次能容纳的更长的录音（约 40 s）会在安静处切开（上游
  qwen-asr 的规则），各段分别转写后拼接。在 AISHELL-1（干净 600 句 + 带噪 600 句）上，强制中文时字错率为
  4.11% / 7.48%，官方运行时为 4.63% / 7.62%；一段 11.4 s 的片段端到端耗时 1499 ms，官方为 1537 ms（1.7B：
  2658 对 2853）。转写与官方并不逐字相同：这个包的音频编码器极其敏感，任意一个输入值差 1 LSB，几乎所有输出值
  都会改变（逐行余弦低至 0.81），而两边的前端在少数几个值的最后一位上不同。
- **自动识别语言不再丢掉转写。** 让这个编译出来的模型自己说出语言时，大约每六句普通话里就有一句它回答
  "没有语音"，然后什么也不写（AISHELL-1：字错率 16%）；官方运行时给它一个空的语言名，丢掉一成（10.0% /
  15.9%），而且始终不知道是什么语言。BLLM 让模型在它认识的 30 种语言里挑一个，再按强制这门语言的方式转写：
  4.57% / 7.76%，600 句里普通话认对 574 / 588 句，静音和噪声仍然返回空。
- **与 Qwen3.5 对话**：`examples/voice_chat.py` 接收一段录音（或麦克风），用 Qwen3-ASR 转写，流式输出
  Qwen3.5 的回答；装了 bcdl 的话，再用 bcdl 的 Matcha-TTS 念出来。`examples/asr_native.cc` 是 C++ 转写命令行工具。
- **bllm-serve 上的 `/v1/audio/transcriptions`**（OpenAI 的 API：`language`、`prompt`、
  `response_format` json / text / verbose_json、`stream`），使用 `BLLM_ASR_MODEL` 指定的 ASR 包——可以与对话
  模型同时提供，也可以单独提供。

### 变更
- **log-mel 前端改为双精度 FFT，快 9 倍。** 它所替换的 400 项 float 直接求和会因相互抵消丢掉安静的频点——与官方
  Qwen3-ASR 运行时的特征相比，最多差到编码器 int16 输入的 21 LSB——而且一段 11.4 s 的片段要花 199 ms；在稀疏
  mel 频带上做混合基 FFT（400 = 4·4·5·5）只要 22 ms。Qwen2.5-Omni 的音频路径共用它。

## [0.9.0] — 2026-11-26

整个 S 系列共用一个 conda 包：包里不再捆绑 hobot 运行时，改用板子自带的。

### 变更
- **conda 包不再捆绑 hobot SDK。** `libbllm` 过去依赖 `hobot-dnn`，它会把 S100 SDK 版本的运行时（`libdnn`、
  `libhbucp`、`libhbrt4`、`libhbtl`）装进环境，排在板子自带的 `/usr/hobot/lib` 之前——在 S100P 上也是如此，
  而 `libdnn`、`libhbucp` 和 `libhbtl` 并不是板子 OS 镜像里的那些文件。这个运行时必须与板子的固件匹配：在
  S600 上它是另一个平台的构建，bcdl 的包正是因此在那里返回错位或 NaN 的输出
  （[bcdl#1](https://github.com/ruisv/bcdl/issues/1)），却没有任何报错。现在 `libbllm` 链接板子的 SDK，并把
  `/usr/hobot/lib` 留在 rpath 上；S 系列的 SDK 头文件相同，BLLM 只调用稳定的 hbDNN / hbUCP C API，所以一个包
  就能服务 S100、S100P 和 S600，OS 升级后也照样能用。升级已有环境时会把其中的 `hobot-dnn` 删掉。在 S100P 上，
  已发布的 0.8.0 二进制在其环境删掉 `hobot-dnn` 后，生成的文本和困惑度逐位相同。
- **`bllm-s600` / `libbllm-s600` 停止提供。** 它们是依赖 `hobot-dnn-s600`（S100 的 `libdnn` / `libhbucp` 加上
  HBRT 4.10.1 的 `libhbrt4`）的 `bllm` 二进制。在 S600 上请替换：
  `conda remove bllm-s600 libbllm-s600 hobot-dnn-s600`，然后像在任何板子上一样 `conda install bllm`。nash-p 的
  `.hbm` 现在必须来自板子自带运行时能加载的工具链；太新的构建会在加载时失败，报 "libhbrt4.so needs to be updated"。
- S600 服务镜像（`docker/Dockerfile.s600`）改为打入 S600 板子的运行时（S100 镜像一直如此），不再安装
  `bllm-s600`。两个镜像都固定 bllm 0.9.0（此前还停在 0.1.6）。
- 运行时留给调用方的 stride（`-1`）按 64 字节行对齐解析（S600 要求如此），不再是 32；64 同样满足 S100/S100P
  （[bcdl#2](https://github.com/ruisv/bcdl/issues/2) 是同一个修复）。我们所知的图都没有这种 stride——我们 S100P
  上的 33 个图，无论是我们的还是官方包里的，每个 stride 都是编译进去并按原样报告的——所以它们都不受影响。

### 新增
- **`bllm.runtime_info()` / `bllm::runtimeInfo()`**——进程实际加载的是哪一份 hobot 运行时：DNN 和 UCP 版本、
  SoC 名称，以及 libdnn 解析自哪个文件。正常情况下这个文件在 `/usr/hobot/lib` 下；在别处就说明有另一份 SDK
  遮住了板子自带的，而这单看结果是看不出来的。报 bug 时请附上它。

## [0.8.0] — 2026-11-13

OpenAI 服务按每个模型自身的对话模板做工具调用和 `reasoning_content`；在重新发布的 Qwen3.5 包上，短提示的
首个 token 快 4–7 倍。

### 新增
- **OpenAI 服务自己就能调用工具、返回思考内容。** 对于模型清单里写明了对话模板家族（`chat.dialect`）的模型，
  请求里的 `tools` 会按该模板自身的格式交给模型，调用以 OpenAI `tool_calls` 返回，并带
  `finish_reason: "tool_calls"`；`role: "tool"` 的结果和助手之前的调用也以同样方式拼回提示，所以
  "调用 → 结果 → 回答"整轮都能走通，流式与否都一样。思考内容拆到 `message.reasoning_content`（以及
  `reasoning_content` 增量）里，不进 `content`。四个家族：`qwen2.5`、`qwen3`、`qwen3-instruct`（Hermes 风格的
  `<tool_call>{json}`）和 `qwen3.5`（它的 XML 调用，参数类型从工具的 schema 读取）。官方 2.0 包的家族从其
  模板得到——Qwen3、Qwen3-2507、Qwen2.5 和 Qwen3-VL；我们自己的包由 `bllm-make-model-dir --dialect` 写入。
  已在官方 Qwen3-1.7B、Qwen3-4B-Instruct-2507 / Thinking-2507 和 Qwen2.5-3B 以及一个 Qwen3.5-0.8B 上做了端到端
  检查。`tool_choice` 接受 `"auto"` 和 `"none"`；强制调用暂时以 400 拒绝，向没有家族的模型（MiniCPM5、
  Phi-4-mini、1.0 包）发送 `tools` 也一样——过去它们会被一声不吭地忽略。`/metrics` 报告 `chat_dialect`、
  `supports_tools` 和 `thinking`。
- `chat_config` 包含清单的 `dialect`。
- **Qwen3.5：短提示的首个 token 不再每个 token 等一步 decode。** 提示中填不满一整个 32-token prefill 块的
  部分——一个典型问题的全部——过去要逐个走 decode 图，所以首字延迟是悬崖而不是斜坡：在 0.8B 上，31 个 token
  的提示要 1.71 s，32 个 token 的只要 0.39 s。现在一个可以接收填充的 prefill 图把这段尾巴作为一个填充过的块
  一次跑完（填充不会改动递推状态，引擎再把它们的 K/V 从注意力窗口里取出）：7–31 个 token 现在稳定在约 0.37 s，
  37–63 个 token 约 0.68 s。短于包的 `prefill_min_tail`（5–8 个 token，逐包实测）的尾巴仍然逐个走，因为那样更快；
  一整块的开销多约 2%。这种图的输入输出与以前相同，所以旧版 bllm 运行重新发布的包，与运行旧包完全一样；包通过
  `model.json` 里的 `prefill_pads: true` 声明它（`bllm-make-model-dir --prefill-pads`），`prefill_min_tail`
  （`--prefill-min-tail`）调整门限。会话报告 `prefill_pads`，并接受 `prefill_min_tail`。需要重新发布的 Qwen3.5 包；
  旧包仍然逐个走尾巴。

### 变更
- 在这些模型上，服务端按模板的方式渲染**整段**对话——与 HF 原版模板逐字节一致，六个模板都用 jinja2 测过（官方
  包内嵌的是同样的模板）。这不止关乎工具：请求里没有 system 提示时，Qwen2.5 会得到它的默认 system 提示；Qwen3
  会丢掉最后一个问题之前各轮的思考内容。没有家族的模型渲染方式与以前完全相同。

## [0.7.0] — 2026-11-12

D-Robotics 官方 LLM 2.0 包解压即用——九个 S100P 文本包和五个图像包——贪心输出与官方运行时逐 token 相同，
解码速度与之持平。S600（`nash-p`）包的 I/O 是 fp16，暂不支持，加载时会以明确的错误拒绝。

### 新增
- **native 引擎驱动官方 D-Robotics LLM 2.0 图。** 2.0 的 language `.hbm` 接收 `input_embeddings`（int16）+
  `position_ids`（int32，`[1,T]`，mrope 为 `[3,1,T]`）+ `attention_mask`，KV 窗口作为输入、只输出新的行——与以前
  的窗口契约相同，只是 token id 换成了来自包里 fp16 `*_embed_tokens.bin` 的主机端 embedding。它的输入数量与 1.0
  的 token 图相同，所以按张量名识别契约；额外的仅 prefill 输入（Qwen3-VL 的 deepstack 特征）会被找出来，纯文本时
  置零。把 `model.json` 的 `embed` 指向这张表，这个包就像任何 dense 模型一样加载。已在 S100P 上对照官方
  `chat_demo` 在 Qwen3-0.6B w8 上检查：中英文 120 个 token 的回复，贪心输出**逐 token 相同**。
- **`bllm.load(<official 2.0 package dir>)` 原样加载。** 包里没有 tokenizer 或配置文件：一切都是内嵌在 language `.hbm`
  里的一份描述 JSON。BLLM 读一次（`read_hbm_desc`），把 `model.json` + `tokenizer.json` 写进一个由符号链接组成的
  缓存目录，第二次加载什么都不用读（Qwen3-0.6B：2.66 s → 1.35 s）。停止 token 来自 `generation_config`；对话格式
  是模板的家族（ChatML 或 Phi 风格），映射到内置格式上（其他的一律拒绝，而不是去猜）；名字里带上下文长度、量化
  和板子。C++ 用户用 `bllm-make-model-dir official <out_dir> <package_dir>` 得到同样的目录。VLM 包会作为图像
  会话加载（下一条）；`vision=False`（`--text-only`）只给出它的文本塔。
- **官方 VLM 包可以就图片对话：Qwen3-VL 和 Qwen2.5-VL。** `bllm.load("Qwen3-VL-2B_w4/")` 返回一个
  `NativeVlmSession`。BLLM 喂给图的每一个输入——`pixel_values`、拼入图像的 embedding、三维位置、mask 以及
  Qwen3-VL 的三张 DeepStack 图——都**与官方运行时逐字节相同**，已在 Qwen3-VL-2B 和 4B 以及 Qwen2.5-VL-3B 上
  对照其自身的张量 dump 检查过；五个 S100P VLM 包上，贪心回复都与官方 `chat_demo` 逐 token 相同。每个包四个
  用例——SDK 自带的照片分别用中文和英文提问、同一张照片拉伸到 1600×400、一张带文字和图形的绘制 PNG：

  | 包（S100P） | 相同 | 视觉塔 | 首字延迟（BLLM） | 解码 官方 / BLLM |
  |---|---:|---:|---:|---:|
  | Qwen3-VL-2B w4 | 4/4 | 180 ms | 404–409 ms | 26.2 / 26.1–26.2 tok/s |
  | Qwen3-VL-4B w4 | 4/4 | 181 ms | 723–728 ms | 14.7 / 14.7 tok/s |
  | Qwen2.5-VL-3B w4 | 4/4 | 302 ms | 902–907 ms | 21.7 / 21.3–21.7 tok/s |
  | Qwen3-VL-8B w4 | 4/4 | 229 ms | 1008–1017 ms | 9.0 / 8.7–9.0 tok/s |
  | Qwen2.5-VL-7B w4 | 4/4 | 304 ms | 1457–1464 ms | 10.4 / 10.2–10.4 tok/s |

  三个图在两个运行时下占用的 BPU 时间相同；首字延迟多出来的是主机端工作（预处理约 8 ms、拼接约 15 ms，官方
  运行时后者约 4 ms）。做到这些用了：
  - 图像用 OpenCV `INTER_CUBIC` 拉伸到塔的边长，与官方运行时一样，不链接 OpenCV 而逐位复现（在缩小、放大、
    奇数尺寸和舍入平局上都对照 cv2 检查过；`bllm._bllm_native.resize_cubic_cv` 暴露了它），然后以能对每个像素值
    复现其 int16 输入的 float 形式归一化；
  - Qwen3-VL 的 DeepStack 图随图像行一起进入 prefill 图的额外输入；带有它们的 token 一律 prefill，从不 decode；
  - 量化的塔按 `rint(q · (s_out/s_in))` 喂给量化的解码器，这是官方公式——在两个 scale 不同的 Qwen3-VL-4B 上，
    我们原来的"先反量化再相除"大约每 600 个值有 1 个不一致，从第一个 token 起就改变了回复；
  - Qwen2.5-VL 的塔按窗口注意力的顺序返回 token（HF 最后那步逆置换不在导出的图里），所以在主机端放回光栅顺序；
    它的位置在 prefill 时是 `[1,3,T]`、decode 时是单行，所以带三维位置的 token 一律 prefill。
  几何参数来自包自身的 `preprocessor_config`；`model.json` 把它记在 `vision` 下（`resize: "cv_cubic"`、`window`）。
- `NativeVlmSession.last_ntok`，与 `NativeSession` 一样；服务端的图像路径现在也用它统计 `usage` 和
  `finish_reason`。
- **思考开关跟随每个模型的模板。** `model.json` 的 `chat` 接受 `think_on` / `think_off`——思考开或关时助手标记
  之后跟什么——以及 `thinking_default`。各模板恰恰在这里不同：Qwen3 不提示也会思考，关闭时补一个空的思考块；
  MiniCPM5 开启时打开 `<think>\n`；只会思考的 checkpoint 无论开关都会打开它；没有思考模式的模型什么都不加。
  官方包这三项都从模板读出，思考默认关闭，与官方运行时一致；没有这些字段的清单行为与以前相同。
  `thinking_switchable` 和 `chat_config` 说明一个会话会渲染成什么样。
- 官方包也会采用模板自身的默认 system 提示（多数没有；Qwen2.5 有固定的一段）、模板总会输出的 BOS（MiniCPM5），
  以及角色标记由字符串拼接构造的 Phi 风格家族（Phi-4-mini）。
- `bllm_selfengine --embed <table>`，用于以 embedding 为输入的图；`bllm_chat_native --no-think`（模板的
  `enable_thinking=False`）。

### 变更
- **每个 BPU 任务句柄晚一步释放，** 在下一个任务占用 BPU 的时间里释放，而不是在关键路径上。`hbUCPReleaseTask`
  实测在 LLM 2.0 图上每个 decode 步 1.25 ms，在 1.0 图上 0.58 ms。S100P 上的解码速度：

  | 模型 | 之前 | 之后 |
  |---|---:|---:|
  | Qwen3-0.6B w8（官方 2.0），对话路径 | 37.8 tok/s | **40.3 tok/s**（官方运行时：40.45） |
  | Qwen2.5-1.5B（1.0 dense） | 23.74 tok/s | 24.06 tok/s |
  | Qwen3.5-2B（混合），对比 0.5.2 的包 | 13.97 tok/s | 14.25 tok/s |

  输出不变——释放从来不在数据路径上。

- system 提示为空的 ChatML 或 Phi 会话不再发送 system 块，与 HF 模板在没有 system 消息时的做法一致（过去会发送
  一个空的）。
- **prefill 原地读取 KV 窗口。** 它的缓存输入绑定到 decode 图读取的同一个环形缓冲上，而不是每个块都清零再重新
  填充——在 36 层的模型上每块要 226 MB 的 memset 和缓存维护——图自己的缓存缓冲也不再分配。Qwen3-VL-4B 上每个
  prefill 块少约 10 ms，ION 也省下这么多。
- 装了 Pillow 时，`NativeVlmSession` 用它解码图像**路径**（libjpeg，与 HF 处理器和官方运行时的解码方式一样），
  并应用 EXIF 方向。stb 仍是回退方案，也是 C++ 示例的解码器，它在一张 JPEG 约 5% 的值上差 1–3 级。数组不受影响。
- 在 LLM 2.0 图上，被遮蔽的注意力值是经 mask 的 scale 量化后的 `mask_pad_value`，与官方运行时写入的一样
  （按 Qwen3-VL 的 scale 1.00026 为 -32760）；数值上同样是 -inf，输入逐字节相同。

### 修复
- **LLM 2.0 图的 embedding 输入现在与官方运行时逐字节相同。** 过去按 `x × (1/scale)` 量化；官方运行时用的是除法，
  只要 `x / scale` 落在比 .5 略小一点的位置，两者就不一致——而 fp16 表里的值经常如此：Qwen3-4B 的
  `0.0294189453125 / 4.16256762e-06 = 7067.4997`，乘以 float 倒数则是 `7067.5000`。一个 19-token 的提示里有 54 个
  这样的元素（对照官方运行时自身的张量 dump 找到），让贪心解码从第三个 token 起走上了另一条回复；prefill 的填充
  位置现在也与官方运行时一样写成 0。（Omni 路径不变。）
- prefill logits 的行 pitch 从张量解析后的 stride 读取，不再假定为按 32 字节填充的词表大小。
- 官方模板以字面文本而非字符串写出其默认 system 轮次时（Qwen2.5-VL："You are a helpful assistant."），过去会被漏掉。
- `bllm_graph_probe` 曾在最后一个任务句柄之前释放模型。
- **`response_format: {"type": "json_object"}` 现在表示一个对象**，与 OpenAI 的定义一致，通过新的
  `bllm.json_object_grammar()` 实现（`json_grammar()` 仍表示任意 JSON 值）。允许任意值时，Qwen3-1.7B 会用一个很长的
  顶层字符串作答。
- **JSON 语法限定空白**——一个空格，或一个换行加最多 20 个缩进字符（llama.cpp 的上限）——且不再允许结尾空白。
  不加限定时，模型可以永远满足语法：Qwen2.5-1.5B 输出了 `1.1`，然后一直换行到 token 上限。
- **`finish_reason` 和 `usage.completion_tokens` 按实际生成的 token 计数**（`last_ntok`）。重新编码回复会把分开生成
  的 token 合并，所以在 `max_tokens` 处截断的回复会少算，并被报告为 `stop`。
- **OpenAI 服务不再向没有思考模式的模型注入空的 `<think></think>`**（Qwen2.5、Qwen3-Instruct-2507、Phi）：助手开头
  现在使用模型自身的思考关闭尾部。`chat_template_kwargs.enable_thinking` 会被采纳，并且优先于顶层字段。

## [0.6.0] — 2026-09-02

### 新增
- **S600 上多核 BPU 端到端可用，而且按图区分。** 自 0.5.0 起引擎就按模型自己的编译核数绑定
  BPU 核，但此前没有任何一个多核产物真正跑起来过：每一个都在提交时被拒。两道墙现在都已查清。

  出货的 S600 包混用核数——视觉塔 **4** 核，prefix 与 expert 单核——因为引擎独立读取每张图的
  核数，而真正的收益就在这里：

  | 图 | 单核 | 多核 | |
  |---|---:|---:|---|
  | 视觉塔，每相机 | 59.0 ms | **29.8 ms**（4 核） | **1.98×** |
  | prefix | 42.6 ms | 40.0 ms（2 核） | 1.07× |
  | expert，每步 | 10.7 ms | — | 编译不过 |

  **每个 50 步动作块：220 → 159 ms**，输出与单核包**逐位相同**（cosine 1.000000000，
  max|Δ| 0.00000）。多核改的是调度，不是算术。

- **多核提交失败时现在会说明原因。** hbDNN 的原始报错说拿不到临时内存空间，读起来像是驱动或
  加载问题；实际上是**编译期预算**。`submitAndWait` 会把这一点讲清楚。

### 修复
- **转换脚本不再照抄 `leap_llm` 的 `max_l2m_size`。** 正是这 24 MB 的预算让所有多核 `.hbm`
  都加载不了：*"Get BPU temporary memspace failed"*，在核 0,1 / 2,3 / 1,2 上一模一样——所以
  它看起来像硬件问题而不是构建问题。把预算交给编译器后，同一张图就能加载并运行。默认值现为
  0（不设置）。

### 已知问题
- **`core_num=4` 仍然编译失败**：完整深度的 prefix 与任意深度的 expert 都报
  `TilingConfig.cpp:1489`。二分排除了显而易见的解释：不是 L2M 预算（去掉它也失败），不是
  RoPE（单层 prefix 带 RoPE 也能编），不单是深度（12 层的视觉塔能编），也不是 cross-attention
  层（只有一层 self-attention 的 expert 也失败）。既然唯一能从多核获益的那张图能编译，这个
  问题先放着，不做解释。

## [0.5.2] — 2026-09-02

**仅文档**——自 0.5.1 以来代码没有改动。之所以记为一个版本，是因为它所带的数字正是 README
现在宣称的数字，打上 tag 才能把这一宣称对应到代码树的某个状态。

### 变更
- **SmolVLA 现已在全部四个 LIBERO suite、两块板上，用同一套 harness 与参考对照验证。** 十二次
  100 集评测：

  | | S100P | S600 | 参考（RTX 4090） |
  |---|---:|---:|---:|
  | libero_spatial | 68 | 71 | 73 |
  | libero_object | 76 | 75 | 72 |
  | libero_goal | 79 | 81 | 79 |
  | libero_10 | 51 | 54 | 48 |
  | **合计 / 400** | **274** | **281** | **272** |

  400 集里两块板分别比参考 +2 和 +9，且**方向随 suite 翻转**——在 `libero_spatial` 上落后，
  在另外三个上领先。系统性的移植损失会让四个 suite 同向变动。逐任务分布也是共享的：
  `libero_goal` 的 task 6 三边都是 5，`libero_10` 的 task 0 三边都是 1。

  `libero_10` 正是参考要在每个 suite 上都跑、而不只跑一个的原因。板子在那里得 51 分，而其他
  suite 是 68–81，单独看会被读成长程任务上的崩溃。参考得分是 **48**。

## [0.5.1] — 2026-09-01

### 变更
- **SmolVLA 包现在默认 4 步去噪，而不是 10 步。** expert 图每步运行一次，所以这是唯一不需要
  重新构建的延迟杠杆：每个 50 步动作块，S100P 上 **741 → 562 ms**，S600 上
  **285 → 220 ms**。

  这是实测打分而不是假设，而且不止一个 suite，因为单次 100 集对比撑不起这个决定——它的区间
  约为 ±9 分：

  | suite / 板 | 10 步 | 4 步 |
  |---|---:|---:|
  | libero_spatial · S100P | 70 | 68 |
  | libero_spatial · S600 | 71 | **71** |
  | libero_object · S100P | 76 | **76** |
  | **合计，300 集** | **217** | **215** |

  三组配对中两组完全相同，合计差两集。`bllm-make-model-dir smolvla --steps 10` 可恢复参考
  调度。

### 新增
- **`bllm-make-model-dir` 拒绝行数与 `--steps` 不符的 `cond_table`。** 这张表编码的是*调度*，
  而不是每步的常量：第 `s` 行存的是 `t = 1 + s·(-1/steps)` 处的时间项，所以 10 步依次经过
  1.0、0.9、0.8 …，而 4 步经过 1.0、0.75、0.5、0.25。因此截断一张更长的表得到的是一个
  *不同*的调度，而不是更短的调度，而且这种不匹配在打包时毫无声息。（运行时在加载时本来就会
  拒绝它；这里提前一步拦下，此时怎么修一目了然。）

## [0.5.0] — 2026-09-01

### 新增
- **SmolVLA 成为一等架构（`arch: "smolvla"`），S100P 与 S600 都支持。** LeRobot 的视觉-语言-
  动作策略被拆成三张图——SmolVLM 视觉塔、作为 prefix 一遍的 SmolLM2 前 16 层、以及交错的
  action expert——打包后运行时不需要 lerobot 或 checkpoint。
  `bllm.load_policy(dir).act(scene, wrist, prompt, state=...)` 返回一个 50 × 7 的动作块，
  S100P 上 **741 ms**，S600 上 **285 ms**。

  expert 是结构上全新的部分：偶数层对 `[prefix KV; their own KV]` 做 self-attention，动作块在
  绝对 RoPE 位置上做因果掩码；奇数层以从 0 开始的位置 cross-attend prefix，所用的 K/V 已由
  prefix 图经过 expert 自己的 k/v 投影好。这两处若按"合理"的方式写，cosine 分别只有 0.32
  和 0.58。timestep 从不进 BPU——`action_time_mlp_in` 对其拼接输入是线性的，所以每步那一项
  是一张常量表。

  与 `pi05_libero` 不同，SmolVLA **会读本体感**：state 被投影成一个 prefix token，所以不传它
  就等于换了一个观测（动作空间里差 2.0，整整一格夹爪）。

  在 `libero_spatial` 上，用同一套 harness、只换计算动作块的那一侧打分：**70/100**（S100P）、
  **71/100**（S600），同一 checkpoint 在 RTX 4090 上为 **73/100**。n = 100 时无法区分，且逐
  任务分布吻合——板子失败的任务正是参考失败的任务。

- `bllm_smolvla_policy`、`bllm_smolvla_probe` 与 `bllm_smolvla_server`，以及
  `bllm-make-model-dir` 的 `smolvla` 模式。
- **`BLLM_BPU_CORES`** 指定多核图跑在哪些 BPU 核上。

### 修复
- **`bllm.load()` 把策略包分流到了对话 VLM 分支。** 那个分支检测有没有视觉塔，而策略包恰好有，
  于是用户得到的是一个来自对话会话深处的报错，而不是"这是策略，请用 `load_policy`"。现在
  先判断策略。
- **`load_policy()` 拒绝一切不是 `pi05` 的包**——而这正是每张 model card 宣传的入口——所以
  新的策略架构根本无法通过文档中的 API 加载。
- **多核 `.hbm` 无法提交。** 为多个 BPU 核编译的图在 `HB_UCP_CORE_ANY` 下会被*拒绝*（"the
  backend must be specified to specific cores"），而不是回退，所以每个调用方都得知道它的核数。
  `Graph::init` 现在从模型读取（`hbDNNGetCompileBpuCoreNum`）并据此绑定；单核与多核产物从
  外部看行为完全一致。

## [0.4.2] — 2026-08-20

### 修复
- **attention 掩码只写了一个 head，所以生成结果取决于板子之前跑过什么。** decode 掩码张量是
  `[n_heads, cache_len]`，图会读每个 head 的那一行；代码只写了第一行，并假定图会广播它。在出货
  的 Qwen3.5-0.8B 上，这让 8 行中的 7 行保存着 `hbUCPMallocCached` 返回时的任意内容——而释放
  的 BPU 内存回来时是脏的，且在同一物理地址上。于是干净的内存读出来是全零掩码（那些 head 上
  完全没有因果掩码）；被无关负载弄脏的内存读出来是垃圾，模型退化成重复。同一个二进制、同一个
  贪心提示词，可能答对，也可能胡言乱语 400 字节，取决于历史——这正是它看起来像偶发故障而不像
  bug 的原因。影响所有 `arch: "hybrid"` 模型——整个 Qwen3.5 家族。板上测试套件因此从
  4 失败 / 151 通过变为 1 失败 / 157 通过，且在三种模式的 2 GB 内存投毒下输出逐位相同。
  分块 prefill 的掩码有同样的缺陷，已用同样方式修复，但这里无法验证：现有模型都不带可选的
  `_prefill` 图，所以那条路径从未执行。

### 变更
- **BPU 分配会清零。** 这并不能让不完整的写入变正确，但能让它变得确定——这是一个能二分定位的
  bug 和一个你一看它就不复现的 bug 之间的区别。代价只是对加载时一次性分配的缓冲区过一遍；大的
  缓存本来就由其所有者清零。decode 吞吐不变（前后实测都是 17.3 tok/s）。
- 掩码构造移到了 `bllm/attn_mask.h`，成为两个具名函数，这样它才有了测试接缝——此前不暴露引擎
  内部就无法从测试触及这段逻辑。`tests/test_attn_mask.cc` 覆盖它，且不需要板子。

## [0.4.1] — 2026-08-19

### 修复
- **rank 为 3 的视觉塔读越了输出缓冲区几百 MB。** `VisionTower` 从最后两个维度推出
  `n_token`/`hidden`——所以 `[1, n_token, hidden]` 的图是明确支持的——但行 pitch 却取自第 0
  维的 stride，而对这个 rank 来说那是整个张量的大小。于是 dense 快速路径没命中，每一行都隔着
  一整个张量去读。现在改用感知 rank 的 `outRowPitch`，与运行时里其他所有读取方一致。
- **输出张量的 stride 在加载时解析，与输入 stride 一样。** 运行时用 `-1` 表示"dense，你自己
  算"；之前只展开了输入，所以任何信任输出 stride 的读取方都会得到负的 pitch，读到缓冲区*之前*。
- **模型加载失败会泄漏整个 `.hbm` 映射。** packed handle 在显式析构函数里释放，而构造函数抛
  异常时析构函数根本不会运行——这里的构造函数在加载之后还要继续校验（图形状、张量类型、可能
  `bad_alloc` 的 KV 窗口）。每一次失败的加载都会泄漏数 GB，Python 绑定和 `bllm-serve` 都一样。
  `Pi05Stack` 则从未释放过它的 handle。现在所有持有者都持有 RAII 的 `PackedHbm`，它也保证
  释放发生在其背后的缓冲区之后。
- **`bllm_vision_probe` 会静默忽略未知的 `--patch-order`**，转而使用 merged 布局。两种布局都会
  产出形状正确、只在数值上不同的张量，所以一个拼写错误会在这个本该区分两者的工具里表现为量化
  故障。现在未知值会被拒绝。
- 有三个头文件使用了 `std::memset`/`std::memcpy`，却依赖 `<cstring>` 被间接引入。

### 变更
- **hbDNN/hbUCP 图层移到了 `bllm/native_graph.h` + `src/native_graph.cc`。** 它是对厂商运行时
  的一层薄而稳定的封装，没有值得内联的东西，却被重新编译进每一个翻译单元。它之上的模型类仍然是
  header-only，所以 CLI、Python 模块和 `bllm-serve` 继续共享同一份实现，无需维护 ABI。构建时间
  不变；收益是每个行为只有一处定义、include 图更窄——视觉塔与音频塔、π0.5 和 hybrid 引擎不再
  为了拿到一个 `Graph` 而把整个自回归引擎和采样器都拉进来。`native_engine.h` 从 963 行降到
  617 行。

### 新增
- **Being-H0.5 被重放为四次普通的共享 KV 推理。** 这个 checkpoint 看起来像两个缠在一起的
  transformer，但推理时它的理解流与生成流在普通因果掩码下占据互不相交、各自连续的位置区间，
  所以一个动作块可以分解为对两张编译图的四次单流调用，每层共用一个 KV 窗口——在 fp32 下与
  packed 参考对照验证完全一致。`bllm::BeingHStack` 负责窗口、掩码以及缓存上传的记忆化；
  `bllm::BeingHHost` 是 flow-matching 动作头，保持 fp32 在 CPU 上运行，因为它是连续的速度
  回归，没有 softmax 来掩盖误差。在 S100P 上，两路相机下一个 15 步动作块耗时 798 ms
  （MPG 关闭），最终动作块与参考的 cosine 为 0.994892。

## [0.4.0] — 2026-08-14

### 新增
- **π0.5（openpi）成为一等模型包。** 输入两帧相机图和一句英文指令，输出一个末端 delta 动作块，
  每个动作块约 1 s，三张图（4.6 GB）全部常驻板上：

      policy = bllm.load_policy("pi05-libero-224-s100p")
      actions = policy.act(scene_rgb, wrist_rgb, "pick up the black bowl ...")

  在 `libero_spatial` 上，模型包通过 openpi 自己的评测 harness 得分 **99/100**，与 x86 参考的
  99/100 持平；`libero_goal` 97/100，`libero_object` 95/100。模型包自带 tokenizer、embedding
  表、动作分位数和 flow-matching 条件项，所以运行时不需要 openpi 或它 14 GB 的 checkpoint。
  新增：由 `bllm-make-model-dir pi05` 生成的 `arch: "pi05"` 清单、`bllm.load_policy()`、
  `bllm_pi05_policy` CLI、C++ 的 `bllm::Pi05Policy`，以及 `docs/PI05.md`。权重发布在
  [ruisv/bllm-pi05-libero-224-s100p](https://huggingface.co/ruisv/bllm-pi05-libero-224-s100p)。

### 修复
- **流式回复可能在句子中途中断**，报 `'utf-8' codec can't decode byte 0x8a in position 0:
  invalid start byte`。字节级 BPE 会把一个多字节字符分散到多个 token 上，解码一个不完整的前缀
  会得到 U+FFFD——三个字节，而完整的字符常常是四个字节。这个占位符本身是合法的 UTF-8，于是被
  输出了，让流的偏移停在一个四字节字符的内部；下一个增量便从续字节开始，单独解码失败。任何
  emoji 与文字混排的回复都可能撞上，所以它看起来是偶发的。现在输出偏移会回退到不完整序列和末尾
  U+FFFD 之前，使每个增量都能独立解码——代价是正在拼装的字符多一个 token 的延迟，且能自我纠正
  （真正的 U+FFFD 仍会在流结束的 flush 时到达）。影响 `NativeSession` 和 `NativeVlmSession`。

## [0.3.0] — 2026-07-29

### 新增
- **VLM 思考开关。** `NativeVlmSession.set_thinking(False)` / `.thinking`，与纯文本的
  `NativeSession` 对应——此前 VLM 会话没有办法强制直接回答，所以一段推理很长的回复可能在给出
  答案之前就耗尽调用方的 token 预算。
- **`bllm.load(path, vision=False)`。** 强制让 VLM/omni 包走 `NativeSession` 而不是
  `NativeVlmSession`，这样视觉/音频塔的权重根本不会碰 BPU/ION——给只要文本、想把那部分内存
  预算拿回来的调用方。`chat(images=[...])` 在没传图时本来就会跳过塔的*计算*；这个开关跳过的是
  *加载*，而不只是某一次调用。
- **C++ API 参考，中英两版。** `docs/CPP_API.zh.md` 与 `docs/CPP_API.en.md` 记录了此前只以头文件
  形式存在的接口：`NativeLlm`（生成、采样、语法、停止串、思考、状态与可复用前缀、服务化原语）、
  `NativeVlm`（媒体类型、实时流、KV 预算、仅 Omni 的视频路径）、`NativeEmbedder`（此前两种语言
  都完全没有文档）、`ModelConfig`、`Tokenizer`、采样参数，以及会话之下的引擎层。包含一张
  C++ ⇄ Python 对照表；Python 参考和两份 README 现在都链接到它。
- **图文现在可以通过 HTTP 服务。** 当 `BLLM_MODEL` 指向 VLM 包时，`/v1/chat/completions` 接受
  OpenAI 的 `image_url` 内容块（内联 base64 或 `http(s)://` 链接）；`/metrics` 暴露 `is_vlm`，
  客户端可据此判断。关于同一张图的多轮对话不会重新跑视觉塔：引擎会记录哪些
  `(user, assistant)` 轮次对已经在会话里，只通过新增的 `NativeVlm::append_turn` /
  `NativeVlmSession.append_turn` 重放新的轮次（只 prefill，不 decode——调用方已经有了回复，
  重新生成既浪费 BPU 时间，而且由于采样是随机的，甚至未必能复现同样的文本）。被编辑过的历史，
  或两个无关对话在这个单会话引擎上交错，会退回完整的 reset+replay，而不冒拼接错误上下文的
  风险。另外新增 `NativeVlm::encode` / `NativeVlmSession.encode`，用于用量统计的 token 计数。

### 变更
- **hybrid decode/prefill 图中的因果掩码去重。** 掩码与 head 无关（因果窗口不随 attention head
  变化），所以固定的掩码缓冲区现在只带一行，由图在内部广播，而不是由主机每一步/每一块写
  `nHead_` 份相同的拷贝。

### 修复
- **S600（`nash-p`）上图像 chat() 首次请求即崩溃。** `NativeVlm` 从未像 `NativeLlm`（纯文本）
  那样把文本解码器绑定到明确的 BPU 核上，所以每个多核 `.hbm` 都会以 `hbUCPSubmitTask`："the
  backend must be specified to specific cores" 拒绝默认的 `HB_UCP_CORE_ANY`。已修复——视觉/音频
  塔保留默认调度（它们不是多核编译的，强行套用解码器的核掩码也没有帮助）。**仍存在的已知问题，
  尚未查明**：崩溃修复后，S600 图像 `chat()` 间歇性出错——在 temp=0、每次都是新进程的条件下，
  大约 40-50% 的调用返回空输出而不是真正的回复。单独的视觉塔和纯文本对话都是 100% 确定的；只有
  组合起来的流水线不稳定。暂时不要把 S600 VLM `chat()` 当作可靠——见
  `include/bllm/native_vlm.h` 中构造函数的注释。

## [0.2.0] — 2026-07-24

**Qwen3.5 图文（VLM）**版本。自建视觉塔加分块 prefill 图，让混合 Qwen3.5 系列在板上能看图；
decode **在同一份 int8 权重上快 1.4–1.7×**；结构化输出与采样新增语法约束、`logit_bias`、
`logprobs`，并修了一批正确性问题。

### 新增

- **Qwen3.5 图文（VLM）—— 本版的重头戏。** Qwen3.5 原生就是图文到文本的模型；BLLM 现在
  编译它的视觉塔（patch-16 / LayerNorm，自建——不是 Omni 的 patch-14 塔），并用交错的 3-D
  mrope 把视觉行拼进混合解码器的嵌入流。`bllm.load(<vlm_dir>)` 路由到 `NativeVlmSession`；
  `chat(text, images=[...])` 接受文件路径或原始 `HxWx3` uint8 数组。视觉塔严格 100% 跑在 BPU 上。
  S100P / 0.8B 上，带 prefill 图时图片 TTFT 为 **1.23 s（320px，100 tok）/ 2.30 s（448px，
  196 tok）**；视觉塔输出与 HF fp32 的余弦为 0.999 / 0.997。混合 VLM 无法取默认值的四项视觉
  设置（patch、mean/std、mrope section、interleaved）由 `bllm-make-model-dir` 强制写入——设错
  一项，只有在喂进图片之后才会得到形状正确的乱码。这一系列拒绝视频（Qwen3.5 用时间戳 token
  分隔帧，而不是 Omni 的 TMRoPE）。
- **分块 prefill 图——提示词 / 图片的摄入摊薄约 76×。** 混合引擎此前逐 token 喂提示词（受
  权重带宽限制，约 69 ms/token——一篇 2048 token 的文档要约 5 分钟）。现在它驱动一张可选的
  seq_len=N prefill 图，放在单独的 `.hbm` 里（`--hbm-prefill`），把摄入变成每 N 个 token 一块
  只读一遍权重。GDN 部分采用并行分块 delta-rule（WY/UT 变换），已与顺序递推逐位一致地验证过。
  没有 prefill 图的模型平稳地沿用原来的逐 token 路径。prefill↔decode 一致性是随发布附带的门禁
  （`scripts/check_prefill_parity.py`），并断言其不是空检查（提示词 > 块大小）。
- **GBNF 语法约束解码——结构化输出是保证，不是请求。** Python 会话上用
  `set_grammar("root ::= ...")`；HTTP 上用 `response_format: {"type": "json_object"}` /
  `{"type": "json_schema", ...}`，或原始的 `grammar` 字段（llama.cpp 的写法）。每一步只有能让
  语法仍可满足的 token 才能被采样，所以声称是 JSON 的回复一定能解析——`json.loads()` 不可能
  失败。`bllm.json_grammar(schema)` 把 JSON Schema 转成语法（`type`、`enum`、`const`、
  `properties`/`required`、`items`、`anyOf`/`oneOf`、指向 `$defs` 的 `$ref`）；可选属性必须按
  schema 中的顺序出现，`additionalProperties` 按 false 处理。S100P / Qwen3.5-0.8B 上的代价：
  每个会话一次性约 315 ms 构建 token→bytes 表，之后 14.3 → 13.4 tok/s（约 6%）。无法解析的
  语法返回 400；不支持语法的旧运行时会被拒绝，而不是不加约束地回答——把一项保证悄悄降级成
  "提示词里请它照做"，正是这个特性要消除的失败模式。

- **`logit_bias`** —— 强推或封禁指定 token，采用 OpenAI 的 `[-100, 100]` 刻度。Python 里
  `set_sampling(logit_bias={id: bias})`，HTTP 上 `logit_bias: {"<id>": bias}`（越界值返回 400，
  而不是悄悄截断）。微妙之处在采样器：它把候选集限制在 top 512 个 logit，而这一截断之所以安全，
  只因为惩罚项只会*降低*某个 token、从不抬高。bias 恰好打破这个假设，所以带 bias 的 id 会在
  截断之后并入候选集——否则一个埋在尾部的 token 上的 `+100` 会在它有机会胜出之前就被悄悄丢掉。
- **`logprobs` / `top_logprobs`** —— 逐 token 对数概率，采用 OpenAI 的
  `choices[].logprobs.content[]` 结构，Python API 另有 `NativeSession.last_logprobs()`。取值是
  对**原始** logits 做的精确全词表 log-softmax（双精度累加，不截断候选）：它描述的是模型本身的
  判断，所以温度、惩罚项和 `logit_bias` 都不会改变它，不同请求之间也可比——这也意味着生成的
  token 不一定是 `top_logprobs[0]`。需要显式开启，因为每个 token 多两遍全词表计算（S100P /
  Qwen3.5-0.8B 实测：5 个候选时 14.3 → 13.6 tok/s，3-5%）。流式时，整个数组在最后一块一次性
  送达：delta 是解码出的一段文本而不是 token，不存在诚实的逐块对齐——客户端把每块的
  `logprobs.content` 拼起来，得到的仍然正好是正确的数组。报告也会裁到返回的文本为止，所以
  停止串截断回复时，不会留下描述客户端从未收到的 token 的 logprobs。

### 变更

- **同一份权重 decode 快 1.4–1.7×——attention 改用分组 matmul，不再用 `expand_win`。** 旧写法把
  KV 窗口从 2 个头扩到 8 个，每层每个 token 物化四个 `[cache_len, n_heads, head_dim]` 张量
  （ctx2k 时 403 MB/token，ctx4k 时比模型自身的权重还多）。现在改为对 query 分组——
  `q[NKV, REP, HDa]` 对 `Kᵀ`，一次批量 matmul，K/V 恰好只读一遍（每 1024 个槽位 201 MB → 25 MB）。
  纯粹是减少访存，int8 权重不变；重新下载最新的 `.hbm` 即可获得。S100P 上实测（decode 图延迟，
  tok/s 上限），同一份权重：

  | 模型 | cache_len | 之前（expand_win） | 之后（matmul） | 加速 |
  |---|---|---|---|---|
  | Qwen3.5-0.8B | ctx2k | 15.7 tok/s | **22.5 tok/s** | 1.43× |
  | Qwen3.5-0.8B | ctx4k | 11.3 tok/s | **19.5 tok/s** | 1.73× |
  | Qwen3.5-2B | ctx4k | 9.1 tok/s | **13.7 tok/s** | 1.51× |

  加速随 `cache_len` 增大，因为旧写法的开销随 KV 窗口增长，而 matmul 写法不会——上下文越长，
  收益越大。
- **采样器 top-k 改为顺序扫描 + 大小为 k 的最小堆，不再用 `nth_element`。** 在 248k token 词表
  上构建候选集原本是个随机访存陷阱（每次比较都是一次 cache miss）：4.25 ms/token，占一次采样
  步的 9%。带堆的单次有序遍历每个 logit 只碰一次 → 1.00 ms，采样 decode 快 7%。贪心不变。
- **prefill 只输出最后一行 logits。** 对 N 行的 prefill，之前的图物化 `[N, 248320]` f32
  （N=128 时占 127 MB ION），并把 254 MB 的 lm_head 跑 N 遍，而只有最后一行用来产生第一个采样
  token。

### 修复

- **把混合 `.hbm` 喂给 dense 引擎时，现在会告诉你该用什么**，而不是一句令人费解的
  `Model not exists: prefill`。`NativeEngine` / `bllm_selfengine` 驱动的是 dense 的
  prefill/decode 格式；混合 Qwen3.5（Gated-DeltaNet/SSM）的 `.hbm`（图名为 `qwen35`，需要宿主
  嵌入表）现在会预先识别出来，报错里指明混合路径（`bllm_qwen35 --hbm .. --embed ..`，或
  `bllm.load(<dir>)`），并干净地退出而不是 abort。
- **量化的视觉塔输出曾被当成 float 读取**——视觉塔可能输出带 scale 的 int16，当 float 读就是
  形状正确的噪声；VLM 路径现在按图声明的 dtype 反量化。
- **服务端曾悄悄丢掉图片部分。** 纯文本模型上的 OpenAI `image_url` 内容部分被忽略，模型只凭
  文本作答；现在明确报错，而不是自信地幻觉。
- **左递归语法曾让整个服务崩溃。** `{"grammar": "root ::= root\n"}`——或任何不消耗输入就能
  到达自身的规则，包括间接地、经由某个分支、或越过可空前缀——会在匹配器里无限展开，以 SIGSEGV
  把进程连同模型一起拖垮。`grammar` 是请求字段，任何客户端都能发；同样的形状也能经由
  `response_format` 里自引用的 `$ref` 触发。现在语法在构造时检查左递归（可空性不动点 + 在
  "可以以…开头"图上找环），以 400 拒绝，匹配器里另有深度上限兜底。右递归——`*`、`+`、`?`
  展开后的形式——不受影响。
- **`logprobs` 对被拆开的字符报告了错误的 `bytes`。** 负载原本由 `decode([id]).encode()` 得出，
  但只含多字节字符一部分的 token 会解码成 U+FFFD：Qwen 词表的 token 94 只贡献单字节 `0xA1`，
  而响应声称是 `[239, 191, 189]`。客户端按逐 token 报告重组文本时，分词器拆开的中日韩字符全都
  被破坏——而这恰恰是 OpenAI 的 `bytes` 字段要防止的。现在取自 token 的真实字节
  （`NativeSession.token_bytes()`）；请求要了 `logprobs` 却什么都没生成时，返回空数组而不是缺失。
- **`bllm.Grammar` 可以被拷贝，但不应如此。** 它的位置是指向自身规则存储的指针，所以拷贝仍指向
  源对象，源对象销毁后就成了悬空指针。拷贝构造函数现已删除（移动不受影响，会话正是靠移动取得
  所有权）。

- **`temperature` 不起作用。** `seed` 默认固定为 `1234`，而采样器每次生成都重新构造——于是每个
  请求都重放同一条随机流。三个相同的 `temperature=0.8` 请求返回的结果逐字节相同；这个旋钮只是
  摆设。默认值现在是 `SEED_RANDOM`（`bllm.SEED_RANDOM`，与 llama.cpp 的 `LLAMA_DEFAULT_SEED`
  是同一个哨兵值），每次生成取一个新种子。显式 `seed` 仍能精确复现，这正是 OpenAI 的 `seed`
  所承诺的——所以可复现路径不变，只是*未指定*的路径不再被钉死。
- **HTTP 上的 `presence_penalty` 和 `frequency_penalty` 被忽略。** 服务端只读非标准的
  `repetition_penalty`，于是 OpenAI 自己的惩罚字段被直接丢掉，而引擎的
  `penalty_present`/`penalty_freq` 闲置不用。两者现在不经缩放直接透传（OpenAI 与引擎都用
  `[-2, 2]`，0 = 关闭）。`min_p` 也一并开放。
- **显式的 `0` 与省略字段无法区分。** `body.get(k, d) or d` 把假值折回默认值，于是
  `temperature: 0`——客户端特意发来强制贪心解码——被读成"未设置"，`top_p: 0` 悄悄变成 `1.0`。
  现在只有 `null`/缺失才算未设置。
- **报告的 TTFT 不含 prefill。** 计时从提示词喂完之后才开始，但 prefill 已经产出了第一个 token
  据以抽取的 logits，所以每个请求测出来都约 0.3 ms，不论是否命中前缀缓存——这个本该体现缓存
  效果的指标恰恰对它视而不见。现在从请求开始被服务时计时：一轮 428 token、复用 256 个缓存 token
  的请求报告 138 ms，同一提示词不走缓存为 274 ms，与文档所述的 2x 相符。

## [0.1.6] — 2026-07-20

### 新增

- **`BLLM_CACHE_MAX_MB` 默认为 `auto`。** 固定默认值无法同时适合两种引擎——两者的 snapshot
  大小相差一个数量级，且随上下文长度变化——而合适的值还取决于加载模型后板子剩下多少内存，
  这只有运行中的进程看得到。服务端现在按 `MemAvailable` 确定预算（取 40%，下限 256 MB、上限
  4 GB），并在日志里打出选定的数值。显式设置的值仍然优先；`0` 仍然关闭缓存。在 8 GB 的板子上
  超额分配不会抛异常——而是直接把板子拖垮——所以比例刻意取得保守。

### 变更

- **混合 snapshot 约小 5x。** 混合模型的缓存集是混合的：GDN 层存的是概括全部历史的固定大小
  递推状态，而 attention 层存的是 K/V 窗口，其首维*就是*窗口本身。保存整个窗口使每个 snapshot
  不论对话多短都是固定大小——Qwen3.5-0.8B（ctx2048）72 MB，Qwen3.5-2B（ctx4096）122 MB——
  这让服务端前缀缓存的容量受上下文窗口而不是对话长度约束。现在只保存已占用的 K/V（它在窗口里
  右对齐，attention mask 可以看出），所以 blob 随对话增长：**17 个 token 时 22 MB，每 token
  +24.6 KB**。一个 136 token 的 snapshot 从 122 MB 降到 25 MB——小 80%——于是 2 GB 的前缀缓存
  能容纳约 66 段对话，而不是 16 段。GDN 状态天然固定大小，仍然整体保存；dense 模型不受影响。

  状态格式变了，所以 magic 从 `BLKH` 升为 `BLKI`，旧的混合 `save_state()` 文件会以明确的错误
  被拒绝，而不是被误读。内存中的 snapshot 不跨版本，所以运行中的服务不受影响。

## [0.1.5] — 2026-07-20

### 新增

- **感知会话的服务层。** `docker/server.py` 现在是 `docker/serving/` 之上一层薄薄的 OpenAI 协议
  门面：一个在单一工作线程上持有模型的引擎，加上一个以所覆盖 token id 为键的引擎 snapshot
  前缀缓存。OpenAI 协议是无状态的——客户端每轮都重发整段历史——所以服务端过去要把它全部重新
  prefill；现在一轮只 prefill 新增的部分。匹配是精确的 token 前缀比较，所以不可能错误复用：
  历史被编辑或分词不同，只会匹配到更短的条目，或者未命中并退回完整 prefill。S100P 上实测——
  混合（Qwen3.5-0.8B）31.6 s → 1.3 s（约 25x），dense（Qwen2.5-1.5B）第 10 轮 TTFT 274 → 141 ms。
  新参数：`BLLM_CACHE_MAX_MB`、`BLLM_CACHE_TTL_S`、`BLLM_CACHE_MAX_ENTRIES`、`BLLM_MAX_QUEUE`；
  新增 `GET /metrics`。
- **会话上的服务原语** —— `snapshot()` / `restore()`（在内存中，不用临时文件）、`feed_ids()` /
  `decode_stream()`（把 generate() 拆开，调用方可以恰好在 prefill/decode 边界处做 snapshot）、
  `request_cancel()` / `canceled`，以及 `context_left`、`prefill_chunk` 和 `cache_align()`。
  `set_prefix()`/`ask()` 服务的是**一个**固定前缀；这些原语让服务端能持有多个状态，并为每个请求
  挑一个。
- **`cache_align()` —— snapshot 必须按块对齐。** dense 引擎以 `prefill_chunk` 个 token 为一段批量
  prefill，余下的逐步 decode，而这两张图在 int8 下数值并不完全相同，所以不在块边界上取的
  snapshot 会改变回复。实测：对齐切分 5/5 复现了不走缓存的答案，未对齐只有 1/18——其中一次把
  *"Venus is the hottest"* 变成了 *"Earth is the hottest"*。这条规则放在库里，调用方不会弄错。
- **真实的 token 计数。** `usage` 现在来自模型的分词器，而不是按每 token 4 个字符估算，并报告
  OpenAI 的 `prompt_tokens_details.cached_tokens`。

### 修复

- **流式中途断开可能破坏下一个请求的回答。** 服务端在 SSE 响应外持有一把 `asyncio.Lock`，而生成
  在每个请求自己的线程上运行；客户端在流式中途消失时，异步生成器被关闭、锁被释放，生成线程却仍在
  驱动引擎，于是下一个请求与它并发运行。在旧服务端上复现过：断开一次之后，*"reply with exactly
  the word: pineapple"* 返回了 `"pineapple1. pineapple1.1. pineapple1. ..."`。现在会话由单一工作
  线程持有，只能经由队列访问，HTTP 侧的任何事件都无法提前释放它。
- **请求悄无声息地无限排队**——客户端只看到超时。准入控制现在在超过 `BLLM_MAX_QUEUE`（默认 8）
  时返回 429。
- **`kPrefillMinTokens` 原为 16，大约高了 5x。** 一次 prefill 不论携带多少 token，都要付出一个
  补齐后整块的代价，所以分界点是（一次 prefill）/（一步 decode）——Qwen2.5-1.5B 实测 2.93，
  Phi-4-mini 实测 3.31，二者都稳定，因为都是单次 BPU 图调用。现在改为 4。一轮 15 个 token
  （像 *"go on"* 这样的简短追问——恰恰是旧值惩罚的常见情形）从 **676 ms 降到 135 ms**。注意这会
  改变 4..15 个 token 的切分方式，因而也会改变回复。

### 说明

0.1.5 打 tag 之后又发现了两个服务层 bug，已在 `main` 上修复（只影响容器镜像——conda 包不带
服务层，所以已发布的 0.1.5 不受影响）：混合引擎根本没有缓存（它不分块，所以
`cache_align(n) == n`，而一个过于积极的守卫把它读成"无事可做"）；以及 snapshot 取在每个请求的
assistant 开头之后，而这部分永远不可能是下一轮提示词的前缀。现在只有对话历史可缓存。S100P 上
修复后的混合 A/B（Qwen3.5-2B，6 轮）：TTFT 从 6.3 s → 31.0 s 逐轮增长变为稳定在约 5.9 s。

Docker 镜像从 conda 频道安装 `bllm`，所以服务端可能比其下的运行时更新。当已安装的构建早于
snapshot 原语时，服务端降级为完整重新 prefill，并在启动时说明，而不是失败——并发修复在两种
情况下都生效，运行时足够新时缓存会自动开启。`GET /metrics` 会报告 `supports_prefix_cache`。

## [0.1.4] — 2026-07-16

### 新增

- **`bllm-make-model-dir` 随包发布** —— 模型目录工具移入 `bllm.make_model_dir`，安装到
  `$PREFIX/bin`，所以 `conda install bllm` 就足以拼出模型目录。它此前只是源码树里的
  `scripts/make_model_dir.py`，从 conda 安装的用户没有办法把官方发布物变成 `bllm.load()` 能接受
  的东西。`scripts/make_model_dir.py` 仍然可用（薄包装），`python -m bllm.make_model_dir` 与之等价。
- **[`docs/MODELS.zh.md`](docs/MODELS.zh.md) / [`docs/MODELS.en.md`](docs/MODELS.en.md)** ——
  端到端部署官方发布物：ION 大小、包里实际有什么、拼 dense 和 Omni 模型目录、Omni 视频预算，
  以及值得了解的失败模式。已在 S100P 上逐条命令验证。

### 修复

- **公开文档里完全缺了 ION carveout 这个前提。** 板上 LLM 需要扩大 carveout
  （`hb_switch_ion.sh balanced` + 重启），否则运行时在分配时段错误——首次运行就失败，且毫无提示
  这是一个设置步骤而不是 bug。现在写进了 README 的安装一节，作为前提而不是脚注。
- **把官方 Omni 路径端到端写进了文档。** 从下载好的 Qwen2.5-Omni 发布物到可加载的模型目录，
  这一段此前没有文档：README 的多模态一节从一个已经拼好的目录讲起，唯一的 `omni` 示例藏在一个
  脚本的 docstring 里。README（zh/en）和新的部署文档现在给出了验证过的命令，"制作模型目录"也
  成了顶层章节而不是"从源码构建"的子节——conda 用户根本走不到那里，而且它也不是从源码构建的事。
- **明确指出 Omni 必须有 `embed_tokens.bin`。** 它在官方列表里是单独的 1.2 GB 下载，放在
  `model/` 下，名字里也不带 Omni——所以三个 `*_Omni_3B_*.hbm` 文件看起来像是完整的一套。缺了它
  现在给出的报错会说明文件从哪来，而不只是报出参数名。
- **Omni 视频预算明确写出，而不是靠暗示。** README 曾说模型目录"只要把 `visual` 指向你想要的
  塔"，但官方发布物只带 448px 的塔：每对帧 256 个 token，对 2048 槽位的窗口，在 2 fps 下约 8 s
  视频。其他分辨率需要离线重新导出，不在本仓范围内。
- **删去过时的"把 `tokenizer.json` 降到 0.19.1"建议**——那是 libxlm 的限制。native 运行时原样读取
  `tokenizers` 0.20+ 的 pair 格式 merges（含 `ignore_merges`）；已在板上验证。
- **`__version__` 卡在了 0.1.2**——于是已发布的 0.1.3 包从 `bllm.__version__` 报出错误的版本。
  现在它跟随 `CMakeLists.txt`。

## [0.1.3] — 2026-07-15

### 修复

- **多核 `.hbm` 绑定（nash-p / S600）。** 以 `core_num > 1` 编译的 `.hbm` 用 `HB_UCP_CORE_ANY`
  提交时被运行时拒绝（*"When infer multi-core model task, the backend must be specified to specific
  cores"*）。`model.json` 早已带有 `bpu_cores`，但 `ModelConfig` 从未解析它，所以引擎总是提交
  `CORE_ANY`。现在会解析，并在加载时把调度核掩码绑定到核 `0..(N-1)`；`set_bpu_priority` 保留该
  掩码。已在 S600 上验证：Qwen3.5-4B nash-p 4 核，22.6 tok/s。

## [0.1.2] — 2026-07-14

### 新增

- **可复用前缀路径** —— `NativeSession.set_prefix()` / `ask()`：把共享前缀（一篇文档、固定的
  system + 工具上下文、few-shot 示例）**只 prefill 一次**，然后针对它回答多个问题而无需重新
  prefill——每次 `ask()` 都恢复前缀，彼此独立。在共享上下文上重复提问，每次快约 3–4×。底层是引擎
  的内存 `snapshot()` / `restore()`（基于文件的 `save_state`/`load_state` 共用同一内核）。
- **思考开关** —— `NativeSession.set_thinking(False)` 预填一个空的 `<think></think>`（即官方模板
  的 `enable_thinking=False`），让 Qwen3.5 推理模型直接作答——更快，token 更少。默认会推理并显示
  该块。（消息内的 `/no_think` 软开关在当前 checkpoint 上不可靠；这种预填是可靠的。）
- 会话上的 **`encode()` / `decode()`**（模型的 C++ 分词器）。
- **Qwen3.5-2B / 4B**（混合 Gated-DeltaNet/SSM）在板上验证通过，100% BPU int8（约 14.5 / 6.9
  tok/s），与 0.8B 并列。
- **S600 多核 BPU** —— 为 N 个 BPU 核编译的 `.hbm`（nash-p，N ∈ {1, 2, 4}）通过
  `make_model_dir.py --bpu-cores N` 记录（`model.json` 的 `bpu_cores`）。大模型能吃满 4 核
  （Qwen3.5-4B，S600 上 27 tok/s）；小模型保持单核。

## [0.1.1] — 2026-07-13

### 新增

- **Phi 对话模板** —— 第三种对话格式（`phi`，与 `chatml` 和原始 `none` 并列），用于以
  `<|user|>` / `<|assistant|>` / `<|system|>` / `<|end|>` token 而不是 ChatML 标记角色的模型。
  `make_model_dir.py` 会自动识别这些标记并生成它，所以 **Phi-4-mini** 现在作为一等的多轮对话包
  发布（此前只能作为原始续写运行）。现有 ChatML 模型不受影响。

## [0.1.0] — 2026-07-12

首个公开版本——面向 D-Robotics RDK S100 / S100P / S600 的板上 LLM / VLM 运行时，在 BPU 上原生
驱动编译好的 `.hbm` 图（hbDNN / hbUCP），无需额外的 LLM SDK。

### 新增

- **native 文本运行时** —— dense 模型（Qwen2.5-1.5B/7B、DeepSeek-R1-Distill、InternLM2、
  GLM-Edge、Phi-4-mini）以及 **Qwen3.5-0.8B 混合 Gated-DeltaNet/SSM**（100% 跑在 BPU 上，约
  21 tok/s），内置 C++ 分词器、ChatML 模板、流式与多轮。
- **多模态** —— 面向 Qwen2.5-Omni 的 `NativeVlm` / `NativeVlmSession`：文本 + 图片 + 音频 + 视频，
  可来自文件路径或原始数组，支持实时流式采集。
- **`bllm.load()`** —— 统一入口，加载 `model.json` 包，**或**裸 `.hbm` + 分词器目录（即时合成
  清单）。
- **采样** —— 贪心 / 温度 / top-k / top-p / min-p / typical-p，加上重复、频率和存在惩罚；设定
  种子即可复现。
- **生产特性** —— 困惑度、异步提交、提示词缓存保存/恢复（逐位一致地续跑）、停止串（提前停止），
  以及 BPU 优先级 / 时间片控制，便于与视觉流水线共享 BPU。
- **打包** —— linux-aarch64 的 conda 包（`bllm`、`libbllm`、`tokenizers-cpp`）；C++ 使用方可用
  `find_package(bllm)`。

[Unreleased]: https://github.com/ruisv/bllm/compare/v0.10.1...HEAD
[0.10.1]: https://github.com/ruisv/bllm/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/ruisv/bllm/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/ruisv/bllm/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/ruisv/bllm/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ruisv/bllm/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ruisv/bllm/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/ruisv/bllm/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/ruisv/bllm/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ruisv/bllm/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/ruisv/bllm/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/ruisv/bllm/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ruisv/bllm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ruisv/bllm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ruisv/bllm/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/ruisv/bllm/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/ruisv/bllm/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/ruisv/bllm/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/ruisv/bllm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ruisv/bllm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ruisv/bllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ruisv/bllm/releases/tag/v0.1.0
