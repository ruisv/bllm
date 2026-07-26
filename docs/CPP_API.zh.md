<!-- 中文 C++ API 文档；English: CPP_API.en.md -->

# BLLM C++ API 参考（中文）

C++ 库，命名空间 `bllm`。[`include/bllm/`](../include/bllm/) 下的公共头文件才是
唯一权威 —— 本文是它们的摘要加用法。Python 绑定见 [`API.zh.md`](API.zh.md)；
两边的名字一一对应（Python `snake_case` ⇄ C++ 同名方法）。
English: [CPP_API.en.md](CPP_API.en.md)。

- [链接与构建](#链接与构建) · [约定](#约定)
- 模型：[ModelConfig](#modelconfig) · [Tokenizer](#tokenizer)
- 会话：[NativeLlm 文本对话](#nativellm-文本对话) ·
  [NativeVlm 多模态](#nativevlm-多模态) · [NativeEmbedder 文本嵌入](#nativeembedder-文本嵌入)
- 生成控制：[采样参数](#采样参数) · [GBNF 语法约束](#gbnf-语法约束)
- 底层：[引擎 NativeEngine 与 NativeHybridEngine](#引擎-nativeengine-与-nativehybridengine) ·
  [媒体读取辅助](#媒体读取辅助)
- [完整示例](#完整示例) · [Python 对照表](#python-对照表)

## 链接与构建

```cpp
#include "bllm/native_llm.h"      // 文本对话
#include "bllm/native_vlm.h"      // 多模态（图 / 音 / 视频）
#include "bllm/native_embedder.h" // 文本嵌入
```

只有一个 CMake 目标 `bllm::native`，它只依赖板载的通用 hobot 运行时
（外加 `tokenizers-cpp` 做字符串 I/O）：

```cmake
find_package(bllm REQUIRED)
target_link_libraries(app PRIVATE bllm::native)
```

conda 包与板上构建见 [README](../README.md#从源码构建)。这里的一切**只能在
RDK S100 / S100P / S600 板子上运行** —— hobot 运行时与 BPU 只存在于板上。
还需要先把 ION carveout 调大并重启，否则运行时会在分配权重处直接段错误
（见 [`MODELS.zh.md`](MODELS.zh.md)）。

## 约定

- **命名空间** `bllm`；头文件后缀 `.h`，按功能分文件，没有总头文件。
- **错误**通过 `std::runtime_error`（消息以 `[bllm]` / `[native]` / `[embed]`
  开头）抛出，包括模型类型用错、KV 窗口越界、快照与模型不匹配等。请用
  `try/catch` 包住。
- **一个模型是一个目录**（`model.json` + `.hbm` + `tokenizer.json`）。所有会话类
  都能直接吃目录路径，也能吃一个已经解析好的 `ModelConfig`。
- **会话类不可拷贝**，各自独占自己的 BPU 图与缓存。
- **一个会话同一时刻只能有一个生成**。唯一可以跨线程调用的是
  `NativeLlm::request_cancel()`。
- **KV / SSM 窗口就是上下文**：越界会抛异常，而不是悄悄丢弃最早的 token
  （对这类全注意力模型，丢弃会让输出直接变乱码）。用 `context_left()` 提前看到。

---

## ModelConfig

`bllm/model_config.h` —— `model.json` 清单，它让模型自描述：架构、`.hbm`、
tokenizer、（hybrid 用的）嵌入表、eos ids、cache_len、rope、对话模板。
这是"离线转换产出 model.json"与"运行时一行加载"之间的接缝。

```cpp
bllm::ModelConfig cfg = bllm::loadModelConfig("/models/qwen3.5-2b");
// 相对路径已按目录解析成绝对路径
```

| 字段 | 说明 |
|---|---|
| `dir` / `name` | 模型目录 / 模型名。 |
| `arch` | `"dense"` / `"hybrid"`（SSM） / `"omni"`（多模态） / `"embed"`（编码器）。 |
| `hbm` | 文本塔 `.hbm` 的绝对路径。 |
| `tokenizer` | `tokenizer.json` 路径。 |
| `embed` | 宿主侧嵌入表（hybrid 与 omni 需要）。 |
| `visual` / `audio` / `mel_filters` | 视觉塔 / 音频塔 / mel 滤波器（多模态）。 |
| `graph` | `.hbm` 内部的图名（hybrid，默认 `"qwen35"`）。 |
| `hbm_prefill` | 可选的 `seq_len=N` 分块 prefill 图。**缺失不是错误** —— 引擎会退回逐 token 摄入。 |
| `eos` | 停止 token id 列表。 |
| `mrope_section` / `mrope_interleaved` | VLM 的 rope 维度在 (t,h,w) 上的切分及其频率布局。 |
| `vision` | `VisionCfg{patch, merge, temporal, mean[3], std[3]}`。 |
| `cache_len` | 仅供参考。 |
| `bpu_cores` | `.hbm` 编译时的 BPU 核数；**> 1 时提交必须绑定到具体核**，会话构造时会自动设好核掩码。 |
| `rope_theta` | rope 底数。 |
| `chat` | `ChatConfig{format, im_start, im_end, bos, r_user, r_assistant, r_system, r_end, system}`。 |

判定用的谓词：`is_hybrid()`、`is_omni()`、`is_embed()`、`has_vision()`。

> **多模态是「包」的属性，不是解码器家族的属性。** `has_vision()`（目录里带
> `visual` 塔）才是"能不能看图"的判据：`"omni"` 是带视觉+音频的 dense 文本塔，
> Qwen3.5 是带视觉的 hybrid 文本塔，两者都走 `NativeVlm`。

> **`VisionCfg` 里的常量必须由清单携带。** 编译好的图里看不出 patch 尺寸与像素
> 归一化 —— 用 patch-14 的预处理去喂 patch-16 的图，会得到**形状正确的垃圾**。

## Tokenizer

`bllm/tokenizer.h` —— HuggingFace `tokenizers`（经 mlc-ai/tokenizers-cpp）的薄封装。
这正是让原生引擎能说**字符串**而不只是 token id 的那一层；encode/decode 与 HF 完全一致。

```cpp
bllm::Tokenizer tk = bllm::Tokenizer::fromFile("/models/x/tokenizer.json");
// 或 bllm::Tokenizer::fromBlob(json_string)
std::vector<int> ids = tk.encode("你好");
std::string text     = tk.decode(ids);
int n = tk.vocab_size();
int id = tk.token_to_id("<|im_start|>");
std::string tok = tk.id_to_token(id);
```

会话类各自持有一个，用 `tokenizer()` 取只读引用。

---

## NativeLlm 文本对话

`bllm/native_llm.h` —— 统一的"字符串进、字符串出"会话，跑在自建原生引擎上。
它加载模型目录、按 `arch` 建好对应引擎（Qwen3.5 走 hybrid Gated-DeltaNet，
Qwen2.5/DeepSeek/… 走 dense KV）、持有 C++ tokenizer，并以字符串暴露
`generate()` / `chat()` / `reset()`，支持流式。

```cpp
#include "bllm/native_llm.h"

bllm::NativeLlm llm("/models/qwen3.5-2b");
std::string reply = llm.chat("你好", 256,
                             [](const std::string& s){ std::cout << s << std::flush; });
```

传进 `arch="omni"` 或 `arch="embed"` 的目录会**明确报错**并告诉你该用哪个类，
而不是给出一个坏掉的会话。多核 `.hbm`（`bpu_cores > 1`）的核掩码在构造时自动
设好 —— 运行时会拒绝多核任务用 `HB_UCP_CORE_ANY`。

### 生成

| 方法 | 说明 |
|---|---|
| `std::string generate(prompt, max_new=256, on_text={}, stop={})` | 原样补全，不套对话模板。 |
| `std::string chat(msg, max_new=256, on_text={}, stop={})` | ChatML / Phi 模板的多轮对话；上下文跨调用保留，每次只喂新的一轮。 |
| `void reset()` | 清空上下文，开始新会话。 |
| `double last_decode_tps() const` | 上一轮的解码 tok/s。 |
| `std::array<double,3> last_timing() const` | 上一轮每 token 的 `{prep, infer, sample}` 毫秒拆分。**仅 hybrid**，dense 上为全零。图本身的耗时可以用 `hrt_model_exec` 单独测，所以这个拆分让**剩下的那部分**变成可归因的，而不是猜的。 |

`on_text` 是 `std::function<void(const std::string&)>`，收到的是**增量**文本。
返回值始终是完整回复。

### 采样

```cpp
llm.set_sampling(/*temp=*/0.7f, /*top_p=*/0.9f, /*top_k=*/0, /*rep_pen=*/1.0f,
                 /*seed=*/bllm::kSeedRandom);
// 完整签名还有：min_p, typ_p, min_keep, penalty_last_n, penalty_freq,
//               penalty_present, logit_bias, logprobs
```

- `temp <= 0` 即贪心（argmax）。
- `logit_bias` 是 `std::unordered_map<int,float>`，即 OpenAI 的同名参数
  （`[-100, 100]` 的加性偏置，用来强推或封禁某个 token）；传空 map 即清除。
- `logprobs >= 0` 让**下一轮**记录每个 token 的对数概率（外加这么多个备选），
  之后用 `last_logprobs()` 读；`-1`（默认）跳过这两趟额外的全词表计算。

| 方法 | 说明 |
|---|---|
| `const std::vector<TokenLogprobs>& last_logprobs() const` | 上一轮逐 token 的对数概率，顺序与生成一致，且**已裁剪**到真正出现在返回文本里的那些 token（停止串截断的部分不会多报）。 |
| `std::string token_bytes(int id) const` | 该 token 真实贡献的字节；控制 token 返回空。**`decode({id})` 给不了这个**：只含半个多字节字符的 token 会解码成 U+FFFD，按 token 重组文本时中文就烂了 —— 这正是 OpenAI logprobs 里 `bytes` 字段存在的原因。 |

`last_logprobs()` 给的是**原始**模型分布（全词表精确 log-softmax），不随
temperature / 惩罚 / `logit_bias` 变化，所以实际生成的 token 未必是 `top[0]`。

### 停止串与思考开关

| 方法 | 说明 |
|---|---|
| `void set_stop(std::vector<std::string>)` | 常驻停止串：回复中一旦出现即结束该轮，并**从流与返回值中裁掉**。它补的是 eos token 停不住的场景（停止串可以跨 token，也不必是一个 token）。传空清除；每次调用的 `stop` 参数只覆盖那一次。 |
| `void set_thinking(bool)` / `bool thinking() const` | Qwen3.5 推理开关。`false` 会在 assistant 标记之后预填一个闭合的空 `<think></think>`（等价于官方模板的 `enable_thinking=False`），模型**直接作答** —— 更快、更省 token。默认 `true`，会推理并显示 `<think>…</think>`。注意消息里的 `/no_think` 软开关在这个 checkpoint 上**不可靠**，预填才可靠。 |

停止串在流式输出时会**先扣住**它自己的前缀，直到该前缀被补全或被排除，
所以客户端不会先看到半个停止串再被回收。

### 上下文、优先级与困惑度

| 方法 | 说明 |
|---|---|
| `int context_left() const` | KV / SSM 窗口里还能放下多少 token。 |
| `void set_bpu_priority(int)` | BPU 队列优先级 0..255（默认 0 最低，让同驻的视觉流水线插队）。它**只排队，不抢占**已经在跑的图 —— 要真正做到图内切换，`.hbm` 编译时还得设 `max_time_per_fc`。这个调用会保留构造时设好的核掩码。 |
| `double perplexity(const std::string&)` | 原文（不套模板）的教师强制困惑度。**会重置会话。** |
| `const ModelConfig& config() const` / `const Tokenizer& tokenizer() const` | 只读访问。 |

### 状态存取与前缀复用

```cpp
llm.save_state("ctx.bin");     // KV(+SSM) 状态 + 末次 logits
llm.load_state("ctx.bin");     // 恢复后生成逐位一致
```

状态与模型绑定，加载别的模型的状态会被拒绝。

针对"多个独立问题打同一份共享上下文"（对一篇文档做 RAG、固定的系统提示 + 工具
定义、少样本示例）的场景，用前缀复用：共享前缀**只 prefill 一次**，之后每次
`ask()` 都从那个快照恢复，而不是重新 prefill。

```cpp
llm.set_prefix(document_text);          // 一次性 prefill，并在内存里快照
std::string a1 = llm.ask("这份报告的结论是什么？");
std::string a2 = llm.ask("它提到了哪些风险？");   // 复用前缀，且不累积
```

| 方法 | 说明 |
|---|---|
| `void set_prefix(const std::string& shared_context = "")` | prefill 系统块 + 可选共享上下文，并快照为可复用的基底。 |
| `std::string ask(query, max_new=256, on_text={}, stop={})` | 针对固定前缀回答，**每次独立不累积**（与 `chat()` 相反）。 |
| `bool has_prefix() const` / `void clear_prefix()` | 是否已设前缀 / 丢弃缓存的前缀。 |

### 服务化原语

`set_prefix()`/`ask()` 服务的是**一个固定前缀**。一个要同时持有很多会话状态、
每个请求挑一个的服务（比如 OpenAI 兼容服务），需要自己驱动快照：

| 方法 | 说明 |
|---|---|
| `std::string snapshot() const` | 把当前上下文快照成字节串（与 `save_state()` 同样的载荷，但在内存里）。 |
| `void restore(const std::string& blob)` | 恢复快照，之后的生成逐位一致。相对 prefill 很便宜，但**不是免费**：`restore()` 会先清空整个缓存窗口再把行拷回去。 |
| `int prefill_chunk() const` | prefill 分块宽度；`0` 表示该引擎不分块。 |
| `void feed_ids(const std::vector<int>& ids)` | 只摄入 token 不解码（`generate()` 的前半）。取 id 而不是文本：服务端已经分过词（缓存也是按 token id 匹配的），把文本增量重新编码可能在接缝处切出不同的 token。 |
| `std::string decode_stream(max_new=256, on_text={}, stop={})` | 从当前上下文开始解码（`generate()` 的后半）。 |
| `void request_cancel()` / `bool canceled() const` | 在 token 边界干净地中断进行中的生成。**线程安全**，可从别的线程调用（比如服务端发现客户端断连）。上下文保持一致 —— 取消点之后不会再生成任何东西；标志是一次性的，落在两轮之间的取消会被丢弃而不会误杀下一轮。 |

> ⚠️ **快照必须打在 `prefill_chunk()` 的整数倍上。** dense 引擎把足够长的一段
> token 走批量 prefill 图、剩下的走单 token decode 图，两张图在 int8 下**数值并不
>相同**。在任意点切分 prompt，就改变了哪些 token 走了哪张图 —— 也就可能改变回答。
> 只在这些倍数点上快照，分段方式才与直通 prefill 一致，复用才是精确的。
> `prefill_chunk() == 0` 表示任意切分都精确（hybrid 引擎在没有 `hbm_prefill` 时
> 就是逐 token 摄入）。

现成实现见 `docker/serving/`（`cache.py` 的前缀缓存 + `engine.py` 的单 worker 线程）。

---

## NativeVlm 多模态

`bllm/native_vlm.h` —— `NativeLlm` 的多模态姊妹类：加载一个带视觉塔的模型目录，
回答"文本 + 图像（+ 音频 / 视频）"的一轮问答。

多模态在这里是怎么真正工作的：文本塔的图吃的是**嵌入向量**（而不是 token id）
以及宿主给的 rope cos/sin。所以会话会

1. 把 ChatML 这一轮分词，并在宿主侧把每个 token 嵌入（`EmbedTable`）；
2. 跑视觉塔，把它的行**拼接**进同一条流中图像所在的位置 —— HF 的 `<|IMAGE|>`
   占位符从来不需要；
3. 分配 3-D mrope 位置 —— 文本是标量，图像 patch 是 `(t, t+row, t+col)` ——
   然后把整条流喂给 prefill/decode。

```cpp
#include "bllm/native_vlm.h"
#include "bllm/image_io.h"

bllm::NativeVlm vlm("/models/qwen3.5-0.8b-vlm448");
bllm::OwnedImage img = bllm::loadImageRGB("bears.jpg");
std::string a = vlm.chat("这张图里有什么？", {{img.rgb.data(), img.width, img.height}},
                         {}, {}, 256,
                         [](const std::string& s){ std::cout << s << std::flush; });
```

数据结构：

```cpp
struct ImageRGB { const uint8_t* rgb; int width, height; };  // 交织 RGB8，像素由调用方持有
struct AudioPCM { const float* samples; size_t count; };     // 16 kHz 单声道 float
struct VideoClip { std::vector<ImageRGB> frames; double fps = 2.0; AudioPCM audio; };
```

| 方法 | 说明 |
|---|---|
| `std::string chat(text, images={}, audios={}, videos={}, max_new=256, on_text={}, stop={})` | 一轮问答；媒体排在 `text` 之前，与 Qwen 的对话模板一致。 |
| `void append_turn(text, images, reply)` | 回放一轮**已经知道回复**的历史——只做 `openTurn`+媒体+文本+`closeTurn`+回复文本的 prefill，不进解码循环。留给下一轮的上下文与真的 `chat()` 过一遍逐位相同；给服务层重放"客户端刚重发、自己刚生成过"的历史用，见 `docker/serving/engine.py` 的 `_serve_vlm`。 |
| `std::vector<int> encode(text) const` | 纯 tokenizer 编码（不套 ChatML），只用于统计 token 数。 |
| `int vision_tokens() const` | 一个 frame-pair / 一张图占多少个 token。 |
| `int vision_image_size() const` | 这个塔编译时的正方形边长 —— 按它采样可以少缩放一次。 |
| `bool has_audio() const` | 这个包带不带音频塔。 |
| `double last_ttft_ms() const` | 上一轮首字延迟，**含媒体编码**。 |
| `double last_decode_tps() const` · `int context_left() const` · `int tokens_used() const` | 同 `NativeLlm`。 |
| `set_sampling(...)` / `set_stop(...)` / `set_bpu_priority(...)` / `reset()` | 与 `NativeLlm` 同义；`set_bpu_priority` 会同时作用于文本、视觉、音频三个塔。 |

构造时会检查视觉塔与文本塔的 hidden 尺寸是否一致 —— 不一致会抛异常，因为那是
一次**静默的垃圾拼接**。

### 实时输入

边采集边把每 2 秒的分块编入 KV 缓存，所以问题到来时大部分 prefill 已经付过了。

```cpp
vlm.stream_begin(/*fps=*/2.0);            // fps <= 0 表示只有音频
while (grabbing) {
  vlm.stream_frame(frame);                // ImageRGB，所有帧尺寸必须一致
  vlm.stream_audio(pcm, n);               // 16 kHz 单声道 float
}
std::string a = vlm.stream_ask("刚才发生了什么？");
bool live = vlm.streaming();
```

> **预算即 KV 窗口。** 一个 frame-pair 跨 `temporal_patch_size`（2）帧，所以
> 2 fps 下视频每秒钟消耗 `vision_tokens()` 个 token，音频每秒约 25 个。
> 一个 2048 槽位的模型因此只装得下约 8 秒视频（用出厂的 448px 塔；换成 224px
> 重导出可到 32 秒），或纯音频约 80 秒。预算耗尽后 `stream_frame` /
> `stream_audio` 会**抛异常而不是淘汰**旧内容；`context_left()` 让调用方提前看到。
> `video_tokens_per_second()` 给出当前设置下视频每秒的 token 开销。

> **视频只在 Omni 包上可用。** 这里的视频路径实现的是 Qwen2.5-Omni 的 TMRoPE
> （每格 `秒数*25` 的时间位置、音视频按 2 秒块交织）；Qwen3.5 用的是**时间戳
> token 分隔帧**，是另一套位置分配。对 hybrid 包传 `videos=` 或
> `stream_begin(fps>0)` 会**明确报错**，而不是悄悄编出一个看着通顺、其实顺序
> 错乱的答案。静止图不受影响：两者的静止图布局是同一套，且已对 HF 参考逐位核过。

## NativeEmbedder 文本嵌入

`bllm/native_embedder.h` —— BPU 上的文本嵌入。嵌入模型比对话模型简单得多：
跨步之间没有 KV 缓存、没有解码循环、没有采样。对 prompt 做一次因果前向，
池化最后一个 token 的 post-norm 隐状态，再 L2 归一化。所以编译出的 `.hbm`
就是单张**编码器**图（把配方里的 `lm_head` 换成 identity 得到）。

```cpp
#include "bllm/native_embedder.h"

bllm::NativeEmbedder emb("/models/qwen3-vl-embedding");
std::vector<float> v = emb.embed("巴黎是法国的首都。");
std::vector<float> small = emb.embed("同一句话", /*dim=*/256);   // Matryoshka 截断
```

| 方法 | 说明 |
|---|---|
| `std::vector<float> embed(text, dim=0, instruction="")` | L2 归一化后的嵌入。`dim > 0` 先截断（Matryoshka）再重新归一化；`0` 保留完整宽度。 |
| `std::vector<float> embed_ids(ids, dim=0)` | 直接吃 token id。 |
| `std::vector<int> build_ids(text, instruction="") const` | 该模型模板下的 token id 序列。**之所以暴露出来，是因为复现这串 id 是把移植结果与参考实现对齐的唯一办法。** |
| `int dimension() const` / `int max_tokens() const` | 向量宽度 / 编码器窗口。 |

> **输入模板不可改。** 它是靠把 token id 与模型自带的 embedder 逐一对拍还原出来的，
> 写错**不会报错**，只会静默返回另一个向量：
>
> ```
> <|im_start|>system\n{instruction}<|im_end|>\n
> <|im_start|>user\n{text}<|im_end|>\n
> <|im_start|>assistant\n<|endoftext|>
>                        ^^^^^^^^^^^^^ 被池化的就是这个 token 的隐状态
> ```

超过编码器窗口的输入会抛异常（提示缩短文本或编译一张更长的图），而不是截断。
`arch` 不是 `"embed"` 的目录会被拒绝。

命令行用法见 [`examples/embed_native.cc`](../examples/embed_native.cc)，
其中的 `--verify` 是**冒烟测试而不是发布闸门**（参考句子彼此过于相似，
成对相似度几乎打平，排序会因噪声翻转 —— 真正区分好坏的是余弦而不是排序）。

---

## 采样参数

`bllm/native_sampler.h` —— 会话的 `set_sampling(...)` 最终写入的就是这个结构；
自己驱动引擎时也可以直接构造它。

```cpp
struct NativeSamplingParams {
  float temp = 0.0f;             // <= 0 => 贪心（argmax）
  float top_p = 1.0f;            // 1.0 = 关闭（nucleus）
  float min_p = 0.0f;            // 0.0 = 关闭
  float typ_p = 1.0f;            // 1.0 = 关闭（locally typical）
  int   top_k = 0;               // <= 0 = 关闭
  int   min_keep = 1;            // 每个过滤器保底留几个
  int   max_new = 200;

  int   penalty_last_n = 64;     // 惩罚回看窗口（<=0 = 关闭）
  float rep_pen = 1.0f;          // 1.0 = 关闭
  float penalty_freq = 0.0f;     // 0.0 = 关闭
  float penalty_present = 0.0f;  // 0.0 = 关闭

  uint64_t seed = kSeedRandom;   // kSeedRandom => 每次重新取种；其他值精确可复现
  std::vector<int> eos;
  std::unordered_map<int, float> logit_bias;   // OpenAI 的 [-100, 100]
  TokenConstraint* constraint = nullptr;       // 语法约束，不持有所有权
  int logprobs = -1;                           // <0 = 关闭
};

struct TokenLogprobs {
  int id;                                      // 实际生成的 token
  float logprob;                               // log p(id)
  std::vector<std::pair<int, float>> top;      // (id, logprob) 备选，按概率降序
};
```

`seed` 默认为 `bllm::kSeedRandom`，即每次生成重新取种子 —— 采样器是每次生成新建
的，固定种子会让每一轮重放同一条随机流，看起来像 temperature 完全不起作用。

## GBNF 语法约束

`bllm/native_grammar.h` —— llama.cpp 那套 GBNF 格式的语法约束解码：**结构是保证
的，不是求来的**。每一步只有仍能让语法可满足的 token 才可能被采样。

```cpp
llm.set_grammar("root ::= \"yes\" | \"no\"\n");
std::string a = llm.chat("这句话是正面的吗？");   // 只可能是 yes 或 no
llm.clear_grammar();
bool on = llm.has_grammar();
```

- 空字符串等价于清除；从**下一轮**开始生效，且每轮都从语法的根重新开始。
- 代价：首次使用时建一张覆盖全词表的 token→字节表（每会话一次，板上约 315 ms），
  之后每步对每个候选做一次语法测试（实测 14.3 → 13.4 tok/s，约 6%）。
- 控制 token（eos、各对话标记、形如 `<|...|>` 的）在这张表里被清空，所以语法
  **不可能**生成它们。
- 底层的 `bllm::Grammar` 与 `bllm::GrammarConstraint` 也可直接使用；
  `Grammar` 会拒绝左递归。

> JSON Schema → GBNF 的转换器目前只在 Python 侧（`bllm.json_grammar(schema)`）。
> C++ 侧请直接写 GBNF，或用 Python 生成后把字符串传进来。

---

## 引擎 NativeEngine 与 NativeHybridEngine

`bllm/native_engine.h`（dense KV）与 `bllm/native_hybrid_engine.h`
（Gated-DeltaNet / SSM）—— 会话类之下的那一层。大部分应用不需要直接碰它们；
需要自己控制摄入与状态时才用。

两者共有的接口：

| 成员 | 说明 |
|---|---|
| `void feed(const std::vector<int>& ids)` | 摄入 token（留下最后一步的 logits 供采样）。 |
| `std::vector<int> generate(const NativeSamplingParams&, on_token)` | 解码循环；`on_token` 返回 `true` 即中断。 |
| `void reset()` | 清空上下文。 |
| `int position()` / `int context_left()` / `int cache_len()` | 已用 / 剩余 / 总窗口。 |
| `std::string snapshot() const` / `void restore(const std::string&)` | 内存快照与恢复。 |
| `void save_state(path)` / `void load_state(path)` | 落盘版本。 |
| `double perplexity(const std::vector<int>&)` | 教师强制困惑度。 |
| `const std::vector<TokenLogprobs>& last_logprobs() const` | 上一轮的逐 token 对数概率。 |
| `void set_sched(const native_detail::BpuSched&)` / `sched()` | BPU 优先级与核掩码。 |
| `void mark_turn_start()` | 让 TTFT 把媒体编码也算进去。 |
| `void feed_embeds(const float* rows, int n, const Pos3* pos)` | 直接喂嵌入行 + 3-D 位置（多模态走的就是这条）。 |

各自特有的：`NativeEngine::chunk()`（批量 prefill 宽度）、`n_layers()`、`n_kv()`、
`head_dim()`、`vocab()`、`embed_input()`；`NativeHybridEngine::prefill_chunk()`、
`n_cache()`、`embed_row()`。会话的 `prefill_chunk()` 就是对这两者的转发。

`native_detail::BpuSched{priority, core_mask}` 是提交参数：多核 `.hbm`
（nash-p / S600）**必须**绑定到具体核 —— 运行时会拒绝多核任务用
`HB_UCP_CORE_ANY`。核掩码取 `(1<<N)-1`（`HB_UCP_BPU_CORE_k == 1<<k`），
单核则保持 `CORE_ANY` 默认。

## 媒体读取辅助

给 CLI / 示例用的最小文件解码，header-only、自带实现，不引入外部依赖。
运行时 API（`VisionTower` / `AudioTower` / `NativeVlm`）本身吃的是裸
RGB8 / 16 kHz float PCM —— 也就是相机与麦克风流水线直接给出的东西。

```cpp
#include "bllm/image_io.h"    // 内含 stb_image
#include "bllm/audio_io.h"    // 内含 dr_wav

bllm::OwnedImage img = bllm::loadImageRGB("photo.jpg");   // {rgb, width, height}
bllm::OwnedAudio au  = bllm::loadAudio16k("clip.wav");    // {samples, sample_rate}, .seconds()
```

`loadAudio16k` 会混成单声道并按需线性重采样。**不是 WAV 的先转码**：
`ffmpeg -i in.mp4 -ac 1 -ar 16000 -f wav out.wav`。

---

## 完整示例

一个最小的流式对话程序（与 [`examples/chat_native.cc`](../examples/chat_native.cc) 同形）：

```cpp
#include "bllm/native_llm.h"
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
  try {
    bllm::NativeLlm llm(argv[1]);              // 模型目录
    llm.set_sampling(0.7f, 0.9f, 0, 1.0f, bllm::kSeedRandom);
    llm.set_stop({"\n\n"});

    for (std::string line; std::getline(std::cin, line) && !line.empty(); ) {
      llm.chat(line, 400, [](const std::string& s){ std::cout << s << std::flush; });
      std::printf("\n[%.1f tok/s, 剩余上下文 %d]\n", llm.last_decode_tps(), llm.context_left());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bllm error: %s\n", e.what());
    return 1;
  }
}
```

更多可运行程序见 [`examples/`](../examples/)：`chat_native.cc`（REPL）、
`qwen35_native.cc`、`vlm_native.cc`（图 / 音 / 视频）、`embed_native.cc`、
`tokenizer_probe.cc`、`vision_probe.cc`。

## Python 对照表

| C++ | Python |
|---|---|
| `bllm::NativeLlm` | `bllm.NativeSession` |
| `bllm::NativeVlm` | `bllm.NativeVlmSession` |
| `bllm::NativeEmbedder` | 暂无绑定（C++ API only） |
| `bllm::loadModelConfig(dir)` | `bllm.load(dir)` 内部完成 |
| `llm.chat(msg, max_new, on_text)` | `llm.chat(msg, max_new, on_text)` / `llm.stream_chat(msg)` |
| `llm.set_sampling(...)` | `llm.set_sampling(...)`（同名关键字参数） |
| `llm.set_grammar(gbnf)` | `llm.set_grammar(gbnf)` + `bllm.json_grammar(schema)` |
| `llm.snapshot()` / `restore()` | `llm.snapshot()` / `llm.restore()` |
| `llm.prefill_chunk()` | `llm.prefill_chunk` / `llm.cache_align(n)` |
| `tk.encode/decode` | `llm.encode/decode` |

Python 侧另有 `cache_align(n)` 这个便利函数，直接给出"可安全快照的最大前缀
长度"；C++ 侧请自己对 `prefill_chunk()` 取整数倍。
