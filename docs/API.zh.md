<!-- 中文 API 文档；English: API.en.md -->

# BLLM API 参考（中文）

Python 包 `bllm`。C++ 对应类型见每节末尾。English: [API.en.md](API.en.md)。

## 顶层函数

### `bllm.load(path, *, backend="auto", **kwargs)`

加载模型，返回相应的会话对象。

- `path` 为**模型目录**（含 `model.json`）→ 稠密/混合返回 `NativeSession`，`arch="omni"` 返回 `NativeVlmSession`。
- `path` 为**裸 `.hbm`** 且传 `tokenizer_dir=<目录>` → 运行时合成清单并返回 `NativeSession`。
- 多余的关键字参数透传给会话构造函数。

```python
llm = bllm.load("/models/qwen2.5-1.5b")
llm = bllm.load("Qwen2.5_1.5B.hbm", tokenizer_dir="Qwen2.5_1.5B_config/")
```

### `bllm.available_backends() -> tuple[str, ...]`

返回可用运行时，原生构建下为 `("native",)`（无扩展的开发机上为 `()`）。

### `bllm.load_video(path, fps=2.0, size=448, max_frames=10, with_audio=True, ffmpeg="ffmpeg") -> dict`

把视频文件采样成 `NativeVlmSession.chat(videos=...)` 所需的字典（内部调用 ffmpeg 抽帧+抽音）。

## `bllm.NativeSession` — 文本对话

```python
llm = bllm.NativeSession(model_dir)          # 或 bllm.load(...)
```

**属性**：`name`、`arch`（`"dense"` / `"hybrid"`）、`vocab_size`、`last_decode_tps`（上次解码 tok/s）。

**方法**（多数返回 `self`，可链式）：

| 方法 | 说明 |
|---|---|
| `set_sampling(temp=0.0, top_p=1.0, top_k=0, rep_pen=1.0, seed=SEED_RANDOM, min_p=0.0, typ_p=1.0, min_keep=1, penalty_last_n=64, penalty_freq=0.0, penalty_present=0.0)` | 配置采样；`temp<=0` 即贪心。 |
| `set_stop(stop: list[str])` | 常驻停止串：回复中一旦出现即结束该轮并从流与返回值中裁掉。 |
| `set_bpu_priority(priority: int)` | BPU 队列优先级 0..255（默认 0 最低，让同驻视觉抢占）。 |
| `reset()` | 清空多轮历史，开始新会话。 |
| `chat(message, max_new=400, on_text=None, stop=None) -> str` | ChatML 多轮；返回完整回复，`on_text` 回调流式增量。 |
| `generate(prompt, max_new=256, on_text=None, stop=None) -> str` | 原样补全（不套对话模板）。 |
| `stream_chat(message, max_new=400, stop=None) -> Iterator[str]` | `chat` 的惰性生成器版本。 |
| `stream(prompt, max_new=256, stop=None) -> Iterator[str]` | `generate` 的生成器版本。 |
| `chat_async(...)` / `generate_async(...)` -> `concurrent.futures.Future[str]` | 非阻塞提交（释放 GIL）；每会话同一时刻一个生成。 |
| `perplexity(text) -> float` | 原文（不套模板）的教师强制困惑度；会重置会话。 |
| `save_state(path)` / `load_state(path)` | 保存/恢复会话的 KV(+SSM) 状态 + 末次 logits，恢复后逐位一致（跳过重复 prefill）。 |
| `set_prefix(shared_context="")` | 把系统提示 + 可选 `shared_context`（一篇文档/固定工具上下文/少样本示例）**prefill 一次**并内存快照。之后每次 `ask()` 复用它。 |
| `ask(query, max_new=400, on_text=None, stop=None) -> str` | 针对 `set_prefix` 的前缀回答 `query`，**每次独立不累积**（每次恢复前缀，从不重复 prefill 前缀）。 |
| `stream_ask(query, max_new=400, stop=None) -> Iterator[str]` | `ask` 的惰性生成器版本。 |
| `has_prefix -> bool` / `clear_prefix()` | 是否已设前缀 / 丢弃缓存的前缀。 |
| `encode(text) -> list[int]` / `decode(ids) -> str` | 用模型的 C++ 分词器编/解码。 |
| `set_thinking(enabled)` | Qwen3.5 思考开关。`False` 在 assistant 处预填空 `<think></think>` → **直接作答**(更快、更省 token);`True`(默认)推理并显示 `<think>…</think>`。(消息里的 `/no_think` 软开关对本 checkpoint 不可靠,预填才可靠。) |

```python
llm.set_thinking(False)                 # 关思考：直接作答,4B 快很多
```

**前缀复用示例**（多问答共享一篇文档，前缀只 prefill 一次）：

```python
llm = bllm.load("/models/qwen3.5-4b")
llm.set_prefix(open("report.txt").read())     # 一次性 prefill 这篇文档
print(llm.ask("这份报告的结论是什么？"))
print(llm.ask("它提到了哪些风险？"))            # 复用前缀，不再重复 prefill
```

C++：`bllm::NativeLlm`（`bllm/native_llm.h`）。`set_prefix`/`ask` 由引擎内存快照
`snapshot()`/`restore()`（`bllm::NativeHybridEngine` 与 `bllm::NativeEngine`）支撑。


### 服务化原语（自建会话池 / 前缀缓存）

`set_prefix()`/`ask()` 服务的是**一个固定前缀**。要同时持有多个会话状态、每个请求挑一个
（比如一个 OpenAI 兼容服务），就自己驱动快照：

| 方法 | 说明 |
|---|---|
| `snapshot() -> bytes` | 把当前上下文快照成字节串（同 `save_state` 的载荷，但不落盘）。 |
| `restore(blob)` | 恢复快照，之后的生成逐位一致。形状不匹配会报错。 |
| `feed_ids(ids)` | 只喂入 token 不解码（`generate()` 的 prefill 那一半）。 |
| `decode_stream(max_new=256, on_text=None, stop=None) -> str` | 从当前上下文解码（`generate()` 的生成那一半）。 |
| `stream_decode(max_new=256, stop=None) -> Iterator[str]` | 同上的惰性生成器版本。 |
| `cache_align(n) -> int` | **可安全快照的最大前缀长度**，见下方警告。 |
| `prefill_chunk -> int` | prefill 分块宽度；`0` 表示该引擎不分块（hybrid）。 |
| `context_left -> int` | KV/SSM 窗口剩余 token。窗口**就是**上下文，越界报错而非静默丢弃。 |
| `request_cancel()` / `canceled -> bool` | 在 token 边界干净地中断进行中的生成。**线程安全**，可从别的线程调用（比如服务端发现客户端断连）。上下文保持一致，取消后会话可立即复用。 |

> ⚠️ **快照必须打在 `prefill_chunk` 的整数倍上。** dense 引擎把足够长的 token 走批量
> prefill 图、剩余的走单 token decode 图，两者在 int8 下**数值不同**。分段只取决于剩余
> token 数，所以未缓存路径的分块边界必落在 `prefill_chunk` 的倍数上；只在这些点快照，
> 尾部分段才与未缓存时一致。板上实测（Qwen2.5-1.5B，greedy）：对齐切分 **5/5** 复现原答案，
> 不对齐 **1/18**，其中一次把 "Venus is the hottest" 变成 "Earth is the hottest"。
> 用 `cache_align()` 即可，别自己算。hybrid 无分块（`prefill_chunk == 0`），任意点都安全。

```python
ids  = sess.encode(prompt)
blob, n = cache.match(ids)             # 你自己的缓存：最长的【已对齐】前缀命中
sess.restore(blob) if blob else sess.reset()

at = sess.cache_align(len(ids))        # 可安全快照的位置
sess.feed_ids(ids[n:at])               # 先喂到边界
cache.put(ids[:at], sess.snapshot())   # 此刻状态 == ids[:at]
sess.feed_ids(ids[at:])                # 再喂剩下的
for piece in sess.stream_decode(256):
    print(piece, end="", flush=True)
```

现成实现见 `docker/serving/`（`cache.py` 的前缀缓存 + `engine.py` 的单 worker 线程）。

## `bllm.NativeVlmSession` — 多模态（Qwen2.5-Omni）

```python
vlm = bllm.NativeVlmSession(model_dir)        # 或 bllm.load(...)（arch="omni"）
```

**属性**：`name`、`vision_tokens`、`vision_image_size`、`has_audio`、`last_decode_tps`、
`last_ttft_ms`（首字延迟，含媒体编码）、`streaming`、`tokens_used`、`context_left`（KV 窗口剩余额度）。

**离线问答**（媒体传文件路径或原始数组：`HxWx3` uint8 帧 / 16 kHz 单声道 float PCM）：

| 方法 | 说明 |
|---|---|
| `chat(text, images=(), audios=(), videos=(), max_new=256, on_text=None, stop=None) -> str` | 文本 + 图/音/视频一轮问答。 |
| `stream_chat(text, images=(), audios=(), videos=(), max_new=256, stop=None) -> Iterator[str]` | 流式版本。 |
| `set_sampling(...)` / `set_stop(...)` / `set_bpu_priority(...)` / `reset()` | 同 `NativeSession`。 |

**实时输入**（边采集边编码）：

| 方法 | 说明 |
|---|---|
| `stream_begin(fps=2.0, with_audio=True)` | 开始一段实时会话。 |
| `stream_frame(frame)` | 送入一帧 `HxWx3` uint8。 |
| `stream_audio(pcm)` | 送入一段 16 kHz 单声道 float PCM。 |
| `stream_ask(text, max_new=256, on_text=None) -> str` | 就已编入的媒体提问。 |

> **上下文即约束**：KV 窗口就是上下文。2 fps 下视频每秒约耗一个 frame-pair 的视觉 token、音频约 25 token/s；
> 越过 `context_left` 会抛异常而非静默丢弃（丢弃最早 token 会让这类全注意力模型输出乱码）。视觉塔分辨率
> 在离线导出时定死，决定视频预算。

C++：`bllm::NativeVlm`（`bllm/native_vlm.h`）。

## 采样参数速览

`temp<=0` → 贪心（argmax）。`top_k`（0=词表大小）、`top_p`、`min_p`、`typ_p` 为候选过滤，`min_keep` 为每个
过滤器保底数量。`rep_pen`（重复）、`penalty_freq`（频率）、`penalty_present`（存在）作用于最近 `penalty_last_n`
个 token。`seed` 默认为 `bllm.SEED_RANDOM`，即每次生成重新取种子 —— 采样器是每次生成新建的，
固定种子会让每一轮重放同一条随机流，temperature 看起来完全不起作用。传入具体的 `seed` 可获得
可复现的采样。
