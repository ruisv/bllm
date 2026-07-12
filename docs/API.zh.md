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
| `set_sampling(temp=0.0, top_p=1.0, top_k=0, rep_pen=1.0, seed=1234, min_p=0.0, typ_p=1.0, min_keep=1, penalty_last_n=64, penalty_freq=0.0, penalty_present=0.0)` | 配置采样；`temp<=0` 即贪心。 |
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

C++：`bllm::NativeLlm`（`bllm/native_llm.h`）。

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
个 token。`seed` 固定随机性。
