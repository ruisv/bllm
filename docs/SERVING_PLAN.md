# 服务层：会话池 + 前缀缓存

> 状态文档 —— 随进展更新。创建于 2026-07-19，基线 0.1.4；**已随 0.1.5 交付并部署**
> （最后更新 2026-07-20）。
>
> 目标：把 `docker/server.py`（275 行 demo 级实现）改造成能承载多轮对话的服务层，
> 消除每轮全量 re-prefill，并修掉一个已存在的并发正确性 bug。

## 预期边界（先说不做什么）

单 BPU 图，一次只能跑一个 decode。**会话池不会带来并发**。它买的是两件事：

1. **TTFT 下降** —— 多轮对话第 N 轮不再重放前 N-1 轮。
2. **多会话互不污染** —— 现在多客户端靠"每次 reset 重放"保证正确性，有了池才能既缓存又不串台。

真并发需要 continuous batching，不在本文档范围。
Omni / VLM 走 embed 模式，`save_state is only for the token path`
（`include/bllm/native_engine.h:365`），**多模态服务不适用本方案**。

## 进度总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P1** | C++ `snapshot`/`restore`/`feed_ids`/`decode_stream` 公开 + 绑定 | ☑ 完成 |
| **P1.5** | 板上验证正确性与收益（决定是否继续） | ☑ 完成，**结论反转，见下** |
| **P0** | 修断连并发 bug + worker 线程模型 + 队列/429 | ☑ 完成 |
| **P2** | `PrefixCache` + 流水线接入 + 真 usage | ☑ 完成 |
| **P3** | LRU/TTL/内存上限 + `/metrics` + 文档 | ☑ 完成 |

| **发版** | conda 0.1.5（24 包，含 S600 变体）+ `kPrefillMinTokens` 调优 + 文档 | ☑ 完成 |
| **上线** | 容器重建、缓存预算 2 GB、板上运行验证 | ☑ 完成 |

全部阶段已实现、发版并部署。测试：`tests/test_prefix_cache.py`(21，任意主机)、
`tests/test_prefix_reuse.py`(8，板上)、`tests/test_serve_e2e.py`(7，需起服务)。

**注意本文档记录了两次"完成"** —— 第一次宣布完成后又在 hybrid 上抓到两个 bug
（见《上线后抓到的两个 bug》）。根因是端到端验证只覆盖了 dense 一条路径。

## 板上实测结论（2026-07-19，S100P）

三条与原计划假设不符，其中两条改变设计：

**1. 收益的主角是 hybrid，不是 dense。** 同口径对比（miss = 全量 prefill，
hit = restore + 只喂增量），模拟客户端每轮重发全历史：

| 引擎 | 模型 | miss → hit | 倍数 |
|---|---|---|---|
| dense | Qwen2.5-1.5B（chunk=256, ctx1024） | 271ms → 138ms | **1.97×**（封顶） |
| hybrid | Qwen3.5-0.8B（ctx2048） | 31.6s → 1.3s | **24.95×** |

hybrid 引擎**没有批量 prefill 图**，逐 token 喂 ~68 ms/token —— 512 token 的 prompt
要 **35 秒** TTFT。对 hybrid 而言前缀缓存不是优化，是可用性前提；而 hybrid 正是 BLLM
独有的 Qwen3.5 系列。dense 则封顶在 ~2×：缓存点只能落在 256 的倍数上，且不足一个 chunk
的增量仍要付一整个 padded chunk（134ms）。

**2. 快照只能打在 `prefill_chunk` 的整数倍上，否则会改变回答。**
dense 的 `feed()` 把 ≥16 token 的部分走批量 prefill 图、其余走单 token decode 图，
两个图在 int8 下数值不同。分段是"剩余 token 数"的纯函数，所以基线的分块边界必落在
`prefill_chunk` 的倍数上；只在这些点快照，尾部分段与未缓存时完全一致。

实测（Qwen2.5-1.5B，greedy，346/514/682 token 三组）：

- 对齐切分：**5/5 精确复现**未缓存的回答
- 不对齐切分：**1/18 精确**，其中一次把 "Venus is the hottest" 变成 **"Earth is the hottest"**

这条规则已收进库里：`NativeSession.cache_align(n)` 返回可安全快照的最大前缀长度。
hybrid 无分块（`prefill_chunk == 0`），任意切分都精确。

**3. `restore()` 的全窗口 memset 不是问题** —— 实测 3.3 ms，相对 prefill 可忽略。
原计划把它列为主要风险，可以撤销。

### 内存实测

| 引擎 | 单条快照 | 说明 |
|---|---|---|
| dense | ~22 KB/token（651 token → 14.3 MB） | 只写已填充行，随上下文增长 |
| hybrid | **恒定 71.97 MB** | 整套 cache 全量 dump，与已用长度无关 |

hybrid 每条 72 MB 是硬约束：512 MB 预算只装得下 7 条。计划里 `BLLM_CACHE_MAX_MB`
按字节计费的设计是对的，但默认条数上限对 hybrid 毫无意义，必须以字节为准。

### 顺带发现的性能问题（不在本方案内，另行处理）

`native_engine.h` 的 `kPrefillMinTokens = 16` 调得不对。实测同一模型：

| 喂入 token 数 | 路径 | 耗时 |
|---|---|---|
| 8 | decode 步 | 360 ms |
| 15 | decode 步 | **676 ms** |
| 16 | padded prefill | **134 ms** |
| 256 | prefill | 134 ms |

单个 decode step ~45 ms（即该模型 ~22 tok/s 的解码速度），而一次 256 宽 prefill 仅 134 ms
—— **盈亏点在 3 个 token 左右，不是 16**。3~15 token 的轮次（"继续""为什么"这类短提问是
高频场景）今天要多付最多 540 ms 的 TTFT。

**但这不是免费的性能修复**：改阈值会改变分段，从而改变输出（见第 2 条）。所以它需要
独立的决策与验证，不在本方案内静默改动。

---

## P0 — 修并发 bug + worker 线程模型

### 问题

`docker/server.py:228` 的 `_sse` 用 `async with engine.lock` 包住整个 SSE 响应，
但生成跑在 `pump` 线程里（`docker/server.py:234`）。客户端**流式中途断连**时：

1. 异步生成器被关闭 → `__aexit__` 释放锁
2. `pump` 线程仍在驱动 engine 跑完 `max_new`（它是 `for piece in gen_factory()`，不会被打断）
3. 下一个请求拿到锁进场 → **两路同时操作同一个 engine**

今天的后果是输出串味。**加了会话池之后后果升级**：快照被写坏，而坏快照会被后续请求
恢复 —— 错误从"一次性"变成"持久化"。所以这一步必须在 P2 之前完成。

### 做法

- 生成收敛到**一个专属 worker 线程**，请求走队列提交，不再每请求起 `threading.Thread`
- 锁改成 worker 持有的 `threading.Lock`，覆盖 `restore → feed → snapshot → decode` 整段
- 断连时设置 cancel flag，`on_text` 回调里检查并中断 decode
- 生成异常时**作废本轮涉及的快照**，宁可 miss 不可用脏状态
- 队列纪律：`BLLM_MAX_QUEUE`（默认 8），超了返回 429 而不是无限等
  （现在是静默排队，客户端只看到超时）

### 待验证 → 已解决

- [x] `streamDecode` 支持中断。两个引擎的 `generate()` 都接受返回 `bool` 的 `on_token`，
      返回 true 即在下一个 decode step 之前干净退出（`native_engine.h:455-459`、
      `native_hybrid_engine.h:231-233`）—— 停止串走的就是这条路。已加
      `NativeLlm::request_cancel()`（原子标志，可从别的线程调）+ `canceled()`。

### 交付内容

- `docker/serving/engine.py` —— 单 worker 线程独占 session，请求经队列提交。
  **这是结构性修复**：HTTP 侧从不持有 engine，所以没有任何 HTTP 事件能提前释放它。
- `docker/serving/cache.py` —— 前缀缓存（纯逻辑，不 import bllm，任意主机可跑单测）
- `docker/server.py` —— 瘦身成纯 OpenAI 协议门面
- 断连时 `engine.cancel(job)`，SSE 侧继续排空直到 worker 报告完成
- `BLLM_MAX_QUEUE`（默认 8），超限返回 429

### 验收结果 ✅

`tests/test_serve_e2e.py::test_repeated_disconnects_do_not_corrupt_later_answers`
反复"流式请求 → 中途断开 → 立刻追问一个确定性问题"5 轮，答案必须逐字不变。

**并已验证该测试能抓到原 bug** —— 把改造前的 `server.py` 从 git 取出来跑在另一个端口：

```
旧实现：'pineapple1. pineapple1.1. pineapple1. pineapple1.1. pineapple1. pineapple'
新实现：'pineapple'
```

被遗弃的生成仍在驱动引擎，上下文串进了下一个请求 —— 正是这条 bug 的直接证据。

---

## P1 — C++ 侧公开接口

### 卡点

服务层需要的原语在 `NativeLlm` 里是 **private**（`include/bllm/native_llm.h:210-212`）：

```cpp
std::string engineSnapshot() const;
void engineRestore(const std::string& blob);
```

现有公开接口都不够用：

| 现有 | 为什么不够 |
|---|---|
| `set_prefix()` / `ask()` | 单个固定前缀，`ask()` 不累积。语义是"多问题打同一份文档"，不是会话 |
| `save_state()` / `load_state()` | 走文件，每轮读写磁盘不可接受 |

### 要加的

```cpp
// native_llm.h —— 从 private 提到 public（实现已有，改可见性）
std::string snapshot() const;
void restore(const std::string& blob);

// 新增：按 token id 续写，绕开"文本→encode"的边界不确定性。
// 拆成两半而不是一个 generate_ids()，这样服务层能在 prefill 完成、decode 开始前
// 精确地打快照 —— 此时状态正好对应完整 prompt。
void feed_ids(const std::vector<int>& ids);
std::string decode_stream(int max_new, const OnText& on_text,
                          const std::vector<std::string>& stop);
```

`encode()` / `decode()` 已经绑定好（`python/_bllm_native.cc:87-88`），不用动。

### 底层机制确认（已在板上复核）

- `writeState` 只写 `D = filled()` 行，快照大小正比于**已用上下文**而非 `cache_len` ✅
  （`include/bllm/native_engine.h:388`）
- `curLogits_` 一并存取，恢复后逐位一致 ✅（`native_engine.h:377`、`:394`）
- **hybrid/SSM 同样成立**：SSM 态是递归摘要，不能截断但可以"恢复快照再追加"。
  本方案只做追加，两种架构统一 ✅
- ✅ `restore()` 内部的全窗口 memset：实测 **3.3 ms**，不是问题，风险撤销。

另外补了两个服务层需要但没绑定的：`context_left`（判上下文溢出）和
`prefill_chunk` / `cache_align()`（对齐规则）。

### 验收结果 ✅ — 但验收标准本身要改

原写的是"restore 后续写与全量 prefill **逐 token 一致**"。这条对 dense **不成立**，
而且不是 bug：分段变了数值就会变（见实测结论第 2 条）。正确的性质拆成两条，
`tests/test_prefix_reuse.py` 分别钉死：

1. **`restore()` 是透明的** —— 恢复快照再追加，等价于"分两次 feed 而从不快照"。
   任意切分点成立（`test_restore_is_transparent`）。这是快照本身的正确性。
2. **对齐的缓存命中等价于不用缓存** —— 只在 `prefill_chunk` 倍数处快照时，
   结果与未缓存路径逐字一致（`test_aligned_reuse_is_exact`）。这是缓存的正确性。

决定性实验：同一切分点下"分两次 feed"与"快照+恢复"输出完全相同（3/3），
证明发散来自分段而非状态处理。

---

## P1.5 — 收益验收（已完成）

### HTTP 层 A/B（dense，同一段 10 轮对话，唯一变量是 `BLLM_CACHE_MAX_MB`）

TTFT（首字延迟，流式测量）—— 总时延会被解码淹没（16 token × ~45ms），
只有 TTFT 才反映缓存作用的那一段：

| 轮次 | prompt | 关缓存 | 开缓存 |
|---|---|---|---|
| 1–5 | <256 tok | 138–153 ms | 138–151 ms（无缓存点，符合预期） |
| 6 | 302 tok | 416 ms | 419 ms（冷，且要打快照） |
| 7–10 | 352–504 tok | **274 ms** | **141 ms** |

**第 10 轮 TTFT 274 → 141 ms，1.94×，降幅 48%。**

### 对照原定验收标准

原定"第 10 轮 TTFT 降幅 ≥ 70%"，**dense 未达标（48%）**，而且这是结构上限，不是实现问题：

- 缓存点只能落在 `prefill_chunk`(256) 的倍数上
- 最后一个对齐点之后的尾巴仍要付一整个 padded chunk（134 ms）
- 所以 504 token 的 prompt 从"2 个 chunk"降到"1 个 chunk"就到头了

ctx 更长的 dense 模型收益会更高（能省的 chunk 更多），但短对话就是这个量级。

hybrid 侧引擎级实测为 **24.95×**（31.6s → 1.3s，见上文实测结论），远超标准；
HTTP 层的 hybrid A/B 未跑（见"未完成事项"）。

## P2 — 前缀缓存与流水线

### 不要用会话 ID

OpenAI 协议是无状态的，客户端每次重发全量 `messages[]`。硬塞 session id 会破坏兼容性
—— Open WebUI / Chatbox 不会发这个字段。

**做法：拿 token 序列本身当 key，匹配即复用，不匹配就退化。**

```python
class PrefixCache:
    """entries: list[(ids: tuple[int,...], blob: bytes, nbytes: int, last_used: float)]"""

    def match(self, ids: list[int]) -> tuple[bytes, int] | None:
        """找出 ids 的最长【完整命中】前缀条目 → (blob, 已覆盖长度)。"""
        best = None
        for e in self.entries:
            n = len(e.ids)
            if n < len(ids) and tuple(ids[:n]) == e.ids:   # 严格前缀，留 ≥1 token 触发 decode
                if best is None or n > best[1]:
                    best = (e.blob, n)
        return best
```

**关键性质：匹配是逐 token 精确比对，猜错不可能。** 任何不一致（客户端改了历史、
重新分词有偏差、换了 system prompt）都只会命中更短的条目或彻底 miss，
退化成今天的全量 prefill —— 永远不会产出错误结果。这让正确性负担降到接近零。

条目数不多时线性扫足够（N ≤ 32，每次比对几百个 int）。真嫌慢再上 trie，**不要一开始就上**。

### 写入时机 —— 两个都存

反正靠匹配验证，存错了只会 miss。

| 存点 | key | 收益 |
|---|---|---|
| prompt prefill 完成、decode 开始前 | 渲染出的 prompt 的 token ids | 稳赢：下一轮 prompt 必然以它为前缀 |
| decode 结束后，补喂 `<\|im_end\|>\n` | 上面 + 回复 + 收尾 | 额外省掉重放助手回复；重编码有偏差时自动 miss，无害 |

### 流水线

替换 `docker/server.py:79` 的 `Engine._stream_sync`：

```python
def _stream_sync(self, prompt, max_new, stop, **samp):
    ids = self.sess.encode(prompt)                    # 真 token，顺便解决 usage
    hit = self.cache.match(ids)

    if hit:
        blob, n = hit
        self.sess.restore(blob)                       # 内存 memcpy，非 BPU
        delta = ids[n:]
        self.stats.hit(n, len(delta))
    else:
        self.sess.reset()
        delta = ids
        self.stats.miss(len(ids))

    self.sess.set_sampling(**samp)
    self.sess.feed_ids(delta)                         # prefill
    self.cache.put(ids, self.sess.snapshot())         # 此刻状态 == 完整 prompt
    yield from self.sess.decode_stream(max_new, stop)
```

### 真 usage

`_approx_tok`（`docker/server.py:273`，4 字符/token 估算）直接删。
`ntok` 现在数的是"文本块"不是 token，一并修。

```python
"usage": {"prompt_tokens": len(ids),
          "completion_tokens": ntok,
          "total_tokens": len(ids) + ntok,
          "prompt_tokens_details": {"cached_tokens": n_hit}}   # OpenAI 标准字段
```

`cached_tokens` 是 OpenAI 官方字段，客户端直接能看到命中率。

### 交付内容 ✅

- `docker/serving/cache.py` —— `PrefixCache`，15 个单测（`tests/test_prefix_cache.py`，
  纯逻辑不依赖板子）
- `docker/serving/engine.py` 的 `_feed_and_cache()` —— 用 `cache_align()` 决定快照点，
  必要时把 prompt 拆成"喂到边界 → 快照 → 喂剩下"两段
- 真 token 计数（`sess.encode()`），`prompt_tokens_details.cached_tokens` 已按
  OpenAI 标准字段上报，客户端直接可见

### 踩到的坑（值得记）

`PrefixCache` 定义了 `__len__`，于是**空缓存是 falsy**，`if self.cache:` 恒假 →
从不插入 → 永远为空，自锁。已加 `__bool__` 恒真 + 三处改成显式 `is not None`。
症状很隐蔽：功能"正常"，只是缓存永不命中。

---

## P3 — 内存治理与可观测

### 内存预算

快照是**宿主内存**（`std::string`），不占 ION。单条大小直接 `len(blob)` 拿，不用估：

```
blob ≈ D × n_layers × (rowK + rowV)  +  vocab × 2
                                        └─ 固定底噪，Qwen 词表 ~152k ⇒ 约 297 KB/条
```

按典型 2B 配置（28 层 / 2 KV 头 / head_dim 128 / int8）粗算约 **14 KB/token**：
400 token 的会话约 5.7 MB + 0.3 MB logits。
⚠️ **这个数是估算，必须在板上实测确认，别照抄。**

### 淘汰策略

- 双上限：`BLLM_CACHE_MAX_MB`（默认 512）、`BLLM_CACHE_MAX_ENTRIES`（默认 16）
- LRU 驱逐，但**长前缀加权** —— 重建 800 token 的快照比 100 token 的贵 8 倍，
  评分用 `last_used + w·len(ids)`
- 每条带 TTL（默认 30 min），避免断线客户端的历史常驻

### 可观测

`/metrics`：命中率、平均复用长度、缓存占用、队列深度、TTFT/TPS 分位数。

### 交付内容 ✅

- 字节预算 + 条数上限 + TTL，淘汰按 `len(ids) / (1 + 空闲秒数)` 打分（长且新鲜的留下）
- 单条超预算直接拒收并计数（`rejected_too_big`），不做无谓抖动 —— hybrid 的 72 MB
  在小预算下就会走这条路
- `GET /metrics`：命中率、token 复用率、缓存字节、队列深度、淘汰/过期计数
- 文档：`docker/README.md`、`docker/USAGE.zh.md`、`docker/USAGE.en.md` 各加一节
  "前缀缓存"，含实测收益表与**什么情况下没有收益**

---

## 运行时版本错配（构建镜像时发现，计划里完全没考虑）

镜像里的 `bllm` 是**从 conda 源装的**，不是本仓源码。所以重建镜像后第一个请求直接 500：

```
'NativeSession' object has no attribute 'cache_align'
```

新服务层依赖的原语（`snapshot`/`restore`/`feed_ids`/`decode_stream`/`cache_align`）
都是本轮才加的，conda 上最新的还是 0.1.4。**服务层比它脚下的运行时新**是常态，
不是意外。

处理方式是**优雅降级**，而不是等发版：

- 启动时检测能力，缺失则关掉缓存、走 `_serve_legacy()`（`reset()` + 从文本 generate，
  即改造前的行为），并在日志里说明缺了什么、怎么启用
- `/metrics` 暴露 `supports_prefix_cache` / `supports_cancel`
- 降级模式下**仍然保留** P0 的 worker 线程修复、队列/429、真 token 计数、metrics

这一点很重要：**P0 的并发修复不需要等新 conda 包就能上线**，缓存则在装上新版 bllm
后自动启用。板上实测降级模式：断连回归测试**通过** —— 即使没有 runtime cancel，
worker 线程的串行化本身就足以阻止并发访问。

## 上线后抓到的两个 bug（都只在 hybrid 上发作）

第一轮"全部完成"的验证是**不充分的**：e2e 只跑了 dense 模型，而这两个 bug 恰好只影响
hybrid —— 也就是收益 25× 的那一类。

**1. hybrid 从不缓存。** `_feed_and_cache` 的守卫写成
`if align <= covered or align >= len(ids): return`。hybrid 不分块 ⇒
`cache_align(n) == n` ⇒ `align >= len(ids)` 恒真 ⇒ **一条都不存**。
修复：抽出纯函数 `snapshot_point()`，`align == n_total` 是 hybrid 的正常情况而非"无事可做"，
并补了 6 个单测钉死（`tests/test_prefix_cache.py`）。

**2. 缓存的位置本身就错了 —— 对所有引擎都错，只是 dense 侥幸躲过。**
`render_chatml` 生成的 prompt 结尾是 `<|im_start|>assistant\n<think>\n\n</think>\n\n`，
而下一轮同样位置渲染成 `<|im_start|>assistant\n{上轮回复}<|im_end|>\n` ——
**第 N 轮的完整 prompt 永远不是第 N+1 轮的前缀**，存了也永不命中。
dense 只是因为 `align` 被截断到 256 的倍数、恰好落在历史内才碰巧能用。

修复：`render_chatml` 改为返回 `(history, opener)`，只有 `history` 可缓存；
服务层把 `cacheable` 上界传给引擎。分开编码两段可能在接缝处分词不同，
所以**验证**它确实是前缀（是特殊 token 边界，但错了会污染缓存，检查很便宜）。

### 修复后的 hybrid HTTP 层 A/B（补上了原先未做的一项）

Qwen3.5-2B ctx4k，容器内，同一段 6 轮对话：

| 轮次 | 关缓存 | 开缓存 |
|---|---|---|
| 1 | 6269 ms | 6280 ms |
| 2 | 11126 ms | 5821 ms |
| 3 | 15991 ms | 5822 ms |
| 6 | **30978 ms** | **5938 ms** |

重点不是第 6 轮的 5.2×，而是 **TTFT 从随历史线性增长变成恒定**。

### 教训

两个 bug 都是"只在一类引擎上发作"，而我只在另一类上做了端到端验证。
`prefill_chunk == 0` 与 `!= 0` 是两条**语义不同的路径**，不是同一条路径的参数差异。

## 部署状态（2026-07-20）

板上 `bllm-serve` 容器，Qwen3.5-2B ctx4k（hybrid），`--restart unless-stopped`：

| 项 | 值 |
|---|---|
| 镜像 | `bllm-serve:qwen3.5-2b-ctx4k-int8-s100p`（内含 conda bllm 0.1.5） |
| 缓存预算 | **2 GB**（`-e BLLM_CACHE_MAX_MB=2048`） |
| 实测容量 | **16 条**独立会话（每条 122 MB），装满占 1957 / 2147 MB |
| 内存余量 | 装满后 `MemAvailable` 仍有 **4.4 GB**（起始 6.3 GB，线性可预测） |
| 能力 | `supports_prefix_cache: true`、`supports_cancel: true` |

调到 2 GB 的原因：默认 512 MB 只装得下 4 条，6 小时 22 个请求就淘汰了 10 次 ——
**并发会话超过 4 个就开始互相踢**。2 GB 下 8 段独立对话全程 0 次淘汰、每段第二轮全部命中。

容量按字节计费而非条数，所以**换 dense 模型不需要重调**：dense 快照约 22 KB/token
（随上下文增长），同样 2 GB 能装的条数要多一个数量级。`BLLM_CACHE_MAX_ENTRIES`
默认 64 在 hybrid 上永远够不着，字节预算先到顶。

再往上加预算是可行的（按当前线性曲线 4 GB 约 32 条、仍余 ~2.4 GB），但**没有实测过**
那个点上 ION 与页缓存的争用，要用先测。

## hybrid 快照瘦身（2026-07-20 完成）

原先每条快照是**恒定**的（0.8b ctx2048 = 72 MB，2b ctx4k = 122 MB），与实际用了多少
上下文无关 —— 等于**用上下文窗口而不是对话长度来限制缓存容量**。

根因：hybrid 的 cache 是混合的，引擎当时把它当作一组不透明缓冲整体 dump。实际分两类：

| 类型 | 形状（0.8b） | 占比 | 能否裁剪 |
|---|---|---|---|
| GDN 递归态（state + conv） | `(16,128,128)` + `(6144,3)` | 30% / 17% | ❌ 固定大小，是全部历史的摘要 |
| 注意力 K/V | `(CL, 2, 256)` —— **首维就是窗口** | 70% / 83% | ✅ 只有已占用的槽有意义 |

按 `stride[0]`（正好是每个位置的字节数，无 padding）只保存已占用的 K/V。
占用部分是**右对齐**的 —— `step()` 的 mask 放行 `[CL_-valid, CL_)`。

**效果**（板上实测）：

| 上下文 | 改前 | 改后 |
|---|---|---|
| 17 token | 72 / 122 MB | **22.1 MB** |
| 136 token | 72 / 122 MB | **25.0 MB** |
| 增长率 | — | +24.6 KB/token |

2b-ctx4k 在 136 token 处省 **80%**；2 GB 预算从 16 条会话变成 **~66 条**。

格式变了，magic `BLKH` → `BLKI`，旧的 `save_state()` 文件会被明确拒绝而不是误读。

正确性由 `test_hybrid_reuse_is_exact_at_any_split` 守住（裁错位置会直接改变输出），
另加 `test_hybrid_snapshot_grows_with_context_not_with_the_window` 防止有人改回全量 dump。

## 风险 —— 结算

| 原列风险 | 结果 |
|---|---|
| `restore()` 的全窗口 memset 吃掉收益 | ❌ 未发生，3.3 ms |
| hybrid 快照体积与 dense 差异大 | ✅ 发生了，且比预想严重：**恒定 72 MB/条**，不随长度缩小 |
| 收益依赖客户端行为 | ✅ 成立，已写进三份用户文档 |
| （未预见）**分段敏感性** | ⚠️ 最重要的一条：不对齐的缓存点会改变回答。已由 `cache_align()` 挡住 |

## 已收尾的原未完成事项

- [x] **hybrid 的 HTTP 层 A/B** —— 修复后在容器上实测，TTFT 从随历史线性增长
      （6.3 s → 31.0 s）变为恒定 ~5.9 s。
- [x] **`kPrefillMinTokens` 调优** —— 16 → 4。盈亏点在三个模型上实测
      2.93 / 3.31 / 2.92，稳定（两者都是一次 BPU 图调用）。15 token 的轮次
      676 ms → 135 ms。会改变 4~15 token 轮次的输出，已额外验证两个模型的多轮连贯性。
- [x] **服务层拆分** —— `docker/serving/`（`cache.py` + `engine.py`），两个 Dockerfile
      的 `COPY` 已同步。
- [x] **发版** —— conda 0.1.5 已发布，容器已用它重建。

## 仍未做

- [ ] **VLM / Omni 不支持**。embed 模式不能快照（`native_engine.h:365`），
      多模态服务仍是每轮全量重算。这是本方案边界内的已知空白。
- [ ] **真并发仍然没有**。单 BPU 图一次一个生成；本方案只解决重复计算，不解决吞吐。
      要吞吐得做 continuous batching。
- [ ] **dense 收益封顶 ~2×**，低于原定的 70% 目标。结构上限（chunk 粒度），
      不是实现问题；hybrid 的 TTFT 恒定化才是本方案的主要价值。
- [ ] **>2 GB 缓存预算未实测**（ION 与页缓存的争用）。

## 后续可做（未排期）

- **缓存预算自适应**：按 `MemAvailable` 和单条快照实测大小推荐一个上限，
  而不是让用户凭感觉填 MB 数 —— 现在 hybrid 与 dense 差 5 倍以上，默认值必然对一边不合适。
- ~~**hybrid 快照瘦身**~~ —— **已完成**，见下节。
