# bllm-serve 容器镜像 · 使用指南

`bllm-serve` 是把 BLLM native 运行时封装成 **OpenAI 兼容 HTTP 服务**的容器镜像,
面向 D-Robotics RDK **S100 / S100P / S600**(仅 aarch64)。拿到镜像即可
`docker run` 起一个本地大模型服务,用任意 OpenAI 客户端对话。

> 这份是**使用**说明。要看镜像怎么构建 / 设计取舍,见 [`README.md`](README.md)。

---

## 0. 前提:这是设备耦合镜像

镜像里**自包含**了我们的运行时栈 + BPU 用户态 `.so` 闭包;但内核 BPU 驱动、设备
节点、`.hbm` 模型不在镜像里,运行时从板子挂进去:

| 东西 | 在哪 |
|---|---|
| 运行时栈(python + `bllm`/`libbllm` + server)、BPU 用户态库 | ✅ 镜像内自带 |
| 内核 BPU 驱动 + `/dev/{bpu,ion,…}` | 板子内核 → `--device` 映射 |
| `.hbm` 模型 + tokenizer + `model.json` | 交付镜像已烤进去;通用镜像用 `-v` 挂载 |

所以**镜像只能在对应板子上跑**(要匹配板上的 BSP/驱动),不是跨机器可移植的普通应用镜像。

---

## 1. 一次性主机准备(容器外,做一次)

在**板子**上执行,不是在容器里:

```bash
# ① 放大 ION carveout(LLM 必须,否则运行时 alloc 时段错误),之后重启
sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot     # ≤3B 用 balanced;7B 用 bpu_first

# ② 若 Docker 守护进程报 iptables(nf_tables)错,切到 legacy(此板内核只有 legacy xtables)
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy

# ③(可选)开 BPU 性能模式,decode 更快 —— 在主机做一次即可
sudo /usr/bin/busybox devmem 0x2b047000 32 0x99
sudo /usr/bin/busybox devmem 0x2b047004 32 0x99
```

---

## 2. 载入镜像

交付通常是一个 `docker save` 出来的 tar:

```bash
docker load -i bllm-serve-qwen3.5-2b-ctx4k-int8-s100p.tar
docker image ls bllm-serve         # 确认 tag
```

镜像 tag 遵循交付命名 `bllm-serve:<模型>-<上下文>-<量化>-<板子>`,例如
`bllm-serve:qwen3.5-2b-ctx4k-int8-s100p` —— 一眼看清模型 / 上下文长度 / 量化 / 目标板。

---

## 3. 运行

### A. 交付镜像(模型已烤进去)—— 直接跑,不用挂模型

```bash
docker run -d --name bllm-serve --network host \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu \
  bllm-serve:qwen3.5-2b-ctx4k-int8-s100p
```

- `--network host` 是**必须**的(此板内核缺 iptables `raw` 表,默认 bridge 起不来)。
- 五个 `--device` 把板上 BPU/ION 节点映射进容器,一个都不能少。

### B. 通用镜像(模型用卷挂载)

镜像不带模型,把板上模型目录挂到 `/models`:

```bash
docker run -d --name bllm-serve --network host \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu \
  -v ~/models/qwen3.5-2b:/models:ro \
  -e BLLM_MODEL=/models \
  bllm-serve:latest
```

`~/models/qwen3.5-2b` 是一个**模型包目录**(含 `model.json` + `.hbm` + tokenizer +
`embed_tokens.bin`)。也可直接指向裸 `.hbm`,此时再加 `-e BLLM_TOKENIZER_DIR=...`。

### C. docker compose(在 `docker/` 目录里)

```bash
cd docker
MODEL_DIR=~/models/qwen3.5-2b docker compose up -d
docker compose logs -f          # 跟随启动 / 模型加载
docker compose down             # 停止 + 删除
```

设备节点、卷、健康检查都写在 `compose.yaml`,按部署改的旋钮走环境变量或 `.env` 文件。

---

## 4. 验证服务起来了

模型加载要几秒到几十秒(看模型大小),`/health` 通了才算就绪:

```bash
curl localhost:8866/health          # {"status":"ok",...}
curl localhost:8866/v1/models       # 列出当前模型 id
```

首次没通别急,`docker logs -f bllm-serve` 看加载进度。

---

## 5. 对话(OpenAI 兼容接口)

### 命令行

```bash
# 非流式
curl localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好,介绍下你自己"}]}'

# 流式(SSE,逐 token)
curl -N localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"从1数到5"}],"stream":true}'
```

支持 `/v1/chat/completions`(流式 + 非流式)、`/v1/models`、`/health`。

**采样字段**:`temperature`(默认 `0`,即贪心)、`top_p`、`presence_penalty`、
`frequency_penalty`、`logit_bias`、`logprobs` / `top_logprobs`、`seed`、
`max_tokens` / `max_completion_tokens`、`stop`、`stream`。
另外接受几个非 OpenAI 标准但通用的扩展:`top_k`、`min_p`、`repetition_penalty`,
以及 `enable_thinking`(Qwen3.5 思考开关,默认 `false`)。没给的字段一律取默认值,
显式传 `0` 与不传是有区别的(`temperature: 0` = 贪心)。

**结构化输出**:`response_format` 与 `grammar` 二选一(同时给是 400 —— 与其定一条
没人记得住的优先级,不如直接报错)。

```jsonc
{"response_format": {"type": "json_object"}}                    // 任意合法 JSON
{"response_format": {"type": "json_schema",
                     "json_schema": {"name": "city", "schema": {...}}}}  // 指定结构
{"grammar": "root ::= \"yes\" | \"no\"\n"}                    // 原始 GBNF(llama.cpp 的字段)
```

这不是"提示模型请输出 JSON",而是**解码期约束**:每一步只有仍能满足语法的 token 能
被采样,所以 `json.loads()` 不会失败。代价实测约 6%(14.3 → 13.4 tok/s)加首次约 315 ms
建表。语法/schema 有问题一律 400,不会静默降级成自由生成——客户端要的是保证。
schema 支持见 API 文档的 `json_grammar`。

`logit_bias` 用 `{"token id": 偏置}`(OpenAI 的 [-100, 100]),超范围直接 400,
不会悄悄截断。`logprobs: true` 时每个 token 回一条对数概率,`top_logprobs: N`
(0..20)再附 N 个备选。给的是**原始**模型分布(全词表精确 log-softmax),不随
temperature / 惩罚 / `logit_bias` 变——所以生成的 token 未必是 `top_logprobs[0]`,
好处是不同采样设置之间可比。**流式**时对数概率**整批挂在最后一个 chunk 上**:
delta 是解码后的文本片段(停止串的回退会合并/延后 token),没有诚实的逐 chunk 对齐;
按惯例把各 chunk 的 `logprobs.content` 拼起来的客户端拿到的数组完全正确。

不传 `seed` 时每个请求重新取种子,所以 `temperature` 会真的产生变化;传了 `seed` 则
相同请求逐字节可复现(OpenAI 的 `seed` 语义)。

### 图文(VLM)

`BLLM_MODEL` 指向一个带 `visual` 塔的目录(见 `docs/MODELS.zh.md` 的 hybrid 打包一节)
时,`/v1/chat/completions` 接受 OpenAI 的 `image_url` content part —— base64 内联或
`http(s)://` 直链都行:

```bash
curl localhost:8866/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "这张图里有什么?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K..."}}
  ]}]
}'
```

`/metrics` 的 `is_vlm` 字段说明当前模型是不是图文包。指向纯文本目录时收到
`image_url` 会直接 400(而不是默默丢图后一本正经地瞎编)。

多轮追问同一张图不会重跑视觉塔:引擎自己记着已经喂进当前会话的历史,续问只是在
末尾追一轮。历史被编辑、换了张图,或两个不相关的对话交替打进来,就整段重来一遍
——答案总是对的,只是这种情况下没有省下重算的收益(图文会话是单进程单会话,没有
纯文本那套 ids 级前缀缓存可用,靠的是回放而不是快照/回滚)。视频输入目前不支持
(Qwen3.5 用时间戳 token 分帧,和 Omni 的 TMRoPE 不是一回事)。

### OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://<板子IP>:8866/v1", api_key="任意")  # 未设 BLLM_API_KEY 时 key 随便填
r = client.chat.completions.create(
    model="bllm", messages=[{"role": "user", "content": "你好"}])
print(r.choices[0].message.content)
```

### 图形客户端(浏览器/桌面)

镜像不带 UI,但接口是 OpenAI 兼容且已开 CORS,任意现成客户端指向
`http://<板子IP>:8866/v1` 即可:

- **[chatbox-lite](https://github.com/lfbear/chatbox-lite)** —— 单个 HTML 文件,纯前端,加一个
  "OpenAI-compatible" provider、base URL 填上即可。
- 桌面 **Chatbox** / **Open WebUI** / **LibreChat** 同理。

> ⚠️ 引擎是**单例**:BPU 同一时刻只有一条 prefill/decode 图,**一次只处理一个生成请求**,
> 并发请求会排队。这是硬件特性,别指望多路并发吞吐。

---

## 6. 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `BLLM_MODEL` | 交付镜像已内置 | 模型目录(含 `model.json`)或裸 `.hbm` 的容器内路径 |
| `BLLM_TOKENIZER_DIR` | — | 仅当 `BLLM_MODEL` 是裸 `.hbm` 时需要 |
| `BLLM_MAX_NEW` | `1024` | 请求没给 `max_tokens` 时的默认生成上限 |
| `BLLM_API_KEY` | — | 设了就强制 `Authorization: Bearer <key>` |
| `BLLM_CORS_ORIGINS` | `*` | 允许的 CORS 源(逗号分隔),默认全开给浏览器客户端 |
| `BLLM_BPU_PRIORITY` | — | 启动时设 BPU 优先级 |
| `BLLM_MAX_QUEUE` | `8` | 排队请求数上限,超了返回 429(单 BPU 图一次只跑一个生成) |
| `BLLM_CACHE_MAX_MB` | `auto` | 前缀缓存预算(MB),或 `auto`(取 `MemAvailable` 的 40%,256 MB–4 GB);`0` 关闭 |
| `BLLM_CACHE_TTL_S` | `1800` | 缓存前缀多久没用就丢弃(秒) |
| `BLLM_CACHE_MAX_ENTRIES` | `64` | 与字节预算并行的条数上限 |
| `BLLM_PERF_MODE` | `0` | `1` = 容器内戳 BPU 性能寄存器(需 `--privileged`;更推荐在主机做) |
| `BLLM_PORT` / `BLLM_HOST` | `8866` / `0.0.0.0` | 监听地址 |


### 前缀缓存(多轮对话免重算)

OpenAI 协议是无状态的:客户端每轮都重发完整 `messages[]`,服务端默认会把整段历史
重新 prefill 一遍。本服务把引擎状态按**它覆盖的 token 序列**做快照缓存,下一轮只
prefill 新增部分。命中情况在返回的 `usage.prompt_tokens_details.cached_tokens` 里,
整体统计看 `GET /metrics`。

匹配是逐 token 精确前缀比对,**猜错不可能**:客户端改了历史、换了 system prompt、
分词有出入,都只会命中更短的条目或彻底不命中,退化成全量 prefill,不会给出错误答案。

**收益因引擎而异**(S100P 实测,客户端每轮重发全历史):

| 引擎 | 模型 | 未命中 → 命中 | 倍数 |
|---|---|---|---|
| dense | Qwen2.5-1.5B | 271ms → 138ms | ~2× |
| hybrid | Qwen3.5-0.8B | 31.6s → 1.3s | ~25× |

hybrid(Qwen3.5)没有批量 prefill 图,逐 token 喂 ~68ms/token,512 token 的 prompt
要 35 秒 —— **对 hybrid 模型这不是优化,是能不能用的问题,别关**。

**什么时候没有收益**(属正常退化,不是故障):

- 单轮问答,每次换新话题 —— 没有可复用的前缀
- 客户端会裁剪/摘要历史(部分 Web UI 会) —— 前缀一变全部落空
- prompt 短于一个 prefill chunk(dense 上通常 256 token) —— 没有安全的快照点,
  但这种 prompt 重算本来就便宜


> **注意**:镜像里的 `bllm` 来自 conda 源。若该版本尚不含快照原语,服务会自动
> **降级为每轮全量重算**(启动日志会说明),此时前缀缓存不生效,但并发修复、队列
> 限流、真 token 计数、`/metrics` 仍然可用。`GET /metrics` 的 `supports_prefix_cache`
> 字段可直接确认当前状态。

**缓存预算**:按字节计费,不按条数 —— 单条快照的大小在两种引擎之间差一个数量级,
且随对话增长。dense 约 22 KB/token;hybrid 带一个约 22 MB 的 GDN 递归态底座,
之上再按 24.6 KB/token 增长(Qwen3.5-2B ctx4k 实测:136 token 的会话约 25 MB,
典型两轮对话约 30 MB)。

**默认是 `auto`,通常不需要动**:不设 `BLLM_CACHE_MAX_MB` 时,服务按启动时
`MemAvailable` 的 40% 自动取值(下限 256 MB、上限 4 GB),并把选定值打进启动日志。
8 GB 板子上实测自选约 2.5 GB,可容纳 **80 条以上**会话。

多个服务/容器共存时这一点尤其重要 —— 各自填一个"看起来合理"的固定值,加起来就会
超过整机内存(实测两个进程分别设 2 GB 和 3 GB 时,`MemAvailable` 掉到 2.4 GB)。
要覆盖就显式给个 MB 数,`0` 关闭缓存:

```bash
docker run -d --name bllm-serve --network host --restart unless-stopped \
  -e BLLM_CACHE_MAX_MB=1024 \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu <image>
```

`GET /metrics` 的 `evictions` 持续增长就是预算偏小的信号;`max_bytes` 是当前实际生效值。

**锁定部署**:设 `BLLM_API_KEY` 并把 `BLLM_CORS_ORIGINS` 收窄到你的前端域名。

---

## 7. 运维

```bash
docker logs -f bllm-serve           # 看日志
docker restart bllm-serve           # 重启
docker stop bllm-serve && docker rm bllm-serve   # 停止 + 删除
docker stats bllm-serve             # 资源占用
```

换模型:换成对应交付镜像重跑,或(通用镜像)改 `-v` 挂载 + `BLLM_MODEL` 重跑。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 启动即退出,日志报 alloc / 段错误 | ION carveout 没放大 → 跑第 1 步 `hb_switch_ion.sh balanced` 后**重启** |
| `docker run` 报 iptables / 网络失败 | 必须 `--network host`;守护进程层面切 `iptables-legacy`(第 1 步 ②) |
| 找不到 `/dev/bpu` 等设备 | 板上 BPU 驱动没加载 / 节点名不同;`ls /dev/ | grep -E 'bpu|ion'` 核对 |
| 加载报 `.so` 找不到符号 / GLIBC | 板子 BSP/驱动升级了,镜像里的 `/usr/hobot/lib` 闭包 ABI 不匹配 → 需重新构建镜像 |
| decode 偏慢 | 主机开 BPU 性能模式(第 1 步 ③);上下文越长 decode 越慢是 BPU 静态图特性 |
| 上下文超了报错 | 全注意力模型越过 KV 窗口会**抛异常**而非丢 token;选更大 ctx 的交付镜像(如 ctx4k) |
| 回答写一半被截断 | 命中生成上限(默认 `BLLM_MAX_NEW=1024`);调大请求的 `max_tokens`,或起容器时 `-e BLLM_MAX_NEW=2048`。返回里 `finish_reason:"length"` 即表示是长度截断(`"stop"` 才是自然结束)。注意 prompt+生成要一起装进 ctx 窗口 |
