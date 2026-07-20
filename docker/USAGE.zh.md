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
| `BLLM_CACHE_MAX_MB` | `512` | 前缀缓存预算(MB),`0` 关闭缓存 |
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

**内存**:按字节计费,不按条数 —— dense 快照约 22 KB/token 且随上下文增长,
hybrid 则是**恒定约 72 MB/条**(整套 cache 全量 dump)。所以 512 MB 预算在 dense 上
能存几十条,在 hybrid 上只有 7 条。内存紧张就调小 `BLLM_CACHE_MAX_MB`,或设 `0` 关掉。

**预算怎么设**:按字节计费,不按条数。hybrid 单条快照是**恒定**的(Qwen3.5-2B ctx4k 实测
122 MB),所以默认 512 MB 只装得下 **4 条会话** —— 并发超过 4 个就开始互相淘汰。
实测把 `BLLM_CACHE_MAX_MB` 调到 2048 后可稳定容纳 **16 条**,占用 1957 MB,
8 GB 板子上装满后仍余 4.4 GB:

```bash
docker run -d --name bllm-serve --network host --restart unless-stopped \
  -e BLLM_CACHE_MAX_MB=2048 \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu <image>
```

dense 模型不需要这么大:快照约 22 KB/token 且随上下文增长,同样预算能装的条数多一个数量级。
`GET /metrics` 的 `evictions` 持续增长就是预算偏小的信号。

> **hybrid 快照大小**:自 0.1.6 起,快照只保存已占用的注意力 K/V,大小随对话增长
> (约 22 MB 起 + 24.6 KB/token),不再是与上下文窗口绑定的固定值。此前 Qwen3.5-2B ctx4k
> 每条恒为 122 MB,现在 136 token 的会话仅 25 MB —— **同样 2 GB 预算从 16 条变约 66 条**。

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
