<!-- 中文；English: README.md -->

# bllm-serve —— 容器化部署

BLLM native 运行时的 OpenAI 兼容 HTTP 推理服务，打包成 Docker 镜像，面向 RDK
**S100 / S100P / S600**（仅 aarch64）。

## 设计

这是一个**设备耦合**的容器（NVIDIA 那种模式），不是隔离的普通应用：

| 层 | 放在哪 | 原因 |
|-------|----------------|-----|
| ubuntu:22.04 + py3.12 + `bllm`/`libbllm`/`tokenizers-cpp`（conda）+ 我们的 server | **烤进镜像** | 我们自己的栈，与板子 OS 弱耦合 |
| BPU 用户态 `.so` 闭包（`libdnn`/`libhbucp`/… → `/usr/hobot/lib`） | **烤进镜像**（构建时从板子上暂存过来） | 自包含；与板上路径一致，RPATH 才能解析。conda 包不带 hobot 运行时，所以这是镜像里唯一的一份，而且就是板子自己的那份 |
| 内核 BPU 驱动 + `/dev/{bpu,ion,…}` 设备节点 | **从板子映射**（`--device`） | 驱动在主机内核里，必须与正在运行的板子匹配 |
| `.hbm` 模型 + tokenizer/`model.json` | **卷挂载**（`-v …:/models`） | GB 级，每次部署自己选 |

基础镜像用 `ubuntu:22.04`，因为 D-Robotics 的 BPU 库硬性要求
**GLIBC_2.34 + GLIBCXX_3.4.30** —— glibc-2.28 的基础镜像（Debian 10）加载不了它们。

## 构建 + 运行（在板上，用 docker compose）

所有东西都在这个 `docker/` 目录里 —— 命令都在这里执行。一次性准备：把板上
`/usr/hobot/lib` 里的闭源 BPU 库闭包暂存到 `_stage/`（已 gitignore）—— compose 会把它打进镜像，
但它没法拷贝主机上的文件：

```bash
cd docker
./stage-libs.sh
```

然后构建 + 启动：

```bash
MODEL_DIR=~/models/qwen3.5-2b docker compose up -d --build
docker compose logs -f          # follow startup / model load
docker compose down             # stop + remove
```

`compose.yaml` 同时设了 `network_mode: host` **和** `build.network: host` —— 这块板子的内核
缺 iptables 的 `raw` 表，Docker 默认的 bridge 网络在构建和运行时都会失败。BPU/ION 设备节点和
模型卷写在 compose 文件里；按部署调整的参数走环境变量（`MODEL_DIR`、`BLLM_PORT`、
`BLLM_API_KEY`、…）或 `compose.yaml` 旁边的 `.env` 文件。

### 自包含镜像（模型烤进去）

要做一个可分发的"直接 `docker run` 就能对话"的交付物，用 `bake-model.sh` 把模型烤进
镜像（先构建与模型无关的基础镜像，再把模型叠在上面一层）。按交付命名规范
`<name>:<model>-<ctx>-<quant>-<board>` 打 tag（各段从 `model.json` 里读）：

```bash
cd docker
./bake-model.sh ~/models/qwen3.5-2b bllm-serve:qwen3.5-2b-ctx512-int8-s100p
docker run -d --name bllm-serve --network host \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu \
  bllm-serve:qwen3.5-2b-ctx512-int8-s100p     # no volume, no MODEL_DIR
```

烤好的镜像 = 基础镜像（~760 MB）+ 模型包（Qwen3.5-2B ≈ 2.8 GB，含
`embed_tokens.bin`），所以 push / `docker save` 约 3.5 GB。`docker image ls` 里显示的磁盘占用
会更大，因为 containerd 存储同时保留了压缩 blob 和解开后的快照。运行时仍然需要 BPU/ION
设备直通。

### 交付镜像

| Tag | 内容 | push / `save` 大小约 |
|-----|----------|-------------------|
| `bllm-serve:0.1.2`（+ `latest`） | 通用运行时 —— 模型走卷挂载 | ~760 MB |
| `bllm-serve:qwen3.5-0.8b-ctx2k-int8-s100p` | + 烤入 Qwen3.5-0.8B | ~2.1 GB |
| `bllm-serve:qwen3.5-2b-ctx512-int8-s100p` | + 烤入 Qwen3.5-2B | ~3.5 GB |

通用镜像就是烤模型镜像所叠的那层基础（`bake-model.sh` 把它构建为
`bllm-serve:latest`；再按 bllm 版本打 tag，如 `bllm-serve:0.1.2`）。

### API

```bash
curl localhost:8866/health
curl localhost:8866/v1/models
curl localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
curl -N localhost:8866/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"数到5"}],"stream":true}'
# speech recognition, when BLLM_ASR_MODEL is set (OpenAI's transcription API)
curl localhost:8866/v1/audio/transcriptions -F file=@question.wav -F language=zh
```

OpenAI 兼容的 `/v1/chat/completions`（流式 + 非流式；对带对话模板家族的模型支持工具调用和
`reasoning_content` —— 见 `USAGE.zh.md`）、`/v1/audio/transcriptions`
（一个官方 Qwen3-ASR 包，与对话模型并列 —— 或者代替它）、`/v1/models`、
`/health`。引擎是**单例** —— 同一时刻只有一个生成在跑（BPU = 单条 prefill/decode 图）；
请求串行排队。只跑**一个** uvicorn worker。

### 环境变量

| 变量 | 默认 | 含义 |
|---------|---------|---------|
| `BLLM_MODEL` | — | 对话模型：一个目录（含 `model.json`）、一个官方包，或一个裸 `.hbm` |
| `BLLM_ASR_MODEL` | — | 一个官方 Qwen3-ASR 包：提供 `/v1/audio/transcriptions`（两者至少设一个） |
| `BLLM_TOKENIZER_DIR` | — | tokenizer 目录，仅当 `BLLM_MODEL` 是裸 `.hbm` 时需要 |
| `BLLM_BACKEND` | `auto` | `bllm.load` 的 backend |
| `BLLM_MAX_NEW` | `1024` | 请求没给 `max_tokens` 时的默认值 |
| `BLLM_API_KEY` | — | 设了就要求 `Authorization: Bearer <key>` |
| `BLLM_BPU_PRIORITY` | — | 启动时调用 `set_bpu_priority()` |
| `BLLM_MAX_QUEUE` | `8` | 排队请求数上限，超了服务返回 429（单 BPU 图一次只跑一个生成） |
| `BLLM_CACHE_MAX_MB` | `auto` | 前缀缓存预算（MB），或 `auto`（`MemAvailable` 的 40%，256 MB–4 GB）；`0` 关闭 |
| `BLLM_CACHE_TTL_S` | `1800` | 缓存的前缀这么久没用就丢弃 |
| `BLLM_CACHE_MAX_ENTRIES` | `64` | 与字节预算并行的条数上限 |
| `BLLM_PERF_MODE` | `0` | `1` = 戳 BPU 性能寄存器（需 `--privileged`；更推荐在主机上做） |
| `BLLM_CORS_ORIGINS` | `*` | 允许的 CORS 源（逗号分隔）；默认全开给浏览器客户端 |
| `BLLM_PORT` / `BLLM_HOST` | `8866` / `0.0.0.0` | 监听地址 |


### 前缀缓存（多轮对话免重算）

OpenAI 协议是无状态的 —— 客户端每轮都重发完整的 `messages[]` —— 所以朴素的服务端每轮都要把
整段历史重新 prefill 一遍。本服务把引擎快照按**它覆盖的 token 序列**做键缓存起来，只 prefill
新增的部分。每个请求的命中量在 `usage.prompt_tokens_details.cached_tokens` 里；
汇总统计在 `GET /metrics`。

匹配是逐 token 的精确前缀比对，所以**不可能错用**：改过的历史、换过的 system prompt 或
分词不同，都只会命中更短的条目或者彻底不命中，退回全量 prefill。

**收益取决于引擎**（S100P 实测，客户端每轮重发历史）：

| 引擎 | 模型 | 未命中 → 命中 | 加速 |
|---|---|---|---|
| dense | Qwen2.5-1.5B | 271ms → 138ms | ~2x |
| hybrid | Qwen3.5-0.8B | 31.6s → 1.3s | ~25x |

hybrid 一行是在不带分块 prefill 图的 Qwen3.5 包上测的：逐 token 喂约 68ms/token，
512 token 的 prompt 要 35 秒。现在发布的 Qwen3.5 包走 prefill 图摄入，未命中的代价小得多，
收益也比表里小；但命中仍省掉整段重新 prefill，**别关掉它**。

**什么时候没有收益**（正常退化，不是故障）：

- 每次都换新话题的单轮问答 —— 没有可复用的前缀
- 会裁剪或摘要历史的客户端（部分 Web UI 会）—— 前缀一变，每个条目都不命中
- 短于一个 prefill chunk 的 prompt（dense 上通常是 256 token）—— 没有安全的快照点，
  不过这种 prompt 重新 prefill 本来就便宜


> **注意**：镜像里的 `bllm` 从 conda 源安装。如果那个版本早于快照原语，服务会
> **降级为全量重新 prefill**（启动时会说明）：没有前缀缓存，但并发修复、准入控制、真实的 token
> 用量和 `/metrics` 照样有效。看 `GET /metrics` 上的 `supports_prefix_cache`。

**缓存预算**：按字节计，不按条数 —— 一份快照的大小在两种引擎之间差一个数量级，
而且随对话增长。dense 约 22 KB/token；hybrid 带一个约 22 MB 的 GDN 递归状态底座，
之上再加约 24.6 KB/token（Qwen3.5-2B ctx4k 实测：136 token 的会话约 25 MB，典型的两轮对话约 30 MB）。

**默认就是 `auto`，通常不用调**：不设 `BLLM_CACHE_MAX_MB` 时，服务在启动时取
`MemAvailable` 的 40%（下限 256 MB，上限 4 GB），并把选定的值写进日志。在 8 GB 的板子上
取到约 2.5 GB，能容纳 **80 条以上**会话。

几个服务或容器共用一块板子时这一点最要紧 —— 各自挑一个看上去合理的固定值，加起来就会超过
整机内存（测试中两个进程分别设 2 GB 和 3 GB，把 `MemAvailable` 压到了 2.4 GB）。要覆盖就显式
给一个 MB 值，或者 `0` 关闭缓存：

```bash
docker run -d --name bllm-serve --network host --restart unless-stopped \
  -e BLLM_CACHE_MAX_MB=1024 \
  --device /dev/bpu --device /dev/bpu_core0 --device /dev/ion \
  --device /dev/ipcdrv --device /dev/dcore0_rpmsg_bpu <image>
```

`GET /metrics` 上的 `evictions` 持续上涨说明预算不够；`max_bytes` 显示实际生效的值。

### 从浏览器 / 现有客户端对话

镜像不带 UI —— 接口是 OpenAI 兼容的，并且已开 CORS，把任意现有客户端指向
`http://<board-ip>:8866/v1` 即可：

- **[chatbox-lite](https://github.com/lfbear/chatbox-lite)** —— 单个 HTML
  文件，完全在客户端运行；添加一个 "OpenAI-compatible" provider，base URL 填
  `http://<board-ip>:8866/v1`（除非设了 `BLLM_API_KEY`，API key 随便填）。
- 桌面版 **Chatbox**、**Open WebUI**、**LibreChat**，或 OpenAI SDK
  （`base_url="http://<board-ip>:8866/v1"`）用法都一样。

要锁定部署，设 `BLLM_API_KEY` 并收窄 `BLLM_CORS_ORIGINS`。

## 主机前提（一次性，在容器外）

- **ION carveout**：`sudo /usr/hobot/bin/hb_switch_ion.sh balanced && sudo reboot`
  （7B 用 `bpu_first`）。这是开机时的内核 carveout —— 在容器里设不了。
- **Docker**：如果守护进程在 `iptables (nf_tables)` 上报错，切到 legacy
  后端：`sudo update-alternatives --set iptables /usr/sbin/iptables-legacy`
  （这个内核只带 legacy xtables 路径）。

## 何时需要重新构建

板子的 **BSP / BPU 驱动**升级后，重新执行 `./stage-libs.sh && docker compose up -d --build`
（在 `docker/` 下）—— 镜像里打包的 `/usr/hobot/lib` 闭包必须与正在运行的内核驱动 ABI 匹配。
