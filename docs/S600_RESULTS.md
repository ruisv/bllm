# BLLM on RDK S600 — 性能实测

BLLM 自研 native runtime 在 **RDK S600**（Nash BPU，4 BPU 核）上的端侧 LLM 推理实测。
表中模型均为 BLLM 手写的 leap 图（Gated-DeltaNet 线性注意力 + 真注意力混合、静态 int8 量化），
全量权重、**100% 落 BPU**（`CPU native ops == {}`）——不是官方 OE-LLM 预置模型。数据为板端真机
实测，单请求、单线程。

## 多核 BPU

S600 有 4 个 BPU 核，BLLM 的 `.hbm` 可在离线编译时指定并行核数（1 / 2 / 4），运行时绑定到对应
核并行执行一次 decode。

- **大模型用满 4 核收益明显**：每层算量大到足以摊薄核间同步开销。
- **小模型单核反而更快**：同步开销大于并行收益，多核得不偿失。

模型目录用 `model.json` 的 `bpu_cores` 声明编译核数；`make_model_dir.py --bpu-cores N` 生成。

## 实测性能（板端真机，单请求）

| 模型 | 参数量 | 结构 | BPU 核 | Decode 吞吐 | 单 token 延迟 |
|---|---|---|---|---|---|
| **Qwen3.5-4B** | 4B | H2560 / 32 层 | 4 | **27.0 tok/s** | 37 ms |
| **Qwen3.5-0.8B** | 0.8B | H1024 / 24 层 | 1 | **42.7 tok/s** | 23 ms |

- 均为 Qwen3.5 混合 Gated-DeltaNet/SSM 架构，全量权重、静态 int8、全算子落 BPU。
- Qwen3.5-4B（4.29 GB）用 4 BPU 核并行，充分发挥 S600 多核算力。
- Qwen3.5-0.8B（784 MB）单核即高吞吐，适合低延迟 / 多实例场景。

## 说明

- 吞吐为板端 decode 阶段实测，单请求、单线程（`hrt_model_exec perf --thread_num 1`，多核加
  `--core_id`）。
- 具体数值随上下文长度、量化位宽、采样配置略有变化。
