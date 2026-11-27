<!-- 中文；English: CONTRIBUTING.md -->

# 参与 BLLM 贡献

感谢关注——欢迎各种贡献：bug 报告、修复、新模型接入，或文档改进。提交贡献即表示你同意
你的贡献按本项目的 [Apache License 2.0](LICENSE) 授权。

## 唯一的硬约束：在板上构建

BLLM 链接 D-Robotics 的 **hobot 运行时**（`hbDNN` / `hbUCP` / `HBRT`），它**只存在于 RDK
S100 / S100P / S600 硬件上**：随板子的系统镜像提供（`/usr/include/hobot`、`/usr/hobot/lib`），
BLLM 链接的就是这一份，自己不打包。因此：

- **C++ 库、Python 扩展和板端测试必须在 S100 / S100P / S600 上构建并运行**（aarch64、
  Ubuntu 22.04、GCC 11、CMake ≥ 3.22）。
- **不要在非板子的主机上尝试编译。** 编辑、代码审查和纯元数据的单元测试
  （`tests/test_modelmeta.py`、`tests/test_backend_routing.py` 以及板外 C++ 测试）在任何地方
  都能做；凡是碰到运行时的都不行。

典型的循环是"在工作站上编辑，同步到板子，在板上构建与测试"。

## 项目结构

```
include/bllm/, src/   the native runtime (header-heavy engines + model.json parsing)
  native_engine.h        dense KV engine over hbDNN
  native_hybrid_engine.h Gated-DeltaNet / SSM engine (Qwen3.5)
  native_vlm.h           multimodal (Qwen2.5-Omni: text + image + audio + video)
  native_llm.h           unified string-in/out session (tokenizer + ChatML + streaming)
  native_sampler.h       sampling (temp / top-k/p / min-p / typ-p / penalties)
  model_config.h         the model.json manifest
python/        nanobind bindings + the pure-Python `bllm` package
examples/      standalone C++ programs (bllm_chat_native, bllm_vlm_native, …)
tests/         pytest + off-board C++ tests
scripts/       sync / build / test / make_model_dir
cmake/         FindHobot / FindTokenizers / package config
```

## 构建与测试

**在板上**克隆并构建（依赖安装见 README 的"从源码构建"一节），然后在板上跑测试：

```bash
scripts/build.sh --python          # cmake + ninja on the board
export PYTHONPATH="$PWD/python"
python -m pytest tests/ -q -rs     # model tests skip cleanly unless BLLM_TEST_* is set
```

板外（任意主机）不需要硬件也能检查元数据/路由逻辑：

```bash
c++ -std=c++17 -I include tests/test_choose_backend.cc -o /tmp/t && /tmp/t
PYTHONPATH=python python -m pytest tests/test_modelmeta.py tests/test_backend_routing.py -q
```

## 约定

- 头文件 `.h`、实现 `.cc`；命名空间 `bllm`；错误经 `BLLM_CHECK(...)`。
- 与周围代码的风格、注释密度和惯用写法保持一致。
- native 模型是一个带 `model.json` 清单的目录——不要手写；用 `scripts/make_model_dir.py`，
  它会从 tokenizer 自带的 `generation_config.json` 解析停止 token。

## 测试

任何行为变更都要新增或更新测试：

- 能离线测的逻辑尽量用**离线**测试钉死（元数据合成、后端路由、采样器、停止串匹配）——
  这些在任何主机上都能跑。
- 凡是涉及模型的，补一个**板端端到端**测试，并在模型不存在时**干净跳过**（模型测试要设置
  `BLLM_TEST_*` 环境变量）。

## 提交与 Pull Request

- 每个提交保持聚焦；遵循日志里已有的 [Conventional Commits](https://www.conventionalcommits.org/)
  风格。
- PR：说明改了什么、怎么验证的（哪块板子、哪个模型）。diff 只围绕一件事。

## 报告 bug

开一个 [Issue](../../issues)，写明：板子（S100 / S100P / S600）与系统、模型、最小复现、实际
输出与期望输出。分配内存时的运行时错误通常是 ION carveout 的问题——见板上说明。

## 范围

BLLM 是面向已完成 `.hbm` 模型的板上**运行时**。离线模型转换（`.hbm` 如何产出）不在本仓库
范围内。批处理以及 InternVL/Qwen-VL 模型类型刻意没有实现（单核部件；没有随附的 `.hbm`）。
