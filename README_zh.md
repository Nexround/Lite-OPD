# Lite-OPD: On-Policy Distillation

[英文](README.md) | [中文](README_zh.md)

Lite-OPD 是一个面向科研场景的在线蒸馏训练框架。Student 模型实时生成 rollout，teacher 模型对 rollout 打分，student 通过 KL 散度损失学习 teacher 的分布。

支持的模型：Qwen2.5 / Qwen3 / Llama 3.x / Gemma 3

支持的损失：forward KL / reverse KL / JSD（全词表，非近似）

## 设计理念

Lite-OPD 面向需要深度定制训练流程的研究工作。框架追求的是**易修改性**和**可维护性**，而非功能全面性。

核心设计选择：

- **单进程同步架构**：训练和推理在同一进程内完成，通过零拷贝权重共享消除通信开销。没有多 worker 协调、没有跨节点调度、没有异步状态同步。
- **最小抽象**：没有 callback 系统、没有 plugin 机制、没有多层配置抽象。核心训练逻辑集中在单个文件中。需要修改行为时直接改代码，不需要理解框架的扩展点设计。
- **低硬件门槛**：单卡即可完成完整的在线蒸馏流程（rollout → teacher scoring → student backward），适合资源有限的实验室环境。

## 代码规模

整个框架 Python 代码约 9000 行，C/CUDA 内核约 2100 行。

```
src/opd/
├── data/       数据加载
├── eval/       评估打分
├── losses/     蒸馏 loss（forward KL / reverse KL / JSD）
├── train/      训练循环、ZeRO-2、sequence packing
├── runtime/    推理引擎生命周期管理
└── inference/  嵌入式推理引擎
```

## 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
│                                                                 │
│  1. Rollout（嵌入式推理引擎）                                    │
│     - Student 生成 N 条 response                                │
│     - 推理引擎与训练共享同一份权重（零拷贝）                      │
│                                                                 │
│  2. Teacher Forward                                             │
│     - Teacher 对 student response 计算 logits                   │
│     - Sequence packing + torch.compile 加速                     │
│                                                                 │
│  3. Loss + Backward（two-stage chunk）                          │
│     - 逐 chunk 计算 KL loss 对 hidden 的梯度                   │
│     - 一次性将梯度反传回 student backbone                        │
│     - 峰值显存 = O(chunk_size) 而非 O(total_response_tokens)   │
│                                                                 │
│  4. Optimizer Step                                              │
│     - 权重更新后推理引擎自动可见（共享内存）                      │
│                                                                 │
│  5. 重复                                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 性能对比

2× H20 96GB，`global_batch_size=128`，`reverse_kl`，500 steps。对比使用基本相同的训练配置，详见 [benchmarks/ms-swift/](benchmarks/ms-swift/README_zh.md)。

<table>
<tr>
<td><img src="asset/benchmark_qwen3_1b7_8b.png" width="400"/></td>
<td><img src="asset/benchmark_qwen25_1b5_7b.png" width="400"/></td>
</tr>
<tr>
<td align="center">Qwen3-1.7B → Qwen3-8B<br/>Lite-OPD ~374s/step vs ms-swift ~474s/step</td>
<td align="center">Qwen2.5-1.5B → Qwen2.5-7B<br/>Lite-OPD ~93s/step vs ms-swift ~96s/step</td>
</tr>
</table>

## 加速技术

详见 [docs/acceleration_techniques_zh.md](docs/acceleration_techniques_zh.md)。

**Lite-OPD 特殊技术：**

| 技术 | 收益 |
|------|------|
| 零拷贝权重共享 | 消除权重同步开销 + 节省一份模型显存 |
| Two-stage chunk backward | 全词表 KL 计算，logits 峰值显存降低约 8x |
| ZeRO-2 梯度 buffer 动态释放 | Rollout 阶段多出 ~3GB/卡给 KV cache |
| Shortest-first 调度 | Batch makespan 减少 ~10% |
| VMM KV cache 释放 | Training 阶段释放全部 KV cache 显存 |
| Teacher compile + bucket packing | 限制编译次数，防止显存增长 |

**常用技术：** Prefix cache (radix tree)、Chunked prefill、Paged attention、CUDA graph、FlashInfer / sgl_kernel、ZeRO-2 数据并行、Sequence packing

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.4+
- CUDA 12.1+

### 安装

```bash
pip install -e .
```

### 训练

```bash
# 单卡
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash scripts/train.sh configs/qwen25_1b5_7b.yaml

# 多卡（ZeRO-2）
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/train.sh configs/qwen25_1b5_7b.yaml
```

### 配置

配置参数详见 [configs/README_zh.md](configs/README_zh.md)。

Smoke test 配置在 `configs/experiment/` 目录下，可用于快速验证环境是否正常：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash scripts/train.sh configs/experiment/smoke_qwen25_1b5_7b.yaml
```

## 文档

- [配置参考](configs/README_zh.md)
- [数据集格式与要求](docs/dataset_format_zh.md)
- [混合 SFT + OPD 训练](docs/hybrid_sft_opd_zh.md)
- [加速技术](docs/acceleration_techniques_zh.md)
- [代码架构](docs/architecture_zh.md)

## 致谢

Lite-OPD 的推理引擎参考或使用了以下开源项目：

- [SGLang](https://github.com/sgl-project/sglang) — 高性能 LLM 推理框架。
- [mini-sglang](https://github.com/EvolvingLMMs-Lab/mini-sglang) — SGLang 核心推理循环的精简复现。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — 高性能算子。
