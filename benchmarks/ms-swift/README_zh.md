# Benchmark: Lite-OPD vs ms-swift

[英文](README.md) | [中文](README_zh.md)

## 环境

- 硬件：2× H20 96GB
- ms-swift 版本：3.5.1（使用 vLLM colocate 模式）
- 损失：reverse KL（ms-swift 中对应 `gkd` + `lmbda=1.0`）
- `global_batch_size=128`（`per_device_train_batch_size=1 × gradient_accumulation_steps=64 × 2 GPUs`）
- `max_completion_length=16384`
- `learning_rate=5e-6`，constant schedule
- ZeRO-2，gradient checkpointing

## 对比条件

两个框架使用基本相同的训练配置（模型、数据、超参数、硬件）。ms-swift 使用 vLLM colocate 模式（推理和训练共卡），与 Lite-OPD 的嵌入式推理引擎架构对应。

Lite-OPD 配置见 [`configs/qwen3_1b7_8b.yaml`](../../configs/qwen3_1b7_8b.yaml) 和 [`configs/qwen25_1b5_7b.yaml`](../../configs/qwen25_1b5_7b.yaml)；ms-swift 配置见本目录下的 `run_qwen3_1b7_8b.sh` 和 `run_qwen25_1b5_7b.sh`。

## 结果

| 配置 | Lite-OPD | ms-swift | 加速比 |
|------|-----|----------|--------|
| Qwen3-1.7B → Qwen3-8B | ~374s/step | ~474s/step | 1.27× |
| Qwen2.5-1.5B → Qwen2.5-7B | ~93s/step | ~96s/step | 1.03× |

## 复现

ms-swift 脚本：
- `run_qwen3_1b7_8b.sh`
- `run_qwen25_1b5_7b.sh`

Lite-OPD 对应配置：
- `configs/qwen3_1b7_8b.yaml`
- `configs/qwen25_1b5_7b.yaml`
