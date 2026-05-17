# Lite-OPD 代码库架构

[英文](architecture.md) | [中文](architecture_zh.md)

Lite-OPD (On-Policy Distillation) 是一个将大模型知识蒸馏到小模型的训练框架。核心特点是 **在线蒸馏**：训练过程中，student 模型实时生成 rollout，teacher 模型对 rollout 打分，student 通过 KL 散度损失学习 teacher 的分布。

## 目录结构

```
src/liteopd/
├── inference/          # 推理引擎（嵌入式，与训练共享进程和权重）
│   ├── engine/         # 引擎核心：前向推理、CUDA graph、采样
│   ├── scheduler/      # 请求调度：shortest-first prefill、decode 管理、preemption
│   ├── attention/      # 注意力后端：FlashInfer、Flash Attention、TRT-LLM
│   ├── kvcache/        # KV cache 池：radix tree cache、VMM-backed cache
│   ├── kernel/         # 底层算子：C++/CUDA 扩展（TVM FFI）
│   ├── models/         # 模型定义：Qwen2/2.5/3、Llama、Gemma3
│   ├── layers/         # 网络层：attention、linear、embedding、RoPE、norm
│   ├── distributed/    # 分布式：TP info、NCCL 通信
│   ├── message/        # 后端消息协议
│   ├── utils/          # 工具函数
│   ├── core.py         # 核心数据结构：Req、Batch、Context、SamplingParams
│   └── env.py          # 环境配置
├── runtime/            # 运行时协调层
│   ├── rollout.py      # InProcessRolloutClient + _EmbeddedScheduler
│   ├── coordinator.py  # RuntimeCoordinator：编排初始化和生命周期
│   └── weight_sync.py  # 权重同步：零拷贝共享 / copy plan 回退
├── train/              # 训练流程
│   ├── run_opd_training.py  # 主入口：数据加载、rollout、loss 计算、优化
│   ├── fused_model.py  # Fused projection：QKV/GateUp 合并，训练推理权重统一
│   ├── packing.py      # Sequence packing：多序列合并为单次 forward
│   ├── zero2.py        # ZeRO-2 optimizer：参数分片 + 梯度 buffer 动态释放
│   ├── config.py       # TrainConfig + YAML 加载
│   ├── launcher.py     # torchrun 启动器
│   └── logging.py      # JSONL 日志
├── losses/             # 损失函数
│   └── kl.py           # forward KL、reverse KL、JSD、chunked 变体
├── data/               # 数据处理
└── eval/               # 评估
```

## 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
│                                                                 │
│  1. Rollout（嵌入式推理引擎）                                    │
│     - student 生成 N 条 response                                │
│     - shortest-first 调度 + continuous batching                 │
│     - 推理引擎与训练共享同一份权重（零拷贝 tensor.set_()）        │
│     - ZeRO-2 梯度 buffer 在此阶段释放，腾出显存给 KV cache       │
│                                                                 │
│  2. Teacher prefill                                             │
│     - teacher 对 student 的 response 计算 hidden states         │
│     - sequence packing 合并多条序列为单次 forward               │
│                                                                 │
│  3. Loss + Backward（two-stage chunk）                          │
│     - Stage 1: 逐 chunk 计算 KL loss 对 hidden 的梯度          │
│     - Stage 2: 一次性将 hidden 梯度反传回 student backbone      │
│     - 峰值显存 = O(chunk_size) 而非 O(total_response_tokens)   │
│                                                                 │
│  4. Optimizer step                                              │
│     - ZeRO-2 梯度 buffer 重新分配                               │
│     - AdamW + gradient accumulation                             │
│     - 权重更新后推理引擎自动可见（共享内存）                      │
│                                                                 │
│  5. 重复                                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 权重统一（Fused Projection）

训练模型和推理引擎使用相同的 fused 权重布局：

```
HF 原始格式:                    Fused 格式（训练 + 推理共用）:
  q_proj.weight [q_size, H]       qkv_proj.weight [q+k+v, H]
  k_proj.weight [k_size, H]  →    （单次 matmul + split）
  v_proj.weight [v_size, H]

  gate_proj.weight [I, H]         gate_up_proj.weight [2*I, H]
  up_proj.weight   [I, H]    →    （单次 matmul + split）
```

- 加载 HF checkpoint 时：`fuse_model_projections()` 将独立权重合并为 fused Parameter
- 训练时：optimizer 直接更新 fused weight，推理引擎通过 `tensor.set_()` 共享同一块显存
- 保存 checkpoint 时：`unfuse_state_dict()` 将 fused weight split 回 HF 格式

这消除了每步训练后的权重同步开销，也消除了模型权重的双份显存占用。

## 推理引擎架构

推理引擎是 Lite-OPD 的核心组件，负责高效生成 student rollout。它是一个嵌入式引擎（与训练共享 GPU 和权重），而非独立服务。

```
InProcessRolloutClient
  └── _EmbeddedScheduler (继承 Scheduler)
        ├── PrefillManager    # shortest-first 调度 + chunked prefill
        ├── DecodeManager     # decode batch 管理
        ├── CacheManager      # KV cache page 分配/释放 + radix prefix cache
        └── Engine
              ├── Model (推理格式，与训练模型共享权重)
              ├── KVCache (MHA / VMM-backed)
              └── GraphRunner (CUDA graph capture/replay)
```

### 内存生命周期（VMM 模式）

```
Engine 初始化:
  cuMemAddressReserve → 虚拟地址（永久）
  cuMemCreate + cuMemMap → 物理内存

Rollout 阶段:
  ZeRO-2 梯度 buffer 已释放
  物理内存已映射，正常推理
  CUDA graph 引用虚拟地址

Training 阶段:
  release() → cuMemUnmap + cuMemRelease
  物理显存归还，用于 activation + teacher forward
  ZeRO-2 梯度 buffer 重新分配

下次 Rollout:
  prepare() → cuMemCreate + cuMemMap（同一虚拟地址）
  CUDA graph 无需重新捕获
```
