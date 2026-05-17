# 加速技术

[英文](acceleration_techniques.md) | [中文](acceleration_techniques_zh.md)

本文档介绍 Lite-OPD 使用的加速技术。分为两部分：Lite-OPD 框架的特殊设计，以及 LLM 训练/推理中的常用技术。

---

## 第一部分：Lite-OPD 框架特殊技术

### 1. 零拷贝权重共享（Rollout ↔ Training）

**问题**：传统方案中推理引擎和训练模型各持有一份权重，每步训练后需要 `copy_()` 同步，浪费显存和时间。

**方案**：
1. 训练模型加载后，将 q/k/v → `qkv_proj`，gate/up → `gate_up_proj`（fused Parameter）
2. 推理引擎的 weight tensor 通过 `tensor.set_()` 直接引用训练模型的 Parameter data
3. Optimizer 更新 fused weight 后，推理引擎自动看到新值（同一块 GPU 内存）

**收益**：
- 消除每步权重同步开销（`refresh_from_model` 变为仅 flush prefix cache）
- 节省一份模型权重的显存（7B 模型约 14 GiB）
- Fused matmul 本身也比两次独立 matmul 更高效

**实现**：
- `train/fused_model.py`：`fuse_model_projections()`、`unfuse_state_dict()`
- `runtime/weight_sync.py`：`share_weights()` 通过 `tensor.set_()` 建立零拷贝共享

---

### 2. Two-Stage Chunk-Level Backward

**问题**：蒸馏 loss 需要同时持有 student 和 teacher 的 logits（vocab_size 维度很大）。如果对整个 response 一次性计算 loss 并 backward，logits 的 activation 峰值显存为 O(response_len × vocab_size)，容易 OOM。

**方案**：将 backward 拆为两个阶段：

```
Stage 1（逐 chunk，不保留计算图）:
  for each chunk of response tokens:
    hidden_leaf = student_hidden[chunk].detach().requires_grad_(True)
    loss = KL(lm_head(hidden_leaf), teacher_logits[chunk])
    grad_hidden = autograd.grad(loss, hidden_leaf)  # 只对 hidden 求梯度
    accumulate grad_hidden into buffer
    # logits 立即释放，峰值 = O(chunk_size × vocab_size)

Stage 2（一次性反传）:
  torch.autograd.backward(student_hidden_packed, grad_tensors=accumulated_grad)
  # student backbone 的完整反向传播
```

**收益**：
- 峰值显存从 O(total_response × vocab_size) 降为 O(chunk_size × vocab_size)
- chunk_size 通常为 response_len / 8，显存节省约 8x
- lm_head 的梯度在 Stage 1 中逐 chunk 累积，无精度损失

**实现**：`train/run_opd_training.py` 中 `batch_rollout_and_loss_with_client()` 的 `kl_backward_mode="two_stage"` 分支。

---

### 3. ZeRO-2 梯度 Buffer 动态释放

**问题**：ZeRO-2 optimizer 需要一个与参数分片等大的梯度 buffer（`_grad_shard_out`）用于 reduce-scatter。在 rollout 阶段，这块 buffer 完全闲置但占用显存。

**方案**：
- Rollout 开始前：`optimizer.release_grad_buffer()` 将 `_grad_shard_out` 设为 None，释放显存
- Training 开始前：`optimizer.prepare_grad_buffer()` 重新分配

**收益**：
- 对于 1.5B 模型 + 2 卡 ZeRO-2，每卡释放约 3 GB
- 这些显存在 rollout 阶段可用于 KV cache，提高并发度

**实现**：`train/zero2.py` 中的 `release_grad_buffer()` / `prepare_grad_buffer()`。

---

### 4. Shortest-First 调度

**问题**：offline batch rollout 中，所有请求同时提交。如果按 FIFO 调度，长请求先 prefill 会占用大量 KV cache，导致短请求排队等待。整个 batch 的完成时间（makespan）由最后完成的请求决定。

**方案**：按请求的 **总目标长度**（input_len + max_output_len）升序调度 prefill：

```python
# scheduler/utils.py
@property
def priority_key(self) -> tuple[int, int, int]:
    return (self.input_len + self.max_output_len, self.input_len, self.uid)
```

短请求优先 prefill → 优先进入 decode → 优先完成释放 KV cache → 为后续长请求腾出空间。

**配合 Longest-First Preemption**：当 KV cache 不足时，驱逐当前 **占用 KV 最多** 的 decode 请求（即已生成 token 最多的）。被驱逐的请求回到 pending 队列重新 prefill。这确保长请求（makespan 关键路径）不会因为短请求的 preemption 而被打断。

**收益**：
- 减少 batch makespan ~10%（取决于序列长度分布的方差）
- 减少 preemption 次数（短请求快速完成，释放空间给长请求）

**实现**：
- `scheduler/utils.py`：`PendingReq.priority_key`
- `scheduler/prefill.py`：`PrefillManager._sort_pending()`
- `scheduler/scheduler.py`：`_preempt_longest()`

---

### 5. KV Cache 训练时释放（VMM）

**问题**：推理引擎的 KV cache 在训练阶段完全闲置，但常规分配方式下无法释放（CUDA graph 引用了这些地址）。

**方案**：使用 CUDA VMM（Virtual Memory Management）API 管理 KV cache：
- 初始化时 `cuMemAddressReserve` 预留虚拟地址空间（永久）
- Rollout 阶段：`cuMemCreate` + `cuMemMap` 映射物理内存，正常推理
- Training 阶段：`cuMemUnmap` + `cuMemRelease` 释放物理内存，显存归还给 activation 和 teacher forward
- 下次 Rollout：重新 `cuMemCreate` + `cuMemMap`（同一虚拟地址），CUDA graph 无需重新捕获

**收益**：
- 训练阶段释放全部 KV cache 显存（7B 模型 + 200 并发约 40-60 GB）
- CUDA graph 不受影响（虚拟地址不变）

**实现**：`kvcache/vmm_pool.py`、`kernel/csrc/src/vmm.cu`

---

### 6. Teacher Compile + Bucket Packing

**问题**：Teacher forward 是训练循环中的计算瓶颈之一。`torch.compile` 可以加速 teacher forward，但 `flex_attention` 内部使用 `torch.compile` 编译 attention kernel 和 `create_block_mask`，它们以 Python int 形式接收 `(Q_LEN, KV_LEN)`。每个新的序列长度组合都会触发 guard failure → retrace → recompile，导致：
- 编译时间累积（每次 recompile 数秒）
- 编译产物（compiled graph）占用 GPU 显存，长时间训练后可能 OOM

**方案**：对 packed batch 进行 bucket padding，将总 token 数向上对齐到 512 的倍数：

```python
_PAD_BUCKET_SIZE = 512

def _round_up_to_bucket(n: int) -> int:
    return ((n + _PAD_BUCKET_SIZE - 1) // _PAD_BUCKET_SIZE) * _PAD_BUCKET_SIZE
```

Padding 使用 `pad_token_id` 填充 `input_ids`，`position_ids` 填充 0。这将可能的 `(Q_LEN, KV_LEN)` 组合从无限多种限制为有限的 bucket 集合（如 512, 1024, 1536, ...），从而限制 `flex_attention` 的编译次数。

**收益**：
- 编译次数从 O(num_steps) 降为 O(max_tokens / bucket_size)，通常 < 64 次
- 消除编译产物的显存增长，防止长时间训练 OOM
- Padding 开销极小（平均浪费 < 256 tokens，占 batch 总量 < 1%）

**实现**：`train/packing.py` 中 `flush()` 函数的 bucket padding 逻辑。

---

## 第二部分：常用技术

以下是 LLM 训练/推理中广泛使用的标准加速技术。

### 7. Prefix Cache（Radix Tree）

用 radix tree 索引已计算的 KV cache。共享前缀的请求复用已有的 KV page，避免重复 prefill。

**实现**：`kvcache/radix_cache.py`

---

### 8. Chunked Prefill

将长 prompt 的 prefill 拆分为多个 chunk（由 `max_extend_tokens` 控制），避免单次 prefill 阻塞 decode 请求。在 Lite-OPD 的 offline batch 场景下，`max_extend_tokens` 设为 65536（几乎不限制），因为不存在在途 decode 被饿死的问题。

**实现**：`scheduler/prefill.py`

---

### 9. Paged Attention

KV cache 划分为固定大小的 page（默认 16 tokens/page），通过 page table 管理。不同长度的序列按需分配 page，无 padding 浪费。

**实现**：`scheduler/cache.py`、`kvcache/mha_pool.py`

---

### 10. CUDA Graph

预先捕获 decode forward pass 为 CUDA graph，replay 时跳过所有 kernel launch 开销。Decode 阶段加速约 2-3x。

**实现**：`engine/graph.py`

---

### 11. FlashInfer / sgl_kernel 高性能算子

使用 FlashInfer 和 sgl_kernel 提供的高效 paged attention kernel，支持 CUDA graph 兼容的 paged KV cache 访问。

**实现**：`attention/fi.py`、`attention/fa.py`

---

### 12. ZeRO-2 数据并行

Optimizer state 分片到各卡，参数通过 all-gather 恢复。支持大模型多卡训练，显存占用约为 DDP 的 1/N。

**实现**：`train/zero2.py`

---

### 13. Sequence Packing

将多条不等长序列打包为一个 packed batch，消除 padding 浪费。Student forward 和 teacher forward 都使用 packing，通过 `position_ids` 区分序列边界。

**实现**：`train/packing.py`

---

## 技术总览

| 技术 | 类别 | 收益 |
|------|------|------|
| 零拷贝权重共享 | Lite-OPD 特殊 | 消除权重同步开销 + 节省一份模型显存 |
| Two-stage chunk backward | Lite-OPD 特殊 | logits 峰值降低约 8x |
| ZeRO-2 buffer 动态释放 | Lite-OPD 特殊 | Rollout 阶段多出 ~3GB/卡 给 KV cache |
| Shortest-first 调度 | Lite-OPD 特殊 | Batch makespan 减少 ~10% |
| VMM KV cache 释放 | Lite-OPD 特殊 | Training 阶段释放全部 KV cache 显存 |
| Teacher compile + bucket packing | Lite-OPD 特殊 | 限制编译次数，防止显存增长 |
| Prefix cache (radix tree) | 常用 | 共享前缀复用 KV |
| Chunked prefill | 常用 | 控制 prefill 延迟和显存峰值 |
| Paged attention | 常用 | 消除 KV cache padding |
| CUDA graph | 常用 | Decode 阶段 2-3x 加速 |
| FlashInfer / sgl_kernel | 常用 | 高效计算 kernel |
| ZeRO-2 数据并行 | 常用 | 大模型多卡训练 |
| Sequence packing | 常用 | 消除 padding，GPU 利用率提升 |
