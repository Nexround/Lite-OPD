# Hybrid SFT + OPD 混合训练

[English](hybrid_sft_opd.md) | [中文](hybrid_sft_opd_zh.md)

## 概述

Hybrid SFT + OPD 混合训练将每条样本的回复区域拆分为两个监督区间：

| 区间 | 来源 | 损失 |
|---|---|---|
| **Gold prefix（金标前缀）** | 由数据集提供 | 交叉熵（SFT），对 gold token 做下一个词预测 |
| **Continuation（续写）** | 训练时由 student 实时生成 | KL 散度（OPD），与 teacher 的分布对齐 |

这样可以将监督学习的稳定性（在已知正确的前缀上）与在线蒸馏的探索能力（在 student 自己的生成上）结合起来。两段 loss 共享同一次反向传播，无额外前向开销。

将 `gold_prefix_field` 设为 `null`（默认值）即可关闭混合模式，退化为标准 OPD。

---

## 动机

标准 OPD 完全在 student 自己的 rollout 上训练。当 student 已能产生合理轨迹时效果良好，但训练早期 student 没有先验锚点，可能不稳定。提供 gold prefix 为 student 提供了每条样本的正确起点，同时仍由 OPD 塑造续写部分。

典型使用场景：

- **思维链锚定** — 将参考答案的前几步推理作为 gold prefix，OPD 从此处接续。
- **格式强制** — 提供建立所需输出结构的 gold prefix（如 `"<think>\n"`），让 student 学习在策略内完成。
- **课程式热身** — 训练初期用较长的 gold prefix，逐步缩短（减小 `gold_prefix_max_tokens`），将监督重心从 SFT 渐进迁移至 OPD。

---

## 工作原理

### Token 层级示意

```
input_ids（一条训练样本）：
┌───────────────────┬─────────────────┬──────────────────────────┐
│    base prompt    │  gold prefix    │      continuation        │
│   （用户消息      │  （来自数据集）  │   （student 实时生成）    │
│    + 对话模板头）  │                 │                          │
└───────────────────┴─────────────────┴──────────────────────────┘
 ◄──────────────── prompt_len ────────►◄──── response_tokens ────►

监督区间：
                    ├─── SFT (CE) ───┤├──── OPD (KL) ───────────┤
```

- **Gold prefix** 通过 `continue_final_message=True` 嵌入到 `apply_chat_template` 的 prompt 字符串中。从 rollout 引擎的视角来看，它属于上下文，不是生成 token。
- **Continuation** 是 student 在 rollout 阶段生成的内容。Teacher 对其做前向推断，提供 KL loss 所需的概率分布。

### 训练步骤

```
1. 从 dataset[gold_prefix_field] 读取每条样本的 gold_prefix。

2. 构建 rollout prompt：
     apply_chat_template([user_msg, assistant(gold_prefix)],
                          continue_final_message=True)
   → prompt 字符串结尾停留在 assistant 轮的中途。

3. Rollout：student 从 gold prefix 末尾开始续写。
   只返回续写部分的 token。

4. 打包序列：
     input_ids = tokenise(rollout_prompt) + tokenise(continuation)
   PackedBatch 记录 gold_prefix_lengths[i] ≈ len(tokenise(gold_prefix))。

5. Teacher 前向（inference_mode）：
     teacher_hidden = teacher.model(input_ids)

6. Student 前向：
     student_hidden = student.model(input_ids)

7. 每条样本的 loss 计算：
   a. SFT 区间  [sft_start : response_start]
        target_ids = input_ids[sft_start+1 : response_start+1]  ← gold token
        loss_sft   = CrossEntropy(lm_head(student_hidden[sft]), target_ids)
   b. OPD 区间  [response_start : response_end]
        loss_opd   = KL(teacher_probs ‖ student_probs)

8. 反向传播（two_stage 模式）：
   Stage 1 — 分别为两段区间累积 ∂loss/∂hidden，写入同一个
             hidden_grad_accum 张量。
   Stage 2 — 对 transformer backbone 做一次反向传播。
```

### Loss 公式

对于含 `G` 个 gold 前缀 token 和 `C` 个续写 token 的单条样本：

```
total_tokens = G + C

loss = Σ_{t=0}^{G-1}  (G / total_tokens) × sft_loss_weight × CE(t)
     + Σ_{t=0}^{C  }  (chunk_len / total_tokens) × KL(t)
```

两项均进一步乘以 `response_tokens / total_response_tokens` 做批次内归一化（与标准 OPD 相同）。

`sft_loss_weight = 1.0` 时，SFT 与 OPD 每 token 的梯度量级相等。增大该值可加强 SFT 信号；减小则让 OPD 主导。

---

## 反向传播细节

混合模式支持全部三种 `kl_backward_mode`。

### `two_stage`（默认，推荐）

SFT 区间与 OPD 区间共用一个 detach 后的**叶张量（leaf tensor）**，覆盖两段区域：

```
hidden_leaf = student_hidden[sft_start : response_end].detach().requires_grad_(True)
                              ├── [:G]  SFT 块 ────┤
                              ├── [G:]  OPD 块 ────┤  （8 个子块）
```

Stage 1 — 按区间累积梯度：
- SFT：`autograd.grad(loss_sft × scale_sft, [hidden_leaf, *lm_head_params])`
- OPD：`autograd.grad(loss_kl_chunk × scale_kl, [hidden_leaf, *lm_head_params])` × 8

两段贡献的梯度按各自位置累积到 `hidden_grad_accum`，LM head 梯度即时写入。

Stage 2 — 单次 backbone 反向传播：
```python
torch.autograd.backward(student_hidden_packed, grad_tensors=packed_grad_accum)
```

Backbone 在一次调用中接收来自两段区间的合并梯度。

### `chunk`

SFT 先用 `retain_graph=True` 反向传播（因为后续 OPD 块仍需图），OPD 块按标准 OPD 流程依次反向。比 `two_stage` 占用更多显存（整张计算图须保留到最后一块）。

### `sample`

SFT loss 与 KL loss 合并为单个标量后一次 `.backward()`。实现最简单；显存开销最高。

---

## 配置项

`TrainConfig` 中新增三个字段控制混合模式：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `gold_prefix_field` | `str \| null` | `null` | 数据集中存放 gold prefix 文本的字段名。`null` 关闭混合模式。 |
| `gold_prefix_max_tokens` | `int \| null` | `null` | Gold prefix 长度上限（token 数）。超出则在 token 边界截断后再打包。`null` 表示不限。 |
| `sft_loss_weight` | `float` | `1.0` | SFT（CE）loss 相对于 OPD（KL）loss 的权重系数。必须为正数。 |

### 示例配置

```yaml
# 标准 OPD 配置（不变）
loss: reverse_kl
student_model: /path/to/student
teacher_model: /path/to/teacher
train_dataset: /path/to/dataset
output_dir: results/hybrid_run
kl_backward_mode: two_stage

# 混合 SFT + OPD
gold_prefix_field: gold_prefix      # 数据集中的字段名
gold_prefix_max_tokens: 512         # null = 不截断
sft_loss_weight: 1.0                # 1.0 = 每 token 等权重
```

完整示例见 `configs/qwen3_5_2b_9b_hybrid.yaml`。

---

## 数据集格式

数据集须包含一个字段，字段名与 `gold_prefix_field` 一致，其余字段与标准 OPD 相同。

```jsonl
{
  "question": "解方程：x² - 5x + 6 = 0",
  "gold_prefix": "对该式进行因式分解。\n(x - 2)(x - 3) = 0\n所以",
  "answer": "x = 2 或 x = 3"
}
```

- **`question`**（或 `problem` / `prompt`）— 用户提示，与标准 OPD 相同。
- **`gold_prefix`** — 锚定回复起点的文本。Student 从该文本末尾续写。
- **`answer`**（或 `solution` / `canonical_solution`）— 仅用于评估准确率，不参与训练 loss。

若某条样本的 gold prefix 为空或字段缺失，该样本自动退化为标准 OPD 样本（无 SFT 区间）。

### 如何构造 gold prefix

Gold prefix 可来自多种来源：

```python
# 方案 A：参考答案的前 N 个 token
gold_prefix = tokenizer.decode(
    tokenizer(reference_solution).input_ids[:256]
)

# 方案 B：思维链标注的前 K 句话
import re
sentences = re.split(r'(?<=[。！？.!?])\s*', chain_of_thought)
gold_prefix = "".join(sentences[:3])

# 方案 C：固定结构头
gold_prefix = "<think>\n"
```

---

## 实现说明

### Tokenisation 边界近似

Gold prefix 被嵌入到对话模板 prompt 字符串中，因此其在 `prompt_ids` 中的精确 token 数可能与 `len(tokenise(gold_prefix_text))` 存在细微差异（BPE 在拼接处的合并效应）。`pack_sequences` 以独立 tokenisation 结果作为近似。偏差通常不超过 1–2 个 token，对训练影响可忽略不计。

### Teacher 对 gold 区间的前向开销

Teacher 对整个打包序列（含 gold prefix 区间）做完整前向推断，但 gold 区间的 hidden state 在 loss 中不被使用——只有 OPD 续写区间的切片被消耗。这带来约 `gold_len / total_len` 比例的额外计算。未来可优化为让 teacher 前向从 gold prefix 边界起步，跳过 gold 区间。

### 向后兼容性

`gold_prefix_field: null`（默认值）使每条样本的 `gold_len = 0`，所有新代码路径短路，行为与标准 OPD 完全相同。现有配置文件无需任何修改。

---

## 超参调优建议

### `sft_loss_weight`

| 值 | 效果 |
|---|---|
| `1.0` | SFT 与 OPD 每 token 梯度量级相等。推荐起点。 |
| `> 1.0` | SFT token 贡献更多梯度；适用于 gold prefix 质量高、student 需要强锚点的场景。 |
| `< 1.0` | OPD token 主导；适用于 gold prefix 存在噪声或 student 已趋于稳定的场景。 |

注意：实际比例还受 gold prefix 与续写相对长度影响。例如前缀 256 token、续写 512 token 时，`sft_loss_weight = 2.0` 可使两段总梯度贡献大致相等。

### `gold_prefix_max_tokens`

截断过长前缀可保证 OPD 续写区间有意义：
- 前缀过长 → student 生成 token 极少 → OPD loss 噪声大。
- 前缀过短 → SFT 锚点弱 → 趋近标准 OPD。

对于数学推理任务（gold prefix 包含初始推理步骤），建议取 **128–512 token**。

### 课程式训练策略

随训练进行逐步缩短 `gold_prefix_max_tokens`，使监督重心从 SFT 向纯 OPD 迁移：

```
Steps   0–100:  gold_prefix_max_tokens: 512   （强 SFT 锚点）
Steps 100–200:  gold_prefix_max_tokens: 256
Steps 200–400:  gold_prefix_max_tokens: null  （纯 OPD，或将字段设为 null）
```

目前需要为每个阶段单独准备配置文件并手动重启，或实现自定义调度器动态修改配置。

---

## 当前限制

- **不支持纯 SFT 模式** — 仍需 `teacher_model`；OPD 分支始终在续写区间运行。若续写区间为空（所有 token 均为 gold prefix），OPD loss 接近零，但 teacher 仍会做一次前向推断。
- **Gold prefix 边界近似** — 由于 BPE tokenisation，SFT/OPD 分割点可能在拼接处偏差 1–2 个 token。
- **Teacher 对 gold 区间的冗余前向** — Teacher 处理 gold prefix token，但其 hidden state 未被使用，产生约 `gold_len / total_len` 的额外计算开销。
