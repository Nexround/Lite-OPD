# Configuration Reference

[英文](README.md) | [中文](README_zh.md)

Lite-OPD 训练配置采用 YAML 格式，所有参数对应 `TrainConfig` dataclass 字段。

## Required

| 参数 | 说明 |
|------|------|
| `loss` | 训练目标：`forward_kl`、`reverse_kl`、`jsd` |
| `student_model` | Student 模型路径（HuggingFace 格式） |
| `teacher_model` | Teacher 模型路径（HuggingFace 格式） |
| `train_dataset` | 训练数据路径，支持 `jsonl`、`parquet`、HF `load_from_disk` 目录 |
| `output_dir` | 输出目录，保存日志、checkpoint 等 |

## Training

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | `2e-6` | AdamW 学习率 |
| `global_batch_size` | `32` | 全局有效 batch size，需被 `world_size` 整除 |
| `max_steps` | `20` | 最大训练步数 |
| `num_epochs` | `1` | 数据重复轮数 |
| `distributed_strategy` | `zero2` | 分布式策略：`ddp` 或 `zero2` |
| `enable_gradient_checkpointing` | `true` | Student 梯度检查点 |
| `kl_backward_mode` | `two_stage` | KL 反向传播模式：`sample`、`chunk`、`two_stage` |
| `max_pack_tokens` | `32768` | 序列 packing 每个 micro-batch 的最大 token 数 |
| `max_prompt_length` | `1024` | Prompt 最大长度（超出截断） |
| `warmup_ratio` | `0.0` | 线性 warmup 占总步数的比例，范围 [0, 1) |
| `cosine_annealing` | `false` | Warmup 后是否使用余弦退火 |
| `offload_teacher` | `false` | 是否将 teacher 权重 offload 到 CPU |
| `compile_teacher` | `true` | 是否对 teacher forward 使用 `torch.compile` |

## Generation (Rollout)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | `0.6` | 采样温度 |
| `top_p` | `0.9` | Top-p 采样 |
| `generation_top_k` | `20` | Top-k 采样；设为 `null` 禁用 |
| `max_total_tokens` | `16384` | Rollout 和 eval 的最大生成 token 数 |
| `generation_mem_fraction_static` | `0.9` | 推理引擎 KV cache 显存占比 |
| `generation_max_running_req` | `128` | 推理引擎最大并发请求数 |
| `generation_page_size` | `1` | KV cache page 大小（token 数） |
| `generation_attention_backend` | `auto` | 注意力后端：`auto`、`fi`（FlashInfer） |
| `generation_cuda_graph_max_bs` | `null` | CUDA graph 最大 batch size；默认等于 `max_running_req` |
| `generation_admission_reserve_tokens` | `2048` | 调度器预留 token 数，防止 preemption |
| `generation_max_preemptions_per_req` | `3` | 单请求最大 preemption 次数 |
| `generation_use_vmm` | `true` | 是否使用 VMM（Virtual Memory Management）管理 KV cache |
| `generation_batch_size` | `null` | 每 rank 每步生成数；默认自动推导为 `global_batch_size // world_size` |

## Evaluation

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eval_dataset` | `null` | 评测数据路径；不设则使用训练数据的 test split |
| `eval_every_steps` | `10` | 每隔多少步执行一次 eval |
| `skip_eval` | `false` | 是否跳过 eval |
| `eval_subset_size` | `-1` | 评测集截断条数；`-1` 不截断 |
| `validation_subset_size` | `0` | 从训练集切出的验证集大小；`0` 不切分 |

## Logging & Checkpointing

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `log_every` | `1` | 每隔多少步写训练日志 |
| `save_every_steps` | `100` | 每隔多少步保存 checkpoint |
| `profile_memory` | `false` | 是否记录 CUDA 内存 profile |

## Constraints

- `global_batch_size` 必须被 `world_size` 整除
- `distributed_strategy=ddp` 时不支持 `kl_backward_mode=chunk`（flex_attention compiled kernels 与 `retain_graph=True` 不兼容）
- `warmup_ratio` 范围 [0, 1)
- `train_subset_size=0` 等价于 `-1`（不截断）

## Example

```yaml
loss: reverse_kl
student_model: /path/to/Qwen2.5-1.5B-Instruct
teacher_model: /path/to/Qwen2.5-7B-Instruct
train_dataset: /path/to/data
eval_dataset: /path/to/eval.jsonl
output_dir: results/my_experiment

distributed_strategy: zero2
kl_backward_mode: two_stage
learning_rate: 5e-6
global_batch_size: 128
max_pack_tokens: 65536
max_steps: 500

generation_mem_fraction_static: 0.9
generation_max_running_req: 200
generation_page_size: 16
generation_attention_backend: fi

enable_gradient_checkpointing: true
temperature: 0.6
top_p: 0.9
generation_top_k: 20
max_total_tokens: 16384

eval_every_steps: 50
save_every_steps: 100
```
