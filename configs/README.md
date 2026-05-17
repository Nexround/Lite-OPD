# Configuration Reference

[English](README.md) | [Chinese](README_zh.md)

Lite-OPD training configuration uses YAML format. All parameters correspond to `TrainConfig` dataclass fields.

## Required

| Parameter | Description |
|-----------|-------------|
| `loss` | Training objective: `forward_kl`, `reverse_kl`, `jsd` |
| `student_model` | Student model path (HuggingFace format) |
| `teacher_model` | Teacher model path (HuggingFace format) |
| `train_dataset` | Training data path; supports `jsonl`, `parquet`, or HF `load_from_disk` directory |
| `output_dir` | Output directory for logs, checkpoints, etc. |

## Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `2e-6` | AdamW learning rate |
| `global_batch_size` | `32` | Effective global batch size; must be divisible by `world_size` |
| `max_steps` | `20` | Maximum training steps |
| `num_epochs` | `1` | Number of data epochs |
| `distributed_strategy` | `zero2` | Distributed strategy: `ddp` or `zero2` |
| `enable_gradient_checkpointing` | `true` | Student gradient checkpointing |
| `kl_backward_mode` | `two_stage` | KL backward mode: `sample`, `chunk`, `two_stage` |
| `max_pack_tokens` | `32768` | Maximum tokens per micro-batch for sequence packing |
| `max_prompt_length` | `1024` | Maximum prompt length (truncated if exceeded) |
| `warmup_ratio` | `0.0` | Fraction of total steps for linear warmup; range [0, 1) |
| `cosine_annealing` | `false` | Whether to use cosine annealing after warmup |
| `offload_teacher` | `false` | Whether to offload teacher weights to CPU |
| `compile_teacher` | `true` | Whether to use `torch.compile` for teacher forward |

## Generation (Rollout)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | `0.6` | Sampling temperature |
| `top_p` | `0.9` | Top-p sampling |
| `generation_top_k` | `20` | Top-k sampling; set to `null` to disable |
| `max_total_tokens` | `16384` | Maximum generation tokens for rollout and eval |
| `generation_mem_fraction_static` | `0.9` | Fraction of GPU memory for inference engine KV cache |
| `generation_max_running_req` | `128` | Maximum concurrent requests in inference engine |
| `generation_page_size` | `1` | KV cache page size (in tokens) |
| `generation_attention_backend` | `auto` | Attention backend: `auto`, `fi` (FlashInfer) |
| `generation_cuda_graph_max_bs` | `null` | CUDA graph max batch size; defaults to `max_running_req` |
| `generation_admission_reserve_tokens` | `2048` | Scheduler reserved tokens to prevent preemption |
| `generation_max_preemptions_per_req` | `3` | Maximum preemptions per request |
| `generation_use_vmm` | `true` | Whether to use VMM (Virtual Memory Management) for KV cache |
| `generation_batch_size` | `null` | Generations per rank per step; defaults to `global_batch_size // world_size` |

## Evaluation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_dataset` | `null` | Evaluation data path; uses training data test split if unset |
| `eval_every_steps` | `10` | Evaluate every N steps |
| `skip_eval` | `false` | Whether to skip evaluation |
| `eval_subset_size` | `-1` | Evaluation set truncation size; `-1` for no truncation |
| `validation_subset_size` | `0` | Validation split from training set; `0` for no split |

## Logging & Checkpointing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_every` | `1` | Log training metrics every N steps |
| `save_every_steps` | `100` | Save checkpoint every N steps |
| `profile_memory` | `false` | Whether to record CUDA memory profile |

## Constraints

- `global_batch_size` must be divisible by `world_size`
- `distributed_strategy=ddp` does not support `kl_backward_mode=chunk` (flex_attention compiled kernels are incompatible with `retain_graph=True`)
- `warmup_ratio` must be in range [0, 1)
- `train_subset_size=0` is equivalent to `-1` (no truncation)

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
