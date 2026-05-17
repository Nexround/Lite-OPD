# src/opd

## Purpose
On-Policy Distillation (Lite-OPD) framework. A student model generates rollouts, a teacher model scores them, and the student learns via KL-divergence loss. The inference engine runs in-process, sharing GPU memory with the training model via zero-copy weight sharing.

## Key Parts
- `data/`: Dataset loading utilities (parquet, JSONL, HF DatasetDict). See `data/FOLDER_MAP.md`.
- `eval/`: Evaluation and scoring (answer extraction, math verification, distributed eval). See `eval/FOLDER_MAP.md`.
- `losses/`: Distillation loss functions (FKL, RKL, JSD, overlap variants, chunked backward). See `losses/FOLDER_MAP.md`.
- `train/`: Training loop, ZeRO-2 optimizer, model fusion, sequence packing, launcher. See `train/FOLDER_MAP.md`.
- `runtime/`: In-process rollout orchestration, weight sharing, VMM lifecycle. See `runtime/FOLDER_MAP.md`.
- `inference/`: Self-contained LLM inference engine (scheduler, KV cache, attention, models, kernels). See `inference/FOLDER_MAP.md`.

## Entry Points
- `python -m liteopd.train.run_opd_training --config <yaml>`: single-GPU training.
- `python -m liteopd.train.launcher --config <yaml> --nproc-per-node N`: multi-GPU training via torchrun.

## Architecture Overview
```
train loop (run_opd_training.py)
  ├── data loading (liteopd.data)
  ├── RuntimeCoordinator (liteopd.runtime)
  │     └── InProcessRolloutClient
  │           └── inference engine (liteopd.inference)
  │                 ├── Scheduler (shortest-first, longest-first preemption)
  │                 ├── Engine (CUDA graph, paged attention)
  │                 └── VMMKVCache (map/unmap around rollout)
  ├── loss computation (liteopd.losses)
  │     └── two-stage chunk backward (hidden → grad → backbone)
  ├── ZeRO2Optimizer (release_grad_buffer during rollout)
  └── evaluation (liteopd.eval)
```

## Key Design Decisions
- Zero-copy weight sharing: inference engine references training model params via `tensor.set_()`.
- Two-stage chunk backward: reduces peak memory by computing loss grad per chunk before backbone backward.
- ZeRO-2 gradient buffer release: frees `_grad_shard_out` during rollout to maximize KV cache memory.
- Shortest-first scheduling: minimizes batch makespan by prioritizing shorter sequences.
- VMM KV cache: CUDA virtual memory allows physical memory release during training without invalidating addresses.
