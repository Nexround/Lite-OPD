# src/opd/inference

## Purpose
Self-contained LLM inference engine used in-process during training rollout. Handles model execution, KV cache management, paged attention, and offline batch scheduling. Not responsible for training, loss computation, or data loading.

## Key Parts
- `core.py`: Fundamental data structures — `Req`, `Batch`, `SamplingParams`, `Context`. Shared across all submodules.
- `env.py`: Runtime environment variables (e.g. `DISABLE_OVERLAP_SCHEDULING`).
- Subdirectories (see their own FOLDER_MAP.md):
  - `engine/`: Forward pass, CUDA graph runner, sampler.
  - `scheduler/`: Admission control, shortest-first scheduling, longest-first preemption.
  - `kvcache/`: Paged KV pool, radix prefix cache, VMM-backed pool.
  - `attention/`: Attention backend abstraction (FlashAttention, FlashInfer, TRT-LLM).
  - `layers/`: Model building blocks (attention, linear, MoE, RoPE, norm, embedding).
  - `models/`: Per-architecture model definitions (Llama, Qwen2, Qwen3, Qwen3-MoE).
  - `kernel/`: JIT-compiled CUDA/Triton kernels (index, MoE, VMM, pynccl, radix).
  - `distributed/`: TP rank/size management, pynccl group setup.
  - `message/`: Serializable message types for scheduler-tokenizer communication.
  - `moe/`: MoE routing backend abstraction.
  - `utils/`: Logging, tokenizer loading, misc torch helpers.

## Entry Points
- `engine.Engine`: constructed by `Scheduler`; drives forward passes.
- `scheduler.Scheduler.normal_loop` / `overlap_loop`: called directly in embedded mode by `InProcessRolloutClient`.

## Outbound Dependencies
- `torch`, `transformers` (HF config/tokenizer loading), optional `flash_attn` / `flashinfer` / `tensorrt_llm`.

## Inbound Dependents
- `liteopd.runtime.rollout`: uses `Engine`, `Scheduler`, model creation, and weight loading.
