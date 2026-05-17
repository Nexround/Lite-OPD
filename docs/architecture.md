# Lite-OPD Codebase Architecture

[English](architecture.md) | [Chinese](architecture_zh.md)

Lite-OPD (On-Policy Distillation) is a training framework for distilling knowledge from large models into smaller ones. Its core feature is **on-policy distillation**: during training, the student model generates rollouts in real time, the teacher model scores them, and the student learns the teacher's distribution via KL divergence loss.

## Directory Structure

```
src/liteopd/
├── inference/          # Inference engine (embedded, shares process and weights with training)
│   ├── engine/         # Engine core: forward inference, CUDA graph, sampling
│   ├── scheduler/      # Request scheduling: shortest-first prefill, decode management, preemption
│   ├── attention/      # Attention backends: FlashInfer, Flash Attention, TRT-LLM
│   ├── kvcache/        # KV cache pools: radix tree cache, VMM-backed cache
│   ├── kernel/         # Low-level operators: C++/CUDA extensions (TVM FFI)
│   ├── models/         # Model definitions: Qwen2/2.5/3, Llama, Gemma3
│   ├── layers/         # Network layers: attention, linear, embedding, RoPE, norm
│   ├── distributed/    # Distributed: TP info, NCCL communication
│   ├── message/        # Backend message protocol
│   ├── utils/          # Utility functions
│   ├── core.py         # Core data structures: Req, Batch, Context, SamplingParams
│   └── env.py          # Environment configuration
├── runtime/            # Runtime coordination layer
│   ├── rollout.py      # InProcessRolloutClient + _EmbeddedScheduler
│   ├── coordinator.py  # RuntimeCoordinator: orchestrates initialization and lifecycle
│   └── weight_sync.py  # Weight sync: zero-copy sharing / copy plan fallback
├── train/              # Training flow
│   ├── run_opd_training.py  # Main entry: data loading, rollout, loss computation, optimization
│   ├── fused_model.py  # Fused projection: QKV/GateUp merge, unified training-inference weights
│   ├── packing.py      # Sequence packing: merge multiple sequences into single forward
│   ├── zero2.py        # ZeRO-2 optimizer: parameter sharding + gradient buffer dynamic release
│   ├── config.py       # TrainConfig + YAML loading
│   ├── launcher.py     # torchrun launcher
│   └── logging.py      # JSONL logging
├── losses/             # Loss functions
│   └── kl.py           # forward KL, reverse KL, JSD, chunked variants
├── data/               # Data processing
└── eval/               # Evaluation
```

## Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
│                                                                 │
│  1. Rollout (embedded inference engine)                         │
│     - Student generates N responses                             │
│     - Shortest-first scheduling + continuous batching           │
│     - Inference engine shares weights with training             │
│       (zero-copy tensor.set_())                                 │
│     - ZeRO-2 gradient buffer released, freeing memory for KV   │
│                                                                 │
│  2. Teacher prefill                                             │
│     - Teacher computes hidden states over student responses     │
│     - Sequence packing merges multiple sequences into one       │
│       forward pass                                              │
│                                                                 │
│  3. Loss + Backward (two-stage chunk)                           │
│     - Stage 1: compute KL loss gradient w.r.t. hidden per chunk │
│     - Stage 2: backpropagate hidden gradients through student   │
│       backbone at once                                          │
│     - Peak memory = O(chunk_size), not O(total_response_tokens) │
│                                                                 │
│  4. Optimizer step                                              │
│     - ZeRO-2 gradient buffer reallocated                        │
│     - AdamW + gradient accumulation                             │
│     - Weight updates instantly visible to inference engine      │
│       (shared memory)                                           │
│                                                                 │
│  5. Repeat                                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Weight Unification (Fused Projection)

The training model and inference engine use the same fused weight layout:

```
HF original format:                Fused format (shared by training + inference):
  q_proj.weight [q_size, H]         qkv_proj.weight [q+k+v, H]
  k_proj.weight [k_size, H]    →    (single matmul + split)
  v_proj.weight [v_size, H]

  gate_proj.weight [I, H]           gate_up_proj.weight [2*I, H]
  up_proj.weight   [I, H]      →    (single matmul + split)
```

- When loading an HF checkpoint: `fuse_model_projections()` merges separate weights into fused Parameters
- During training: the optimizer directly updates fused weights; the inference engine shares the same memory via `tensor.set_()`
- When saving checkpoints: `unfuse_state_dict()` splits fused weights back to HF format

This eliminates per-step weight synchronization overhead and the duplicate memory usage of model weights.

## Inference Engine Architecture

The inference engine is Lite-OPD's core component, responsible for efficiently generating student rollouts. It is an embedded engine (sharing GPU and weights with training), not a standalone service.

```
InProcessRolloutClient
  └── _EmbeddedScheduler (extends Scheduler)
        ├── PrefillManager    # Shortest-first scheduling + chunked prefill
        ├── DecodeManager     # Decode batch management
        ├── CacheManager      # KV cache page allocation/release + radix prefix cache
        └── Engine
              ├── Model (inference format, shares weights with training model)
              ├── KVCache (MHA / VMM-backed)
              └── GraphRunner (CUDA graph capture/replay)
```

### Memory Lifecycle (VMM Mode)

```
Engine initialization:
  cuMemAddressReserve → virtual address (permanent)
  cuMemCreate + cuMemMap → physical memory

Rollout phase:
  ZeRO-2 gradient buffer already released
  Physical memory mapped, normal inference
  CUDA graphs reference virtual addresses

Training phase:
  release() → cuMemUnmap + cuMemRelease
  Physical memory returned for activations + teacher forward
  ZeRO-2 gradient buffer reallocated

Next rollout:
  prepare() → cuMemCreate + cuMemMap (same virtual address)
  CUDA graphs need no recapture
```
