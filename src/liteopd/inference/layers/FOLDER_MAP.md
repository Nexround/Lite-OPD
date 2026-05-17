# src/opd/inference/layers

## Purpose
Reusable model layer building blocks for the inference engine. Provides TP-aware linear, attention, MoE, embedding, norm, and RoPE layers that compose into full model architectures.

## Key Parts
- `attention.py`: `AttentionLayer` — dispatches to the configured attention backend; handles RoPE application and KV write-back.
- `linear.py`: `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear` — tensor-parallel linear layers.
- `embedding.py`: `VocabParallelEmbedding`, `ParallelLMHead`.
- `norm.py`: `RMSNorm`, `RMSNormFused`.
- `rotary.py`: RoPE implementation; `set_rope_device()` / `get_rope()`.
- `moe.py`: MoE dispatch layer wrapping `BaseMoeBackend`.
- `activation.py`: Gated activation functions (SiLU, GELU).
- `base.py`: `BaseOP`, `StateLessOP`, `OPList` — base classes for composable layer ops.

## Entry Points
- Layers are instantiated inside model `__init__` methods in `liteopd.inference.models`.
- `set_rope_device(device)`: called at engine startup before model creation.

## Outbound Dependencies
- `liteopd.inference.distributed`: TP rank/size for weight sharding.
- `liteopd.inference.attention`: attention kernel dispatch.
- `liteopd.inference.moe`: MoE routing.

## Inbound Dependents
- `liteopd.inference.models.*`: all model architectures use these layers.
