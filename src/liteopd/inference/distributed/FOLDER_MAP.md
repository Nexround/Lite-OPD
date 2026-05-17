# src/opd/inference/distributed

## Purpose
Tensor-parallel rank/size management and intra-node collective communication setup. Provides a global TP info singleton and pynccl-based distributed group initialization.

## Key Parts
- `info.py`: `DistributedInfo` — frozen dataclass holding `(rank, size)`; global singleton accessed via `get_tp_info()` / `set_tp_info()`.
- `impl.py`: `enable_pynccl_distributed()` / `destroy_distributed()` — initializes the pynccl communicator for TP all-reduce.
- `__init__.py`: re-exports `DistributedInfo`, `get_tp_info`, `set_tp_info`, `enable_pynccl_distributed`, `destroy_distributed`.

## Entry Points
- `set_tp_info(rank, size)`: called at engine startup (and set to `(0, 1)` in embedded single-GPU mode).
- `enable_pynccl_distributed(group)`: called by `Engine.__init__` when `tp_size > 1`.

## Outbound Dependencies
- `liteopd.inference.kernel.pynccl`: underlying NCCL wrapper.

## Inbound Dependents
- `liteopd.inference.engine.Engine`: sets up TP group.
- `liteopd.inference.layers.*`: reads `get_tp_info()` for weight sharding.
- `liteopd.runtime.rollout.InProcessRolloutClient`: calls `set_tp_info(0, 1)` for single-GPU embedded mode.
