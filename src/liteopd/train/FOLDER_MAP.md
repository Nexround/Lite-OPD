# src/opd/train

## Purpose
Training loop, optimizer, model fusion, and experiment launching for Lite-OPD (On-Policy Distillation). Orchestrates rollout generation, loss computation, and checkpointing.

## Key Parts
- `run_opd_training.py`: Main training script. Loads data via `liteopd.data`, runs rollout generation via `RuntimeCoordinator`, computes chunked KL/hidden loss via `liteopd.losses`, handles gradient accumulation, checkpointing, and evaluation via `liteopd.eval`. Contains prompt-building helpers (`build_messages`, `get_prompt_text`, `get_gold_text`) and thin wrappers that bind them to the generic eval functions.
- `zero2.py`: `ZeRO2Optimizer` — manual ZeRO-2 implementation. Shards optimizer moments across ranks; reuses `_flat_params_padded` as both param storage and grad pack buffer. Provides `release_grad_buffer()` / `prepare_grad_buffer()` to free gradient memory during rollout.
- `fused_model.py`: `FusedQKVLinear`, `FusedGateUpLinear` — fuses Q/K/V and gate/up projections into single matmuls. Required before `share_weights()`.
- `launcher.py`: Spawns training via `torchrun`; copies the input YAML to the run output directory.
- `config.py`: `load_train_config()` — loads and validates the YAML experiment config.
- `logging.py`: `JsonlLogger` — appends training metrics as JSON lines; records `relative_param_update_norm = ||delta_theta|| / ||theta||` each step.
- `packing.py`: Sequence packing for efficient variable-length batching (multiple sequences in one forward pass).

## Entry Points
- `python -m liteopd.train.run_opd_training --config <yaml>`: single-GPU direct launch.
- `python -m liteopd.train.launcher --config <yaml> --nproc-per-node N`: multi-GPU launch via torchrun.

## Outbound Dependencies
- `liteopd.losses`: distillation loss functions.
- `liteopd.runtime`: `RuntimeCoordinator` / `InProcessRolloutClient`.
- `liteopd.data`: dataset loading (`load_examples_from_path`, `split_validation_examples`).
- `liteopd.eval.scoring`: evaluation functions (accuracy computation, distributed eval).
- `transformers`, `torch.distributed`: model loading, DDP.

## Inbound Dependents
- CLI entry points (launcher, direct invocation).

## Notes
- `ZeRO2Optimizer` checkpoint format: keys `m`, `v`, `step_count`, `param_groups`.
- `_flat_params_padded` is overwritten with packed gradients during `reduce_scatter_grads()`; a `_param_shard_snapshot` clone is saved beforehand for weight decay.
- The inference engine references training model parameters via zero-copy `share_weights()`; only a prefix cache flush is needed after each optimizer step.
