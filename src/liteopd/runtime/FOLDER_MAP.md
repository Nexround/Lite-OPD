# src/opd/runtime

## Purpose
Orchestrates the in-process rollout engine: lifecycle management, weight sharing, and the embedded scheduler that runs inference in the same process as training.

## Key Parts
- `rollout.py`: `InProcessRolloutClient` — creates the inference model, shares weights with the HF training model, and drives the offline batch scheduler. `_EmbeddedScheduler` subclasses `Scheduler` for single-process offline use.
- `coordinator.py`: `RuntimeCoordinator` — thin wrapper that constructs `InProcessRolloutClient` and exposes `prepare()` / `release()` / `refresh_from_model()` to the training loop.
- `weight_sync.py`: `share_weights()` — makes inference engine weight tensors reference the same GPU memory as HF model parameters (zero-copy via `tensor.set_()`; requires fused projections). Handles Qwen2.5, Qwen3, Llama, and Gemma3. For Gemma3, norm weights use copy+transform (`weight + 1`) instead of zero-copy; `refresh_derived_weights()` re-applies the transform after optimizer steps.

## Entry Points
- `RuntimeCoordinator.__init__`: called once at training startup to build the engine.
- `InProcessRolloutClient.generate_messages` / `generate_request_batch`: called each rollout phase.
- `InProcessRolloutClient.refresh_from_model`: called after each optimizer step to flush the prefix cache (and refresh Gemma3 derived weights).
- `InProcessRolloutClient.prepare` / `release`: map/unmap VMM physical memory around each rollout phase.

## Outbound Dependencies
- `liteopd.inference.*`: engine, scheduler, models, weight loading.
- `liteopd.train.fused_model`: expects fused QKV/gate-up projections before `share_weights()`.

## Inbound Dependents
- `liteopd.train.run_opd_training`: constructs `RuntimeCoordinator`, calls rollout methods each iteration.

## Notes
- `_EmbeddedScheduler` bypasses the IPC message queue; `receive_msg` / `send_result` are replaced with in-memory list operations.
- `flush_cache` resets the radix prefix cache and page table but preserves the dummy request's table entry.
- Emits a warning when the model is not a dense Qwen2.5/Qwen3/Llama/Gemma3, because the fused-weight-sharing path is only validated on those families.
