# src/opd/inference/engine

## Purpose
Executes forward passes for prefill and decode batches. Owns the model instance, KV cache pool, CUDA graph runner, and token sampler. Does not handle scheduling or admission control.

## Key Parts
- `engine.py`: `Engine` — initializes model, KV cache, attention backend, graph runner, and sampler. Exposes `forward_batch()`.
- `graph.py`: `GraphRunner` — captures and replays CUDA graphs for decode batches; handles batch padding to graph-captured sizes.
- `sample.py`: `Sampler` / `BatchSamplingArgs` — top-k/top-p/greedy sampling after each forward pass. Current default is a vLLM-style native sampler (`temperature -> top-k/top-p -> softmax -> exponential-race sample`); `flashinfer` sampler is now opt-in via `OPD_USE_FLASHINFER_SAMPLER=1`.
- `config.py`: `EngineConfig` — all engine-level parameters (dtype, memory_ratio, page_size, cuda_graph_max_bs, etc.).

## Entry Points
- `Engine.forward_batch(batch, sample_args)`: called by `Scheduler._forward()` each step.
- `Engine.shutdown()`: releases distributed resources.

## Outbound Dependencies
- `liteopd.inference.kvcache`: KV pool allocation.
- `liteopd.inference.attention`: attention metadata and kernel dispatch.
- `liteopd.inference.models`: model forward.
- `liteopd.inference.distributed`: TP group setup.

## Inbound Dependents
- `liteopd.inference.scheduler.scheduler`: constructs `Engine`, calls `forward_batch`.
- `liteopd.runtime.rollout._EmbeddedScheduler`: constructs `Engine` directly with an externally-provided model.

## Notes
- `sample.py` no longer routes sampled decode through the old `flashinfer.softmax(...)->*_from_probs(...)` path by default. The default path is intentionally closer to vLLM for easier sampler-level debugging and A/B comparison.
- If runtime output still collapses to repeated token `0` / `"!"` after sampler-only validation passes, the remaining suspects move upstream/downstream of the sampler: model forward, weight sharing, or token result plumbing.
