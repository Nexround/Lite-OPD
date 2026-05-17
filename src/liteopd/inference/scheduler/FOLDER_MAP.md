# src/opd/inference/scheduler

## Purpose
Drives the prefill/decode scheduling loop: admission control, chunked prefill, preemption, and result dispatch. Sits between the engine (forward passes) and the I/O layer (message queues or in-memory lists).

## Key Parts
- `scheduler.py`: `Scheduler` — main loop (`normal_loop`, `overlap_loop`, `run_forever`), batch preparation, preemption logic (`_preempt_longest`), result processing.
- `prefill.py`: `PrefillManager` / `PrefillAdder` — greedy admission controller. `_kv_tokens_needed()` computes KV reservation per request: `(input_len - cached_len) + max(decoded_len, min(max_output_len, reserve))`. Admission now distinguishes `ADMITTED / SKIP / STOP` so a single unfittable pending request does not necessarily block later candidates in the same round.
- `decode.py`: `DecodeManager` — tracks running decode requests; `find_longest_preemptable()` selects the eviction victim.
- `cache.py`: `CacheManager` — paged KV allocation, prefix cache match/insert/evict, `available_size` (uses `evictable_leaf_size`), `total_capacity`.
- `table.py`: `TableManager` — per-request slot allocation and the shared `token_pool` / `page_table` tensors.
- `config.py`: `SchedulerConfig` — extends `EngineConfig` with scheduling parameters.
- `utils.py`: `PendingReq` — pending request dataclass with `decoded_len` (non-zero after preemption) and a shortest-seq-first `priority_key`.

## Entry Points
- `Scheduler.run_forever()`: out-of-process server mode.
- `Scheduler.normal_loop()`: called directly by `_EmbeddedScheduler.generate()`.
- `Scheduler.flush_cache()` (overridden in `_EmbeddedScheduler`): resets state between rollout batches.

## Outbound Dependencies
- `liteopd.inference.engine`: `Engine` for forward passes.
- `liteopd.inference.kvcache`: prefix cache creation.
- `liteopd.inference.core`: `Req`, `Batch`, `SamplingParams`.

## Inbound Dependents
- `liteopd.runtime.rollout._EmbeddedScheduler`: subclasses `Scheduler` for in-process offline use.

## Notes
- Decode-side preemption evicts the request with the largest `device_len` (input + decoded tokens), freeing the most KV pages per eviction.
- After preemption, the victim's already-generated tokens are preserved in `input_ids`; only the KV cache is freed, so re-admission costs one prefill rather than a full re-decode.
- All pending requests now share a single shortest-seq-first ordering rule based on `current_context_len + remaining_decode_budget`; this applies to fresh requests, chunked-prefill continuations, and preempted requests reinserted into the queue.
- Prefill admission can now skip a request that currently lacks KV headroom and continue scanning later requests; only hard round limits such as exhausted prefill budget or table slots stop the scan immediately.
- Periodic `loop=... kv=...` logs use `active_kv_tokens = sum(req.device_len for req in running_reqs)` as the numerator. Retained prefix-cache occupancy is reported separately in `prefix_cache(protected/evictable/evictable_leaf)`; the redundant `cache=...` percentage field has been removed from the periodic log line.
