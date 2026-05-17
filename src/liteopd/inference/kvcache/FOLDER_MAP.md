# src/opd/inference/kvcache

## Purpose
KV cache storage and prefix reuse. Provides a paged pool abstraction and a radix-tree prefix cache that deduplicates common prompt prefixes across requests.

## Key Parts
- `base.py`: `BasePrefixCache`, `BaseCacheHandle`, `BaseKVCachePool`, `SizeInfo`, `MatchResult`, `InsertResult` — abstract interfaces and shared data types.
- `radix_cache.py`: `RadixPrefixCache` — LRU radix tree; `evict()` is best-effort (stops when no more evictable leaves remain). `evictable_leaf_size` counts only leaf nodes with `ref_count==0`.
- `mha_pool.py`: Standard dense KV cache pool (one contiguous tensor per layer).
- `vmm_pool.py`: `VMMKVCache` — CUDA VMM-backed pool; physical memory is mapped/unmapped around rollout phases to reclaim GPU memory during training without invalidating CUDA graph addresses.
- `__init__.py`: `create_kvcache_pool()` / `create_prefix_cache()` factory functions.

## Entry Points
- `create_kvcache_pool(config)`: called by `Engine.__init__`.
- `create_prefix_cache(device, type)`: called by `CacheManager.__init__` and `flush_cache`.
- `VMMKVCache.map_physical()` / `unmap_physical()`: called by `InProcessRolloutClient.prepare()` / `release()`.

## Outbound Dependencies
- `liteopd.inference.kernel.vmm`: VMM allocation primitives.
- `liteopd.inference.distributed`: TP rank for per-layer shard sizing.

## Inbound Dependents
- `liteopd.inference.scheduler.cache.CacheManager`: uses prefix cache for match/insert/evict.
- `liteopd.inference.engine.Engine`: allocates the KV pool.

## Notes
- `evictable_leaf_size` (not `evictable_size`) is the correct signal for admission control: only leaf nodes can actually be freed without evicting their children first.
