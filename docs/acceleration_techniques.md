# Acceleration Techniques

[English](acceleration_techniques.md) | [Chinese](acceleration_techniques_zh.md)

This document describes the acceleration techniques used in Lite-OPD. It is divided into two parts: techniques specific to the Lite-OPD framework, and standard techniques commonly used in LLM training/inference.

---

## Part 1: Lite-OPD-Specific Techniques

### 1. Zero-Copy Weight Sharing (Rollout ↔ Training)

**Problem**: In conventional approaches, the inference engine and training model each hold a separate copy of the weights, requiring `copy_()` synchronization after each training step — wasting both memory and time.

**Solution**:
1. After loading the training model, fuse q/k/v → `qkv_proj` and gate/up → `gate_up_proj` (fused Parameters)
2. The inference engine's weight tensors reference the training model's Parameter data via `tensor.set_()`
3. After the optimizer updates the fused weights, the inference engine automatically sees the new values (same GPU memory)

**Benefits**:
- Eliminates per-step weight sync overhead (`refresh_from_model` becomes just a prefix cache flush)
- Saves one copy of model weights in memory (~14 GiB for a 7B model)
- Fused matmul is also more efficient than two separate matmuls

**Implementation**:
- `train/fused_model.py`: `fuse_model_projections()`, `unfuse_state_dict()`
- `runtime/weight_sync.py`: `share_weights()` establishes zero-copy sharing via `tensor.set_()`

---

### 2. Two-Stage Chunk-Level Backward

**Problem**: Distillation loss requires holding both student and teacher logits simultaneously (the vocab_size dimension is large). Computing loss and backward over the entire response at once results in peak activation memory of O(response_len × vocab_size), easily causing OOM.

**Solution**: Split backward into two stages:

```
Stage 1 (per-chunk, no retained graph):
  for each chunk of response tokens:
    hidden_leaf = student_hidden[chunk].detach().requires_grad_(True)
    loss = KL(lm_head(hidden_leaf), teacher_logits[chunk])
    grad_hidden = autograd.grad(loss, hidden_leaf)  # gradient w.r.t. hidden only
    accumulate grad_hidden into buffer
    # logits freed immediately; peak = O(chunk_size × vocab_size)

Stage 2 (single backward pass):
  torch.autograd.backward(student_hidden_packed, grad_tensors=accumulated_grad)
  # full backward through student backbone
```

**Benefits**:
- Peak memory reduced from O(total_response × vocab_size) to O(chunk_size × vocab_size)
- chunk_size is typically response_len / 8, yielding ~8x memory savings
- lm_head gradients accumulate across chunks in Stage 1 with no precision loss

**Implementation**: The `kl_backward_mode="two_stage"` branch in `batch_rollout_and_loss_with_client()` in `train/run_opd_training.py`.

---

### 3. ZeRO-2 Gradient Buffer Dynamic Release

**Problem**: The ZeRO-2 optimizer requires a gradient buffer (`_grad_shard_out`) equal in size to the parameter shard for reduce-scatter. During rollout, this buffer is completely idle but still occupies memory.

**Solution**:
- Before rollout: `optimizer.release_grad_buffer()` sets `_grad_shard_out` to None, freeing memory
- Before training: `optimizer.prepare_grad_buffer()` reallocates it

**Benefits**:
- For a 1.5B model + 2-GPU ZeRO-2, frees ~3 GB per GPU
- This memory becomes available for KV cache during rollout, increasing concurrency

**Implementation**: `release_grad_buffer()` / `prepare_grad_buffer()` in `train/zero2.py`.

---

### 4. Shortest-First Scheduling

**Problem**: In offline batch rollout, all requests are submitted simultaneously. With FIFO scheduling, long requests prefill first and occupy large amounts of KV cache, causing short requests to queue. The batch completion time (makespan) is determined by the last request to finish.

**Solution**: Schedule prefills in ascending order of **total target length** (input_len + max_output_len):

```python
# scheduler/utils.py
@property
def priority_key(self) -> tuple[int, int, int]:
    return (self.input_len + self.max_output_len, self.input_len, self.uid)
```

Short requests prefill first → enter decode first → finish first and release KV cache → free space for subsequent long requests.

**Combined with Longest-First Preemption**: When KV cache is insufficient, evict the decode request currently **occupying the most KV** (i.e., the one that has generated the most tokens). Evicted requests return to the pending queue for re-prefill. This ensures long requests (on the makespan critical path) are not interrupted by preemption of short requests.

**Benefits**:
- Reduces batch makespan by ~10% (depending on sequence length distribution variance)
- Reduces preemption count (short requests finish quickly, freeing space for long requests)

**Implementation**:
- `scheduler/utils.py`: `PendingReq.priority_key`
- `scheduler/prefill.py`: `PrefillManager._sort_pending()`
- `scheduler/scheduler.py`: `_preempt_longest()`

---

### 5. KV Cache Release During Training (VMM)

**Problem**: The inference engine's KV cache is completely idle during training, but conventional allocation cannot release it (CUDA graphs reference those addresses).

**Solution**: Use CUDA VMM (Virtual Memory Management) API to manage KV cache:
- At initialization: `cuMemAddressReserve` reserves virtual address space (permanent)
- During rollout: `cuMemCreate` + `cuMemMap` maps physical memory for normal inference
- During training: `cuMemUnmap` + `cuMemRelease` releases physical memory back for activations and teacher forward
- Next rollout: `cuMemCreate` + `cuMemMap` again (same virtual address) — CUDA graphs need no recapture

**Benefits**:
- Frees all KV cache memory during training (~40-60 GB for a 7B model + 200 concurrent requests)
- CUDA graphs are unaffected (virtual addresses remain unchanged)

**Implementation**: `kvcache/vmm_pool.py`, `kernel/csrc/src/vmm.cu`

---

### 6. Teacher Compile + Bucket Packing

**Problem**: Teacher forward is one of the compute bottlenecks in the training loop. `torch.compile` can accelerate teacher forward, but `flex_attention` internally uses `torch.compile` to compile attention kernels and `create_block_mask`, receiving `(Q_LEN, KV_LEN)` as Python ints. Each new sequence length combination triggers a guard failure → retrace → recompile, causing:
- Cumulative compilation time (several seconds per recompile)
- Compiled graph artifacts consuming GPU memory, potentially causing OOM during long training runs

**Solution**: Apply bucket padding to packed batches, aligning total token count up to multiples of 512:

```python
_PAD_BUCKET_SIZE = 512

def _round_up_to_bucket(n: int) -> int:
    return ((n + _PAD_BUCKET_SIZE - 1) // _PAD_BUCKET_SIZE) * _PAD_BUCKET_SIZE
```

Padding uses `pad_token_id` for `input_ids` and 0 for `position_ids`. This limits the possible `(Q_LEN, KV_LEN)` combinations from infinitely many to a finite set of buckets (e.g., 512, 1024, 1536, ...), bounding `flex_attention` compilation count.

**Benefits**:
- Compilation count reduced from O(num_steps) to O(max_tokens / bucket_size), typically < 64
- Eliminates memory growth from compiled artifacts, preventing OOM during long training
- Padding overhead is minimal (average waste < 256 tokens, < 1% of batch total)

**Implementation**: Bucket padding logic in the `flush()` function in `train/packing.py`.

---

## Part 2: Standard Techniques

The following are widely-used standard acceleration techniques in LLM training/inference.

### 7. Prefix Cache (Radix Tree)

Uses a radix tree to index computed KV cache. Requests sharing a common prefix reuse existing KV pages, avoiding redundant prefill.

**Implementation**: `kvcache/radix_cache.py`

---

### 8. Chunked Prefill

Splits long prompt prefill into multiple chunks (controlled by `max_extend_tokens`), preventing a single prefill from blocking decode requests. In Lite-OPD's offline batch scenario, `max_extend_tokens` is set to 65536 (effectively unlimited) since there are no in-flight decode requests that could be starved.

**Implementation**: `scheduler/prefill.py`

---

### 9. Paged Attention

KV cache is divided into fixed-size pages (default 16 tokens/page), managed via a page table. Sequences of different lengths allocate pages on demand with no padding waste.

**Implementation**: `scheduler/cache.py`, `kvcache/mha_pool.py`

---

### 10. CUDA Graph

Pre-captures the decode forward pass as a CUDA graph; replay skips all kernel launch overhead. Provides ~2-3x speedup during the decode phase.

**Implementation**: `engine/graph.py`

---

### 11. FlashInfer / sgl_kernel High-Performance Operators

Uses efficient paged attention kernels from FlashInfer and sgl_kernel, supporting CUDA graph-compatible paged KV cache access.

**Implementation**: `attention/fi.py`, `attention/fa.py`

---

### 12. ZeRO-2 Data Parallelism

Optimizer states are sharded across GPUs; parameters are recovered via all-gather. Supports multi-GPU training of large models with ~1/N memory usage compared to DDP.

**Implementation**: `train/zero2.py`

---

### 13. Sequence Packing

Packs multiple variable-length sequences into a single packed batch, eliminating padding waste. Both student forward and teacher forward use packing, with `position_ids` distinguishing sequence boundaries.

**Implementation**: `train/packing.py`

---

## Summary

| Technique | Category | Benefit |
|-----------|----------|---------|
| Zero-copy weight sharing | Lite-OPD-specific | Eliminates weight sync + saves one model copy |
| Two-stage chunk backward | Lite-OPD-specific | ~8x reduction in logits peak memory |
| ZeRO-2 buffer dynamic release | Lite-OPD-specific | ~3GB/GPU freed for KV cache during rollout |
| Shortest-first scheduling | Lite-OPD-specific | ~10% reduction in batch makespan |
| VMM KV cache release | Lite-OPD-specific | Frees all KV cache memory during training |
| Teacher compile + bucket packing | Lite-OPD-specific | Bounds compilation count, prevents memory growth |
| Prefix cache (radix tree) | Standard | Reuses KV for shared prefixes |
| Chunked prefill | Standard | Controls prefill latency and peak memory |
| Paged attention | Standard | Eliminates KV cache padding |
| CUDA graph | Standard | 2-3x decode phase speedup |
| FlashInfer / sgl_kernel | Standard | High-performance compute kernels |
| ZeRO-2 data parallelism | Standard | Multi-GPU large model training |
| Sequence packing | Standard | Eliminates padding, improves GPU utilization |
