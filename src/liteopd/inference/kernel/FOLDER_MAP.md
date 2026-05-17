# src/opd/inference/kernel

## Purpose
Low-level CUDA and Triton kernels used by the inference engine. Provides JIT-compiled index/scatter operations, MoE routing kernels, VMM memory management, and a pynccl wrapper for intra-node collective communication.

## Key Parts
- `index.py`: JIT-compiled gather/scatter index kernels (used for KV cache page table reads/writes).
- `moe_impl.py`: Triton MoE token-dispatch and combine kernels.
- `pynccl.py`: Python wrapper around NCCL for TP all-reduce / all-gather without spawning extra processes.
- `vmm.py`: CUDA Virtual Memory Management helpers — `create_vmm_allocation()`, `tensor_from_vmm_ptr()`.
- `store.py`: Kernel artifact cache / JIT compilation store.
- `radix.py`: Triton kernels supporting radix cache operations.
- `utils.py`: Shared kernel utilities.
- `triton/`: Triton kernel implementations (e.g. `fused_moe.py`).
- `csrc/`: C++/CUDA JIT source files compiled at runtime.

## Entry Points
- `index.py` functions: called by `CacheManager._page_to_token` and page table writes.
- `vmm.create_vmm_allocation` / `tensor_from_vmm_ptr`: called by `VMMKVCache.__init__`.
- `pynccl`: used by `distributed` module for TP collectives.

## Outbound Dependencies
- `torch`, `triton`, optional TVM FFI for JIT compilation.

## Inbound Dependents
- `liteopd.inference.kvcache.vmm_pool`: VMM allocation.
- `liteopd.inference.distributed`: pynccl collectives.
- `liteopd.inference.scheduler.cache`: index kernels for page table writes.
