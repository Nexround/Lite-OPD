# Environment Dependencies

[English](dependencies.md) | [Chinese](dependencies_zh.md)

This document lists the environment dependencies for Lite-OPD's inference engine components and their severity levels.

## Dependency Classification

- **Weak dependency**: Only depends on Python/C language version or common library versions. Any modern training environment satisfies these.
- **Strong dependency**: Depends on specific hardware, CUDA driver version, or GPU architecture. Requires confirming target environment support.

---

## Core Framework Dependencies (Weak)

| Dependency | Version | Purpose | Notes |
|------------|---------|---------|-------|
| Python | ≥ 3.10 | Global | `match` statements, `X | None` type annotations |
| PyTorch | ≥ 2.0 | Training + Inference | `torch.from_dlpack`, `torch.compile` compatibility |
| transformers | ≥ 4.37 | Model loading | Qwen2/3 support |
| tvm_ffi | Bundled | C++ extension compilation | JIT/AOT compilation of CUDA kernels |
| flashinfer | ≥ 0.1 | Attention kernels | Paged attention + CUDA graph |
| triton | ≥ 2.2 | Triton kernels | store_cache and other operators |

---

## CUDA Graph (Weak Dependency)

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA Toolkit | ≥ 10.0 | CUDA graph API introduced in 10.0 |
| PyTorch | ≥ 1.10 | `torch.cuda.CUDAGraph` API |

CUDA graph is a standard CUDA feature supported by all modern GPUs (Volta+) and drivers. **No strong dependencies**.

---

## VMM KV Cache (Strong Dependency)

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA Driver | ≥ 11.2 (driver ≥ 470.42) | VMM API introduced in 11.2 |
| GPU Architecture | Compute Capability ≥ 7.0 (Volta+) | Hardware must support virtual memory management |
| `libcuda.so.1` | System-level | C++ extension links `-lcuda` |

### Details

VMM uses the following CUDA Driver API (not Runtime API):

```
cuMemAddressReserve   — Reserve virtual address space
cuMemCreate           — Create physical memory allocation handle
cuMemMap              — Map physical memory to virtual address
cuMemSetAccess        — Set access permissions
cuMemUnmap            — Unmap
cuMemRelease          — Release physical memory
cuMemAddressFree      — Free virtual address space
cuMemGetAllocationGranularity — Query alignment granularity
```

These APIs are available starting from CUDA 11.2. On older drivers:
- Compilation may succeed (declarations exist in headers)
- At runtime, `dlopen` will fail to find symbols, or APIs return `CUDA_ERROR_NOT_SUPPORTED`

### Compatibility Matrix

| GPU | Compute Capability | VMM Support |
|-----|-------------------|-------------|
| V100 | 7.0 | Supported |
| A100 | 8.0 | Supported |
| H100 | 9.0 | Supported |
| H20 | 9.0 | Supported |
| RTX 3090 | 8.6 | Supported |
| RTX 4090 | 8.9 | Supported |
| T4 | 7.5 | Supported |
| P100 | 6.0 | Not supported |
| K80 | 3.7 | Not supported |

### Fallback Strategy

VMM is an optional feature (`use_vmm: false` is the default). When disabled:
- KV cache uses standard `torch.empty()` allocation
- Memory remains resident throughout the rollout + training cycle
- CUDA graphs work normally (addresses don't change anyway)
- Only cost: reduced available memory during training phase

---

## FlashInfer (Medium Dependency)

| Dependency | Version | Notes |
|------------|---------|-------|
| GPU Architecture | SM ≥ 80 (Ampere+) | FlashInfer's efficient kernels require SM80+ |
| CUDA Toolkit | ≥ 11.8 | Compilation requirement |

On GPUs with SM < 80, FlashInfer may fall back to slower implementations or be unavailable. In that case, switch to `attention_backend: "fa"` (Flash Attention).

---

## Triton Kernels (Weak Dependency)

| Dependency | Version | Notes |
|------------|---------|-------|
| triton | ≥ 2.2 | JIT compilation |
| GPU Architecture | SM ≥ 70 (Volta+) | Triton support range |

Triton kernels are JIT-compiled on first invocation and have relaxed CUDA version requirements.

---

## TVM FFI Build System (Weak Dependency)

| Dependency | Version | Notes |
|------------|---------|-------|
| C++ compiler | C++17 | `g++ ≥ 7` or `clang++ ≥ 5` |
| nvcc | Matches PyTorch CUDA version | Compiles .cu files |
| tvm_ffi | pip install | Provides build framework and FFI bindings |

Build artifacts are cached under `~/.cache/`; no recompilation after first build.

---

## Distributed Training (Weak Dependency)

| Dependency | Version | Notes |
|------------|---------|-------|
| NCCL | ≥ 2.10 | Usually installed with PyTorch |
| Multi-GPU | Optional | Single GPU also works |

---

## Summary

```
                    Strong dependencies (hardware/driver)
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    VMM KV Cache    FlashInfer      (none other)
    CUDA ≥ 11.2    SM ≥ 80
    CC ≥ 7.0
         │               │
         │               │
    Can fall back to   Can switch to
    torch.empty        Flash Attention
         │               │
         └───────────────┼───────────────┘
                         │
                    Weak dependencies (universal)
                         │
    Python ≥ 3.10, PyTorch ≥ 2.0, C++17
    Triton ≥ 2.2, tvm_ffi, NCCL
```

All strong-dependency components have fallback paths. On standard training clusters (A100/H100/H20 + CUDA 12+), all features are available.
