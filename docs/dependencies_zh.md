# 环境依赖

[英文](dependencies.md) | [中文](dependencies_zh.md)

本文档列出 Lite-OPD 推理引擎各组件的环境依赖及其强弱程度。

## 依赖分类

- **弱依赖**：仅依赖 Python/C 语言版本或通用库版本。任何现代训练环境都满足。
- **强依赖**：依赖特定硬件、CUDA 驱动版本或 GPU 架构。需要确认目标环境是否支持。

---

## 核心框架依赖（弱）

| 依赖 | 版本要求 | 用途 | 说明 |
|------|----------|------|------|
| Python | ≥ 3.10 | 全局 | `match` 语句、`X \| None` 类型注解 |
| PyTorch | ≥ 2.0 | 训练 + 推理 | `torch.from_dlpack`、`torch.compile` 兼容 |
| transformers | ≥ 4.37 | 模型加载 | Qwen2/3 支持 |
| tvm_ffi | 项目内置 | C++ 扩展编译 | JIT/AOT 编译 CUDA kernel |
| flashinfer | ≥ 0.1 | 注意力 kernel | Paged attention + CUDA graph |
| triton | ≥ 2.2 | Triton kernel | store_cache 等算子 |

---

## CUDA Graph（弱依赖）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| CUDA Toolkit | ≥ 10.0 | CUDA graph API 从 10.0 引入 |
| PyTorch | ≥ 1.10 | `torch.cuda.CUDAGraph` API |

CUDA graph 是标准 CUDA 功能，所有现代 GPU（Volta+）和驱动都支持。**无强依赖**。

---

## VMM KV Cache（强依赖）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| CUDA Driver | ≥ 11.2 (driver ≥ 470.42) | VMM API 从 11.2 引入 |
| GPU 架构 | Compute Capability ≥ 7.0 (Volta+) | 硬件需支持虚拟内存管理 |
| `libcuda.so.1` | 系统级 | C++ 扩展链接 `-lcuda` |

### 详细说明

VMM 使用以下 CUDA Driver API（非 Runtime API）：

```
cuMemAddressReserve   — 预留虚拟地址空间
cuMemCreate           — 创建物理内存分配句柄
cuMemMap              — 将物理内存映射到虚拟地址
cuMemSetAccess        — 设置访问权限
cuMemUnmap            — 解除映射
cuMemRelease          — 释放物理内存
cuMemAddressFree      — 释放虚拟地址空间
cuMemGetAllocationGranularity — 查询对齐粒度
```

这些 API 从 CUDA 11.2 开始可用。在更早的驱动上：
- 编译可以通过（头文件中有声明）
- 运行时 `dlopen` 会找不到符号，或 API 返回 `CUDA_ERROR_NOT_SUPPORTED`

### 兼容性矩阵

| GPU | Compute Capability | VMM 支持 |
|-----|-------------------|-----------|
| V100 | 7.0 | 支持 |
| A100 | 8.0 | 支持 |
| H100 | 9.0 | 支持 |
| H20 | 9.0 | 支持 |
| RTX 3090 | 8.6 | 支持 |
| RTX 4090 | 8.9 | 支持 |
| T4 | 7.5 | 支持 |
| P100 | 6.0 | 不支持 |
| K80 | 3.7 | 不支持 |

### 降级策略

VMM 是可选功能（`use_vmm: false` 为默认值）。不启用时：
- KV cache 使用标准 `torch.empty()` 分配
- 显存在整个 rollout + training 周期内常驻
- CUDA graph 正常工作（地址本来就不变）
- 唯一代价：training 阶段可用显存减少

---

## FlashInfer（中等依赖）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| GPU 架构 | SM ≥ 80 (Ampere+) | FlashInfer 的高效 kernel 需要 SM80+ |
| CUDA Toolkit | ≥ 11.8 | 编译要求 |

在 SM < 80 的 GPU 上，FlashInfer 可能回退到较慢的实现或不可用。此时可切换到 `attention_backend: "fa"`（Flash Attention）。

---

## Triton Kernels（弱依赖）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| triton | ≥ 2.2 | JIT 编译 |
| GPU 架构 | SM ≥ 70 (Volta+) | Triton 支持范围 |

Triton kernel 在首次调用时 JIT 编译，对 CUDA 版本要求宽松。

---

## TVM FFI 编译系统（弱依赖）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| C++ 编译器 | C++17 | `g++ ≥ 7` 或 `clang++ ≥ 5` |
| nvcc | 与 PyTorch CUDA 版本匹配 | 编译 .cu 文件 |
| tvm_ffi | pip 安装 | 提供编译框架和 FFI 绑定 |

编译产物缓存在 `~/.cache/` 下，首次编译后不再重复。

---

## 分布式训练（弱依赖）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| NCCL | ≥ 2.10 | 通常随 PyTorch 安装 |
| 多 GPU | 可选 | 单卡也能运行 |

---

## 总结

```
                    强依赖（硬件/驱动）
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    VMM KV Cache    FlashInfer      (无其他)
    CUDA ≥ 11.2    SM ≥ 80
    CC ≥ 7.0
         │               │
         │               │
    可降级为            可切换为
    torch.empty        Flash Attention
         │               │
         └───────────────┼───────────────┘
                         │
                    弱依赖（通用）
                         │
    Python ≥ 3.10, PyTorch ≥ 2.0, C++17
    Triton ≥ 2.2, tvm_ffi, NCCL
```

所有强依赖组件都有降级路径。在标准训练集群（A100/H100/H20 + CUDA 12+）上，所有功能均可使用。
