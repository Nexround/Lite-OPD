# src/opd/inference/kernel/csrc/src

## Purpose
Native C++/CUDA kernel source implementations for high-performance operators.

## Key Parts
- Source files implementing low-level CUDA kernels (index operations, memory management).

## Entry Points
- Compiled as Python extensions and called by the `kernel` Python wrappers above.

## Outbound Dependencies
- CUDA, PyTorch C++ extension API.

## Inbound Dependents
- `liteopd.inference.kernel`: Python-side wrappers load the compiled extensions.
