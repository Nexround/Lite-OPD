from __future__ import annotations

import torch
from liteopd.inference.distributed import get_tp_info
from liteopd.inference.kernel.vmm import create_vmm_allocation, tensor_from_vmm_ptr, VMMAllocation
from liteopd.inference.utils import div_even

from .base import BaseKVCachePool


class VMMKVCache(BaseKVCachePool):
    """KV cache backed by CUDA Virtual Memory Management.

    Physical memory is only allocated when mapped. Between rollout phases,
    physical memory can be released (unmap) while the virtual address stays
    stable, allowing CUDA graphs to be reused without re-capture.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)

        shape = (2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
        element_size = torch.tensor([], dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= s
        total_bytes = numel * element_size

        self._vmm_alloc = create_vmm_allocation(total_bytes, device)
        self._vmm_alloc.map()

        self._kv_buffer = tensor_from_vmm_ptr(
            self._vmm_alloc.get_ptr(), shape, dtype, device
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def map_physical(self) -> None:
        if not self._vmm_alloc.is_mapped():
            torch.cuda.empty_cache()
            self._vmm_alloc.map()

    def unmap_physical(self) -> None:
        if self._vmm_alloc.is_mapped():
            torch.cuda.synchronize(self._device)
            self._vmm_alloc.unmap()

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from liteopd.inference.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
