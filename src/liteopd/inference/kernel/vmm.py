from __future__ import annotations

import ctypes
import functools
from typing import TYPE_CHECKING, Any, Tuple

import torch

from .utils import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module

    class VMMAllocation:
        def map(self) -> None: ...
        def unmap(self) -> None: ...
        def get_ptr(self) -> int: ...
        def get_size(self) -> int: ...
        def is_mapped(self) -> bool: ...

else:
    VMMAllocation = Any


@functools.cache
def _load_vmm_module() -> Module:
    return load_aot("vmm", cuda_files=["vmm.cu"], extra_ldflags=["-lcuda"])


@functools.cache
def _get_vmm_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.VMMAllocation")
    class VMMAllocationImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return VMMAllocationImpl


def create_vmm_allocation(size_bytes: int, device: torch.device) -> VMMAllocation:
    _load_vmm_module()
    cls = _get_vmm_cls()
    return cls(size_bytes, device.index)


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DELETER_FUNC = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))


_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER_FUNC),
]

_DTYPE_MAP = {
    torch.float16: _DLDataType(2, 16, 1),  # kDLFloat = 2
    torch.bfloat16: _DLDataType(4, 16, 1),  # kDLBfloat = 4
    torch.float32: _DLDataType(2, 32, 1),
    torch.int32: _DLDataType(0, 32, 1),  # kDLInt = 0
    torch.int64: _DLDataType(0, 64, 1),
}


@_DELETER_FUNC
def _noop_deleter(ptr):
    pass


def tensor_from_vmm_ptr(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a PyTorch tensor backed by a VMM virtual address.

    The returned tensor's storage is NOT managed by PyTorch's caching allocator.
    The caller must ensure the VMM mapping is valid during any access.
    """
    dl_dtype = _DTYPE_MAP.get(dtype)
    if dl_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(2, device.index)  # kDLCUDA = 2
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = dl_dtype
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    managed.deleter = _noop_deleter

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(ctypes.addressof(managed), b"dltensor", None)

    tensor = torch.from_dlpack(capsule)

    # prevent GC of the ctypes structs while tensor is alive
    tensor.__vmm_prevent_gc = (managed, shape_arr)  # type: ignore[attr-defined]
    return tensor
