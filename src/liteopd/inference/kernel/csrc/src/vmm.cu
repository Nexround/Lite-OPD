#include <minisgl/utils.h>

#include <cuda.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <string>

namespace {

#define CU_CHECK(expr)                                                         \
  do {                                                                         \
    CUresult __err = (expr);                                                   \
    if (__err != CUDA_SUCCESS) {                                               \
      const char *__msg = nullptr;                                             \
      cuGetErrorString(__err, &__msg);                                         \
      host::RuntimeCheck(false,                                                \
                         std::string("CUDA Driver Error: ") +                  \
                             (__msg ? __msg : "unknown"));                      \
    }                                                                          \
  } while (0)

struct VMMAllocation : public tvm::ffi::Object {
public:
  VMMAllocation(int64_t size_bytes, int device_ordinal)
      : m_device(device_ordinal), m_mapped(false), m_va(0), m_size(0),
        m_granularity(0), m_phys{} {
    host::RuntimeCheck(size_bytes > 0, "VMM allocation size must be positive");

    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, device_ordinal));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal;

    CU_CHECK(cuMemGetAllocationGranularity(
        &m_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    m_size =
        ((static_cast<size_t>(size_bytes) + m_granularity - 1) / m_granularity) *
        m_granularity;

    CU_CHECK(cuMemAddressReserve(&m_va, m_size, m_granularity, 0, 0));
    m_prop = prop;
  }

  ~VMMAllocation() {
    if (m_mapped) {
      cuMemUnmap(m_va, m_size);
      cuMemRelease(m_phys);
    }
    if (m_va != 0) {
      cuMemAddressFree(m_va, m_size);
    }
  }

  void map() {
    host::RuntimeCheck(!m_mapped, "VMM allocation is already mapped");
    CU_CHECK(cuMemCreate(&m_phys, m_size, &m_prop, 0));
    CU_CHECK(cuMemMap(m_va, m_size, 0, m_phys, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = m_device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(m_va, m_size, &access, 1));

    m_mapped = true;
  }

  void unmap() {
    host::RuntimeCheck(m_mapped, "VMM allocation is not mapped");
    CU_CHECK(cuMemUnmap(m_va, m_size));
    CU_CHECK(cuMemRelease(m_phys));
    m_mapped = false;
  }

  auto get_ptr() const -> int64_t {
    return static_cast<int64_t>(m_va);
  }

  auto get_size() const -> int64_t {
    return static_cast<int64_t>(m_size);
  }

  auto is_mapped() const -> bool { return m_mapped; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("minisgl.VMMAllocation", VMMAllocation,
                                    tvm::ffi::Object);

private:
  int m_device;
  bool m_mapped;
  CUdeviceptr m_va;
  size_t m_size;
  size_t m_granularity;
  CUmemGenericAllocationHandle m_phys;
  CUmemAllocationProp m_prop;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VMMAllocation>()
      .def(refl::init<int64_t, int>(), "__init__")
      .def("map", &VMMAllocation::map)
      .def("unmap", &VMMAllocation::unmap)
      .def("get_ptr", &VMMAllocation::get_ptr)
      .def("get_size", &VMMAllocation::get_size)
      .def("is_mapped", &VMMAllocation::is_mapped);
}

} // namespace
