#pragma once

#include "hetu/core/memory_pool.h"

namespace hetu {
namespace impl {

class CUDAMemoryPool : public MemoryPool {
 public:
  CUDAMemoryPool(DeviceIndex device_id, std::string&& name)
  : MemoryPool(Device(kCUDA, device_id), std::move(name)) {}

  ~CUDAMemoryPool() = default;

  inline size_t get_data_alignment() const noexcept {
    return 256;
  }
};

} // namespace impl
} // namespace hetu
