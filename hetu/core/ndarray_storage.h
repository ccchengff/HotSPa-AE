#pragma once

#include "hetu/core/memory_pool.h"
#include "hetu/core/device.h"
#include <functional>

namespace hetu {

class NDArrayStorage {
 public:
  
  NDArrayStorage(DataPtr ptr, bool in_mempool = true): 
    _ptr(ptr), 
    _in_mempool(in_mempool) {
  } 

  ~NDArrayStorage() {
    if (_in_mempool) {
      FreeToMemoryPool(_ptr);
    } else {
      // deprecated: 这一部分使用mempool的borrow data
      // 内存由外界维护
      // 例如ncclMemAlloc和ncclMemFree
      return;
    }
  }

  inline size_t size() const {
    return _ptr.size;
  }

  inline void* mutable_data() {
    return _ptr.ptr;
  }

  inline const void* data() const {
    return _ptr.ptr;
  }

  inline const Device& device() const {
    return _ptr.device;
  }

  inline Device& device() {
    return _ptr.device;
  }

  inline DataPtr data_ptr() const {
    return _ptr;
  }

  inline bool is_new_malloc() const {
    return _ptr.is_new_malloc;
  }

 protected:
  DataPtr _ptr;
  bool _in_mempool;
  bool _writable{true};
};

} // namespace hetu
