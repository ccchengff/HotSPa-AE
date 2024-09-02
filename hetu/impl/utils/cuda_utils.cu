#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace cuda {

namespace {
thread_local int current_device_id = -1;
} // namespace

#if CUDA_VERSION >= 12000
void CudaTryGetDevice(int* device_id) {
  *device_id = current_device_id;
}

void CudaSetDevice(int device_id) {
  if (current_device_id != device_id) {
    CUDA_CALL(cudaSetDevice(device_id));
    current_device_id = device_id;
  }
}
#endif

NDArray to_int64_ndarray(const std::vector<int64_t>& vec,
                         DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(vec.size())},
                            Device(kCUDA, device_id), kInt64, kBlockingStream);
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CudaMemcpy(ret->raw_data_ptr(), vec.data(), vec.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  return ret;
}

NDArray to_int64_ndarray(const int64_t* from, size_t n, DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(n)}, Device(kCUDA, device_id),
                            kInt64, kBlockingStream);
  CudaMemcpy(ret->raw_data_ptr(), from, n * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  return ret;
}

NDArray to_byte_ndarray(const std::vector<uint8_t>& vec,
                        DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(vec.size())},
                            Device(kCUDA, device_id), kByte, kBlockingStream);
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CudaMemcpy(ret->raw_data_ptr(), vec.data(), vec.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  return ret;
}

NDArray to_byte_ndarray(const uint8_t* from, size_t n, DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(n)}, Device(kCUDA, device_id),
                            kByte, kBlockingStream);
  CudaMemcpy(ret->raw_data_ptr(), from, n * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  return ret;
}

} // namespace cuda
} // namespace hetu
