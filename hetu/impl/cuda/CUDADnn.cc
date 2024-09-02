#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/stream/CUDAStream.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {

static std::once_flag cudnn_device_init_flags[HT_MAX_GPUS_COMPILE_TIME];
static cudnnHandle_t cudnn_handles[HT_MAX_GPUS_COMPILE_TIME];

static void InitCudnn(int32_t device_id) {
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDNN_CALL(cudnnCreate(&cudnn_handles[device_id]));
  cudaStream_t cuda_stream = GetCUDAComputingStream(device_id).cuda_stream();
  if (cuda_stream)
    CUDNN_CALL(cudnnSetStream(cudnn_handles[device_id], cuda_stream));
}

void InitCudnnOnce(int32_t device_id) {
  std::call_once(cudnn_device_init_flags[device_id], InitCudnn, device_id);
}

} // namespace

cudnnHandle_t GetCudnnHandle(int32_t device_id) {
  InitCudnnOnce(device_id);
  return cudnn_handles[device_id];
}

} // namespace impl
} // namespace hetu
