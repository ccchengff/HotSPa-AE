#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void update_scale_kernel(spec_t* scale, int* growth_tracker,
                                    float* found_inf, double growth_factor,
                                    double backoff_factor, int growth_interval,
                                    size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (*found_inf) {
    *scale = (*scale) * backoff_factor;
    *growth_tracker = 0;
  } else {
    int successful = (*growth_tracker) + 1;
    if (successful == growth_interval) {
      *scale = (*scale) * growth_factor;
      *growth_tracker = 0;
    } else 
      *growth_tracker = successful;
  }
}

void UpdateScaleCuda(NDArray& scale, NDArray& growth_tracker,
                     const NDArray& found_inf, double growth_factor,
                     double backoff_factor, int growth_interval, 
                     const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(scale);
  HT_ASSERT_SAME_DEVICE(scale, growth_tracker);

  size_t size = scale->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    scale->dtype(), spec_t, "CheckFiniteCuda", [&]() {
      update_scale_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        scale->data_ptr<spec_t>(), growth_tracker->data_ptr<int>(),
        found_inf->data_ptr<float>(), growth_factor, backoff_factor,
        growth_interval, size);
    });
  NDArray::MarkUsedBy({scale, growth_tracker, found_inf}, stream); 
}

} // namespace impl
} // namespace hetu
