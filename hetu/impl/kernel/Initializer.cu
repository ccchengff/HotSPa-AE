#include "hetu/core/ndarray.h"
#include "hetu/impl/cuda/CUDARand.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/random/CUDARandomState.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void NormalInitsCuda(NDArray& data, double mean, double stddev, uint64_t seed,
                     const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  CUDAStream cuda_stream(stream);
  CUDARandomState rand_state = GetCUDARandomState(cuda_stream.device_id(), seed, 4);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "NormalInitsCuda", [&]() {
      launch_loop_kernel<spec_t>(data, size, stream,
                                 [=] __device__ (int idx) -> spec_t {
                                   curandStatePhilox4_32_10_t state;
                                   curand_init(rand_state.seed, idx, rand_state.offset, &state);
                                   return curand_normal(&state) *
                                          static_cast<spec_t>(stddev) +
                                          static_cast<spec_t>(mean);
                                 });
    });
  NDArray::MarkUsedBy(data, stream);
}

void UniformInitsCuda(NDArray& data, double lb, double ub, uint64_t seed,
                      const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  HT_ASSERT(lb < ub) << "Invalid range for uniform random init: "
                     << "[" << lb << ", " << ub << ").";
  size_t size = data->numel();
  if (size == 0)
    return;
  CUDAStream cuda_stream(stream);
  CUDARandomState rand_state = GetCUDARandomState(cuda_stream.device_id(), seed, 4);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "UniformInitCuda", [&]() {
      launch_loop_kernel<spec_t>(data, size, stream,
                                 [=] __device__ (int idx) -> spec_t {
                                   curandStatePhilox4_32_10_t state;
                                   curand_init(rand_state.seed, idx, rand_state.offset, &state);
                                   return curand_uniform(&state) *
                                          (static_cast<spec_t>(ub) - static_cast<spec_t>(lb)) +
                                          static_cast<spec_t>(lb);
                                 });
  });
  NDArray::MarkUsedBy(data, stream);
}

void TruncatedNormalInitsCuda(NDArray& data, double mean, double stddev,
                              double lb, double ub, uint64_t seed,
                              const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  CUDAStream cuda_stream(stream);
  CUDARandomState rand_state = GetCUDARandomState(cuda_stream.device_id(), seed, 32);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "TruncatedNormalInitsCuda", [&]() {
      launch_loop_kernel<spec_t>(data, size, stream,
                                 [=] __device__ (int idx) -> spec_t {
                                   curandStatePhilox4_32_10_t state;
                                   curand_init(rand_state.seed, idx, rand_state.offset, &state);
                                   spec_t val;
                                   do {
                                     val = curand_normal(&state) *
                                           static_cast<spec_t>(stddev) +
                                           static_cast<spec_t>(mean);
                                   } while (val < static_cast<spec_t>(lb) ||
                                            val > static_cast<spec_t>(ub));
                                   return val;
                                 });
  });
  NDArray::MarkUsedBy(data, stream);
}

} // namespace impl
} // namespace hetu
