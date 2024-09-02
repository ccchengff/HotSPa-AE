#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void SinCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SinCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t>(input, output, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return hetu::cuda::cuda_sin(x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void CosCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "CosCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t>(input, output, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return hetu::cuda::cuda_cos(x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void SinGradientCuda(const NDArray& input, const NDArray& output_grad,
                     NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_SAME_SHAPE(input, output_grad);
  HT_ASSERT_SAME_SHAPE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SinGradientCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t, spec_t>(input, output_grad, input_grad, size, stream,
                                                 [] __device__ (spec_t x, spec_t y) -> spec_t {
                                                   return y * hetu::cuda::cuda_cos(x);
                                                });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

void CosGradientCuda(const NDArray& input, const NDArray& output_grad,
                     NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_SAME_SHAPE(input, output_grad);
  HT_ASSERT_SAME_SHAPE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "CosGradientCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t, spec_t>(input, output_grad, input_grad, size, stream,
                                                 [] __device__ (spec_t x, spec_t y) -> spec_t {
                                                   return -y * hetu::cuda::cuda_sin(x);
                                                });
  });
  NDArray::MarkUsedBy({input, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
