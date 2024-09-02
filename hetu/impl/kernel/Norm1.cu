#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__device__ spec_t sgn(spec_t x) {
    if (x == 0.0)
        return 0.0;
    return x / hetu::cuda::cuda_abs(x);
}

template <typename spec_t>
__global__ void norm_kernel(const spec_t* input, spec_t* output, size_t size, 
                            int64_t p, size_t before_dim_size,
                            size_t reduce_dim_size, size_t after_dim_size,
                            const OffsetCalculator* in_offset_calculator,
                            const OffsetCalculator* out_offset_calculator) {
  __shared__ spec_t shared_sum[32];

  size_t x = blockIdx.x / after_dim_size;
  size_t y = blockIdx.x % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  if (after_dim_size > 1) {
      stride = after_dim_size * blockDim.x;
      start_ptr = x * reduce_dim_size * after_dim_size + y
                  + threadIdx.x * after_dim_size;
      end_ptr = x * reduce_dim_size * after_dim_size + y
                + reduce_dim_size * after_dim_size;
  } else {
      size_t cols_per_thread =
          (reduce_dim_size + blockDim.x - 1) / blockDim.x;
      size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y
                              + reduce_dim_size * after_dim_size;
      start_ptr = x * reduce_dim_size * after_dim_size + y
                  + threadIdx.x * cols_per_thread * after_dim_size;
      end_ptr =
          min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
      stride = after_dim_size;
  }
  size_t output_ptr = x * after_dim_size + y;
  if (start_ptr >= end_ptr)
      return;

  spec_t sum_thread = 0;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
      auto in_offset = in_offset_calculator->get(ptr);
      sum_thread += hetu::cuda::cuda_pow(spec_t(hetu::cuda::cuda_abs(input[in_offset])), spec_t(p));
  }
  hetu::cuda::BlockReduceSum(sum_thread, shared_sum);
  if (threadIdx.x == 0) {
      auto out_offset = out_offset_calculator->get(output_ptr);
      output[out_offset] = hetu::cuda::cuda_pow(spec_t(sum_thread), spec_t(1.0 / p));
  }
}

template <typename spec_t>
__global__ void norm_gradient_kernel(const spec_t *input, const spec_t *norm,
                                     const spec_t *grad, spec_t *output, int64_t p,
                                     size_t reduce_dim_size,
                                     size_t after_dim_size, size_t size,
                                     const OffsetCalculator* in_offset_calculator,
                                     const OffsetCalculator* norm_offset_calculator,
                                     const OffsetCalculator* grad_offset_calculator,
                                     const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int na = idx / (reduce_dim_size * after_dim_size);
  int nc = (idx % (reduce_dim_size * after_dim_size)) % after_dim_size;
  int idx_y = na * after_dim_size + nc;

  auto in_offset = in_offset_calculator->get(idx);
  auto grad_offset = grad_offset_calculator->get(idx_y);
  spec_t input_val = input[in_offset];
  spec_t grad_val = grad[grad_offset];
  auto out_offset = out_offset_calculator->get(idx);

  if (p == 1) {
      output[out_offset] = sgn(input_val) * grad_val;
  } else if (p == 2) {
      auto norm_offset = norm_offset_calculator->get(idx_y);
      spec_t norm_val = norm[norm_offset];
      if (norm_val == 0)
          output[out_offset] = 0;
      else
          output[out_offset] = grad_val * input_val / norm_val;
  } else if (p > 2) {
      auto norm_offset = norm_offset_calculator->get(idx_y);
      spec_t norm_val = norm[norm_offset];
      if (norm_val == 0)
          output[out_offset] = 0;
      else
          output[out_offset] = input_val * hetu::cuda::cuda_pow(hetu::cuda::cuda_abs(input_val), spec_t(p - 2)) * grad_val
                        / hetu::cuda::cuda_pow(norm_val, spec_t(p - 1));
  }
}



void NormCuda(const NDArray& input, NDArray& output, int64_t dim, int64_t p, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t before_dim_size, reduce_dim_size, after_dim_size;
  before_dim_size = reduce_dim_size = after_dim_size = 1;
  for (int i = 0; i < input->ndim(); ++i) {
      if (i < dim)
          before_dim_size *= input->shape(i);
      else if (i == dim)
          reduce_dim_size = input->shape(i);
      else
          after_dim_size *= input->shape(i);
  }

  size_t size = before_dim_size * after_dim_size;

  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = 32;
  blocks.x = before_dim_size * after_dim_size;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "NormCuda", [&]() {
      norm_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size, p, 
        before_dim_size, reduce_dim_size, after_dim_size,
        in_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, in_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

void NormGradientCuda(const NDArray& input, const NDArray& output, const NDArray& output_grad,
                      NDArray& input_grad, int64_t dim, int64_t p, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);

  size_t reduce_dim_size, after_dim_size, size;
  reduce_dim_size = after_dim_size = size = 1;
  for (int i = 0; i < input->ndim(); ++i) {
      size *= input->shape(i);
      if (i == dim)
          reduce_dim_size = input->shape(i);
      else if (i > dim)
          after_dim_size *= input->shape(i);
  }

  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_offset_calculator_arr,
          out_grad_offset_calculator_arr, in_grad_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator,
                   *out_grad_offset_calculator, *in_grad_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) = 
    AllocOffsetCalculator(output_grad, stream);
  std::tie(in_grad_offset_calculator_arr, in_grad_offset_calculator) = 
    AllocOffsetCalculator(input_grad, stream);
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "NormGradientCuda", [&]() {
      norm_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), 
      input_grad->data_ptr<spec_t>(), p, reduce_dim_size, after_dim_size, size,
      in_offset_calculator, out_offset_calculator,
      out_grad_offset_calculator, in_grad_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, input_grad, output_grad, in_offset_calculator_arr,
                      out_offset_calculator_arr, out_grad_offset_calculator_arr,
                      in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
