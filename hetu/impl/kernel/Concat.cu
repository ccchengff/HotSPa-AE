#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void Concat_kernel(const spec_t* inputA, const spec_t* inputB,
                              size_t size, size_t concat_size, size_t offset1,
                              size_t offset2, spec_t* output,
                              const OffsetCalculator* A_offset_calculator,
                              const OffsetCalculator* B_offset_calculator,
                              const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int all_offset = offset1 + offset2;
  int post_ind = idx % concat_size;
  int temp = idx / concat_size;
  int mid_ind = temp % all_offset;
  int pre_ind = temp / all_offset;
  spec_t val;
  if (mid_ind < offset1) {
    int x_ind = (pre_ind * offset1 + mid_ind) * concat_size + post_ind;
    auto A_offset = A_offset_calculator->get(x_ind);
    val = inputA[A_offset];
  } else {
    int y_ind =
      (pre_ind * offset2 + mid_ind - offset1) * concat_size + post_ind;
    auto B_offset = B_offset_calculator->get(y_ind);
    val = inputB[B_offset];
  }
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = val;
}

template <typename spec_t>
__global__ void Concat_gradient_kernel(const spec_t* output_grad, size_t size,
                                       size_t concat_size, size_t concat_offset,
                                       size_t small_offset, size_t big_offset,
                                       spec_t* input_grad,
                                       const OffsetCalculator* out_grad_offset_calculator,
                                       const OffsetCalculator* in_grad_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int post_ind = idx % concat_size;
  int temp = idx / concat_size;
  int mid_ind = temp % small_offset + concat_offset;
  int pre_ind = temp / small_offset;
  int o_idx = (pre_ind * big_offset + mid_ind) * concat_size + post_ind;
  auto out_grad_offset = out_grad_offset_calculator->get(o_idx);
  auto in_grad_offset = in_grad_offset_calculator->get(idx);
  input_grad[in_grad_offset] = output_grad[out_grad_offset];
}

void ConcatCuda(const NDArray& inputA, const NDArray& inputB, NDArray& output,
                size_t axis, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, inputB);
  HT_ASSERT_SAME_DEVICE(inputA, output);

  size_t size = output->numel();
  size_t offset1 = inputA->shape(axis);
  size_t offset2 = inputB->shape(axis);
  int concat_size = 1;
  int now_ndim = inputA->ndim();
  for (int i = axis + 1; i < now_ndim; i++) {
    int cur_dim = inputA->shape(i);
    concat_size *= cur_dim;
  }
  if (size == 0 || offset1 == 0 || offset2 == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray A_offset_calculator_arr, B_offset_calculator_arr,
          out_offset_calculator_arr;
  OffsetCalculator *A_offset_calculator, *B_offset_calculator,
                   *out_offset_calculator;
  std::tie(A_offset_calculator_arr, A_offset_calculator) =
    AllocOffsetCalculator(inputA, stream);
  std::tie(B_offset_calculator_arr, B_offset_calculator) =
    AllocOffsetCalculator(inputB, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) =
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "ConcatCuda", [&]() {
      Concat_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(), size,
        concat_size, offset1, offset2, output->data_ptr<spec_t>(),
        A_offset_calculator, B_offset_calculator,
        out_offset_calculator);
    });
  NDArray::MarkUsedBy({inputA, inputB, output, A_offset_calculator_arr,
                      B_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

void ConcatGradientCuda(const NDArray& output_grad, NDArray& input_grad,
                        size_t axis, size_t id, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);

  size_t size = input_grad->numel();
  size_t big_offset = output_grad->shape(axis);
  size_t small_offset = input_grad->shape(axis);
  size_t concat_offset = (id == 1) ? (big_offset - small_offset) : 0;
  size_t concat_size = 1;
  size_t now_ndim = output_grad->ndim();
  for (int i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output_grad->shape(i);
    concat_size *= cur_dim;
  }
  if (size == 0 || small_offset == 0 || big_offset == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray out_grad_offset_calculator_arr, in_grad_offset_calculator_arr;
  OffsetCalculator *out_grad_offset_calculator, *in_grad_offset_calculator;
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) =
    AllocOffsetCalculator(output_grad, stream);
  std::tie(in_grad_offset_calculator_arr, in_grad_offset_calculator) = 
    AllocOffsetCalculator(input_grad, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ConcatGradientCuda", [&]() {
      Concat_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        output_grad->data_ptr<spec_t>(), size, concat_size, concat_offset,
        small_offset, big_offset, input_grad->data_ptr<spec_t>(),
        out_grad_offset_calculator, in_grad_offset_calculator);
    });
  NDArray::MarkUsedBy({output_grad, input_grad, out_grad_offset_calculator_arr,
                      in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
