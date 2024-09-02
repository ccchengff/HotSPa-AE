#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void gather_kernel(const spec_t* input, const int64_t* ids, size_t size, 
                              size_t after_stride, size_t cur_stride,
                              size_t after_stride_out, size_t cur_stride_out, spec_t* output,
                              const OffsetCalculator* in_offset_calculator,
                              const OffsetCalculator* ids_offset_calculator,
                              const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t b_index = idx / (cur_stride_out * after_stride_out);
  size_t p_index = idx % (cur_stride_out * after_stride_out);
  size_t a_index = p_index % after_stride_out;
  auto ids_offset = ids_offset_calculator->get(idx);
  size_t id_num = int(ids[ids_offset]);
  size_t i_index =
    b_index * (cur_stride * after_stride) + id_num * after_stride + a_index;
  auto in_offset = in_offset_calculator->get(i_index);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = input[in_offset];
}

template <typename spec_t>
__global__ void gather_gradient_kernel(const spec_t* grad_output, const int64_t* ids,
                                       size_t size, size_t after_stride, size_t cur_stride,
                                       size_t after_stride_out, size_t cur_stride_out,
                                       spec_t* grad_input,
                                       const OffsetCalculator* grad_out_offset_calculator,
                                       const OffsetCalculator* ids_offset_calculator,
                                       const OffsetCalculator* grad_in_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t b_index = idx / (cur_stride_out * after_stride_out);
  size_t p_index = idx % (cur_stride_out * after_stride_out);
  size_t a_index = p_index % after_stride_out;
  auto ids_offset = ids_offset_calculator->get(idx);
  size_t id_num = int(ids[ids_offset]);
  size_t i_index =
    b_index * (cur_stride * after_stride) + id_num * after_stride + a_index;
  auto grad_out_offset = grad_out_offset_calculator->get(idx);
  auto grad_in_offset = grad_in_offset_calculator->get(i_index);
  hetu::cuda::AtomicAdd(grad_input + grad_in_offset, grad_output[idx]);
}

void GatherCuda(const NDArray& input, const NDArray& id, NDArray& output,
                size_t dim, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, id);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(id->ndim() == input->ndim())
    << "invalid index shape.Expect dim=1, but get" << id->ndim();
  size_t after_stride = 1, after_stride_out = 1;
  size_t cur_stride = input->shape(dim), cur_stride_out = output->shape(dim);
  HT_ASSERT(id->shape() == output->shape())
    << "Invalid shapes.Index shape:" << id->shape()
    << "Input shape:" << input->shape() << "Output shape:" << output->shape();
  for (size_t i = dim + 1; i < input->ndim(); ++i) {
    after_stride *= input->shape(i);
    after_stride_out *= output->shape(i);
  }
  size_t size = output->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, id_offset_calculator_arr,
          out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *id_offset_calculator,
                   *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(id_offset_calculator_arr, id_offset_calculator) =
    AllocOffsetCalculator(id, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "GatherCuda", [&]() {
      gather_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), id->data_ptr<int64_t>(), size,
        after_stride, cur_stride, after_stride_out, cur_stride_out, output->data_ptr<spec_t>(),
        in_offset_calculator, id_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, id, output, in_offset_calculator_arr,
                      id_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

void GatherGradientCuda(const NDArray& grad_output, const NDArray& id, NDArray& grad_input,
                        size_t dim, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad_output);
  HT_ASSERT_SAME_DEVICE(grad_output, id);
  HT_ASSERT_SAME_DEVICE(grad_output, grad_input);
  HT_ASSERT(id->ndim() == grad_input->ndim())
    << "invalid index shape.Expect dim=1, but get" << id->ndim();
  size_t after_stride = 1, after_stride_out = 1;
  size_t cur_stride = grad_input->shape(dim), cur_stride_out = grad_output->shape(dim);
  for (size_t i = dim + 1; i < grad_input->ndim(); ++i) {
    after_stride *= grad_input->shape(i);
    after_stride_out *= grad_output->shape(i);
  }
  size_t size = grad_output->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray grad_out_offset_calculator_arr, id_offset_calculator_arr,
          grad_in_offset_calculator_arr;
  OffsetCalculator *grad_out_offset_calculator, *id_offset_calculator,
                   *grad_in_offset_calculator;
  std::tie(grad_out_offset_calculator_arr, grad_out_offset_calculator) =
    AllocOffsetCalculator(grad_output, stream);
  std::tie(id_offset_calculator_arr, id_offset_calculator) =
    AllocOffsetCalculator(id, stream);
  std::tie(grad_in_offset_calculator_arr, grad_in_offset_calculator) = 
    AllocOffsetCalculator(grad_input, stream);
  NDArray::zeros_(grad_input, stream.stream_index());
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_FLOATING_TYPES(
    grad_output->dtype(), spec_t, "GatherGradientCuda", [&]() {
      gather_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad_output->data_ptr<spec_t>(), id->data_ptr<int64_t>(), size, 
        after_stride, cur_stride, after_stride_out, cur_stride_out, grad_input->data_ptr<spec_t>(),
        grad_out_offset_calculator, id_offset_calculator, grad_in_offset_calculator);
    });
  NDArray::MarkUsedBy({grad_output, id, grad_input, grad_out_offset_calculator_arr,
                      id_offset_calculator_arr, grad_in_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
