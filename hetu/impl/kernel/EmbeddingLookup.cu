#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include <chrono>

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void embedding_lookup_kernel(const spec_t* input, const int64_t* ids,
                                        size_t size, size_t length,
                                        size_t input_row, spec_t* output,
                                        const OffsetCalculator* in_offset_calculator,
                                        const OffsetCalculator* ids_offset_calculator,
                                        const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto ids_offset = ids_offset_calculator->get(idx / length);
  int64_t id = ids[ids_offset];
  auto out_offset = out_offset_calculator->get(idx);
  if (id < 0 || id >= input_row) {
    output[out_offset] = 0;
  } else {
    auto in_offset = in_offset_calculator->get(length * id + (idx % length));
    output[out_offset] = input[in_offset];
  }
}

template <typename spec_t>
__global__ void embedding_lookup_gradient_kernel(const spec_t* output_grad,
                                                 const int64_t* ids, size_t size,
                                                 size_t length, size_t input_row,
                                                 spec_t* input_grad,
                                                 const OffsetCalculator* out_grad_offset_calculator,
                                                 const OffsetCalculator* ids_offset_calculator,
                                                 const OffsetCalculator* in_grad_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto ids_offset = ids_offset_calculator->get(idx / length);
  int id = int(ids[ids_offset]);
  // for distributed gradient compute
  if (id < 0 || id >= input_row)
    return;  
  auto in_grad_offset = in_grad_offset_calculator->get(length * id + (idx % length));
  auto out_grad_offset = out_grad_offset_calculator->get(idx);
  hetu::cuda::AtomicAdd((input_grad + in_grad_offset), (output_grad[out_grad_offset]));
}

void EmbeddingLookupCuda(const NDArray& input, const NDArray& id,
                         NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, id);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->ndim() == 2)
    << "input_dim is invalid.Expect 2,but get " << input->ndim();

  for (int i = 0; i < output->ndim(); i++) {
    if (i < output->ndim() - 1) {
      HT_ASSERT(id->shape(i) == output->shape(i));
    } else if (i == output->ndim() - 1) {
      HT_ASSERT(input->shape(1) == output->shape(i));
    }
  }
  size_t input_row = input->shape(0);
  size_t length = input->shape(1);
  size_t size = id->numel() * input->shape(1);
  if (size == 0 || input_row == 0 || length == 0)
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
    input->dtype(), spec_t, "EmbbedingLookupCuda", [&]() {
      embedding_lookup_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), id->data_ptr<int64_t>(), size, length,
        input_row, output->data_ptr<spec_t>(), in_offset_calculator,
        id_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, id, output, in_offset_calculator_arr,
                      id_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

void EmbeddingLookupGradientCuda(const NDArray& output_grad, const NDArray& id,
                                 NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, id);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);
  HT_ASSERT(input_grad->ndim() == 2)
    << "input_dim is invalid.Expect 2,but get " << input_grad->ndim();

  for (int i = 0; i < output_grad->ndim(); i++) {
    if (i < output_grad->ndim() - 1) {
      HT_ASSERT(id->shape(i) == output_grad->shape(i));
    } else if (i == output_grad->ndim() - 1) {
      HT_ASSERT(input_grad->shape(1) == output_grad->shape(i));
    }
  }
  size_t input_row = input_grad->shape(0);  
  size_t length = input_grad->shape(1);
  size_t size = input_grad->numel();
  if (size == 0 || length == 0)
    return;
  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_grad_offset_calculator_arr, id_offset_calculator_arr,
          out_grad_offset_calculator_arr;
  OffsetCalculator *in_grad_offset_calculator, *id_offset_calculator,
                   *out_grad_offset_calculator;
  std::tie(in_grad_offset_calculator_arr, in_grad_offset_calculator) =
    AllocOffsetCalculator(input_grad, stream);
  std::tie(id_offset_calculator_arr, id_offset_calculator) =
    AllocOffsetCalculator(id, stream);
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) = 
    AllocOffsetCalculator(output_grad, stream);
  NDArray::zeros_(input_grad, stream.stream_index());
  size_t size2 = output_grad->numel();
  threads.x = MIN(size2, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size2, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "EmbeddingLookupGradientCuda", [&]() {
      embedding_lookup_gradient_kernel<spec_t>
        <<<blocks, threads, 0, cuda_stream>>>(
          output_grad->data_ptr<spec_t>(), id->data_ptr<int64_t>(), size2, length,
          input_row, input_grad->data_ptr<spec_t>(), out_grad_offset_calculator,
          id_offset_calculator, in_grad_offset_calculator);
    });
  NDArray::MarkUsedBy({output_grad, id, input_grad, out_grad_offset_calculator_arr,
                      id_offset_calculator_arr, in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu