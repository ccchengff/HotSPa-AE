#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void outer_kernel(const spec_t* inputA, const spec_t* inputB, size_t sizeB,
                             size_t size, spec_t* output,
                             const OffsetCalculator* A_offset_calculator,
                             const OffsetCalculator* B_offset_calculator,
                             const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int64_t A_idx = idx / sizeB; 
  int64_t B_idx = idx % sizeB;
  auto A_offset = A_offset_calculator->get(A_idx);
  auto B_offset = B_offset_calculator->get(B_idx);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = inputA[A_offset] * inputB[B_offset];
}

void OuterCuda(const NDArray& inputA, const NDArray& inputB, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, inputB);
  HT_ASSERT_SAME_DEVICE(inputA, output);

  size_t size = output->numel();
  size_t sizeB = inputB->numel();
  if (size == 0)
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
    inputA->dtype(), spec_t, "OuterCuda", [&]() {
      outer_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(), sizeB, size, output->data_ptr<spec_t>(),
        A_offset_calculator, B_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({inputA, inputB, output, A_offset_calculator_arr,
                      B_offset_calculator_arr, out_offset_calculator_arr}, stream);
}


} // namespace impl
} // namespace hetu
