#include "hetu/core/ndarray.h"
#include "hetu/impl/cuda/CUDARand.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void dropout2d_kernel(const spec_t* input, spec_t* output,
                                 float drop_rate, size_t size,
                                 size_t last_two,
                                 const OffsetCalculator* in_offset_calculator,
                                 const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t leader = size_t(idx / last_two) * last_two;
  auto out_offset = out_offset_calculator->get(leader);
  float keep_mask = (float) (output[out_offset] >= drop_rate);
  auto in_offset = in_offset_calculator->get(idx);
  out_offset = out_offset_calculator->get(idx);
  output[out_offset] = input[in_offset] * keep_mask / (1 - drop_rate);
}

void Dropout2dCuda(const NDArray& input, double drop_rate, uint64_t seed,
                   NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(output, output);
  HT_ASSERT_SAME_SHAPE(input, output);
  size_t size = input->numel();
  if (size == 0)
    return;
  HT_ASSERT(input->ndim() == 4);
  size_t last_two_size = 1;
  last_two_size *= input->shape(input->ndim() - 1);
  last_two_size *= input->shape(input->ndim() - 2);

  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CURAND_CALL(curandSetStream(gen, cuda_stream));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "Dropout2dCuda", [&]() {
    curand_gen_uniform<spec_t>(gen, output->data_ptr<spec_t>(), size);
    dropout2d_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
      static_cast<float>(drop_rate), size, last_two_size,
      in_offset_calculator, out_offset_calculator);
  });
  CURAND_CALL(curandDestroyGenerator(gen));
  NDArray::MarkUsedBy({input, output, in_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

void Dropout2dGradientWithRecomputationCuda(const NDArray& grad,
                                            double drop_rate, uint64_t seed,
                                            NDArray& output,
                                            const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_SAME_DEVICE(grad, output);
  HT_ASSERT_SAME_SHAPE(grad, output);
  size_t size = grad->numel();
  if (size == 0)
    return;
  size_t last_two_size = 1;
  last_two_size *= grad->shape(grad->ndim() - 1);
  last_two_size *= grad->shape(grad->ndim() - 2);

  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray grad_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *grad_offset_calculator, *out_offset_calculator;
  std::tie(grad_offset_calculator_arr, grad_offset_calculator) =
    AllocOffsetCalculator(grad, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CURAND_CALL(curandSetStream(gen, cuda_stream));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "Dropout2dCuda", [&]() {
    curand_gen_uniform<spec_t>(gen, output->data_ptr<spec_t>(), size);
    dropout2d_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
      static_cast<float>(drop_rate), size, last_two_size,
      grad_offset_calculator, out_offset_calculator);
  });
  CURAND_CALL(curandDestroyGenerator(gen));
  NDArray::MarkUsedBy({grad, output, grad_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
