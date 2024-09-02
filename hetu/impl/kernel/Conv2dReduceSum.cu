#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void conv2d_reduce_kernel(const spec_t* input, spec_t* output,
                                     size_t input_size, size_t output_size,
                                     size_t batch_size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= output_size)
    return;
  spec_t temp = 0;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < input_size; j++) {
      temp += input[i * input_size * output_size + idx * input_size + j];
    }
  }
  output[idx] = temp;
}

void Conv2dReduceSumCuda(const NDArray& input, NDArray& output,
                         const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  HT_ASSERT(input->shape(1) == output->shape(0));
  size_t batch_size = input->shape(0);
  size_t input_size = input->shape(2) * input->shape(3);
  size_t output_size = output->shape(0);
  size_t size = output_size;
  if (input_size == 0 || output_size == 0 || batch_size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "Conv2dReduceCuda", [&]() {
      conv2d_reduce_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), input_size,
        output_size, batch_size);
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
