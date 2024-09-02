#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void conv2d_broadcast_kernel(const spec_t* input, spec_t* output,
                                        size_t input_size, size_t output_size,
                                        size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t input_id = (idx % (input_size * output_size)) / output_size;
  output[idx] = input[input_id];
}

void Conv2dBroadcastCuda(const NDArray& input, NDArray& output,
                         const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  (input->shape(1) == output->shape(0));
  HT_ASSERT(input->shape(0) == output->shape(1));
  size_t batch_size = output->shape(0);
  size_t input_size = input->shape(0);
  size_t output_size = (output->shape(2)) * (output->shape(3));
  size_t size = batch_size * input_size * output_size;
  if (input_size == 0 || output_size == 0 || batch_size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "Conv2dBroadcastCuda", [&]() {
      conv2d_broadcast_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), input_size,
        output_size, size);
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
