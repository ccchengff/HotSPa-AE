#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void dynamic_concatenate_kernel(const spec_t* input, spec_t* output,
                                   int dynamic_input_width, int input_width, int output_width,
                                   int offset, int concat_size, size_t size,
                                   const OffsetCalculator* in_offset_calculator,
                                   const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int post_ind = idx % concat_size;
  int prev_ind = idx / concat_size;
  if (prev_ind % input_width >= dynamic_input_width) // the paddings
    return;
  int mid_ind = prev_ind % input_width + offset;
  prev_ind = prev_ind / input_width;
  int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
  auto in_offset = in_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(out_ind);
  output[out_offset] = input[in_offset];
}

void DynamicConcatenateCuda(const NDArray& input, NDArray& output, size_t axis,
                     size_t offset, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = input->numel();
  int now_ndim = output->ndim();
  HT_ASSERT(input->ndim() == now_ndim);
  int num_concats = 1;
  for (int i = 0; i < axis; ++i) {
    int cur_dim = output->shape(i);
    HT_ASSERT(input->dynamic_shape(i) == cur_dim);
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (int i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output->shape(i);
    HT_ASSERT(input->dynamic_shape(i) == cur_dim);
    concat_size *= cur_dim;
  }
  int input_width = input->shape(axis);
  int dynamic_input_width = input->dynamic_shape(axis);
  int output_width = output->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
    return;
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
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "DynamicConcatenateCuda", [&]() {
      dynamic_concatenate_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
        dynamic_input_width, input_width, output_width, offset, concat_size, size,
        in_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, in_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
