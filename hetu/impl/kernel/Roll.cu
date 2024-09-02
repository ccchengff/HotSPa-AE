#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void roll_kernel(const spec_t* input, spec_t* output, size_t size,
                            int rank, const int64_t* shifts,
                            const int64_t* strides, const int64_t* sizes,
                            const OffsetCalculator* in_offset_calculator,
                            const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  int output_idx = idx;
  int new_dim_idx = 0;

#pragma unroll
  for (int i = 0; i < rank; i++) {
    new_dim_idx = (idx / strides[i]) % sizes[i] + shifts[i];
    if (new_dim_idx >= sizes[i])
      output_idx += (shifts[i] - sizes[i]) * strides[i];
    else
      output_idx += shifts[i] * strides[i];
  }
  auto in_offset = in_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(output_idx);
  output[out_offset] = input[in_offset];
}

void RollCuda(const NDArray& input, const HTShape& shift, const HTAxes& axis,
              NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t len = input->numel();
  int64_t nums = shift.size();
  int64_t n_dims = input->ndim();

  HTStride stride_dim(n_dims, 0);
  stride_dim[n_dims - 1] = 1;
  for (int i = 1; i < n_dims; i++) {
    stride_dim[n_dims - i - 1] =
      input->shape(n_dims - i) * stride_dim[n_dims - i];
  }

  HTStride strides(MAX(1, nums));
  HTShape sizes(MAX(1, nums));
  HTShape shifts(MAX(1, nums));
  if (axis.size() == 0) {
    strides[0] = 1;
    sizes[0] = len;
    shifts[0] = (shift[0] % len + len) % len;
  } else {
    for (int i = 0; i < nums; i++) {
      int dim = axis[i] >= 0 ? axis[i] : axis[i] + n_dims;
      int size = input->shape(dim);
      if (size != 0) {
        strides[i] = stride_dim[dim];
        sizes[i] = size;
        shifts[i] = (shift[i] % size + size) % size;
      }
    }
  }

  auto device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  auto strides_arr = hetu::cuda::to_int64_ndarray(strides, device_id);
  auto sizes_arr = hetu::cuda::to_int64_ndarray(sizes, device_id);
  auto shifts_arr = hetu::cuda::to_int64_ndarray(shifts, device_id);
  dim3 blocks, threads;
  threads.x = MIN(len, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(len, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RollCuda", [&]() {
      roll_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
        len, nums, 
        shifts_arr->data_ptr<int64_t>(), 
        strides_arr->data_ptr<int64_t>(), 
        sizes_arr->data_ptr<int64_t>(),
        in_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, shifts_arr, strides_arr, sizes_arr,
                       in_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
