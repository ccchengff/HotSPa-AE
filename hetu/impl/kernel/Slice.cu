#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

// Out-of-place version of slice and its gradient
/* It is replaced with in-place version. */
template <typename spec_t>
__global__ void slice_kernel(const spec_t* input, spec_t* output,
                             const int64_t* output_shape,
                             const int64_t* input_shape,
                             const int64_t* begin_pos, 
                             size_t ndim,
                             size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t tmp_index = idx;
  size_t i_index = 0;
  int64_t i_mat = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    int64_t offset = begin_pos[i] + tmp_index % output_shape[i];
    tmp_index /= output_shape[i];
    i_index += offset * i_mat;
    i_mat *= input_shape[i];
  }
  output[idx] = input[i_index];
}

template <typename spec_t>
__global__ void slice_gradient_kernel(const spec_t* input, spec_t* output,
                                      const int64_t* output_shape,
                                      const int64_t* input_shape,
                                      const int64_t* begin_pos,
                                      size_t ndim, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = 0;
  size_t tmp_index = idx;
  size_t i_index = 0;
  int64_t i_mat = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    int64_t offset = tmp_index % output_shape[i];
    if (offset < begin_pos[i] || offset >= begin_pos[i] + input_shape[i])
      return;
    tmp_index /= output_shape[i];
    i_index += (offset - begin_pos[i]) * i_mat;
    i_mat *= input_shape[i];
  }
  output[idx] = input[i_index];
}

void SliceCuda(const NDArray& input, NDArray& output, const HTShape& begin_pos,
               const Stream& stream) {
  HT_ASSERT(input->is_cuda()) << "Input is not on a host device.";
  HT_ASSERT(output->is_cuda()) << "Output is not on a host device.";
  HT_ASSERT(input->device() == output->device())
    << "input and output are not on the same host device. "
    << "Devices: (input) " << input->device() << " vs. (output) "
    << output->device();
  HT_ASSERT(input->ndim() == output->ndim())
    << "input and output has different dims. "
    << "input shape: " << input->shape()
    << ", output shape: " << output->shape();
  size_t ndim = input->ndim();
  size_t o_size = 1;
  for (int i = 0; i < ndim; ++i) {
    HT_ASSERT(begin_pos[i] >= 0);
    HT_ASSERT(begin_pos[i] + output->shape(i) <= input->shape(i))
    << begin_pos[i] << "," << output->shape(i) << "," << input->shape(i);
    o_size *= output->shape(i);
  }
  
  size_t size = o_size;
  if (size == 0)
    return;
  
  auto device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  auto pos_arr = hetu::cuda::to_int64_ndarray(begin_pos, device_id);
  auto i_shape_arr = hetu::cuda::to_int64_ndarray(input->shape(), device_id);
  auto o_shape_arr = hetu::cuda::to_int64_ndarray(output->shape(), device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SliceCuda", [&]() {
      slice_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
        o_shape_arr->data_ptr<int64_t>(), 
        i_shape_arr->data_ptr<int64_t>(), 
        pos_arr->data_ptr<int64_t>(), 
        ndim, size);
    });
  NDArray::MarkUsedBy({input, output, pos_arr, i_shape_arr, o_shape_arr}, stream);
}

void SliceGradientCuda(const NDArray& output_grad, NDArray& input_grad,
                       const HTShape& begin_pos, const Stream& stream) {
  HT_ASSERT(output_grad->is_cuda()) << "Output_grad is not on a host device.";
  HT_ASSERT(input_grad->is_cuda()) << "Input_grad is not on a host device.";
  HT_ASSERT(input_grad->device() == output_grad->device())
    << "input_grad and output_grad are not on the same host device. "
    << "Devices: (input_grad) " << input_grad->device() << " vs. (output_grad) "
    << output_grad->device();
  HT_ASSERT(input_grad->ndim() == output_grad->ndim())
    << "input and output grad has different dims. ";
  size_t ndim = output_grad->ndim();
  size_t o_size = 1;
  for (int i = 0; i < ndim; ++i) {
    HT_ASSERT(begin_pos[i] >= 0);
    HT_ASSERT(begin_pos[i] + output_grad->shape(i) <= input_grad->shape(i));
    o_size *= input_grad->shape(i);
  }

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  
  auto device_id = output_grad->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  auto pos_arr = hetu::cuda::to_int64_ndarray(begin_pos, device_id);
  auto i_shape_arr =
    hetu::cuda::to_int64_ndarray(output_grad->shape(), device_id);
  auto o_shape_arr =
    hetu::cuda::to_int64_ndarray(input_grad->shape(), device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output_grad->dtype(), spec_t, "SliceGradientCuda", [&]() {
      slice_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        output_grad->data_ptr<spec_t>(), input_grad->data_ptr<spec_t>(),
        o_shape_arr->data_ptr<int64_t>(), 
        i_shape_arr->data_ptr<int64_t>(), 
        pos_arr->data_ptr<int64_t>(), 
        ndim, size);
    });
  NDArray::MarkUsedBy({output_grad, input_grad, pos_arr, i_shape_arr, o_shape_arr}, stream);
}

} // namespace impl
} // namespace hetu
