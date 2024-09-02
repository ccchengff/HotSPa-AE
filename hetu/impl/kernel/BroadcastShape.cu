#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void broadcast_shape_kernel(const spec_t* input, spec_t* output,
                                       const int64_t* out_strides, const int64_t* in_dims,
                                       size_t ndims, size_t size,
                                       const OffsetCalculator* in_offset_calculator,
                                       const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t i_ind = 0;
  size_t temp = idx;
  for (int i = 0; i < ndims; ++i) {
    i_ind *= in_dims[i];
    i_ind += (in_dims[i] > 1) * temp / out_strides[i];
    temp %= out_strides[i];
  }
  auto in_offset = in_offset_calculator->get(i_ind);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = input[in_offset];
}

void BroadcastShapeCuda(const NDArray& input, NDArray& output,
                        const HTShape& add_axes, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  size_t size = output->numel();
  size_t input_size = input->numel();
  if (size == 0 || input_size == 0)
    return;

  int input_dim = input->ndim();
  int output_dim = output->ndim();
  HTShape in_dims(HT_MAX_NDIM);
  HTStride out_strides(HT_MAX_NDIM);
  size_t output_size = 1;
  size_t diff = output_dim - input_dim;
  if (add_axes.empty()) {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = i < diff ? 1 : input->shape(i - diff);
    }
  } else {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = 0;
    }
    for (int i = 0; i < diff; ++i) {
      in_dims[add_axes[i]] = 1;
    }
    int o_ind = 0;
    for (int i = 0; i < input->ndim(); ++i) {
      while (in_dims[o_ind++] == 1) {}
      in_dims[o_ind - 1] = input->shape(i);
    }
  }

  
  auto device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  auto in_dims_arr = hetu::cuda::to_int64_ndarray(in_dims, device_id);
  auto out_strides_arr = hetu::cuda::to_int64_ndarray(out_strides, device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastShapeCuda", [&]() {
      broadcast_shape_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
        out_strides_arr->data_ptr<int64_t>(),
        in_dims_arr->data_ptr<int64_t>(), 
        output_dim, size, in_offset_calculator,
        out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, in_dims_arr, out_strides_arr,
                      in_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

template <typename spec_t>
__global__ void
broadcast_shape_mul_kernel(const spec_t* input, spec_t const_value,
                           spec_t* output, const int64_t* out_strides,
                           const int64_t* in_dims, size_t ndims, size_t size,
                           const OffsetCalculator* in_offset_calculator,
                           const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t i_ind = 0;
  size_t temp = idx;
  for (int i = 0; i < ndims; ++i) {
    i_ind *= in_dims[i];
    i_ind += (in_dims[i] > 1) * temp / out_strides[i];
    temp %= out_strides[i];
  }
  auto in_offset = in_offset_calculator->get(i_ind);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = input[in_offset] * const_value;
}

void BroadcastShapeMulCuda(const NDArray& input, double const_value,
                           NDArray& output, const HTShape& add_axes,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  size_t size = output->numel();
  size_t input_size = input->numel();
  if (size == 0 || input_size == 0)
    return;

  int input_dim = input->ndim();
  int output_dim = output->ndim();
  HTStride out_strides(HT_MAX_NDIM);
  HTShape in_dims(HT_MAX_NDIM);
  int64_t output_size = 1;
  int diff = output_dim - input_dim;
  if (add_axes.empty()) {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = i < diff ? 1 : input->shape(i - diff);
    }
  } else {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = 0;
    }
    for (int i = 0; i < diff; ++i) {
      in_dims[add_axes[i]] = 1;
    }
    int o_ind = 0;
    for (int i = 0; i < input->ndim(); ++i) {
      while (in_dims[o_ind++] == 1) {}
      in_dims[o_ind - 1] = input->shape(i);
    }
  }

  auto device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  auto in_dims_arr = hetu::cuda::to_int64_ndarray(in_dims, device_id);
  auto out_strides_arr = hetu::cuda::to_int64_ndarray(out_strides, device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastShapeMulCuda", [&]() {
      broadcast_shape_mul_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), static_cast<spec_t>(const_value),
        output->data_ptr<spec_t>(), 
        out_strides_arr->data_ptr<int64_t>(), 
        in_dims_arr->data_ptr<int64_t>(), 
        output_dim, size, in_offset_calculator,
        out_offset_calculator);
    });

  NDArray::MarkUsedBy({input, output, in_dims_arr, out_strides_arr,
                      in_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
