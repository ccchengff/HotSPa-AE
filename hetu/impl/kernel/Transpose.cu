#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <chrono>

namespace hetu {
namespace impl {

// Out-of-place version of transpose and its gradient
/* It is replaced with in-place version. */
template <typename spec_t>
__global__ void transpose_kernel(const spec_t* input, spec_t* output,
                                 const int64_t* buf, uint32_t ndims,
                                 size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  const auto* in_strides = buf;
  const auto* out_strides = buf + ndims;
  const auto* perm = buf + ndims * 2;
  uint32_t i_idx = 0;
  uint32_t t = idx;
#pragma unroll
  for (uint32_t i = 0; i < ndims; ++i) {
    const uint32_t ratio = t / out_strides[i];
    t -= ratio * out_strides[i];
    i_idx += ratio * in_strides[perm[i]];
  }
  output[idx] = input[i_idx];
}

bool BatchTranspose(size_t ndims, int64_t* perm) {
  for (int i = 0; i < ndims - 2; ++i) {
    if (perm[i] != i)
      return false;
  }
  if (perm[ndims - 1] == ndims - 2 && perm[ndims - 2] == ndims - 1)
    return true;
  return false;
}

void TransposeCuda(const NDArray& input, NDArray& output, const HTAxes& perm,
                   const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->numel() == output->numel());

  auto ndim = static_cast<uint32_t>(input->ndim());
  auto ndim_ = static_cast<uint32_t>(output->ndim());
  HT_ASSERT(ndim == ndim_);
  const auto& in_dims = input->shape();
  const auto& out_dims = output->shape();
  HTShape buf(3 * ndim);

  int64_t in_stride = 1;
  int64_t out_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buf[i] = in_stride;
    buf[ndim + i] = out_stride;
    buf[ndim * 2 + i] = perm[i];
    in_stride *= in_dims[i];
    out_stride *= out_dims[i];
  }
  HT_ASSERT(in_stride == out_stride);
  size_t size = in_stride;

  if (size == 0)
    return;

  int device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  auto buf_arr = hetu::cuda::to_int64_ndarray(buf, device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TransposeCuda", [&]() {
      transpose_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
        buf_arr->data_ptr<int64_t>(), ndim, size);
    });
  NDArray::MarkUsedBy({input, output, buf_arr}, stream);
}

} // namespace impl
} // namespace hetu
