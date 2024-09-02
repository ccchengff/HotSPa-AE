#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/numeric_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__forceinline__ __device__ void WarpReduceArgmax(spec_t& val, int used_warp_size) {
  spec_t tmp_val;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  unsigned int active_warp_size = used_warp_size;
  while (active_warp_size >= 2) {
    unsigned int k = active_warp_size;
    k >>= 1;
    tmp_val = hetu::cuda::shfl_down_sync(mask, val, k, warpSize);
    if (tmp_val > val) {
      val = tmp_val;
    }
    active_warp_size = active_warp_size - k;
  }
}

template <>
__forceinline__ __device__ void WarpReduceArgmax(bfloat16& val, int used_warp_size) {
  #if (__CUDA_ARCH__ >= 800)
  bfloat16 tmp_val;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  unsigned int active_warp_size = used_warp_size;
  while (active_warp_size >= 2) {
    unsigned int k = active_warp_size;
    k >>= 1;
    tmp_val = hetu::cuda::shfl_down_sync(mask, val, k, warpSize);
    if (tmp_val > val) {
      val = tmp_val;
    }
    active_warp_size = active_warp_size - k;
  }
  #else
  float val_f = float(val);
  float tmp_val;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  unsigned int active_warp_size = used_warp_size;
  while (active_warp_size >= 2) {
    unsigned int k = active_warp_size;
    k >>= 1;
    tmp_val = hetu::cuda::shfl_down_sync(mask, val_f, k, warpSize);
    if (tmp_val > val_f) {
      val = bfloat16(tmp_val);
    }
    active_warp_size = active_warp_size - k;
  }
  #endif
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceArgmax(spec_t& val,
                                                  spec_t* shared_value,
                                                  spec_t* wrap_max,
                                                  size_t idx,
                                                  size_t threads_per_pos,
                                                  size_t used_threads_per_pos) {
  int thread_id = threadIdx.x % threads_per_pos;
  int wid = thread_id / warpSize;
  int tid = thread_id % warpSize;
  int epilogue_size = used_threads_per_pos - wid * warpSize;
  int used_warp_size = min(epilogue_size, warpSize);

  WarpReduceArgmax(val, used_warp_size);

  __syncthreads();
  if (tid == 0) {
    shared_value[idx * warpSize + wid] = val;
  }

  __syncthreads();
  val = (thread_id < threads_per_pos / warpSize) ? shared_value[idx * warpSize + tid] : numeric_limits<spec_t>::lowest();

  if (wid == 0) {
    WarpReduceArgmax(val, used_warp_size);
    if (thread_id == 0)
      wrap_max[idx] = val;
  }
}

// QUESTION: how to fix the epilogue?
template <typename spec_t>
__forceinline__ __device__ spec_t WarpReduceSumExp(spec_t val, int used_warp_size) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += hetu::cuda::shfl_down_sync(mask, val, k, warpSize);
  return val;
}

// QUESTION: how to fix the epilogue?
template <>
__forceinline__ __device__ bfloat16 WarpReduceSumExp(bfloat16 val, int used_warp_size) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  #if(__CUDA_ARCH__ >= 800)
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += hetu::cuda::shfl_down_sync(mask, val, k, warpSize);
  #else
  float val_f = float(val);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val_f += hetu::cuda::shfl_down_sync(mask, val_f, k, warpSize);    
  val = bfloat16(val_f);
  #endif
  return val;
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceSumExp(spec_t& val,
                                                  spec_t* shared,
                                                  spec_t* wrap_sum,
                                                  size_t idx,
                                                  size_t threads_per_pos,
                                                  size_t used_threads_per_pos) {
  int thread_id = threadIdx.x % threads_per_pos;
  int tid = thread_id % warpSize;
  int wid = thread_id / warpSize;
  int epilogue_size = used_threads_per_pos - wid * warpSize;
  int used_warp_size = min(epilogue_size, warpSize);

  val = WarpReduceSumExp(val, used_warp_size);

  __syncthreads();
  if (tid == 0)
    shared[idx * threads_per_pos + wid] = val;

  __syncthreads();
  spec_t zero = 0;
  val = (thread_id < threads_per_pos / warpSize) ? shared[idx * threads_per_pos + tid] : zero;

  if (wid == 0) {
    val = WarpReduceSumExp(val, used_warp_size);
    if (thread_id == 0)
      wrap_sum[idx] = spec_t(val);
  }
}


template <typename spec_t>
__global__ void softmax_kernel(const spec_t* input, spec_t* output,
                               size_t before_dim_size, size_t reduce_dim_size,
                               size_t after_dim_size,
                               const OffsetCalculator* in_offset_calculator,
                               const OffsetCalculator* out_offset_calculator) {

  size_t pos_per_block = blockDim.x;
  size_t total_idx = blockIdx.x * pos_per_block + threadIdx.x;
  size_t x = total_idx / after_dim_size;
  size_t y = total_idx % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  size_t pos_tid = 0;
  size_t pos_idx = threadIdx.x;
  if (after_dim_size > 1) {
    stride = after_dim_size;
    start_ptr =
      x * reduce_dim_size * after_dim_size + y + pos_tid * after_dim_size;
    end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
  } else {
    size_t cols_per_thread = reduce_dim_size;
    size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
    start_ptr = x * reduce_dim_size * after_dim_size + y +
      pos_tid * cols_per_thread * after_dim_size;
    end_ptr = min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
    stride = after_dim_size;
  }
  if (start_ptr >= end_ptr)
    return;

  spec_t max_thread = numeric_limits<spec_t>::lowest();
  spec_t sum_thread = 0;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
    auto in_offset = in_offset_calculator->get(ptr);
    max_thread = hetu::cuda::cuda_max(input[in_offset], max_thread);
  }

  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
    auto in_offset = in_offset_calculator->get(ptr);
    sum_thread += hetu::cuda::cuda_exp(input[in_offset] - max_thread);
  }

  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
    auto in_offset = in_offset_calculator->get(ptr);
    auto out_offset = out_offset_calculator->get(ptr);
    output[out_offset] = hetu::cuda::cuda_exp(input[in_offset] - max_thread) / sum_thread;
  }
}

template <typename spec_t>
__global__ void softmax_kernel2(const spec_t* input, spec_t* output,
                               size_t before_dim_size, size_t reduce_dim_size,
                               size_t after_dim_size, size_t threads_per_pos,
                               const OffsetCalculator* in_offset_calculator,
                               const OffsetCalculator* out_offset_calculator) {
  __shared__ spec_t shared_sum[1024];
  __shared__ spec_t wrap_max[1024];
  __shared__ spec_t wrap_sum[1024];

  size_t pos_per_block = blockDim.x / threads_per_pos;
  size_t total_idx = blockIdx.x * pos_per_block + threadIdx.x / threads_per_pos;
  size_t x = total_idx / after_dim_size;
  size_t y = total_idx % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  size_t pos_tid = threadIdx.x % threads_per_pos;
  size_t pos_idx = threadIdx.x / threads_per_pos;
  size_t cols_per_thread = (reduce_dim_size + threads_per_pos - 1) / threads_per_pos;
  // important: remove the epilogue!
  size_t used_threads_per_pos = (reduce_dim_size + cols_per_thread - 1) / cols_per_thread;
  size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y +
    reduce_dim_size * after_dim_size;
  start_ptr = x * reduce_dim_size * after_dim_size + y +
    pos_tid * cols_per_thread * after_dim_size;
  end_ptr = min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
  stride = after_dim_size;
  if (start_ptr >= end_ptr)
    return;

  spec_t max_thread = numeric_limits<spec_t>::lowest();
  spec_t sum_thread = 0;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
    auto in_offset = in_offset_calculator->get(ptr);
    max_thread = hetu::cuda::cuda_max(input[in_offset], max_thread);
  }
  
  BlockReduceArgmax(max_thread, shared_sum, wrap_max, pos_idx, threads_per_pos, used_threads_per_pos);

  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
    auto in_offset = in_offset_calculator->get(ptr);
    sum_thread += hetu::cuda::cuda_exp(input[in_offset] - wrap_max[pos_idx]);
  }

  BlockReduceSumExp(sum_thread, shared_sum, wrap_sum, pos_idx, threads_per_pos, used_threads_per_pos);
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
    auto in_offset = in_offset_calculator->get(ptr);
    auto out_offset = out_offset_calculator->get(ptr);
    output[out_offset] = hetu::cuda::cuda_exp(input[in_offset] - wrap_max[pos_idx]) / wrap_sum[pos_idx];
  }
}

void SoftmaxCuda(const NDArray& input, NDArray& output, int64_t dim, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  if (dim < 0) {
    dim = dim + input->ndim();
    HT_ASSERT(dim >= 0 && dim < input->ndim());
  }
  size_t before_dim_size = 1, reduce_dim_size, after_dim_size = 1;
  reduce_dim_size = input->shape(dim);
  for (size_t i = 0; i < input->ndim(); ++i) {
    if (i < dim)
      before_dim_size *= input->shape(i);
    else if (i > dim)
      after_dim_size *= input->shape(i);
  }

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  if (dim != input->ndim() - 1) {
    int blocks = before_dim_size * after_dim_size;
    int threads_per_pos = 1;
    int threads = threads_per_pos;
    while (threads * 2 <= HT_DEFAULT_NUM_THREADS_PER_BLOCK && blocks % 2 == 0) {
      threads *= 2;
      blocks /= 2;
    }
    HT_DISPATCH_FLOATING_TYPES(
      input->dtype(), spec_t, "SoftMaxCuda", [&]() {
        softmax_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
          before_dim_size, reduce_dim_size, after_dim_size,
          in_offset_calculator, out_offset_calculator);
      });
  }
  else {
    HT_ASSERT(after_dim_size == 1);
    int blocks = before_dim_size;
    int threads_per_pos = hetu::impl::GetThreadNum(reduce_dim_size);
    int threads = threads_per_pos;
    while (threads * 2 <= HT_DEFAULT_NUM_THREADS_PER_BLOCK && blocks % 2 == 0) {
      threads *= 2;
      blocks /= 2;
    }
    HT_DISPATCH_FLOATING_TYPES(
      input->dtype(), spec_t, "SoftMaxCuda", [&]() {
        softmax_kernel2<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
          before_dim_size, reduce_dim_size, after_dim_size, threads_per_pos,
          in_offset_calculator, out_offset_calculator);
      }); 
  }
  NDArray::MarkUsedBy({input, output, in_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

template <typename spec_t>
__global__ void softmax_grad_kernel(const spec_t* output, const spec_t* output_grad,
                                    spec_t* input_grad, size_t before_dim_size,
                                    size_t reduce_dim_size, size_t after_dim_size,
                                    const OffsetCalculator* out_offset_calculator,
                                    const OffsetCalculator* out_grad_offset_calculator,
                                    const OffsetCalculator* in_grad_offset_calculator) {
  __shared__ spec_t shared_sum[32];
  __shared__ spec_t wrap_sum[1];

  // here threads_per_pos = blockDim.x
  size_t used_threads_per_pos = blockDim.x;
  size_t x = blockIdx.x / after_dim_size;
  size_t y = blockIdx.x % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  if (after_dim_size > 1) {
    stride = after_dim_size * blockDim.x;
    start_ptr =
      x * reduce_dim_size * after_dim_size + y + threadIdx.x * after_dim_size;
    end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
  } else {
    size_t cols_per_thread = (reduce_dim_size + blockDim.x - 1) / blockDim.x;
    // important: remove the epilogue!
    used_threads_per_pos = (reduce_dim_size + cols_per_thread - 1) / cols_per_thread;
    size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
    start_ptr = x * reduce_dim_size * after_dim_size + y +
      threadIdx.x * cols_per_thread * after_dim_size;
    end_ptr = min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
    stride = after_dim_size;
  }
  if (start_ptr >= end_ptr)
    return;

  spec_t sum_thread = 0;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
    auto out_offset = out_offset_calculator->get(ptr);
    auto out_grad_offset = out_grad_offset_calculator->get(ptr);
    sum_thread += output_grad[out_grad_offset] * output[out_offset];
  }

  BlockReduceSumExp(sum_thread, shared_sum, wrap_sum, 0, blockDim.x, used_threads_per_pos);
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) {
    auto out_grad_offset = out_grad_offset_calculator->get(ptr);
    auto out_offset = out_offset_calculator->get(ptr);
    auto in_grad_offset = in_grad_offset_calculator->get(ptr);
    input_grad[in_grad_offset] = output_grad[out_grad_offset] * output[out_offset]
                               - output[out_offset] * wrap_sum[0];
  }
}

void SoftmaxGradientCuda(const NDArray& input_Y, const NDArray& output_grad,
                         NDArray& input_grad, int64_t dim, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_Y);
  HT_ASSERT_SAME_DEVICE(input_Y, output_grad);
  HT_ASSERT_SAME_DEVICE(input_Y, input_grad);

  size_t before_dim_size = 1, reduce_dim_size, after_dim_size = 1;
  reduce_dim_size = input_Y->shape(dim);
  for (size_t i = 0; i < input_Y->ndim(); ++i) {
    if (i < dim)
      before_dim_size *= input_Y->shape(i);
    else if (i > dim)
      after_dim_size *= input_Y->shape(i);
  }

  int blocks = before_dim_size * after_dim_size;
  int threads = hetu::impl::GetThreadNum(reduce_dim_size);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_grad_offset_calculator_arr,
          in_grad_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_grad_offset_calculator,
                   *in_grad_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input_Y, stream);
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) = 
    AllocOffsetCalculator(output_grad, stream);
  std::tie(in_grad_offset_calculator_arr, in_grad_offset_calculator) = 
    AllocOffsetCalculator(input_grad, stream);
  HT_DISPATCH_FLOATING_TYPES(
    input_Y->dtype(), spec_t, "SoftMaxCuda", [&]() {
      softmax_grad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input_Y->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(),
        input_grad->data_ptr<spec_t>(),
        before_dim_size, reduce_dim_size, after_dim_size,
        in_offset_calculator, out_grad_offset_calculator,
        in_grad_offset_calculator);
    });
  NDArray::MarkUsedBy({input_Y, output_grad, input_grad, in_offset_calculator_arr,
                      out_grad_offset_calculator_arr, in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
