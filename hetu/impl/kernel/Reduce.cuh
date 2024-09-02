#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/kernel/Vectorized.cuh"
#include <numeric>

namespace hetu {
namespace impl {

// Data type during computation for high precision.
template <typename spec_t>
struct OpMathType {
  using type = spec_t;
};
template <>
struct OpMathType<hetu::bfloat16> {
  using type = float;
};
template <>
struct OpMathType<hetu::float16> {
  using type = float;
};
template <typename T>
using opmath_type = typename OpMathType<T>::type;

template <typename spec_t, typename arg_t, typename func_t>
struct func_wrapper_t {
  func_t combine;
  func_wrapper_t(const func_t& op) : combine(op) {}

  static inline __device__ spec_t project(arg_t val) {
    return static_cast<spec_t>(val);
  }

  __device__ arg_t reduce(arg_t acc, arg_t val) const {
    return combine(acc, val);
  }
};

template <typename spec_t, typename arg_t, typename func_t>
func_wrapper_t<spec_t, arg_t, func_t> func_wrapper(const func_t& op) {
  return func_wrapper_t<spec_t, arg_t, func_t> { op };
}

static constexpr int MAX_NUM_THREADS = 512;

// Return floor(log2(n))
static inline int last_pow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

template <typename spec_t>
int get_output_vec_size(size_t reduce_ndim, size_t ndim, const HTShape& in_shape,
                        const HTStride& in_strides, spec_t* input) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](uint64_t n) {
    while (n % vec_size != 0) {
      vec_size /= 2;
    }
  };

  uint64_t base_address = reinterpret_cast<uint64_t>(input) / sizeof(spec_t);
  update_vec_size(base_address);

  const int output_idx = reduce_ndim;
  update_vec_size(in_shape[output_idx]);

  for (int i = 0; i < ndim; i++) {
    if (i != output_idx) {
      update_vec_size(in_strides[i]);
    }
  }
  return vec_size;
}

struct ReduceConfig {
  static constexpr int input_vec_size = 4;

  ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
    : element_size_bytes(element_size_bytes)
    , num_inputs(num_inputs)
    , num_outputs(num_outputs) {}
  ReduceConfig() {};

  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  int output_vec_size = 1;
  bool vectorize_input = false;

  void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_num_threads = MAX_NUM_THREADS / output_vec_size;
    int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(last_pow2(dim0)) : max_num_threads;
    int dim1_pow2 = dim1 < max_num_threads ? static_cast<int>(last_pow2(dim1)) : max_num_threads;

    block_width = std::min(dim0_pow2, static_cast<int>(HT_WARP_SIZE));
    block_height = std::min(dim1_pow2, static_cast<int>(max_num_threads / block_width));
    block_width = std::min(dim0_pow2, static_cast<int>(max_num_threads / block_height));
    num_threads = block_width * block_height;
  }

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  dim3 block() const {
    return dim3(block_width, block_height);
  }

  dim3 grid() const {
    return dim3(DIVUP(num_outputs / output_vec_size, step_output), ctas_per_output);
  }

  __host__ __device__ bool should_block_x_reduce() const {
    return input_mult[0] != 0;
  }

  __host__ __device__ bool should_block_y_reduce() const {
    return input_mult[1] != 0;
  }

  __host__ __device__ bool should_global_reduce() const {
    return input_mult[2] != 0;
  }

  __device__ bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
      (!should_block_x_reduce() || threadIdx.x == 0) &&
      (!should_block_y_reduce() || threadIdx.y == 0);
  }

  __device__ bool should_reduce_tail() const {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
      (!should_global_reduce() || blockIdx.y == 0);
  }

  __host__ __device__ int64_t input_idx() const {
    int64_t lane = threadIdx.x;
    int64_t warp = threadIdx.y;
    int64_t cta2 = blockIdx.y;
    return (lane * input_mult[0] +
            warp * input_mult[1] +
            cta2 * input_mult[2]);
  }

  template <int output_vec_size>
  __host__ __device__ int64_t output_idx() const {
    int64_t lane = threadIdx.x;
    int64_t warp = threadIdx.y;
    int64_t cta1 = blockIdx.x;
    return (lane * output_mult[0] +
            warp * output_mult[1] +
            cta1 * step_output) * output_vec_size;
  }

  __device__ int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  __device__ int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() ||
         block_width <= HT_WARP_SIZE)) {
      return 0;
    }
    return element_size_bytes * num_threads * output_vec_size;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce()) {
      size *= block().x * output_vec_size;
    }
    return size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  int values_per_thread() const {
    return DIVUP(num_inputs, step_input);
  }
};

template <typename arg_t, typename spec_t>
static ReduceConfig setReduceConfig(size_t reduce_ndim, size_t ndim, const HTShape& in_shape,
                                    const HTStride& in_strides, spec_t* input, const CUDAStream& cuda_stream) {
  int maxThreadsPerMultiProcessor;
  int multiProcessorCount;
  CudaDeviceGetAttribute(&maxThreadsPerMultiProcessor,
    cudaDevAttrMaxThreadsPerMultiProcessor, cuda_stream.device_id());
  CudaDeviceGetAttribute(&multiProcessorCount,
    cudaDevAttrMultiProcessorCount, cuda_stream.device_id());

  int64_t num_outputs = 1;
  for (int i = reduce_ndim; i < ndim; i++) {
    num_outputs *= in_shape[i];
  }
  int64_t inputs_per_output = 1;
  for (int i = 0; i < reduce_ndim; i++) {
    inputs_per_output *= in_shape[i];
  }
  
  auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output);

  // Assume that each thread handles a single output as a start
  // NOTE: dim0 & dim1 does not guarantee any reduce config
  int64_t dim0;
  int64_t dim1;
  int64_t fastest_moving_stride;
  bool contiguous_reduction;

  if (ndim > 0) {
    // Adjust block size to map block width to contiguous dimension
    // of tensor to get the best memory accessing pattern
    contiguous_reduction = 
      (reduce_ndim == ndim) ||
      (in_strides[0] < in_strides[reduce_ndim]);
    if (contiguous_reduction) {
      // Map block.y to every output
      // block_x_reduce is required
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = in_strides[0];
    } else {
      // Map block.x to every output
      dim0 = num_outputs;
      dim1 = inputs_per_output;
      fastest_moving_stride = in_strides[reduce_ndim];
    }
  } else {
    contiguous_reduction = true;
    fastest_moving_stride = 1;
    dim0 = 1;
    dim1 = 1;
  }

  // We do vectorization to gain better memory access
  if (fastest_moving_stride == 1) {
    if (contiguous_reduction && dim0 > 128 && reduce_ndim == 1) {
      config.vectorize_input = true;
      dim0 /= config.input_vec_size;
    } else if (!contiguous_reduction) {
      config.output_vec_size = get_output_vec_size<spec_t>(reduce_ndim, ndim, in_shape,
                                                          in_strides, input);
      dim0 /= config.output_vec_size;
    }
  }
  
  // Adjust block_width and block_height
  config.set_block_dimension(dim0, dim1);
  
  int block_width = config.block_width;
  int block_height = config.block_height;

  if (ndim == 0 || contiguous_reduction) {
    config.input_mult[0] = config.split_input(block_width);
  } else {
    config.output_mult[0] = config.split_output(block_width);
  }

  constexpr int min_values_per_thread = 16;
  constexpr int max_values_per_thread = 256;

  if (config.values_per_thread() >= block_height * 16 || config.values_per_thread() >= max_values_per_thread) {
    config.input_mult[1] = config.split_input(block_height);
  } else {
    config.output_mult[1] = config.split_output(block_height);
  }
  
  const int blocks_per_sm = maxThreadsPerMultiProcessor / config.num_threads;
  const int num_mp = multiProcessorCount;
  const int target_grid_size = num_mp * blocks_per_sm;
  int grid = config.grid().x;
  if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
    int ctas_per_output1 = DIVUP(target_grid_size, grid);
    int ctas_per_output2 = DIVUP(config.values_per_thread(), min_values_per_thread);
    int ctas_per_output3 = DIVUP(config.values_per_thread(), max_values_per_thread);
    config.ctas_per_output = std::max(std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);
    if (config.ctas_per_output > 1) {
      config.input_mult[2] = config.split_input(config.ctas_per_output);
    }
  }
  return config;
}

__device__ static bool mark_block_finished(int* semaphores) {
  __shared__ bool is_last_block_done_shared;

  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
    is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
  }

  __syncthreads();

  return is_last_block_done_shared;
}

__device__ static int64_t calc_offset(int64_t index, const int64_t* strides, const int64_t* shape,
                                      size_t start_dim, size_t end_dim) {
  int64_t offset = 0;
  for (int i = start_dim; i < end_dim; i++) {
    offset += (index % shape[i]) * strides[i];
    index /= shape[i];
  }
  return offset;
}

template <typename spec_t, typename arg_t, typename op_t, typename ident_t=double>
__device__ void input_vectorized_thread_reduce_impl(const spec_t* data, arg_t* value, ReduceConfig config,
                                                    op_t ops, ident_t ident) {
  int64_t end = config.num_inputs;
  constexpr int input_vec_size = config.input_vec_size;

  arg_t acc_value = ident;
  constexpr int align_bytes = alignof(aligned_vector<spec_t, input_vec_size>);
  constexpr int align_elements = align_bytes / sizeof(spec_t);
  int shift = ((uint64_t)data) % align_bytes / sizeof(spec_t);
  if (shift > 0) {
    data -= shift;
    end += shift;
    if (threadIdx.x >= shift && threadIdx.x < align_elements && config.should_reduce_tail()) {
      acc_value = ops.reduce(acc_value, static_cast<arg_t>(data[threadIdx.x]));
    }
    end -= align_elements;
    data += align_elements;
    shift = align_elements - shift;
  }

  int64_t idx = config.input_idx();
  const int64_t stride = config.step_input;

  arg_t value_list[input_vec_size];
  value_list[0] = acc_value;

  #pragma unroll
  for (int i = 1; i < input_vec_size; i++) {
    value_list[i] = ident;
  }

  while (idx * input_vec_size + input_vec_size - 1 < end) {
    using vec_t = aligned_vector<spec_t, input_vec_size>;
    auto *base = reinterpret_cast<const vec_t*>(data);
    const auto values_vec = base[idx];
    #pragma unroll
    for (int64_t i = 0; i < input_vec_size; i++) {
      value_list[i] = ops.reduce(value_list[i], static_cast<arg_t>(values_vec.val[i]));
    }
    idx += stride;
  }

  // tail
  int64_t tail_start = end - end % input_vec_size;
  if (config.should_reduce_tail()) {
    int idx = tail_start + threadIdx.x;
    if (idx < end) {
      value_list[0] = ops.reduce(value_list[0], static_cast<arg_t>(data[idx]));
    }
  }

  // combine accumulators
  #pragma unroll
  for (int i = 1; i < input_vec_size; i++) {
    value_list[0] = ops.reduce(value_list[0], value_list[i]);
  }
  *value = static_cast<arg_t>(value_list[0]);
}

template <typename spec_t, typename arg_t, int output_vec_size, typename op_t,
          typename ident_t, typename offset_calc_t, int list_size=4>
__device__ void thread_reduce_impl(const spec_t* data, arg_t* value, ReduceConfig config,
                                   op_t ops, ident_t ident, offset_calc_t offset_calc) {
  int64_t idx = config.input_idx();
  const int64_t end = config.num_inputs;
  const int64_t stride = config.step_input;

  // multiple accumulators to remove dependency between unrolled loops
  arg_t value_list[list_size][output_vec_size];

  #pragma unroll
  for (int i = 0; i < list_size; i++) {
    #pragma unroll
    for (int j = 0; j < output_vec_size; j++) {
      value_list[i][j] = ident;
    }
  }

  aligned_vector<spec_t, output_vec_size> values[list_size];

  while (idx + (list_size - 1) * stride < end) {
    #pragma unroll
    for (int64_t i = 0; i < list_size; i++) {
      const auto offset = offset_calc(idx + i * stride) / output_vec_size;
      using vec_t = aligned_vector<spec_t, output_vec_size>;
      auto *base = reinterpret_cast<const vec_t*>(data);
      values[i] = base[offset];
    }
    #pragma unroll
    for (int64_t i = 0; i < list_size; i++) {
      #pragma unroll
      for (int64_t j = 0; j < output_vec_size; j++) {
        value_list[i][j] = ops.reduce(value_list[i][j], static_cast<arg_t>(values[i].val[j]));
      }
    }
    idx += stride * list_size;
  }

  // tail
  int idx_ = idx;
  #pragma unroll
  for (int64_t i = 0; i < list_size; i++) {
    if (idx >= end) {
      break;
    }
    const auto offset = offset_calc(idx) / output_vec_size;
    using vec_t = aligned_vector<spec_t, output_vec_size>;
    auto *base = reinterpret_cast<const vec_t*>(data);
    values[i] = base[offset];
    idx += stride;
  }
  idx = idx_;
  #pragma unroll
  for (int64_t i = 0; i < list_size; i++) {
    if (idx >= end) {
      break;
    }
    #pragma unroll
    for (int64_t j = 0; j < output_vec_size; j++) {
      value_list[i][j] = ops.reduce(value_list[i][j], static_cast<arg_t>(values[i].val[j]));
    }
    idx += stride;
  }

  // combine accumulators
  #pragma unroll
  for (int i = 1; i < list_size; i++) {
    #pragma unroll
    for (int64_t j = 0; j < output_vec_size; j++) {
      value_list[0][j] = ops.reduce(value_list[0][j], value_list[i][j]);
    }
  }

  #pragma unroll
  for (int i = 0; i < output_vec_size; i++) {
    value[i] = static_cast<arg_t>(value_list[0][i]);
  }
}

template <typename spec_t, typename arg_t, int output_vec_size, typename op_t, typename ident_t=double>
__device__ void thread_reduce(const spec_t* data, const int64_t* in_strides, const int64_t* in_shape,
                              size_t reduce_ndim, arg_t* value, ReduceConfig config,
                              op_t ops, ident_t ident) {
  if (config.vectorize_input) {
    assert(output_vec_size == 1);
    input_vectorized_thread_reduce_impl<spec_t, arg_t>(data, value, config, ops, ident);
  } else {
    int64_t element_stride = in_strides[0];
    bool is_contiguous = (reduce_ndim == 1 && element_stride == 1);
    if (is_contiguous) {
      thread_reduce_impl<spec_t, arg_t, output_vec_size>(data, value, config, ops, ident,
                                                                        [](int64_t idx) { return idx; });
    } else if (reduce_ndim == 1) {
      thread_reduce_impl<spec_t, arg_t, output_vec_size>(data, value, config, ops, ident,
                                                                        [&](int64_t idx) { return idx * element_stride; });
    } else {
      thread_reduce_impl<spec_t, arg_t, output_vec_size>(data, value, config, ops, ident,
                                                                        [&](int64_t idx) { return calc_offset(idx, in_strides, in_shape,
                                                                                                              0, reduce_ndim); });
    }
  }
}

template <typename arg_t, int output_vec_size, typename op_t>
__device__ void block_x_reduce(arg_t* value, char* shared_memory, op_t ops) {
  using args_vec_t = arg_t[output_vec_size];
  args_vec_t* shared = reinterpret_cast<args_vec_t*>(shared_memory);
  int dim_x = blockDim.x;

  if (dim_x > warpSize) {
    int address_base = threadIdx.x + threadIdx.y * blockDim.x;
    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      shared[address_base][i] = value[i];
    }
    for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1) {
      __syncthreads();
      if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
        arg_t* other = shared[address_base + offset];
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = ops.reduce(value[i], other[i]);
          shared[address_base][i] = value[i];
        }
      }
    }
    dim_x = warpSize;
  }

  __syncthreads();
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (int offset = 1; offset < dim_x; offset <<= 1) {
    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      arg_t other = hetu::cuda::shfl_down_sync(mask, value[i], offset, warpSize);
      value[i] = ops.reduce(value[i], other);
    }
  }
}

template <typename arg_t, int output_vec_size, typename op_t>
__device__ void block_y_reduce(arg_t* value, char* shared_memory, ReduceConfig config,
                               op_t ops) {
  using args_vec_t = arg_t[output_vec_size];
  args_vec_t* shared = reinterpret_cast<args_vec_t*>(shared_memory);
  #pragma unroll
  for (int i = 0; i < output_vec_size; i++) {
    shared[config.shared_memory_offset(0)][i] = value[i];
  }
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      arg_t* other = shared[config.shared_memory_offset(offset)];
      #pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        value[i] = ops.reduce(value[i], other[i]);
        shared[config.shared_memory_offset(0)][i] = value[i];
      }
    }
  }
}

template <typename spec_t, typename arg_t, typename out_t, int output_vec_size, typename op_t, typename ident_t=double>
__device__ void global_reduce(arg_t* value, char* shared_memory, void* cta_buf, const int64_t* out_strides,
                              const int64_t* in_shape, size_t reduce_ndim, size_t ndim,
                              out_t* output, int* semaphores, ReduceConfig config,
                              op_t ops, ident_t ident) {
  using args_vec_t = arg_t[output_vec_size];
  args_vec_t* reduce_buffer = reinterpret_cast<args_vec_t*>(cta_buf);
  int64_t output_idx = config.output_idx<output_vec_size>();
  int64_t out_base_offsets[output_vec_size];
  out_t* out_addr[output_vec_size];

  #pragma unroll
  for (int i = 0; i < output_vec_size; i++) {
    out_base_offsets[i] = calc_offset(output_idx + i, out_strides, in_shape,
                                  reduce_ndim, ndim);
    out_addr[i] = reinterpret_cast<out_t*>((char*)output + out_base_offsets[i] * sizeof(out_t));
  }

  bool should_store = config.should_store(output_idx);
  if (should_store) {
    int64_t offset = config.staging_memory_offset(blockIdx.y);
    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      reduce_buffer[offset][i] = value[i];
    }
  }

  __threadfence(); // make sure writes are globally visible
  __syncthreads(); // make sure writes to staging are all done
  bool is_last_block_done = mark_block_finished(semaphores);

  if (is_last_block_done) {
    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      value[i] = ident;
    }
    
    int64_t input_offset, step;
    if (config.should_block_x_reduce()) {
      input_offset = threadIdx.x + threadIdx.y * blockDim.x;
      step = blockDim.x * blockDim.y;
    } else {
      input_offset = threadIdx.y;
      step = blockDim.y;
    }

    for (; input_offset < config.ctas_per_output; input_offset += step) {
      int64_t idx = config.staging_memory_offset(input_offset);
      #pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        value[i] = ops.reduce(value[i], reduce_buffer[idx][i]);
      }
    }
    block_y_reduce<arg_t, output_vec_size>(value, shared_memory, config, ops);
    if (config.should_block_x_reduce()) {
      block_x_reduce<arg_t, output_vec_size>(value, shared_memory, ops);
    }
    if (should_store) {
      #pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        *(out_addr[i]) = ops.project(value[i]);
      }
    }
  }
}

static void setReduceMeta(const NDArray& in_arr, size_t in_ndim, int64_t num_ax,
                          int64_t* reduce_axes, size_t& reduce_ndim, size_t& merge_ndim,
                          HTShape& in_merge_shape, HTStride& in_strides, HTStride& out_strides) {
  size_t rest_ndim = 0;
  int64_t merge_size, stride = in_arr->stride(in_ndim - 1);
  HTShape in_rest_shape;
  HTStride in_rest_strides;

  for (int i = in_ndim - 1, j = num_ax - 1; i >= 0;) {
    if (j >= 0 && reduce_axes[j] < i) {
      merge_size = in_arr->shape(i);
      i--;
      while (i > reduce_axes[j] && in_arr->stride(i) == in_arr->stride(i + 1) * in_arr->shape(i + 1)) {
        merge_size *= in_arr->shape(i);
        i--;
      }
      in_rest_shape.emplace_back(merge_size);
      in_rest_strides.emplace_back(stride);
      stride = in_arr->stride(i);
      rest_ndim++;
    }
    else if (j >= 0 && reduce_axes[j] == i) {
      merge_size = in_arr->shape(i);
      i--;
      j--;
      while (j >= 0 && reduce_axes[j + 1] == reduce_axes[j] + 1 && i >= 0 &&
             in_arr->stride(i) == in_arr->stride(i + 1) * in_arr->shape(i + 1)) {
        merge_size *= in_arr->shape(i);
        i--;
        j--;
      }
      in_merge_shape.emplace_back(merge_size);
      in_strides.emplace_back(stride);
      stride = in_arr->stride(i);
      reduce_ndim++;
    }
    else {
      merge_size = in_arr->shape(i);
      i--;
      while (i >= 0 && in_arr->stride(i) == in_arr->stride(i + 1) * in_arr->shape(i + 1)) {
        merge_size *= in_arr->shape(i);
        i--;
      }
      in_rest_shape.emplace_back(merge_size);
      in_rest_strides.emplace_back(stride);
      stride = in_arr->stride(i);
      rest_ndim++;
    }
  }
  merge_ndim = reduce_ndim + rest_ndim;

  // Merge reduce dims and rest dims
  in_merge_shape.insert(in_merge_shape.end(), in_rest_shape.begin(), in_rest_shape.end());
  in_strides.insert(in_strides.end(), in_rest_strides.begin(), in_rest_strides.end());
  
  out_strides = HTShape(merge_ndim, 0);
  stride = 1;
  for (int i = reduce_ndim; i < merge_ndim; i++) {
    out_strides[i] = stride;
    stride *= in_merge_shape[i];
  }
}

template <typename spec_t, typename arg_t, typename out_t, int output_vec_size, typename op_t, typename ident_t=double>
__global__ void reduce_kernel(const int64_t* in_strides, const int64_t* out_strides, const int64_t* in_shape, op_t ops,
                              size_t reduce_ndim, size_t ndim, ident_t ident, const spec_t* input,
                              out_t* output, void* cta_buf, int* semaphores, ReduceConfig config) {
  extern __shared__ char shared_memory[];

  int64_t output_idx = config.output_idx<output_vec_size>();
  int64_t input_idx = config.input_idx();
  int64_t in_base_offset = calc_offset(output_idx, in_strides, in_shape,
                                       reduce_ndim, ndim);
  using arg_vec_t = arg_t[output_vec_size];
  arg_vec_t value;

  if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
    const spec_t* input_slice = reinterpret_cast<const spec_t*>((const char*)input + in_base_offset * sizeof(spec_t));
    thread_reduce<spec_t, arg_t, output_vec_size>(input_slice, in_strides, in_shape,
                                                  reduce_ndim, value, config, ops, ident);
  }

  if (config.should_block_y_reduce()) {
    block_y_reduce<arg_t, output_vec_size>(value, shared_memory, config, ops);
  }

  if (config.should_block_x_reduce()) {
    block_x_reduce<arg_t, output_vec_size>(value, shared_memory, ops);
  }

  int64_t out_base_offsets[output_vec_size];
  out_t* out_addr[output_vec_size];

  #pragma unroll
  for (int i = 0; i < output_vec_size; i++) {
    out_base_offsets[i] = calc_offset(output_idx + i, out_strides, in_shape,
                                      reduce_ndim, ndim);
    out_addr[i] = reinterpret_cast<out_t*>((char*)output + out_base_offsets[i] * sizeof(out_t));
  }

  if (config.should_global_reduce()) {
    // global_reduce will write back to output
    global_reduce<spec_t, arg_t, out_t, output_vec_size, op_t, ident_t>(value, shared_memory, cta_buf, out_strides,
                                                                        in_shape, reduce_ndim, ndim, output,
                                                                        semaphores, config, ops, ident);
  } else if (config.should_store(output_idx)) {
    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      *(out_addr[i]) = ops.project(value[i]);
    }
  }
}

template <typename spec_t, typename out_t, typename arg_t, typename op_t, typename ident_t=double>
void launch_reduce_kernel(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                          int64_t num_ax, const op_t& ops, ident_t ident, const Stream& stream) {
  if (num_ax <= 0)
    return;
  for (int i = 0; i < num_ax; ++i)
    HT_ASSERT(axes[i] >= 0 && axes[i] < in_arr->ndim());

  int64_t* reduce_axes = (int64_t*) malloc(num_ax * sizeof(int64_t));
  memcpy(reduce_axes, axes, num_ax * sizeof(int64_t));
  std::sort(reduce_axes, reduce_axes + num_ax);
  num_ax = std::unique(reduce_axes, reduce_axes + num_ax) - reduce_axes;

  // Merge contiguous reduce / rest dims
  size_t in_ndim = in_arr->ndim();
  size_t reduce_ndim = 0;
  size_t merge_ndim = 0;
  HTShape in_merge_shape;
  HTStride in_strides, out_strides;

  setReduceMeta(in_arr, in_ndim, num_ax, reduce_axes, reduce_ndim, merge_ndim,
                in_merge_shape, in_strides, out_strides);
  
  auto device_id = in_arr->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);

  ReduceConfig config;
  // Merge into one H2D data transfer
  auto reduce_meta = in_merge_shape;
  reduce_meta.insert(reduce_meta.end(), in_strides.begin(), in_strides.end());
  reduce_meta.insert(reduce_meta.end(), out_strides.begin(), out_strides.end());
  auto reduce_meta_arr = hetu::cuda::to_int64_ndarray(reduce_meta, device_id);
  int64_t *in_merge_shape_ptr = reduce_meta_arr->data_ptr<int64_t>();
  int64_t *in_strides_ptr = in_merge_shape_ptr + in_merge_shape.size();
  int64_t *out_strides_ptr = in_strides_ptr + in_strides.size();

  config = setReduceConfig<arg_t, spec_t>(reduce_ndim, merge_ndim, in_merge_shape,
                                          in_strides, in_arr->data_ptr<spec_t>(), cuda_stream);

  void* cta_buf_ptr = nullptr;
  int* semaphores_ptr = nullptr;
  NDArray cta_buf_arr, semaphores_arr;
  if (config.should_global_reduce()) {
    auto cta_buf_size = config.global_memory_size();
    auto semaphores_size = config.semaphore_size();
    cta_buf_arr = NDArray::empty({static_cast<int64_t>(cta_buf_size)},
                                 Device(kCUDA, stream.device_index()), kByte, stream.stream_index());
    semaphores_arr = NDArray::zeros({static_cast<int64_t>(semaphores_size)},
                                    Device(kCUDA, stream.device_index()), kByte, stream.stream_index());
    cta_buf_ptr = cta_buf_arr->raw_data_ptr();
    semaphores_ptr = semaphores_arr->data_ptr<int>();
  }

  dim3 block = config.block();
  dim3 grid = config.grid();
  int shared_memory = config.shared_memory_size();

  switch(config.output_vec_size) {
    case 4:
      reduce_kernel<spec_t, arg_t, out_t, 4><<<grid, block, shared_memory, cuda_stream>>>(
        in_strides_ptr, out_strides_ptr, in_merge_shape_ptr,
        ops, reduce_ndim, merge_ndim, ident,
        in_arr->data_ptr<spec_t>(), out_arr->data_ptr<out_t>(),
        cta_buf_ptr, semaphores_ptr, config);
      break;
    case 2:
      reduce_kernel<spec_t, arg_t, out_t, 2><<<grid, block, shared_memory, cuda_stream>>>(
        in_strides_ptr, out_strides_ptr, in_merge_shape_ptr,
        ops, reduce_ndim, merge_ndim, ident,
        in_arr->data_ptr<spec_t>(), out_arr->data_ptr<out_t>(),
        cta_buf_ptr, semaphores_ptr, config);
      break;
    default:
      reduce_kernel<spec_t, arg_t, out_t, 1><<<grid, block, shared_memory, cuda_stream>>>(
        in_strides_ptr, out_strides_ptr, in_merge_shape_ptr,
        ops, reduce_ndim, merge_ndim, ident,
        in_arr->data_ptr<spec_t>(), out_arr->data_ptr<out_t>(),
        cta_buf_ptr, semaphores_ptr, config);
  }

  free(reduce_axes);
  if (config.should_global_reduce()) {
    NDArray::MarkUsedBy({cta_buf_arr, semaphores_arr}, stream);
  }
  NDArray::MarkUsedBy({in_arr, out_arr, reduce_meta_arr}, stream);
  return;
}

} // namespace impl
} // namespace hetu
