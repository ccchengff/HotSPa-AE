#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include <thrust/tuple.h>

namespace hetu {
namespace impl {

static constexpr uint32_t NUM_THREADS = 256;
static constexpr uint32_t THREAD_WORK_SIZE = 4;
static constexpr uint32_t BLOCK_WORK_SIZE = NUM_THREADS * THREAD_WORK_SIZE;

template <typename spec_t>
int get_vectorize_size(const spec_t* ptr) {
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<spec_t, 2>>::value;
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<spec_t, 4>>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template <typename spec_t, typename T, typename... Args>
int get_vectorize_size(const spec_t* ptr, const T* arg, const Args*... args) {
  return std::min(get_vectorize_size(ptr),
                  get_vectorize_size(arg, args...));
}

// TODO: Merge 0-3 input kernels into one general kernel

template <typename func_t, typename out_t, typename... IN_t>
__device__ void unroll_kernel_impl(func_t op, int remaining, int base_idx, out_t* output, const IN_t*... inputs) {
  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    results[i] = op(inputs[linear_idx]...);
  }
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    output[linear_idx] = results[i];
  }
}

template <typename func_t, typename out_t>
__device__ void unroll_kernel_impl(func_t op, int remaining, int base_idx, out_t* output) {
  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    results[i] = op(linear_idx);
  }
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    output[linear_idx] = results[i];
  }
}

template <typename func_t, typename out_t, typename... IN_t>
__global__ void unroll_kernel(func_t op, size_t size, out_t* output, const IN_t*... inputs) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  unroll_kernel_impl(op, remaining, base_idx, output, inputs...);
}

template <int vec_size, typename func_t, typename out_t, typename... IN_t>
__device__ void vectorize_kernel_impl(func_t op, aligned_vector<out_t, vec_size>* output,
                                      const aligned_vector<IN_t, vec_size>*... inputs) {
  int loop_size = THREAD_WORK_SIZE / vec_size;
  #pragma unroll
  for (int i = 0; i < loop_size; i++) {
    int index = threadIdx.x + i * NUM_THREADS;
    aligned_vector<out_t, vec_size> ret;
    #pragma unroll
    for (int j = 0; j < vec_size; j++) {
      ret.val[j] = op(inputs[index].val[j]...);
    }
    output[index] = ret;
  }
}

template <int vec_size, typename func_t, typename out_t>
__device__ void vectorize_kernel_impl(func_t op, aligned_vector<out_t, vec_size>* output) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int loop_size = THREAD_WORK_SIZE / vec_size;
  #pragma unroll
  for (int i = 0; i < loop_size; i++) {
    int index = threadIdx.x + i * NUM_THREADS;
    int linear_idx = base_idx + index * vec_size;
    aligned_vector<out_t, vec_size> ret;
    #pragma unroll
    for (int j = 0; j < vec_size; j++) {
      ret.val[j] = op(linear_idx + j);
    }
    output[index] = ret;
  }
}

template <int vec_size, typename func_t, typename out_t, typename... IN_t>
__global__ void vectorize_kernel(func_t op, size_t size, out_t* output, const IN_t*... inputs) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  if (remaining < BLOCK_WORK_SIZE) {
    // do a naive unrolled loop to handle the reminder
    unroll_kernel_impl(op, remaining, base_idx, output, inputs...);
  } else {
    // use vectorize memory load/store to handle a full `block_work_size` data
    vectorize_kernel_impl<vec_size>(op, reinterpret_cast<aligned_vector<out_t, vec_size>*>(output + base_idx),
                                    reinterpret_cast<const aligned_vector<IN_t, vec_size>*>(
                                    const_cast<IN_t*>(inputs) + base_idx)...);
  }
}

template <typename func_t, typename out_t, typename... IN_t>
static inline void launch_vectorized_kernel(const func_t& op, size_t size, const Stream& stream,
                                            out_t* output, const IN_t*... inputs) {
  int64_t grid = DIVUP(size, BLOCK_WORK_SIZE);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  int vec_size = get_vectorize_size(output, inputs...);
  switch (vec_size) {
    case 4:
      vectorize_kernel<4><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    case 2:
      vectorize_kernel<2><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    case 1:
      unroll_kernel<<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    default:
      HT_RUNTIME_ERROR << "Unexpected vectorization size";
      __builtin_unreachable();
  }
}

// Ternary loop kernel
template <int nt, int vt, typename in_a_t, typename in_b_t, typename in_c_t,
          typename out_t, typename func_t>
__global__ void ternary_kernel(const in_a_t* inputA,
                               const in_b_t* inputB,
                               const in_c_t* inputC,
                               size_t size, out_t* output, func_t op,
                               const OffsetCalculator* A_offset_calculator,
                               const OffsetCalculator* B_offset_calculator,
                               const OffsetCalculator* C_offset_calculator,
                               const OffsetCalculator* out_offset_calculator) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      auto A_offset = A_offset_calculator->get(idx);
      auto B_offset = B_offset_calculator->get(idx);
      auto C_offset = C_offset_calculator->get(idx);
      auto out_offset = out_offset_calculator->get(idx);
      output[out_offset] = op(inputA[A_offset], inputB[B_offset], inputC[C_offset]);
      idx += nt;
    }
  }
}

template <typename in_a_t, typename in_b_t, typename in_c_t,
          typename out_t, typename func_t>
void launch_loop_kernel(const NDArray& inputA, const NDArray& inputB,
                        const NDArray& inputC, NDArray& output, size_t size,
                        const Stream& stream, const func_t& op) {
  bool contiguous = inputA->is_contiguous() &&
                    inputB->is_contiguous() &&
                    inputC->is_contiguous() &&
                    output->is_contiguous();
  if (contiguous) {
    launch_vectorized_kernel(op, size, stream, output->data_ptr<out_t>(),
                             inputA->data_ptr<in_a_t>(),
                             inputB->data_ptr<in_b_t>(),
                             inputC->data_ptr<in_c_t>());
  } else {
    constexpr int unroll_factor = sizeof(DataType2Size(output->dtype())) >= 4 ? 2 : 4;
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(size, unroll_factor * block.x));
    NDArray A_offset_calculator_arr, B_offset_calculator_arr,
            C_offset_calculator_arr, out_offset_calculator_arr;
    OffsetCalculator *A_offset_calculator, *B_offset_calculator,
                     *C_offset_calculator, *out_offset_calculator;
    std::tie(A_offset_calculator_arr, A_offset_calculator) =
      AllocOffsetCalculator(inputA, stream);
    std::tie(B_offset_calculator_arr, B_offset_calculator) =
      AllocOffsetCalculator(inputB, stream);
    std::tie(C_offset_calculator_arr, C_offset_calculator) =
      AllocOffsetCalculator(inputC, stream);
    std::tie(out_offset_calculator_arr, out_offset_calculator) =
      AllocOffsetCalculator(output, stream);
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    ternary_kernel<NUM_THREADS, unroll_factor><<<grid, block, 0, cuda_stream>>>(
      inputA->data_ptr<in_a_t>(), inputB->data_ptr<in_b_t>(),
      inputC->data_ptr<in_c_t>(), size, output->data_ptr<out_t>(), op,
      A_offset_calculator, B_offset_calculator,
      C_offset_calculator, out_offset_calculator);
    NDArray::MarkUsedBy({A_offset_calculator_arr, B_offset_calculator_arr,
                        C_offset_calculator_arr, out_offset_calculator_arr}, stream);
  }
}

// Binary loop kernel
template <int nt, int vt, typename spec_a_t, typename spec_b_t,
          typename out_t, typename func_t>
__global__ void binary_kernel(const spec_a_t* inputA,
                              const spec_b_t* inputB,
                              size_t size, out_t* output, func_t op,
                              const OffsetCalculator* A_offset_calculator,
                              const OffsetCalculator* B_offset_calculator,
                              const OffsetCalculator* out_offset_calculator) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      auto A_offset = A_offset_calculator->get(idx);
      auto B_offset = B_offset_calculator->get(idx);
      auto out_offset = out_offset_calculator->get(idx);
      output[out_offset] = op(inputA[A_offset], inputB[B_offset]);
      idx += nt;
    }
  }
}

template <typename in_a_t, typename in_b_t, typename out_t, typename func_t>
void launch_loop_kernel(const NDArray& inputA, const NDArray& inputB,
                        NDArray& output, size_t size,
                        const Stream& stream, const func_t& op) {
  bool contiguous = inputA->is_contiguous() &&
                    inputB->is_contiguous() &&
                    output->is_contiguous();
  if (contiguous) {
    launch_vectorized_kernel(op, size, stream, output->data_ptr<out_t>(),
                             inputA->data_ptr<in_a_t>(),
                             inputB->data_ptr<in_b_t>());
  } else {
    constexpr int unroll_factor = sizeof(DataType2Size(output->dtype())) >= 4 ? 2 : 4;
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(size, unroll_factor * block.x));
    NDArray A_offset_calculator_arr, B_offset_calculator_arr, out_offset_calculator_arr;
    OffsetCalculator *A_offset_calculator, *B_offset_calculator, *out_offset_calculator;
    std::tie(A_offset_calculator_arr, A_offset_calculator) =
      AllocOffsetCalculator(inputA, stream);
    std::tie(B_offset_calculator_arr, B_offset_calculator) =
      AllocOffsetCalculator(inputB, stream);
    std::tie(out_offset_calculator_arr, out_offset_calculator) =
      AllocOffsetCalculator(output, stream);
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    binary_kernel<NUM_THREADS, unroll_factor><<<grid, block, 0, cuda_stream>>>(
      inputA->data_ptr<in_a_t>(), inputB->data_ptr<in_b_t>(),
      size, output->data_ptr<out_t>(), op,
      A_offset_calculator, B_offset_calculator, out_offset_calculator);
    NDArray::MarkUsedBy({A_offset_calculator_arr, B_offset_calculator_arr,
                        out_offset_calculator_arr}, stream);
  }
}

// Unary loop kernel
template <int nt, int vt, typename in_a_t, typename out_t, typename func_t>
__global__ void unary_kernel(const in_a_t* input, size_t size,
                             out_t* output, func_t op,
                             const OffsetCalculator* in_offset_calculator,
                             const OffsetCalculator* out_offset_calculator) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      auto in_offset = in_offset_calculator->get(idx);
      auto out_offset = out_offset_calculator->get(idx);
      output[out_offset] = op(input[in_offset]);
      idx += nt;
    }
  }
}

template <typename in_t, typename out_t, typename func_t>
void launch_loop_kernel(const NDArray& input, NDArray& output, size_t size,
                        const Stream& stream, const func_t& op) {
  bool contiguous = input->is_contiguous() && output->is_contiguous();
  if (contiguous) {
    launch_vectorized_kernel(op, size, stream, output->data_ptr<out_t>(),
                             input->data_ptr<in_t>());
  } else {
    constexpr int unroll_factor = sizeof(DataType2Size(output->dtype())) >= 4 ? 2 : 4;
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(size, unroll_factor * block.x));
    NDArray in_offset_calculator_arr, out_offset_calculator_arr;
    OffsetCalculator *in_offset_calculator, *out_offset_calculator;
    std::tie(in_offset_calculator_arr, in_offset_calculator) =
      AllocOffsetCalculator(input, stream);
    std::tie(out_offset_calculator_arr, out_offset_calculator) =
      AllocOffsetCalculator(output, stream);
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    unary_kernel<NUM_THREADS, unroll_factor><<<grid, block, 0, cuda_stream>>>(
      input->data_ptr<in_t>(), size, output->data_ptr<out_t>(), op,
      in_offset_calculator, out_offset_calculator);
    NDArray::MarkUsedBy({in_offset_calculator_arr, out_offset_calculator_arr}, stream);
  }
}

// Index (Zero-operand) loop kernel
template <int nt, int vt, typename out_t, typename func_t>
__global__ void index_kernel(size_t size, out_t* output, func_t op,
                             const OffsetCalculator* out_offset_calculator) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      auto out_offset = out_offset_calculator->get(idx);
      output[out_offset] = op(idx);
      idx += nt;
    }
  }
}

template <typename out_t, typename func_t>
void launch_loop_kernel(NDArray& output, size_t size,
                        const Stream& stream, const func_t& op) {
  bool contiguous = output->is_contiguous();
  if (contiguous) {
    launch_vectorized_kernel(op, size, stream, output->data_ptr<out_t>());
  } else {
    constexpr int unroll_factor = sizeof(DataType2Size(output->dtype())) >= 4 ? 2 : 4;
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(size, unroll_factor * block.x));
    NDArray out_offset_calculator_arr;
    OffsetCalculator *out_offset_calculator;
    std::tie(out_offset_calculator_arr, out_offset_calculator) =
      AllocOffsetCalculator(output, stream);
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    index_kernel<NUM_THREADS, unroll_factor><<<grid, block, 0, cuda_stream>>>(
      size, output->data_ptr<out_t>(), op, out_offset_calculator);
    NDArray::MarkUsedBy({out_offset_calculator_arr}, stream);
  }
}

template <typename ins_t, int end, int current=0>
struct load_multiple_inputs {
  __device__ static void apply(void** input_ptrs, ins_t* inputs, int linear_idx, int j) {
    using arg_t = std::tuple_element_t<current, ins_t>;
    std::get<current>(inputs[j]) = *(reinterpret_cast<arg_t*>(input_ptrs[current]) + linear_idx);
    load_multiple_inputs<ins_t, end, current + 1>::apply(input_ptrs, inputs, linear_idx, j);
  }

  __device__ static void apply(void** input_ptrs, ins_t* inputs, int* linear_idx, int j) {
    using arg_t = std::tuple_element_t<current, ins_t>;
    std::get<current>(inputs[j]) = *(reinterpret_cast<arg_t*>(input_ptrs[current]) + linear_idx[current]);
    load_multiple_inputs<ins_t, end, current + 1>::apply(input_ptrs, inputs, linear_idx, j);
  }
};

template <typename ins_t, int end>
struct load_multiple_inputs<ins_t, end, end> {
  __device__ static void apply(void** input_ptrs, ins_t* inputs, int linear_idx, int j) {}

  __device__ static void apply(void** input_ptrs, ins_t* inputs, int* linear_idx, int j) {}
};

template <typename outs_t, int num_outputs, int current=0>
struct store_multiple_outputs {
  __device__ static void apply(void** output_ptrs, outs_t outputs, int linear_idx) {
    using out_t = typename thrust::tuple_element<current, outs_t>::type;
    out_t *to = reinterpret_cast<out_t*>(output_ptrs[current]) + linear_idx;
    *to = thrust::get<current>(outputs);
    store_multiple_outputs<outs_t, num_outputs, current + 1>::apply(output_ptrs, outputs, linear_idx);
  }

  __device__ static void apply(void** output_ptrs, outs_t outputs, int* linear_idx) {
    using out_t = typename thrust::tuple_element<current, outs_t>::type;
    out_t *to = reinterpret_cast<out_t*>(output_ptrs[current]) + linear_idx[current];
    *to = thrust::get<current>(outputs);
    store_multiple_outputs<outs_t, num_outputs, current + 1>::apply(output_ptrs, outputs, linear_idx);
  }
};

template <typename outs_t, int num_outputs>
struct store_multiple_outputs<outs_t, num_outputs, num_outputs> {
  __device__ static void apply(void** output_ptrs, outs_t outputs, int linear_idx) {}

  __device__ static void apply(void** output_ptrs, outs_t outputs, int* linear_idx) {}
};

template <typename in_t, typename out_t, typename func_t>
__device__ inline void unroll_kernel_for_multi_outputs_impl(void** input_ptrs, void** output_ptrs,
                                                            int remaining, int base_idx, func_t op) {
  constexpr int num_outputs = thrust::tuple_size<out_t>::value;
  constexpr int nargs = std::tuple_size<in_t>::value;

  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  in_t inputs[THREAD_WORK_SIZE];
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    load_multiple_inputs<in_t, nargs>::apply(input_ptrs, inputs, linear_idx, i);
    results[i] = hetu::impl::apply(op, inputs[i]);
  }
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    store_multiple_outputs<out_t, num_outputs>::apply(output_ptrs, results[i], linear_idx);
  }
}

template <typename in_t, typename out_t, int num_outputs, int num_inputs, typename func_t>
__global__ void unroll_kernel_for_multi_outputs(void** input_ptrs, void** output_ptrs,
                                                size_t size, func_t op) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  unroll_kernel_for_multi_outputs_impl<in_t, out_t>(input_ptrs, output_ptrs, remaining, base_idx, op);
}

template <int nt, int vt, int num_outputs, int num_inputs,
          typename in_t, typename out_t, typename func_t>
__global__ void loop_kernel_for_multi_outputs(void** input_ptrs, void** output_ptrs,
                                              size_t size, func_t op,
                                              OffsetCalculator** in_offset_calculators,
                                              OffsetCalculator** out_offset_calculators) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  in_t inputs[vt];
  out_t results[vt];
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      int in_offsets[num_inputs];
      int out_offsets[num_outputs];
      #pragma unroll
      for (int j = 0; j < num_inputs; j++) {
        in_offsets[j] = in_offset_calculators[j]->get(idx);
      }
      #pragma unroll
      for (int j = 0; j < num_outputs; j++) {
        out_offsets[j] = out_offset_calculators[j]->get(idx);
      }
      load_multiple_inputs<in_t, num_inputs>::apply(input_ptrs, inputs, in_offsets, i);
      results[i] = hetu::impl::apply(op, inputs[i]);
      store_multiple_outputs<out_t, num_outputs>::apply(output_ptrs, results[i], out_offsets);
      idx += nt;
    }
  }
}

template <typename in_t, typename out_t, typename func_t>
void launch_loop_kernel_multiple_outputs(const NDArrayList& inputs, const NDArrayList& outputs, size_t size,
                                         const Stream& stream, const func_t& op) {
  constexpr int num_outputs = thrust::tuple_size<out_t>::value;
  constexpr int num_inputs = std::tuple_size<in_t>::value;
  HT_ASSERT(num_outputs == outputs.size());
  HT_ASSERT(num_inputs == inputs.size());
  auto device_id = stream.device_index();
  std::vector<void*> data_ptrs(num_outputs + num_inputs);
  for (int i = 0; i < num_outputs; i++) {
    data_ptrs[i] = outputs[i]->raw_data_ptr();
  }
  for (int i = 0; i < num_inputs; i++) {
    data_ptrs[num_outputs + i] = inputs[i]->raw_data_ptr();
  }
  NDArray data_ptrs_arr =
    hetu::cuda::to_ptr_ndarray(data_ptrs, device_id);
  void** output_ptrs = data_ptrs_arr->data_ptr<void*>();
  void** input_ptrs = output_ptrs + num_outputs;
  bool contiguous = IsContiguous(inputs) && IsContiguous(outputs);
  if (contiguous) {
    int64_t grid = DIVUP(size, BLOCK_WORK_SIZE);
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    unroll_kernel_for_multi_outputs<in_t, out_t, num_outputs, num_inputs><<<grid, NUM_THREADS, 0, cuda_stream>>>(
      input_ptrs, output_ptrs, size, op);
  } else {
    constexpr int unroll_factor = 2;
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(size, unroll_factor * block.x));
    NDArrayList data = inputs;
    data.insert(data.end(), outputs.begin(), outputs.end());
    NDArrayList offset_calculator_arrs(num_inputs + num_outputs);
    std::vector<OffsetCalculator*> offset_calculators(num_inputs + num_outputs);
    std::tie(offset_calculator_arrs, offset_calculators) =
      AllocOffsetCalculator(data, stream);
    NDArray offset_calculators_arr =
      hetu::cuda::to_ptr_ndarray(offset_calculators, device_id);
    OffsetCalculator** in_offset_calculators = offset_calculators_arr->data_ptr<OffsetCalculator*>();
    OffsetCalculator** out_offset_calculators = in_offset_calculators + num_inputs;
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    loop_kernel_for_multi_outputs<NUM_THREADS, unroll_factor, num_outputs, num_inputs, in_t, out_t>
                                 <<<grid, block, 0, cuda_stream>>>
                                 (input_ptrs, output_ptrs, size, op,
                                  in_offset_calculators, out_offset_calculators);
    offset_calculator_arrs.push_back(offset_calculators_arr);
    NDArray::MarkUsedBy(offset_calculator_arrs, stream);
  }
  NDArray::MarkUsedBy({data_ptrs_arr}, stream);
}

} // namespace impl
} // namespace hetu
