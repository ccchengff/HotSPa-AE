#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void rotary_kernel(const spec_t* x1, const spec_t* x2,
                              const spec_t* cos, const spec_t* sin,
                              spec_t* out1, spec_t* out2, size_t size,
                              size_t seq_len, size_t nheads, size_t rotary_dim,
                              const OffsetCalculator* x1_offset_calculator,
                              const OffsetCalculator* x2_offset_calculator,
                              const OffsetCalculator* cos_offset_calculator,
                              const OffsetCalculator* sin_offset_calculator,
                              const OffsetCalculator* out1_offset_calculator,
                              const OffsetCalculator* out2_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t rotary_idx = idx % rotary_dim;
  size_t seq_idx = (idx / (nheads * rotary_dim)) % seq_len;
  size_t cos_idx = seq_idx * rotary_dim + rotary_idx;
  auto x1_offset = x1_offset_calculator->get(idx);
  auto x2_offset = x2_offset_calculator->get(idx);
  auto cos_offset = cos_offset_calculator->get(cos_idx);
  auto sin_offset = sin_offset_calculator->get(cos_idx);
  auto out1_offset = out1_offset_calculator->get(idx);
  auto out2_offset = out2_offset_calculator->get(idx);
  out1[out1_offset] = x1[x1_offset] * cos[cos_offset] - x2[x2_offset] * sin[sin_offset];
  out2[out2_offset] = x1[x1_offset] * sin[sin_offset] + x2[x2_offset] * cos[cos_offset];
}

template <typename spec_t>
__global__ void rotary_conj_kernel(const spec_t* x1, const spec_t* x2,
                                   const spec_t* cos, const spec_t* sin,
                                   spec_t* out1, spec_t* out2, size_t size,
                                   size_t seq_len, size_t nheads, size_t rotary_dim,
                                   const OffsetCalculator* x1_offset_calculator,
                                   const OffsetCalculator* x2_offset_calculator,
                                   const OffsetCalculator* cos_offset_calculator,
                                   const OffsetCalculator* sin_offset_calculator,
                                   const OffsetCalculator* out1_offset_calculator,
                                   const OffsetCalculator* out2_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t rotary_idx = idx % rotary_dim;
  size_t seq_idx = (idx / (nheads * rotary_dim)) % seq_len;
  size_t cos_idx = seq_idx * rotary_dim + rotary_idx;
  auto x1_offset = x1_offset_calculator->get(idx);
  auto x2_offset = x2_offset_calculator->get(idx);
  auto cos_offset = cos_offset_calculator->get(cos_idx);
  auto sin_offset = sin_offset_calculator->get(cos_idx);
  auto out1_offset = out1_offset_calculator->get(idx);
  auto out2_offset = out2_offset_calculator->get(idx);
  out1[out1_offset] = x1[x1_offset] * cos[cos_offset] + x2[x2_offset] * sin[sin_offset];
  out2[out2_offset] = -x1[x1_offset] * sin[sin_offset] + x2[x2_offset] * cos[cos_offset];
}

void RotaryCuda(const NDArray& x1, const NDArray& x2,
                const NDArray& cos, const NDArray& sin,
                NDArray& out1, NDArray& out2,
                bool conj, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(x1);
  HT_ASSERT_SAME_DEVICE(x1, x2);
  HT_ASSERT_SAME_DEVICE(x1, cos);
  HT_ASSERT_SAME_DEVICE(x1, sin);
  HT_ASSERT_SAME_DEVICE(x1, out1);
  HT_ASSERT_SAME_DEVICE(x1, out2);

  size_t size = x1->numel();
  if (size == 0)
    return;
    
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  NDArray x1_offset_calculator_arr, x2_offset_calculator_arr,
          sin_offset_calculator_arr, cos_offset_calculator_arr,
          out1_offset_calculator_arr, out2_offset_calculator_arr;
  OffsetCalculator *x1_offset_calculator, *x2_offset_calculator,
                   *sin_offset_calculator, *cos_offset_calculator,
                   *out1_offset_calculator, *out2_offset_calculator;
  std::tie(x1_offset_calculator_arr, x1_offset_calculator) =
    AllocOffsetCalculator(x1, stream);
  std::tie(x2_offset_calculator_arr, x2_offset_calculator) =
    AllocOffsetCalculator(x2, stream);
  std::tie(sin_offset_calculator_arr, sin_offset_calculator) = 
    AllocOffsetCalculator(sin, stream);
  std::tie(cos_offset_calculator_arr, cos_offset_calculator) = 
    AllocOffsetCalculator(cos, stream);
  std::tie(out1_offset_calculator_arr, out1_offset_calculator) = 
    AllocOffsetCalculator(out1, stream);
  std::tie(out2_offset_calculator_arr, out2_offset_calculator) = 
    AllocOffsetCalculator(out2, stream);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  if (!conj) {
    HT_DISPATCH_FLOATING_TYPES(
    x1->dtype(), spec_t, "RotaryCuda", [&]() {
      rotary_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        x1->data_ptr<spec_t>(), x2->data_ptr<spec_t>(),
        cos->data_ptr<spec_t>(), sin->data_ptr<spec_t>(),
        out1->data_ptr<spec_t>(), out2->data_ptr<spec_t>(), size,
        x1->shape(1), x1->shape(2), x1->shape(3),
        x1_offset_calculator, x2_offset_calculator,
        cos_offset_calculator, sin_offset_calculator,
        out1_offset_calculator, out2_offset_calculator);
    });
  } else {
    HT_DISPATCH_FLOATING_TYPES(
    x1->dtype(), spec_t, "RotaryCuda", [&]() {
      rotary_conj_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        x1->data_ptr<spec_t>(), x2->data_ptr<spec_t>(),
        cos->data_ptr<spec_t>(), sin->data_ptr<spec_t>(),
        out1->data_ptr<spec_t>(), out2->data_ptr<spec_t>(), size,
        x1->shape(1), x1->shape(2), x1->shape(3),
        x1_offset_calculator, x2_offset_calculator,
        cos_offset_calculator, sin_offset_calculator,
        out1_offset_calculator, out2_offset_calculator);
    });
  }
  NDArray::MarkUsedBy({x1, x2, cos, sin, out1, out2}, stream);
}

} // namespace impl
} // namespace hetu