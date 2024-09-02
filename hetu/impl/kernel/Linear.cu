#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

extern NDArray prepare_for_cublas(const NDArray& a, bool& transpose, const Stream& stream);

template <typename spec_t>
__global__ void bias_set_kernel(const spec_t* input, spec_t* output,
                                size_t input_size, size_t size,
                                const OffsetCalculator* in_offset_calculator,
                                const OffsetCalculator* out_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto in_offset = in_offset_calculator->get(idx % input_size);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = input[in_offset];
}

void LinearCuda(const NDArray& a, bool trans_a, const NDArray& b, bool trans_b,
                const NDArray& bias, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(b, 2);
  HT_ASSERT_NDIM(output, 2);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  size_t size = output->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index()); 
  
  bool bias_exist = bias.is_defined();
  if (bias_exist) {
    HT_ASSERT_SAME_DEVICE(a, bias);
    HT_ASSERT_NDIM(bias, 1);
    HT_ASSERT_SAME_DTYPE(a, bias);
    size_t input_size = bias->numel();
    NDArray bias_offset_calculator_arr, out_offset_calculator_arr;
    OffsetCalculator *bias_offset_calculator, *out_offset_calculator;
    std::tie(bias_offset_calculator_arr, bias_offset_calculator) =
      AllocOffsetCalculator(bias, stream);
    std::tie(out_offset_calculator_arr, out_offset_calculator) = 
      AllocOffsetCalculator(output, stream);
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      bias->dtype(), spec_t, "BiasSetCuda", [&]() {
        bias_set_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          bias->data_ptr<spec_t>(), output->data_ptr<spec_t>(), input_size, size,
          bias_offset_calculator, out_offset_calculator);
      });
    NDArray::MarkUsedBy({bias_offset_calculator_arr, out_offset_calculator_arr}, stream);
  }
  HTAxes trans_axes = {1, 0};
  NDArray a_trans = trans_a ? NDArray::permute(a, trans_axes, stream.stream_index())
                            : a;
  NDArray b_trans = trans_b ? NDArray::permute(b, trans_axes, stream.stream_index())
                            : b;
  bool trans_a_;
  bool trans_b_;
  NDArray a_ = prepare_for_cublas(b_trans, trans_a_, stream);
  NDArray b_ = prepare_for_cublas(a_trans, trans_b_, stream);

  int64_t m = a_->shape(1);
  int64_t n = b_->shape(0);
  int64_t k = a_->shape(0);
  int64_t lda = a_->stride(!trans_a_ ? 1 : 0);
  int64_t ldb = b_->stride(!trans_b_ ? 1 : 0);
  int64_t ldc = output->stride(0);

  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "MatMul", [&]() {
    spec_t alpha = 1, beta = bias_exist ? 1 : 0;
    float alpha_f = 1, beta_f = bias_exist ? 1 : 0;
    if (output->dtype() == DataType::FLOAT16 || output->dtype() == DataType::BFLOAT16) {
      cublas_gemm<spec_t>(cublas_handle, !trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
                          !trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, static_cast<const void*>(&alpha_f),
                          a_->data_ptr<spec_t>(), lda,
                          b_->data_ptr<spec_t>(), ldb, static_cast<const void*>(&beta_f),
                          output->data_ptr<spec_t>(), ldc);
    } else {
      cublas_gemm<spec_t>(cublas_handle, !trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
                          !trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, static_cast<const void*>(&alpha),
                          a_->data_ptr<spec_t>(), lda,
                          b_->data_ptr<spec_t>(), ldb, static_cast<const void*>(&beta),
                          output->data_ptr<spec_t>(), ldc);
    }
  });
  NDArray::MarkUsedBy({a_trans, b_trans, a_, b_, bias, output}, stream);
}

} // namespace impl
} // namespace hetu
