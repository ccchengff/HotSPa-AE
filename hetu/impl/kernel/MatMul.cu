#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

NDArray prepare_for_cublas(const NDArray& a, bool& transpose, const Stream& stream) {
  NDArray a_ = a;
  if (a->is_contiguous()) {
    transpose = true;
  }
  if (a->stride(0) == 1 && (a->stride(1) >= std::max<int64_t>(1, a->shape(0)))) {
    transpose = false;
  }
  else if (a->stride(1) == 1 && (a->stride(0) >= std::max<int64_t>(1, a->shape(1)))) {
    transpose = true;
  }
  else {
    transpose = true;
    a_ = NDArray::contiguous(a, stream.stream_index());
  }
  return a_;
}

void MatMulCuda(const NDArray& a, bool trans_a, const NDArray& b, bool trans_b,
                NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(b, 2);
  HT_ASSERT_NDIM(output, 2);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index());
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
    spec_t alpha = 1, beta = 0;
    float alpha_f = 1, beta_f = 0;
    if (output->dtype() == DataType::FLOAT16 || output->dtype() == DataType::BFLOAT16) {
      cublas_gemm<spec_t>(cublas_handle, !trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
                          !trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, static_cast<const void*>(&alpha_f),
                          a_->data_ptr<spec_t>(), lda,
                          b_->data_ptr<spec_t>(), ldb, static_cast<const void*>(&beta_f),
                          output->data_ptr<spec_t>(), ldc);
    }
    else {
      cublas_gemm<spec_t>(cublas_handle, !trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
                          !trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, static_cast<const void*>(&alpha),
                          a_->data_ptr<spec_t>(), lda,
                          b_->data_ptr<spec_t>(), ldb, static_cast<const void*>(&beta),
                          output->data_ptr<spec_t>(), ldc);
    }
  });
  NDArray::MarkUsedBy({a_trans, b_trans, a_, b_, output}, stream);
}

} // namespace impl
} // namespace hetu
