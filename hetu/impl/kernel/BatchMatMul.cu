#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <numeric>
#include <iterator>

namespace hetu {
namespace impl {

NDArray prepare_for_cublas(const NDArray& a, bool& transpose, int64_t& lda,
                           int64_t m, int64_t n, const Stream& stream) {
  size_t ndim = a->ndim();
  NDArray a_ = a;
  if (a->stride(ndim - 1) == 1 && (a->stride(ndim - 2) >= std::max<int64_t>(1, m))) {
    transpose = false;
    lda = a->stride(ndim - 2);
  } else if (a->stride(ndim - 2) == 1 && (a->stride(ndim - 1) >= std::max<int64_t>(1, n))) {
    transpose = true;
    lda = a->stride(ndim - 1);
  } else {
    transpose = false;
    if (!a->is_contiguous()) {
      a_ = NDArray::contiguous(a, stream.stream_index());
    }
    lda = a_->stride(ndim - 2);
  }
  return a_;
}

void BatchMatMulCuda(const NDArray& a, bool trans_a, const NDArray& b,
                     bool trans_b, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);
  HT_ASSERT_NDIM(a, 3);
  HT_ASSERT_NDIM(b, 3);
  HT_ASSERT_NDIM(output, 3);

  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index());
  HTAxes trans_axes = HTAxes(a->ndim());
  std::iota(trans_axes.begin(), trans_axes.end(), 0);
  std::iter_swap(trans_axes.end() - 2, trans_axes.end() - 1);
  NDArray a_trans = trans_a ? NDArray::permute(a, trans_axes, stream.stream_index())
                            : a;
  NDArray b_trans = trans_b ? NDArray::permute(b, trans_axes, stream.stream_index())
                            : b;
  int64_t m = output->shape(2);
  int64_t n = output->shape(1);
  int64_t k = b_trans->shape(1);
  HT_ASSERT(a_trans->shape(0) == b_trans->shape(0));
  HT_ASSERT(a_trans->shape(0) == output->shape(0));
  int64_t batchCount = a_trans->shape(0);
  int64_t lda, ldb, ldc = output->stride(1);
  bool trans_a_, trans_b_;
  NDArray a_ = prepare_for_cublas(b_trans, trans_a_, lda, m, k, stream);
  NDArray b_ = prepare_for_cublas(a_trans, trans_b_, ldb, k, n, stream);

  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "BatchMatMul", [&]() {
    spec_t alpha = 1, beta = 0;
    float alpha_f = 1, beta_f = 0;
    if (output->dtype() == DataType::FLOAT16 || output->dtype() == DataType::BFLOAT16) {
      cublas_batch_gemm<spec_t>(
        cublas_handle, trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, static_cast<const void*>(&alpha_f),
        a_->data_ptr<spec_t>(), lda, a_->stride(0), b_->data_ptr<spec_t>(),
        ldb, b_->stride(0), static_cast<const void*>(&beta_f), output->data_ptr<spec_t>(), ldc, output->stride(0),
        batchCount);
    }
    else {
      cublas_batch_gemm<spec_t>(
        cublas_handle, trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha,
        a_->data_ptr<spec_t>(), lda, a_->stride(0), b_->data_ptr<spec_t>(),
        ldb, b_->stride(0), &beta, output->data_ptr<spec_t>(), ldc, output->stride(0),
        batchCount);
    }
  });
  NDArray::MarkUsedBy({a_trans, b_trans, a_, b_, output}, stream);
}

} // namespace impl
} // namespace hetu
