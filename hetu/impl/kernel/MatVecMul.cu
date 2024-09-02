#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

NDArray prepare_for_cublas(const NDArray& a, bool& trans, int64_t& m,
                           int64_t& n, int64_t& lda, const Stream& stream) {
  NDArray a_ = a;
  if (a->stride(0) == 1 && (a->stride(1) >= std::max<int64_t>(1, a->shape(0)))) {
    m = a->shape(0);
    n = a->shape(1);
    lda = a->stride(1);
    trans = !trans;
  } else if (a->stride(1) == 1 && (a->stride(0) >= std::max<int64_t>(1, a->shape(1)))) {
    m = a->shape(1);
    n = a->shape(0);
    lda = a->stride(0);
  } else {
    m = a->shape(1);
    n = a->shape(0);
    lda = a->stride(0);
    a_ = NDArray::contiguous(a, stream.stream_index());
  }
  return a_;
}

void MatVecMulCuda(const NDArray& a, bool trans, const NDArray& x,
                NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, x);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(x, 1);
  HT_ASSERT_NDIM(output, 1);
  HT_ASSERT_SAME_DTYPE(a, x);
  HT_ASSERT_SAME_DTYPE(a, output);

  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index());
  const auto x_ = x->stride(0) == 0 ? NDArray::contiguous(x, stream.stream_index()) : x;
  auto x_stride = x_->stride(0);
  int64_t m, n, lda;
  NDArray a_ = prepare_for_cublas(a, trans, m, n, lda, stream);

  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "MatVecMul", [&]() {
    spec_t alpha = 1, beta = 0;
    float alpha_f = 1, beta_f = 0;
    if (output->dtype() == DataType::FLOAT16 || output->dtype() == DataType::BFLOAT16) {
      cublas_gemv<spec_t>(cublas_handle, !trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                          m, n, static_cast<const void*>(&alpha_f),
                          a_->data_ptr<spec_t>(), lda,
                          x_->data_ptr<spec_t>(), x_stride, static_cast<const void*>(&beta_f),
                          output->data_ptr<spec_t>(), output->stride(0));
    }
    else {
      cublas_gemv<spec_t>(cublas_handle, !trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                          m, n, static_cast<const void*>(&alpha),
                          a_->data_ptr<spec_t>(), lda,
                          x_->data_ptr<spec_t>(), x_stride, static_cast<const void*>(&beta),
                          output->data_ptr<spec_t>(), output->stride(0));
    }
  });
  NDArray::MarkUsedBy({a, x, a_, x_, output}, stream);
}

} // namespace impl
} // namespace hetu
