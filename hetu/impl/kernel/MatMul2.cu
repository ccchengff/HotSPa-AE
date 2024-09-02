#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void MatMul2Cuda(const NDArray& a, bool trans_a, const NDArray& b,
                 bool trans_b, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index());

  HTShape broadcast_shape = NDArrayMeta::Broadcast(a->shape(), b->shape());

  int ndim = a->ndim();
  int m = output->shape(ndim - 1);
  int n = output->shape(ndim - 2);
  int k = trans_a ? a->shape(ndim - 2) : a->shape(ndim - 1);
  long long int strideA = a->shape(ndim - 2) * a->shape(ndim - 1);
  long long int strideB = b->shape(ndim - 2) * b->shape(ndim - 1);
  long long int strideC = output->shape(ndim - 2) * output->shape(ndim - 1);
  int batchCount = 1;
  for (int i = 0; i < ndim - 2; ++i) {
    HT_ASSERT(a->shape(i) == b->shape(i));
    HT_ASSERT(a->shape(i) == output->shape(i));
    batchCount *= a->shape(i);
  }

  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "BatchMatMul", [&]() {
    spec_t alpha = 1, beta = 0;
    cublas_batch_gemm<spec_t>(
      cublas_handle, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
      trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha,
      b->data_ptr<spec_t>(), trans_b ? k : m, strideB, a->data_ptr<spec_t>(),
      trans_a ? n : k, strideA, &beta, output->data_ptr<spec_t>(), m, strideC,
      batchCount);
  });
  NDArray::MarkUsedBy({a, b, output}, stream);
}

} // namespace impl
} // namespace hetu
