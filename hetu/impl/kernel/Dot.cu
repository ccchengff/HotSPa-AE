#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void DotCuda(const NDArray& x, const NDArray& y, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(x);
  HT_ASSERT_SAME_DEVICE(x, y);
  HT_ASSERT_SAME_DEVICE(x, output);
  HT_ASSERT_NDIM(x, 1);
  HT_ASSERT_NDIM(output, 0);
  HT_ASSERT_SAME_DTYPE(x, y);
  HT_ASSERT_SAME_DTYPE(x, output);

  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index());

  int32_t n = x->numel();
  int32_t incx = x->stride(0);
  int32_t incy = y->stride(0);

  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "Dot", [&]() {
    cublas_dot<spec_t>(cublas_handle, n, x->data_ptr<spec_t>(), incx,
                       y->data_ptr<spec_t>(), incy,
                       output->data_ptr<spec_t>());
  });
  NDArray::MarkUsedBy({x, y, output}, stream);
}

} // namespace impl
} // namespace hetu
