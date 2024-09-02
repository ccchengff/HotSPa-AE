#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

// Out-of-place version of transpose and its gradient
/* It is replaced with in-place version. */
template <typename spec_t>
void transpose_cpu(const spec_t* input, spec_t* output, const int64_t* buf,
                   uint32_t ndims, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    const auto* in_strides = buf;
    const auto* out_strides = buf + ndims;
    const auto* perm = buf + ndims * 2;
    uint32_t i_idx = 0;
    uint32_t t = idx;
    for (uint32_t i = 0; i < ndims; ++i) {
      const uint32_t ratio = t / out_strides[i];
      t -= ratio * out_strides[i];
      i_idx += ratio * in_strides[perm[i]];
    }
    output[idx] = input[i_idx];
  }
}

void TransposeCpu(const NDArray& input, NDArray& output, const HTAxes& perm,
                  const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  auto ndim = static_cast<uint32_t>(input->ndim());
  auto ndim_ = static_cast<uint32_t>(output->ndim());
  HT_ASSERT(ndim == ndim_);
  HTShape buf(3 * ndim);
  int64_t in_stride = 1;
  int64_t out_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buf[i] = in_stride;
    buf[ndim + i] = out_stride;
    buf[ndim * 2 + i] = perm[i];
    in_stride *= input->shape(i);
    out_stride *= output->shape(i);
  }
  HT_ASSERT(in_stride == out_stride);
  size_t size = in_stride;
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TransposeCpu", [&]() {
      cpu_stream.EnqueueTask(
        [input, output, buf, ndim ,size]() {
        transpose_cpu<spec_t>(input->data_ptr<spec_t>(),
                              output->data_ptr<spec_t>(), buf.data(), ndim, size);
        },"Transpose");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
