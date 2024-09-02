#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void triutril_cpu(const spec_t* input, spec_t* output, bool lower,
                  int64_t H, int64_t W, int64_t diagonal, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    int row = (idx / W) % H;
    int col = idx % W;
    bool mask = lower ? (col - row > diagonal) : (col - row < diagonal);
    output[idx] = mask ? spec_t(0) : input[idx];
  }
}

template <typename spec_t>
void triutril_cpu(const spec_t* input, spec_t* output, bool lower,
                  int64_t H, int64_t W, int64_t diagonal, size_t size,
                  int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                  const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    int row = (idx / W) % H;
    int col = idx % W;
    bool mask = lower ? (col - row > diagonal) : (col - row < diagonal);
    output[o_idx] = mask ? spec_t(0) : input[i_idx];
  }
}

void TriuTrilCpu(const NDArray& input, NDArray& output, bool lower,
                 int64_t diagonal, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  int64_t ndim = input->ndim();
  int64_t H = input->shape(ndim - 2);
  int64_t W = input->shape(ndim - 1);
  if (size == 0)
    return;
  if (input->is_contiguous() && output->is_contiguous()) {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "TriuTrilCpu", [&]() {
        auto _future = cpu_stream.EnqueueTask(
          [input, output, lower, H, W, diagonal, size]() {
          triutril_cpu<spec_t>(
            input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
            lower, H, W, diagonal, size);
          },"TriuTril");
      });
  }
  else {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "TriuTrilCpu", [&]() {
        auto _future = cpu_stream.EnqueueTask(
          [input, output, lower, H, W, diagonal, size]() {
          triutril_cpu<spec_t>(
            input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
            lower, H, W, diagonal, size, input->ndim(),
            input->stride().data(), output->stride().data(),
            input->shape().data());
          },"TriuTril");
      });
  }
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
