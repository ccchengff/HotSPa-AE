#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void eye_cpu(spec_t* output, size_t size, size_t ncols) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t row = idx / ncols;
    size_t col = idx % ncols;
    if (row == col)
      output[idx] = 1;
    else
      output[idx] = 0;
  }
}

void EyeCpu(NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output);
  HT_ASSERT(output->ndim() == 2);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  size_t ncols = output->shape(1);
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "EyeCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [output, size, ncols]() {
      eye_cpu<spec_t>(
        output->data_ptr<spec_t>(), size, ncols);
      },"Eye");
    });
  NDArray::MarkUsedBy({output}, stream);
}

} // namespace impl
} // namespace hetu
