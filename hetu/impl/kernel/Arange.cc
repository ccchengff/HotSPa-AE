#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void range_cpu(spec_t start, spec_t step, size_t size, spec_t* output) {
  for (size_t idx = 0; idx < size; idx++) 
    output[idx] = start + step * idx;
}

void ArangeCpu(double start, double step, NDArray& output, const Stream& stream) {
  CPUStream cpu_stream(stream);
  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "RangeCpu", [&]() {
      auto _range_future = cpu_stream.EnqueueTask(
      [start, step, output, size]() {
      range_cpu<spec_t>(
        static_cast<spec_t>(start), static_cast<spec_t>(step), size, output->data_ptr<spec_t>());
      },
      "Arange");
    });
  NDArray::MarkUsedBy({output}, stream);
}

} // namespace impl
} // namespace hetu
