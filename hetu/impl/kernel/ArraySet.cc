#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void array_set_cpu(spec_t* arr, spec_t value, size_t size) {
  std::fill(arr, arr + size, value);
}

void ArraySetCpu(NDArray& data, double value, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(data);
  CPUStream cpu_stream(stream);
  size_t size = data->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "ArraySetCpu", [&]() {
      auto _arrayset_future = cpu_stream.EnqueueTask(
      [data, value, size]() {
        array_set_cpu<spec_t>(data->data_ptr<spec_t>(),
                              static_cast<spec_t>(value), size);
      },"ArraySet");
    });
  NDArray::MarkUsedBy({data}, stream);
}

} // namespace impl
} // namespace hetu
