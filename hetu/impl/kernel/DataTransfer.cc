#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

void DataTransferCpu(const NDArray& from, NDArray& to, const Stream& stream) {
  HT_ASSERT_COPIABLE(from, to);
  size_t numel = from->numel();
  if (numel == 0)
    return;
  void* to_ptr = to->raw_data_ptr();
  void* from_ptr = from->raw_data_ptr();
  if (to_ptr == from_ptr) {
    HT_ASSERT(from->dtype() == to->dtype())
      << "NDArrays with " << from->dtype() << " and " << to->dtype()
      << " types are sharing the same storage, which is not allowed.";
    return;
  }
  CPUStream cpu_stream(stream);
  auto _future = cpu_stream.EnqueueTask(
  [from, to, to_ptr, from_ptr, numel]() {
    if (from->dtype() == to->dtype()) {
      memcpy(to_ptr, from_ptr, numel * DataType2Size(from->dtype()));
    } else {
      HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(
        from->dtype(), to->dtype(), spec_a_t, spec_b_t, "DataTransferCpu", [&]() {
          auto* typed_from_ptr = reinterpret_cast<spec_a_t*>(from_ptr);
          auto* typed_to_ptr = reinterpret_cast<spec_b_t*>(to_ptr);
          std::copy(typed_from_ptr, typed_from_ptr + numel, typed_to_ptr);
        });
    }
  },
  "DataTransfer");
  NDArray::MarkUsedBy({from, to}, stream);
}

} // namespace impl
} // namespace hetu
