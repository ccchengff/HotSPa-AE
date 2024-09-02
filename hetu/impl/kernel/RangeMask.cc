#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void rangemask_cpu(const spec_t* input, int64_t min, 
                   int64_t max, int64_t* output, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = (static_cast<int64_t>(input[idx]) >= min) && (static_cast<int64_t>(input[idx]) <= max) ? 0 : 1;
  }
}

void RangeMaskCpu(const NDArray& input, int64_t min, int64_t max,
                  NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);
  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "RangeMaskCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, min, max, output, size]() {
        rangemask_cpu<spec_t>(
        input->data_ptr<spec_t>(), min, max, 
        output->data_ptr<int64_t>(), size);
      },"RangeMask");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu
