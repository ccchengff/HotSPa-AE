#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void maskedfill_cpu(const spec_t* input, const int64_t* mask, 
                                spec_t val, spec_t* output, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    bool mask_bit = bool(mask[idx]);
    output[idx] = mask_bit ? val : input[idx];
  }
}

void MaskedfillCpu(const NDArray& input, const NDArray& mask,
                   double val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, mask);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "MaskfillCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, mask, output, val, size]() {
        maskedfill_cpu<spec_t>(
        input->data_ptr<spec_t>(), mask->data_ptr<int64_t>(),
        static_cast<spec_t>(val), output->data_ptr<spec_t>(), size);
      },"Maskfill"); 
    });
  NDArray::MarkUsedBy({input, mask, output}, stream);
}

} // namespace impl
} // namespace hetu
