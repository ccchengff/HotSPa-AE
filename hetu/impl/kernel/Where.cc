#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void where_cpu(const int64_t* cond, const spec_t* arr1, const spec_t* arr2,
               spec_t* output, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = cond[idx] ? arr1[idx] : arr2[idx];
  }
}

void WhereCpu(const NDArray& cond, const NDArray& inputA, const NDArray& inputB,
              NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(cond);
  HT_ASSERT_SAME_DEVICE(cond, inputA);
  HT_ASSERT_SAME_DEVICE(cond, inputB);
  HT_ASSERT_SAME_DEVICE(cond, output);
  HT_ASSERT_EXCHANGABLE(inputA, inputB);

  CPUStream cpu_stream(stream);

  size_t size = cond->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "WhereCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [cond, inputA, inputB, output, size]() {
        where_cpu<spec_t>(cond->data_ptr<int64_t>(), inputA->data_ptr<spec_t>(),
                          inputB->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
                          size);
        },"Where");
    });
  NDArray::MarkUsedBy({cond, inputA, inputB, output}, stream);
}

} // namespace impl
} // namespace hetu
