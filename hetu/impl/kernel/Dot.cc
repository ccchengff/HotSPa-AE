#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void dot_cpu(const spec_t* inputA, const spec_t* inputB, size_t size,
             spec_t* output) {
  spec_t out = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    out += inputA[idx] * inputB[idx];
  *output = out;
}

void DotCpu(const NDArray& inputA, const NDArray& inputB, NDArray& output,
               const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, output);
  HT_ASSERT_SAME_DEVICE(inputB, output);
  HT_ASSERT_NDIM(inputA, 1);
  HT_ASSERT_NDIM(inputB, 1);
  HT_ASSERT_NDIM(output, 0);

  CPUStream cpu_stream(stream);

  size_t size = inputA->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "DotCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [inputA, inputB, output, size]() {
      dot_cpu<spec_t>(inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(),
                      size, output->data_ptr<spec_t>());
      },"Dot");
    });
  NDArray::MarkUsedBy({inputA, inputB, output}, stream);
}

} // namespace impl
} // namespace hetu
