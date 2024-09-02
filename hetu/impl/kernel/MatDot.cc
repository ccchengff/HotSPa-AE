#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void dot_cpu(const spec_t* inputA, const spec_t* inputB, size_t size,
             size_t size2, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = inputA[idx] * inputB[(int) (idx % size2)];
}

void MatDotCpu(const NDArray& inputA, const NDArray& inputB, NDArray& output,
               const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, output);
  HT_ASSERT_SAME_DEVICE(inputB, output);
  HT_ASSERT_EXCHANGABLE(inputA, output);

  CPUStream cpu_stream(stream);

  size_t size = inputA->numel();
  size_t size2 = inputB->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "MatDotCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [inputA, inputB, output, size, size2]() {
      dot_cpu<spec_t>(inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(),
                      size, size2, output->data_ptr<spec_t>());
      },"MatDot");   
    });
  NDArray::MarkUsedBy({inputA, inputB, output}, stream);
}

} // namespace impl
} // namespace hetu
