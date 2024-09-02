#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void outer_cpu(const spec_t* inputA, const spec_t* inputB, size_t sizeB, size_t size, spec_t* output) {
  for (size_t idx = 0; idx < size; ++idx) {
    int64_t A_idx = idx / sizeB; 
    int64_t B_idx = idx % sizeB;
    output[idx] = inputA[A_idx] * inputB[B_idx];
  }
}


void OuterCpu(const NDArray& inputA, const NDArray& inputB, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, inputB);
  HT_ASSERT_SAME_DEVICE(inputA, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  size_t sizeB = inputB->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "OuterCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [inputA, inputB, output, sizeB, size]() {
      outer_cpu<spec_t>(
        inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(), sizeB, size, output->data_ptr<spec_t>());
      },"Outer");
    });
  NDArray::MarkUsedBy({inputA, inputB, output}, stream);
}


} // namespace impl
} // namespace hetu
