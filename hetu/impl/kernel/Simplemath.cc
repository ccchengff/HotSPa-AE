#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void floor_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = std::floor(input[idx]);
}

void FloorCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "FloorCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
        floor_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                          output->data_ptr<spec_t>());
      }, "Floor");    
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void ceil_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = std::ceil(input[idx]);
}

void CeilCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "CeilCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      ceil_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                       output->data_ptr<spec_t>());
      }, "Ceil");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void round_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = std::round(input[idx]);
}

void RoundCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "RoundCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      round_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                        output->data_ptr<spec_t>());
      }, "Round");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
