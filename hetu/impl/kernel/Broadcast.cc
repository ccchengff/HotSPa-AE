#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void broadcast_cpu(const spec_t* input, size_t input_size, size_t size,
                   spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx)
    output[idx] = input[idx % input_size];
}

template <typename spec_t>
void broadcast_gradient_cpu(const spec_t* input, size_t input_size, size_t size,
                            spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx)
    output[idx] = input[idx];
}

void BroadcastCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  size_t size = output->numel();
  size_t input_size = input->numel();
  if (size == 0 || input_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, input_size, size]() {
      broadcast_cpu<spec_t>(input->data_ptr<spec_t>(), input_size, size,
                            output->data_ptr<spec_t>());
      },
      "Broadcast");     
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void BroadcastGradientCpu(const NDArray& input, NDArray& output,
                          const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  size_t size = output->numel();
  size_t input_size = input->numel();
  if (size == 0 || input_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, input_size, size]() {
      broadcast_gradient_cpu<spec_t>(input->data_ptr<spec_t>(), input_size,
                                     size, output->data_ptr<spec_t>());
      },
      "BroadcastGradient");     
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
