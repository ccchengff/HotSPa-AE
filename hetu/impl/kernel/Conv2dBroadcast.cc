#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void conv2d_broadcast_cpu(const spec_t* input, spec_t* output,
                          size_t input_size, size_t output_size, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < output_size; idx++) {
    size_t input_id = (idx % (input_size * output_size)) / output_size;
    output[idx] = input[input_id];
  }
}

void Conv2dBroadcastCpu(const NDArray& input, NDArray& output,
                        const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->shape(0) == output->shape(1));

  CPUStream cpu_stream(stream);
  
  size_t batch_size = output->shape(0);
  size_t input_size = input->shape(0);
  size_t output_size = (output->shape(2)) * (output->shape(3));
  size_t size = batch_size * input_size * output_size;
  if (input_size == 0 || output_size == 0 || batch_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "Conv2dBroadcastCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, input_size, output_size, size]() {
      conv2d_broadcast_cpu<spec_t>(input->data_ptr<spec_t>(),
                                   output->data_ptr<spec_t>(), input_size,
                                   output_size, size);
      },
      "Conv2dBroadcast");
       
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
