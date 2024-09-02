#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void conv2d_reduce_cpu(const spec_t* input, spec_t* output, size_t input_size,
                       size_t output_size, size_t batch_size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < output_size; idx++) {
    spec_t temp = 0;
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < input_size; j++) {
        temp += input[i * input_size * output_size + idx * input_size + j];
      }
    }
    output[idx] = temp;
  }
}

void Conv2dReduceSumCpu(const NDArray& input, NDArray& output,
                        const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  HT_ASSERT(input->shape(1) == output->shape(0));

  CPUStream cpu_stream(stream);
  
  size_t batch_size = input->shape(0);
  size_t input_size = input->shape(2) * input->shape(3);
  size_t output_size = output->shape(0);
  if (input_size == 0 || output_size == 0 || batch_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "Conv2dReduceCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, input_size, output_size, batch_size]() {
      conv2d_reduce_cpu<spec_t>(input->data_ptr<spec_t>(),
                                output->data_ptr<spec_t>(), input_size,
                                output_size, batch_size);
      },
      "Conv2dReduce");   
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
