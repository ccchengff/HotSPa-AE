#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void onehot_cpu(const spec_t* input, size_t size, size_t last_dim,
                spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int offset = (int) (idx % last_dim);
    float writein = 0;
    if (offset == (int) input[idx / last_dim]) {
      writein = 1;
    } else {
      writein = 0;
    }
    output[idx] = writein;
  }
}

void OnehotCpu(const NDArray& input, size_t num_classes, NDArray& output,
               const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  size_t input_dim = input->ndim();
  size_t last_dim = output->shape(input_dim);
  HT_ASSERT(num_classes == last_dim) << "The last dim of output is invalid.";
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "OnehotCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size, last_dim]() {
      onehot_cpu<spec_t>(input->data_ptr<spec_t>(), size, last_dim,
                         output->data_ptr<spec_t>());
      },"Onehot");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
