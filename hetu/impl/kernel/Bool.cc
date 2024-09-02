#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/omp_utils.h"


namespace hetu {
namespace impl {

template <typename spec_t>
void bool_cpu(const spec_t* input, size_t size, bool* output) {
  for (size_t idx = 0; idx < size; ++idx) {
    if (input[idx] > 0)
      output[idx] = 1;
    else
      output[idx] = 0;
  }
}

template <typename spec_t>
void bool_cpu(const spec_t* input, size_t size, bool* output,
              int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    if (input[i_idx] > 0)
      output[i_idx] = 1;
    else
      output[i_idx] = 0;
  }
}

template <typename spec_t>
void bool_cpu(const spec_t* input, size_t size, bool* output,
              int64_t ndims, const int64_t* stride, const int64_t* stride_out,
              const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    if (input[i_idx] > 0)
      output[o_idx] = 1;
    else
      output[o_idx] = 0;
  }
}

void BoolCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(output->dtype() == DataType::BOOL);

  CPUStream cpu_stream(stream);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BoolCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      if (input->is_contiguous() && output->is_contiguous()) {
        bool_cpu<spec_t>(
          input->data_ptr<spec_t>(), size, output->data_ptr<bool>());
      }
      else {
        bool_cpu<spec_t>(
          input->data_ptr<spec_t>(), size, output->data_ptr<bool>(),
          input->ndim(), input->stride().data(), output->stride().data(),
          input->shape().data());
      }
      },
      "Bool");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
