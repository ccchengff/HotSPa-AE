#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void asstrided_cpu(const spec_t *input, spec_t *output, size_t size,
                   const int64_t *stride_in, const int64_t *stride_out, int ndim) {
  for (size_t idx = 0; idx < size; ++idx) {
    int index = 0;
    size_t ind = idx;
    for (int i = 0; i < ndim; i++) {
      int tmp_index = ind / stride_out[i];
      index += tmp_index * stride_in[i];
      ind = ind % stride_out[i];
    }
    output[idx] = input[index];
  }
}

void AsStridedCpu(const NDArray& input, NDArray& output, const HTShape& stride,
                  const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  int ndim = output->ndim();

  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "AsStridedCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, stride, size, ndim]() {
        asstrided_cpu<spec_t>(
          input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size, 
          stride.data(), output->stride().data(), ndim);
      },"AsStrided");     
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void array_zero_set_cpu(spec_t* input, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    input[idx] = 0;
  }
}

template <typename spec_t>
void asstrided_gradient_cpu(const spec_t *input, spec_t *output, size_t size,
                            const int64_t *stride_in, const int64_t *stride_out, int ndim) {
  for (size_t idx = 0; idx < size; ++idx) {
    int index = 0;
    size_t ind = idx;
    for (int i = 0; i < ndim; i++) {
      int tmp_index = ind / stride_out[i];
      index += tmp_index * stride_in[i];
      ind = ind % stride_out[i];
    }
    output[index] += input[idx];
  }
}

void AsStridedGradientCpu(const NDArray& output, NDArray& input,
                          const HTShape& stride, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);
  size_t size = output->numel();
  int ndim = output->ndim();

  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "AsStridedGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, stride, size, ndim]() {
        array_zero_set_cpu<spec_t>(
          input->data_ptr<spec_t>(), input->numel());
        asstrided_gradient_cpu<spec_t>(
          output->data_ptr<spec_t>(), input->data_ptr<spec_t>(), size, 
          stride.data(), output->stride().data(), ndim);
      },
      "AsStridedGradient");     
    });
  NDArray::MarkUsedBy({output, input}, stream);
}

} // namespace impl
} // namespace hetu
