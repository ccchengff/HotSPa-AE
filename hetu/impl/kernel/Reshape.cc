#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void memory_copy_cpu(const spec_t* input, spec_t* output, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = input[idx];
  }
}

template <typename spec_t>
void memory_copy_cpu(const spec_t* input, spec_t* output, size_t size,
                     int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                     const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = input[i_idx];
  }
}

void ReshapeCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  size_t input_size = input->numel();
  size_t size = output->numel();
  HT_ASSERT(input_size == size) << "input size and output size are different. ";
  if (input_size == 0)
    return;
  if (input->is_contiguous() && output->is_contiguous()) {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "ReshapeCpu", [&]() {
        auto _future = cpu_stream.EnqueueTask(
          [input, output, size]() {
          memory_copy_cpu<spec_t>(input->data_ptr<spec_t>(),
                                  output->data_ptr<spec_t>(), size);
          },"Reshape");
      });
  }
  else {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "ReshapeCpu", [&]() {
        auto _future = cpu_stream.EnqueueTask(
          [input, output, size]() {
          memory_copy_cpu<spec_t>(input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
                                  size, input->ndim(),
                                  input->stride().data(), output->stride().data(),
                                  input->shape().data());
          },"Reshape");
      });    
  }
  NDArray::MarkUsedBy({input, output}, stream);  
}

} // namespace impl
} // namespace hetu
