#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void gather_cpu(const spec_t* input, const int64_t* ids, size_t size, 
                size_t after_stride, size_t cur_stride,
                size_t after_stride_out, size_t cur_stride_out,
                spec_t* output) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t b_index = idx / (cur_stride_out * after_stride_out);
    size_t p_index = idx % (cur_stride_out * after_stride_out);
    size_t a_index = p_index % after_stride_out;
    size_t id_num = int(ids[idx]);
    size_t i_index =
      b_index * (cur_stride * after_stride) + id_num * after_stride + a_index;
    output[idx] = input[i_index];
  }
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
void gather_gradient_cpu(const spec_t* grad_output, const int64_t* ids,
                    size_t size, 
                    size_t after_stride, size_t cur_stride,
                    size_t after_stride_out, size_t cur_stride_out,
                    spec_t* grad_input) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t b_index = idx / (cur_stride_out * after_stride_out);
    size_t p_index = idx % (cur_stride_out * after_stride_out);
    size_t a_index = p_index % after_stride_out;
    size_t id_num = int(ids[idx]);
    size_t i_index =
      b_index * (cur_stride * after_stride) + id_num * after_stride + a_index;
    grad_input[i_index] += grad_output[idx];
  }
}

void GatherCpu(const NDArray& input, const NDArray& id, NDArray& output,
               size_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, id);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(id->ndim() == input->ndim())
    << "invalid index shape.Expect dim=1, but get" << id->ndim();
  size_t after_stride = 1, after_stride_out = 1;
  size_t cur_stride = input->shape(dim), cur_stride_out = output->shape(dim);
  HT_ASSERT(id->shape() == output->shape())
    << "Invalid shapes.Index shape:" << id->shape()
    << "Input shape:" << input->shape() << "Output shape:" << output->shape();

  CPUStream cpu_stream(stream);

  for (size_t i = dim + 1; i < input->ndim(); ++i) {
    after_stride *= input->shape(i);
    after_stride_out *= output->shape(i);
  }
  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "GatherCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, id, output, size, after_stride, cur_stride, after_stride_out, cur_stride_out]() {
        gather_cpu<spec_t>(
        input->data_ptr<spec_t>(), id->data_ptr<int64_t>(), size,
        after_stride, cur_stride, after_stride_out, cur_stride_out, output->data_ptr<spec_t>());
      },"Gather");
    });
  NDArray::MarkUsedBy({input, id, output}, stream);
}

void GatherGradientCpu(const NDArray& grad_output, const NDArray& id, NDArray& grad_input,
                        size_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(grad_output);
  HT_ASSERT_SAME_DEVICE(grad_output, id);
  HT_ASSERT_SAME_DEVICE(grad_output, grad_input);
  HT_ASSERT(id->ndim() == grad_input->ndim())
    << "invalid index shape.Expect dim=1, but get" << id->ndim();

  CPUStream cpu_stream(stream);

  size_t after_stride = 1, after_stride_out = 1;
  size_t cur_stride = grad_input->shape(dim), cur_stride_out = grad_output->shape(dim);
  for (size_t i = dim + 1; i < grad_input->ndim(); ++i) {
    after_stride *= grad_input->shape(i);
    after_stride_out *= grad_output->shape(i);
  }
  size_t size = grad_output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    grad_output->dtype(), spec_t, "GatherGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [grad_output, id, grad_input, size, after_stride, cur_stride, after_stride_out, cur_stride_out]() {
        array_zero_set_cpu<spec_t>(
        grad_input->data_ptr<spec_t>(), grad_input->numel());
        gather_gradient_cpu<spec_t>(
        grad_output->data_ptr<spec_t>(), id->data_ptr<int64_t>(), size, 
        after_stride, cur_stride, after_stride_out, cur_stride_out, grad_input->data_ptr<spec_t>());
      },"GatherGradient");
    });
  NDArray::MarkUsedBy({grad_output, id, grad_input}, stream);
}

} // namespace impl
} // namespace hetu
