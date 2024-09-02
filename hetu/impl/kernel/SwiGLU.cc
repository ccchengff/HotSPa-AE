#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

/*
  假设输入dim [..., d * 2] -> [...., d]
*/
template <typename spec_t>
void swiglu_cpu(const spec_t* input, size_t size, size_t d, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    auto in_idx = (idx / d) * (2 * d) + (idx % d);
    spec_t x = input[in_idx] / (1.0f + std::exp(-1 * input[in_idx]));
    spec_t y = input[in_idx + d];
    output[idx] = x * y;
  }
}

template <typename spec_t>
void swiglu_cpu(const spec_t* input, size_t size, size_t d, spec_t* output, int64_t ndims,
                const int64_t* stride, const int64_t* stride_out,
                const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    auto in_idx = (idx / d) * (2 * d) + (idx % d);
    int64_t i_x_idx = hetu::impl::get_index(in_idx, ndims, stride, c_shape);
    int64_t i_y_idx = hetu::impl::get_index(in_idx + d, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);

    spec_t x = input[i_x_idx] / (1.0f + std::exp(-1 * input[i_x_idx]));
    spec_t y = input[i_y_idx];
    output[o_idx] = x * y;
  }
}

void SwigluCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  size_t size = output->numel();
  size_t d_size = output->shape().back();
  if (size == 0)
    return;

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SwigluCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, output, size, d_size]() {
          if (input->is_contiguous() && output->is_contiguous()) {
            swiglu_cpu<spec_t>(input->data_ptr<spec_t>(), size, d_size,
                               output->data_ptr<spec_t>());
          } else {
            swiglu_cpu<spec_t>(input->data_ptr<spec_t>(), size, d_size,
                               output->data_ptr<spec_t>(), input->ndim(),
                               input->stride().data(), output->stride().data(),
                               output->shape().data());
          }
        },
        "SwiGLU");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void swiglu_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                         size_t size, size_t d, spec_t* input_grad) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    auto in_idx = (idx / d) * (2 * d) + (idx % d);
    spec_t x = input[in_idx];
    spec_t y = input[in_idx + d];

    spec_t sigmoid_x = 1 / (1.0f + exp(-1 * x));
    input_grad[in_idx + d] = output_grad[idx] * x * sigmoid_x;
    input_grad[in_idx] = output_grad[idx] * y * sigmoid_x *
      (1.0f + sigmoid_x * x * std::exp(-1 * x));
  }
}

template <typename spec_t>
void swiglu_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                       size_t size, size_t d, spec_t* input_grad, int64_t ndims,
                       const int64_t* stride, const int64_t* stride_out,
                       const int64_t* stride_in, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    auto in_idx = (idx / d) * (2 * d) + (idx % d);
    int64_t i_x_idx = hetu::impl::get_index(in_idx, ndims, stride, c_shape);
    int64_t i_y_idx = hetu::impl::get_index(in_idx + d, ndims, stride, c_shape);
    int64_t ig_x_idx = hetu::impl::get_index(in_idx, ndims, stride_in, c_shape);
    int64_t ig_y_idx =
      hetu::impl::get_index(in_idx + d, ndims, stride_out, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);

    spec_t x = input[i_x_idx];
    spec_t y = input[i_y_idx];

    spec_t sigmoid_x = 1 / (1.0f + exp(-1 * x));
    input_grad[ig_y_idx] = output_grad[o_idx] * x * sigmoid_x;
    input_grad[ig_x_idx] = output_grad[o_idx] * y * sigmoid_x *
      (1.0f + sigmoid_x * x * std::exp(-1 * x));
  }
}

void SwigluGradientCpu(const NDArray& input, const NDArray& output_grad,
                       NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);

  size_t size = output_grad->numel();
  size_t d_size = output_grad->shape().back();
  if (size == 0)
    return;

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SwigluGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, input_grad, output_grad, size, d_size]() {
          if (input->is_contiguous() && input_grad->is_contiguous() 
                  && output_grad->is_contiguous()) {
            swiglu_gradient_cpu<spec_t>(input->data_ptr<spec_t>(),
                                        output_grad->data_ptr<spec_t>(), size,
                                        d_size, input_grad->data_ptr<spec_t>());
          } else {
            swiglu_gradient_cpu<spec_t>(
              input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size,
              d_size, input_grad->data_ptr<spec_t>(), input->ndim(),
              input->stride().data(), output_grad->stride().data(),
              input_grad->stride().data(), input->shape().data());
          }
        },
        "SwiGLU");
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu