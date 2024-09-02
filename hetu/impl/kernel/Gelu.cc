#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

#define SQRT_1_2  0.70710678118654757274f
#define pi 3.14159265358979323846f
#define e  2.71828182845904523536f

namespace hetu {
namespace impl {

template <typename spec_t>
void gelu_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = input[idx] * 0.5f * (1.0f + std::erf(input[idx] * SQRT_1_2));
  }
}

template <typename spec_t>
void gelu_cpu(const spec_t* input, size_t size, spec_t* output,
              int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    output[i_idx] = input[i_idx] * 0.5f * (1.0f + std::erf(input[i_idx] * SQRT_1_2));
  }
}

template <typename spec_t>
void gelu_cpu(const spec_t* input, size_t size, spec_t* output,
              int64_t ndims, const int64_t* stride, const int64_t* stride_out,
              const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = input[i_idx] * 0.5f * (1.0f + std::erf(input[i_idx] * SQRT_1_2));
  }
}

template <typename spec_t>
void gelu_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                       size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = output_grad[idx]*(0.5f + 0.5f * std::erf(input[idx] / std::sqrt(2.0)) + 
                  0.5f * input[idx] * (std::sqrt(2.0f) * std::pow(e, (-0.5f * std::pow(input[idx] , 2.0f))) / 
                  std::sqrt(pi)));
  }
}

template <typename spec_t>
void gelu_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                       size_t size, spec_t* output,
                       int64_t ndims, const int64_t* stride, const int64_t* stride_out, const int64_t* stride_in,
                       const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t og_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    int64_t ig_idx = hetu::impl::get_index(idx, ndims, stride_in, c_shape);
    output[ig_idx] = output_grad[og_idx]*(0.5f + 0.5f * std::erf(input[i_idx] / std::sqrt(2.0)) + 
                  0.5f * input[i_idx] * (std::sqrt(2.0f) * std::pow(e, (-0.5f * std::pow(input[i_idx] , 2.0f))) / 
                  std::sqrt(pi)));
  }
}

void GeluCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "GeluCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
        if (input->is_contiguous() && output->is_contiguous()) {
          gelu_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                           output->data_ptr<spec_t>());
        }
        else {
          gelu_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                           output->data_ptr<spec_t>(), input->ndim(),
                           input->stride().data(), output->stride().data(),
                           input->shape().data());
        }
      },"Gelu");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void GeluGradientCpu(const NDArray& input, const NDArray& output_grad,
                     NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  CPUStream cpu_stream(stream);
  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "GeluGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output_grad, input_grad, size]() {
        if (input->is_contiguous() && output_grad->is_contiguous() && input_grad->is_contiguous()) {
          gelu_gradient_cpu<spec_t>(input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(),
                                    size, input_grad->data_ptr<spec_t>());
        }
        else {
          gelu_gradient_cpu<spec_t>(input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(),
                                    size, input_grad->data_ptr<spec_t>(), input->ndim(),
                                    input->stride().data(), output_grad->stride().data(),
                                    input_grad->stride().data(), input->shape().data());
        }
      },"Gelu");
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
