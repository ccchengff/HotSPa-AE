#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void sin_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::sin(input[idx]);
  }
}

template <typename spec_t>
void sin_cpu(const spec_t* input, size_t size, spec_t* output,
             int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    output[i_idx] = std::sin(input[i_idx]);
  }
}

template <typename spec_t>
void sin_cpu(const spec_t* input, size_t size, spec_t* output,
             int64_t ndims, const int64_t* stride, const int64_t* stride_out,
             const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = std::sin(input[i_idx]);
  }
}

void SinCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SinCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      if (input->is_contiguous() && output->is_contiguous()) {
        sin_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                        output->data_ptr<spec_t>());
      }
      else {
        sin_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                        output->data_ptr<spec_t>(), input->ndim(),
                        input->stride().data(), output->stride().data(),
                        input->shape().data());
      }
      }, "Sin");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void cos_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::cos(input[idx]);
  }
}

template <typename spec_t>
void cos_cpu(const spec_t* input, size_t size, spec_t* output,
             int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    output[i_idx] = std::cos(input[i_idx]);
  }
}

template <typename spec_t>
void cos_cpu(const spec_t* input, size_t size, spec_t* output,
             int64_t ndims, const int64_t* stride, const int64_t* stride_out,
             const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = std::cos(input[i_idx]);
  }
}

void CosCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "CosCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      if (input->is_contiguous() && output->is_contiguous()) {
        cos_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                        output->data_ptr<spec_t>());
      }
      else {
        cos_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                        output->data_ptr<spec_t>(), input->ndim(),
                        input->stride().data(), output->stride().data(),
                        input->shape().data());
      }
      }, "Cos");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void sin_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                      size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = output_grad[idx] * std::cos(input[idx]);
  }
}

template <typename spec_t>
void sin_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                      size_t size, spec_t* output, 
                      int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                      const int64_t* stride_in, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t og_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    int64_t ig_idx = hetu::impl::get_index(idx, ndims, stride_in, c_shape);
    output[ig_idx] = output_grad[og_idx] * std::cos(input[i_idx]);
  }
}

template <typename spec_t>
void cos_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                      size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = -output_grad[idx] * std::sin(input[idx]);
  }
}

template <typename spec_t>
void cos_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                      size_t size, spec_t* output, 
                      int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                      const int64_t* stride_in, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t og_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    int64_t ig_idx = hetu::impl::get_index(idx, ndims, stride_in, c_shape);
    output[ig_idx] = -output_grad[og_idx] * std::sin(input[i_idx]);
  }
}

void SinGradientCpu(const NDArray& input, const NDArray& output_grad,
                    NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SinGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output_grad, input_grad, size]() {
      if (input->is_contiguous() && output_grad->is_contiguous() && input_grad->is_contiguous()) {
        sin_gradient_cpu<spec_t>(
          input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size,
          input_grad->data_ptr<spec_t>());
      }
      else {
        sin_gradient_cpu<spec_t>(
          input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size,
          input_grad->data_ptr<spec_t>(), input->ndim(),
          input->stride().data(), output_grad->stride().data(),
          input_grad->stride().data(), input->shape().data());
      }
      }, "SinGradient");
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

void CosGradientCpu(const NDArray& input, const NDArray& output_grad,
                    NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "CosGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output_grad, input_grad, size]() {
      if (input->is_contiguous() && output_grad->is_contiguous() && input_grad->is_contiguous()) {
        cos_gradient_cpu<spec_t>(
          input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size,
          input_grad->data_ptr<spec_t>());
      }
      else {
        cos_gradient_cpu<spec_t>(
          input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size,
          input_grad->data_ptr<spec_t>(), input->ndim(),
          input->stride().data(), output_grad->stride().data(),
          input_grad->stride().data(), input->shape().data());
      }
      }, "CosGradient");
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
