#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void broadcast_shape_cpu(const spec_t* input, spec_t* output,
                         const int64_t* out_strides, const int64_t* in_dims,
                         size_t ndims, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    size_t i_ind = 0;
    size_t temp = idx;
    for (size_t i = 0; i < ndims; ++i) {
      i_ind *= in_dims[i];
      i_ind += (in_dims[i] > 1) * temp / out_strides[i];
      temp %= out_strides[i];
    }
    output[idx] = input[i_ind];
  }
}

void BroadcastShapeCpu(const NDArray& input, NDArray& output,
                       const HTShape& add_axes, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  size_t size = output->numel();
  size_t input_size = input->numel();
  if (size == 0 || input_size == 0)
    return;

  int input_dim = input->ndim();
  int output_dim = output->ndim();
  HTStride out_strides(output_dim);
  HTShape in_dims(output_dim);
  int64_t output_size = 1;
  int diff = output_dim - input_dim;
  if (add_axes.empty()) {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = i < diff ? 1 : input->shape(i - diff);
    }
  } else {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = 0;
    }
    for (int i = 0; i < diff; ++i) {
      in_dims[add_axes[i]] = 1;
    }
    int o_ind = 0;
    for (int i = 0; i < input_dim; ++i) {
      while (in_dims[o_ind++] == 1) {}
      in_dims[o_ind - 1] = input->shape(i);
    }
  }

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastShapeCpu", [&]() {
      cpu_stream.EnqueueTask(
        [input, output, out_strides, in_dims, output_dim, size]() {
          broadcast_shape_cpu<spec_t>(
            input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
            out_strides.data(), in_dims.data(), output_dim, size);
        },
        "BroadcastShape");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void broadcast_shape_mul_cpu(const spec_t* input, spec_t const_value,
                             spec_t* output, const int64_t* out_strides,
                             const int64_t* in_dims, size_t ndims,
                             size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    size_t i_ind = 0;
    size_t temp = idx;
    for (size_t i = 0; i < ndims; ++i) {
      i_ind *= in_dims[i];
      i_ind += (in_dims[i] > 1) * temp / out_strides[i];
      temp %= out_strides[i];
    }
    output[idx] = input[i_ind] * const_value;
  }
}

void BroadcastShapeMulCpu(const NDArray& input, double const_value,
                          NDArray& output, const HTShape& add_axes,
                          const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  size_t size = output->numel();
  size_t input_size = input->numel();
  if (size == 0 || input_size == 0)
    return;

  int input_dim = input->ndim();
  int output_dim = output->ndim();
  HTStride out_strides(output_dim);
  HTShape in_dims(output_dim);
  int64_t output_size = 1;
  int diff = output_dim - input_dim;
  if (add_axes.empty()) {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = i < diff ? 1 : input->shape(i - diff);
    }
  } else {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = 0;
    }
    for (int i = 0; i < diff; ++i) {
      in_dims[add_axes[i]] = 1;
    }
    int o_ind = 0;
    for (int i = 0; i < input_dim; ++i) {
      while (in_dims[o_ind++] == 1) {}
      in_dims[o_ind - 1] = input->shape(i);
    }
  }

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastShapeMulCpu", [&]() {
      cpu_stream.EnqueueTask(
        [input, output, out_strides, const_value, in_dims, output_dim, size]() {
          broadcast_shape_mul_cpu<spec_t>(
            input->data_ptr<spec_t>(), static_cast<spec_t>(const_value),
            output->data_ptr<spec_t>(), out_strides.data(), in_dims.data(),
            output_dim, size);
        },
        "BroadcastShapeMul");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
