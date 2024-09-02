#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void roll_cpu(const spec_t *input, spec_t *output, size_t size, int rank,
              const int64_t *shifts, const int64_t *strides, const int64_t *sizes) {
  for (size_t idx = 0; idx < size; ++idx) {

    int output_idx = idx;
    int new_dim_idx = 0;

  #pragma unroll
    for (int i = 0; i < rank; i++) {
      new_dim_idx = (idx / strides[i]) % sizes[i] + shifts[i];
      if (new_dim_idx >= sizes[i])
        output_idx += (shifts[i] - sizes[i]) * strides[i];
      else
        output_idx += shifts[i] * strides[i];
    }
    output[output_idx] = input[idx];
  }
}


void RollCpu(const NDArray& input, const HTShape& shift, const HTAxes& axis,
             NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  size_t len = input->numel();
  int64_t nums = shift.size();
  int64_t n_dims = input->ndim();

  HTAxes stride_dim(n_dims);
  stride_dim[n_dims - 1] = 1;
  for (int i = 0; i < n_dims; i++) {
    if (i > 0)
      stride_dim[n_dims - i - 1] =
        input->shape(n_dims - i) * stride_dim[n_dims - i];
  }

  HTStride strides(n_dims);
  HTShape sizes(n_dims);
  HTShape shifts(n_dims);

  if (axis.size() == 0) {
    strides[0] = 1;
    sizes[0] = len;
    shifts[0] = (shift[0] % len + len) % len;
  } else {
    for (int i = 0; i < nums; i++) {
      int dim = axis[i] >= 0 ? axis[i] : axis[i] + n_dims;
      int size = input->shape(dim);
      if (size != 0) {
        strides[i] = stride_dim[dim];
        sizes[i] = size;
        shifts[i] = (shift[i] % size + size) % size;
      }
    }
  }

  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RollCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, output, len, nums, shifts, strides, sizes]() {
        roll_cpu<spec_t>(
          input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
          len, nums, shifts.data(), strides.data(), sizes.data());
        },"Roll"); 
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
