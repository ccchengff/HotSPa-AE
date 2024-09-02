#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void diagonal_cpu(const spec_t* input, size_t size, int strideA, int strideB,
                  int strideC, int dim1_len, int dim2_len, int dim_len,
                  int offset, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int diag_idx = idx % dim_len;
    int ori_idx = idx / dim_len;
    int indexC = ori_idx % strideC;
    ori_idx /= strideC;
    int indexB = ori_idx % strideB;
    ori_idx /= strideB;
    int indexA = ori_idx % strideA;
    int input_idx = indexA;
    input_idx = input_idx * dim1_len + diag_idx;
    input_idx = input_idx * strideB + indexB;
    input_idx = input_idx * dim2_len + diag_idx + offset;
    input_idx = input_idx * strideC + indexC;
    output[idx] = input[input_idx];
  }
}

void DiagonalCpu(const NDArray& input, NDArray& output, int dim1, int dim2,
                 int offset, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  if (dim1 > dim2) {
    int tmp = dim1;
    dim1 = dim2;
    dim2 = tmp;
  }
  int strideA = 1;
  int strideB = 1;
  int strideC = 1;
  for (int i = 0; i < dim1; ++i) {
    strideA *= input->shape(i);
  }
  for (int i = dim1 + 1; i < dim2; ++i) {
    strideB *= input->shape(i);
  }
  for (size_t i = dim2 + 1; i < input->ndim(); ++i) {
    strideC *= input->shape(i);
  }
  int dim_len = std::min(input->shape(dim1), input->shape(dim2));
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "DiagonalCpu", [&]() {
    auto _binary_future = cpu_stream.EnqueueTask(
      [input, output, size, strideA, strideB,
        strideC, dim1, dim2, dim_len, offset]() {
        diagonal_cpu<spec_t>(input->data_ptr<spec_t>(), size, strideA, strideB,
                             strideC, static_cast<int>(input->shape(dim1)),
                             static_cast<int>(input->shape(dim2)), dim_len,
                             offset, output->data_ptr<spec_t>());}, "Diagonal");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void diagonal_gradient_cpu(const spec_t* input, size_t size, int strideA,
                           int strideB, int strideC, int dim_len,
                           spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int ori_idx = idx;
    int indexC = ori_idx % strideC;
    ori_idx /= strideC;
    int index_dim2 = ori_idx % dim_len;
    ori_idx /= dim_len;
    int indexB = ori_idx % strideB;
    ori_idx /= strideB;
    int index_dim1 = ori_idx % dim_len;
    ori_idx /= dim_len;
    int indexA = ori_idx % strideA;
    output[idx] = 0;
    if (index_dim1 == index_dim2) {
      int input_idx = indexA;
      input_idx = input_idx * dim_len + index_dim1;
      input_idx = input_idx * strideB + indexB;
      input_idx = input_idx * strideC + indexC;
      output[idx] = input[input_idx];
    }
  }
}

void DiagonalGradientCpu(const NDArray& input, NDArray& output, int dim1,
                         int dim2, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  if (dim1 > dim2) {
    int tmp = dim1;
    dim1 = dim2;
    dim2 = tmp;
  }
  int strideA = 1;
  int strideB = 1;
  int strideC = 1;
  for (int i = 0; i < dim1; ++i) {
    strideA *= input->shape(i);
  }
  for (int i = dim1 + 1; i < dim2; ++i) {
    strideB *= input->shape(i);
  }
  for (size_t i = dim2; i < input->ndim(); ++i) {
    strideC *= input->shape(i);
  }
  int dim_len = input->shape(dim1);
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "DiagonalGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size, strideA, strideB, strideC, dim_len]() {
              diagonal_gradient_cpu<spec_t>(input->data_ptr<spec_t>(), size, strideA,
                                    strideB, strideC, dim_len,
                                    output->data_ptr<spec_t>());
      }, "DiagonalGradient");          
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
