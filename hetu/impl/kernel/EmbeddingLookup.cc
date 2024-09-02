#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void embedding_lookup_cpu(const spec_t* input, const int64_t* ids, size_t size,
                          size_t length, size_t input_row, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int64_t id = ids[idx];
    spec_t* output_ptr = output + length * idx;
    if (id < 0 || id >= (int) input_row) {
      for (size_t i = 0; i < length; i++)
        output_ptr[i] = 0;
    } else {
      const spec_t* input_ptr = input + length * id;
      for (size_t i = 0; i < length; i++)
        output_ptr[i] = input_ptr[i];
    }
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
void embedding_lookup_gradient_cpu(const spec_t* output_grad, const int64_t* ids,
                                   size_t size, size_t length,
                                   spec_t* input_grad) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size / length; ++idx) {
    int id = int(ids[idx]);
    for (size_t i = 0; i < length; i++) {
      input_grad[length * id + i] +=  output_grad[length * idx + i];
    }
  }
}

void EmbeddingLookupCpu(const NDArray& input, const NDArray& id,
                        NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, id);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->ndim() == 2)
    << "input_dim is invalid.Expect 2,but get " << input->ndim();
  
  CPUStream cpu_stream(stream);

  for (size_t i = 0; i < output->ndim(); i++) {
    if (i + 1 < output->ndim()) {
      HT_ASSERT(id->shape(i) == output->shape(i));
    } else if (i + 1 == output->ndim()) {
      HT_ASSERT(input->shape(1) == output->shape(i));
    }
  }
  size_t input_row = input->shape(0);
  size_t length = input->shape(1);
  size_t size = id->numel();
  if (size == 0 || input_row == 0 || length == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "EmbbedingLookupCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, id, output, size, length, input_row]() {
      embedding_lookup_cpu(input->data_ptr<spec_t>(), id->data_ptr<int64_t>(),
                           size, length, input_row, output->data_ptr<spec_t>());
      },
      "EmbbedingLookup");
    });
  NDArray::MarkUsedBy({input, id, output}, stream);
}

void EmbeddingLookupGradientCpu(const NDArray& output_grad, const NDArray& id,
                                NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, id);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);
  HT_ASSERT(input_grad->ndim() == 2)
    << "input_dim is invalid.Expect 2,but get " << input_grad->ndim();

  CPUStream cpu_stream(stream);

  for (size_t i = 0; i < output_grad->ndim(); i++) {
    if (i < output_grad->ndim() - 1) {
      HT_ASSERT(id->shape(i) == output_grad->shape(i));
    } else if (i == output_grad->ndim() - 1) {
      HT_ASSERT(input_grad->shape(1) == output_grad->shape(i));
    }
  }
  size_t length = input_grad->shape(1);
  size_t size = input_grad->numel();
  if (size == 0 || length == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "EmbeddingLookupGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input_grad, output_grad, id, size, length]() {
      array_zero_set_cpu(input_grad->data_ptr<spec_t>(), size);
      embedding_lookup_gradient_cpu(output_grad->data_ptr<spec_t>(),
                                    id->data_ptr<int64_t>(), output_grad->numel(), length,
                                    input_grad->data_ptr<spec_t>());
      },
      "EmbbedingLookupGradient");  
    });
  NDArray::MarkUsedBy({output_grad, id, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
