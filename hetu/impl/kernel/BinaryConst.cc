#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t, typename Operator>
void binary_const_cpu(const spec_t* input, spec_t value, size_t size,
                      Operator op, spec_t* output) {
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = op(value, input[idx]);
}

template <typename spec_t>
void add_const_cpu(const spec_t* input, spec_t value, size_t size,
                   spec_t* output) {
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = value + input[idx];
}

template <typename spec_t>
void sub_const_cpu(const spec_t* input, spec_t value, size_t size,
                   spec_t* output) {
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = input[idx] - value;
}

template <typename spec_t>
void mul_const_cpu(const spec_t* input, spec_t value, size_t size,
                   spec_t* output) {
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = value * input[idx];
}

template <typename spec_t>
void div_const_cpu(const spec_t* input, spec_t value, size_t size,
                   spec_t* output) {
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = input[idx] / value;
}

template<typename Operator>
void BinaryConstToolCpu(const NDArray& input, double value,
                        NDArray& output, Operator op, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);

  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BinaryConstCpu", [&]() {
      binary_const_cpu<spec_t>(
        input->data_ptr<spec_t>(), static_cast<spec_t>(value), size, op,
        output->data_ptr<spec_t>());
    });
}

void AddConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  CPUStream cpu_stream(stream);
  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "AddConstCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [stream, input, output, value, size]() {
        add_const_cpu<spec_t>(
          input->data_ptr<spec_t>(), static_cast<spec_t>(value), size, 
          output->data_ptr<spec_t>());
    },
    "AddConst");
  });
  NDArray::MarkUsedBy({input, output}, stream);
}

void SubConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  CPUStream cpu_stream(stream);
  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "SubConstCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [stream, input, output, value, size]() {
        sub_const_cpu<spec_t>(
          input->data_ptr<spec_t>(), static_cast<spec_t>(value), size, 
          output->data_ptr<spec_t>());
    },
    "SubConst");
  });
  NDArray::MarkUsedBy({input, output}, stream);
}

void MulConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  CPUStream cpu_stream(stream);
  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "MulConstCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [stream, input, output, value, size]() {
        mul_const_cpu<spec_t>(
          input->data_ptr<spec_t>(), static_cast<spec_t>(value), size, 
          output->data_ptr<spec_t>());
    },
    "MulConst");
  });
  NDArray::MarkUsedBy({input, output}, stream);
}

void DivConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  CPUStream cpu_stream(stream);
  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "DivConstCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [stream, input, output, value, size]() {
        div_const_cpu<spec_t>(
          input->data_ptr<spec_t>(), static_cast<spec_t>(value), size, 
          output->data_ptr<spec_t>());
    },
    "DivConst");
  });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
