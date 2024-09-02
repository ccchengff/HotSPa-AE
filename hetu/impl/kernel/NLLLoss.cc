#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void nllloss_cpu(const spec_t* pred, const int64_t* label, 
                    size_t n_rows, size_t n_cols, spec_t* loss) {
  for (size_t idx = 0; idx < n_rows; ++idx) {
    int64_t id = label[idx];
    if (id < 0 || id >= int64_t(n_cols)) {
      loss[idx] = 0;
    } else {
      loss[idx] = - pred[n_cols * idx + id];
    }
  }
}

template <typename spec_t>
void nllloss_gradient_cpu(const spec_t* pred, const int64_t* label,
                          const spec_t* grad_loss, size_t n_rows, size_t n_cols,
                          spec_t* output) {
  for (size_t idx = 0; idx < n_rows; ++idx) {
    int64_t id = label[idx];
    for (size_t i = 0; i < n_cols; ++i) {
      output[n_cols * idx + i] = 0; 
    }
    if (id < 0 || id >= int64_t(n_cols)) {
      output[n_cols * idx + id] = 0;
    } else {
      output[n_cols * idx + id] = - grad_loss[idx] * n_cols;
    }
  }
}


void NLLLossCpu(const NDArray& pred, const NDArray& label,
                NDArray& loss, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_SHAPE(label, loss);
  HT_ASSERT(pred->ndim() == label->ndim() + 1);

  CPUStream cpu_stream(stream);

  for (size_t i = 0; i < label->ndim(); i++)
    HT_ASSERT(pred->shape(i) == label->shape(i));

  size_t n_rows = 1, n_cols;
  for (size_t i = 0; i < label->ndim(); i++)
    n_rows *= label->shape(i);
  n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;

  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "NLLLossCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, loss, n_rows, n_cols]() {
      nllloss_cpu(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(), n_rows, n_cols,
        loss->data_ptr<spec_t>());
      },"NLLLoss");
    });
  NDArray::MarkUsedBy({pred, label, loss}, stream);
}

template <typename spec_t>
void array_zero_set_cpu(spec_t* input, size_t size) {
  for (size_t idx = 0; idx < size; ++idx)
    input[idx] = 0;
}

void NLLLossGradientCpu(const NDArray& pred, const NDArray& label,
                        const NDArray& grad_loss, NDArray& output,
                        const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, grad_loss);
  HT_ASSERT_SAME_DEVICE(pred, output);

  CPUStream cpu_stream(stream);

  size_t n_rows = 1, n_cols;
  for (size_t i = 0; i < pred->ndim() - 1; ++i)
    n_rows *= pred->shape(i);
  n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "NLLLossGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, grad_loss, output, n_rows, n_cols]() {
      array_zero_set_cpu(output->data_ptr<spec_t>(), output->numel());
      nllloss_gradient_cpu(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, n_cols, output->data_ptr<spec_t>());
      },"NLLLossGradient");    
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output}, stream);
}

} // namespace impl
} // namespace hetu
