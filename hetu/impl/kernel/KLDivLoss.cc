#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t> 
void kldivloss_cpu(const spec_t* pred,
                   const spec_t* label, size_t n_rows,
                   spec_t* loss) {
  for (size_t idx = 0; idx < n_rows; ++idx) {
    spec_t lglabel = std::log(label[idx]);
    // clip to -100 following PyTorch
    spec_t min_value = -100;
    loss[idx] = label[idx] * (MAX(lglabel, min_value) - pred[idx]); 
  }
}

template <typename spec_t> 
void kldivloss_cpu(const spec_t* pred,
                   const spec_t* label, size_t n_rows,
                   spec_t* loss, int64_t ndims, const int64_t* stride_pred, 
                   const int64_t* stride_label,
                   const int64_t* stride_loss, const int64_t* c_shape) {
  for (size_t idx = 0; idx < n_rows; ++idx) {
    int64_t pred_idx = hetu::impl::get_index(idx, ndims, stride_pred, c_shape);
    int64_t label_idx = hetu::impl::get_index(idx, ndims, stride_label, c_shape);
    int64_t loss_idx = hetu::impl::get_index(idx, ndims, stride_loss, c_shape);
    spec_t lglabel = std::log(label[label_idx]);
    // clip to -100 following PyTorch
    spec_t min_value = -100;
    loss[loss_idx] = label[label_idx] * (MAX(lglabel, min_value) - pred[pred_idx]); 
  }
}

template <typename spec_t>
void kldivloss_gradient_cpu(const spec_t* pred, const spec_t* label,
                            const spec_t* grad_loss, size_t n_rows,
                            spec_t* output) {
  for (size_t idx = 0; idx < n_rows; ++idx) {
    output[idx] = - grad_loss[idx] * label[idx];
  }
}

template <typename spec_t>
void kldivloss_gradient_cpu(const spec_t* pred, const spec_t* label,
                            const spec_t* grad_loss, size_t n_rows,
                            spec_t* output, int64_t ndims, const int64_t* stride_pred, 
                            const int64_t* stride_label, const int64_t* stride_loss, 
                            const int64_t* stride_out, const int64_t* c_shape) {
  for (size_t idx = 0; idx < n_rows; ++idx) {
    int64_t label_idx = hetu::impl::get_index(idx, ndims, stride_label, c_shape);
    int64_t loss_idx = hetu::impl::get_index(idx, ndims, stride_loss, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = - grad_loss[loss_idx] * label[label_idx];
  }
}

void KLDivLossCpu(const NDArray& pred, const NDArray& label,
                            NDArray& loss, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, loss);

  CPUStream cpu_stream(stream);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "KLDivLossCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, loss, n_rows]() {
      if (pred->is_contiguous() && label->is_contiguous() && loss->is_contiguous()) {
        kldivloss_cpu<spec_t>(
          pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(), n_rows,
          loss->data_ptr<spec_t>());
      }
      else {
        kldivloss_cpu<spec_t>(
          pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(), n_rows,
          loss->data_ptr<spec_t>(), pred->ndim(),
          pred->stride().data(), label->stride().data(),
          loss->stride().data(), pred->shape().data());
      }
      },"KLDivLoss");
    });
  NDArray::MarkUsedBy({pred, label, loss}, stream);
}

void KLDivLossGradientCpu(const NDArray& pred, const NDArray& label,
                          const NDArray& grad_loss, NDArray& output,
                          const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, grad_loss);
  HT_ASSERT_SAME_DEVICE(pred, output);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, grad_loss);
  HT_ASSERT_SAME_NDIM(pred, output);

  CPUStream cpu_stream(stream);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "KLDivLossGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, grad_loss, output, n_rows]() {
      if (pred->is_contiguous() && label->is_contiguous() && grad_loss->is_contiguous()) {
        kldivloss_gradient_cpu<spec_t>(
          pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(),
          grad_loss->data_ptr<spec_t>(), n_rows, output->data_ptr<spec_t>());
      }
      else {
        kldivloss_gradient_cpu<spec_t>(
          pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(),
          grad_loss->data_ptr<spec_t>(), n_rows, output->data_ptr<spec_t>(), pred->ndim(),
          pred->stride().data(), label->stride().data(), grad_loss->stride().data(),
          output->stride().data(), pred->shape().data());        
      }
      },"KLDivLossGradient");
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output}, stream);
}

} // namespace impl
} // namespace hetu
