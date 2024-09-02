#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void mseloss_cpu(const spec_t* pred, const spec_t* label,
                 size_t n_rows, spec_t* loss) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    loss[idx] = (pred[idx] - label[idx]) * (pred[idx] - label[idx]);
  }
}

template <typename spec_t>
void mseloss_cpu(const spec_t* pred, const spec_t* label,
                 size_t n_rows, spec_t* loss,
                 int64_t ndims, const int64_t* stride_pred, const int64_t* stride_label,
                 const int64_t* stride_loss, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    int64_t pred_idx = hetu::impl::get_index(idx, ndims, stride_pred, c_shape);
    int64_t label_idx = hetu::impl::get_index(idx, ndims, stride_label, c_shape);
    int64_t loss_idx = hetu::impl::get_index(idx, ndims, stride_loss, c_shape);
    loss[loss_idx] = (pred[pred_idx] - label[label_idx]) * (pred[pred_idx] - label[label_idx]);
  }
}

template <typename spec_t>
void mseloss_gradient_cpu(const spec_t* pred, const spec_t* label,
                          const spec_t* grad_loss, size_t n_rows,
                          spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    output[idx] = 2 * grad_loss[idx] * (pred[idx] - label[idx]);
  }
}

template <typename spec_t>
void mseloss_gradient_cpu(const spec_t* pred, const spec_t* label,
                          const spec_t* grad_loss, size_t n_rows,
                          spec_t* output, int64_t ndims, const int64_t* stride_pred, 
                          const int64_t* stride_label, const int64_t* stride_loss, 
                          const int64_t* stride_out, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    int64_t pred_idx = hetu::impl::get_index(idx, ndims, stride_pred, c_shape);
    int64_t label_idx = hetu::impl::get_index(idx, ndims, stride_label, c_shape);
    int64_t loss_idx = hetu::impl::get_index(idx, ndims, stride_loss, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = 2 * grad_loss[loss_idx] * (pred[pred_idx] - label[label_idx]);
  }
}

void MSELossCpu(const NDArray& pred, const NDArray& label,
                           NDArray& loss, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, loss);

  CPUStream cpu_stream(stream);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "MSELossCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, loss, n_rows]() {
      if (pred->is_contiguous() && label->is_contiguous() && loss->is_contiguous()) {
        mseloss_cpu(pred->data_ptr<spec_t>(),
                    label->data_ptr<spec_t>(), n_rows,
                    loss->data_ptr<spec_t>());
      }
      else {
        mseloss_cpu(pred->data_ptr<spec_t>(),
                    label->data_ptr<spec_t>(), n_rows,
                    loss->data_ptr<spec_t>(), pred->ndim(),
                    pred->stride().data(), label->stride().data(),
                    loss->stride().data(), pred->shape().data());
      }
      },"MSELoss");
    });
   NDArray::MarkUsedBy({pred, label, loss}, stream);
}

void MSELossGradientCpu(const NDArray& pred, const NDArray& label,
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
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "MSELossGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, grad_loss, output, n_rows]() {
      if (pred->is_contiguous() && label->is_contiguous() && grad_loss->is_contiguous() && output->is_contiguous()) {
        mseloss_gradient_cpu(
          pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(),
          grad_loss->data_ptr<spec_t>(), n_rows, output->data_ptr<spec_t>());
      }
      else {
        mseloss_gradient_cpu(
          pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(),
          grad_loss->data_ptr<spec_t>(), n_rows, output->data_ptr<spec_t>(),
          pred->ndim(), pred->stride().data(), label->stride().data(), 
          grad_loss->stride().data(), output->stride().data(), pred->shape().data());        
      }
      },"MSELossGradient");
    });
   NDArray::MarkUsedBy({pred, label, grad_loss, output}, stream);
}

} // namespace impl
} // namespace hetu
