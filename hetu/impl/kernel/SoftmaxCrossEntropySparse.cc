#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void softmax_cross_entropy_sparse_cpu(const spec_t* pred,
                                      const int64_t* label, 
                                      size_t n_rows, size_t n_cols,
                                      const int64_t ignored_index,
                                      spec_t* loss) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    if(int64_t(label[idx])==ignored_index) {
      loss[idx] = 0;
      return;
    }  
    spec_t maxval = pred[idx * n_cols];
    for (size_t i = 1; i < n_cols; ++i) {
      maxval = MAX(maxval, pred[idx * n_cols + i]);
    }

    spec_t sum = 0;
    for (size_t i = 0; i < n_cols; ++i) {
      sum += std::exp(pred[idx * n_cols + i] - maxval);
    }

    size_t curid = idx * n_cols + int64_t(label[idx]);
    loss[idx] = -(pred[curid] - maxval) + std::log(sum);
  }
}

template <typename spec_t>
void softmax_cross_entropy_sparse_gradient_cpu(const spec_t* pred, const int64_t* label,
                                               const spec_t* grad_loss, size_t n_rows, size_t n_cols,
                                               const int64_t ignored_index,
                                               spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    if(int64_t(label[idx]) == ignored_index) {
      for (size_t i = 0; i < n_cols; ++i) {
          size_t curid = idx * n_cols + i;
          output[curid] = 0;
      }
      return;        
    }
    
    spec_t maxval = pred[idx * n_cols];

    for (size_t i = 1; i < n_cols; ++i) {
        maxval = MAX(maxval, pred[idx * n_cols + i]);
    }

    spec_t sum = 0;
    for (size_t i = 0; i < n_cols; ++i) {
        sum += std::exp(pred[idx * n_cols + i] - maxval);
    }
    for (size_t i = 0; i < n_cols; ++i) {
        size_t curid = idx * n_cols + i;
        if(int64_t(i) == int64_t(label[idx]))
          output[curid] = (std::exp(pred[curid] - maxval) / sum - 1.0) * grad_loss[idx];
        else
          output[curid] = (std::exp(pred[curid] - maxval) / sum) * grad_loss[idx];
    }
  }
}

void SoftmaxCrossEntropySparseCpu(const NDArray& pred, const NDArray& label,
                                  NDArray& loss, const int64_t ignored_index, 
                                  const Stream& stream) {

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  size_t n_cols = pred->shape(pred->ndim() - 1);

  CPUStream cpu_stream(stream);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "SoftmaxCrossEntropySparseCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [pred, label, loss, n_rows, n_cols, ignored_index]() {
        softmax_cross_entropy_sparse_cpu(
          pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(), n_rows, n_cols,
          ignored_index, loss->data_ptr<spec_t>());
        },"SoftmaxCrossEntropySparse");
    });
  NDArray::MarkUsedBy({pred, label, loss}, stream);
}

void SoftmaxCrossEntropySparseGradientCpu(const NDArray& pred, const NDArray& label,
                                          const NDArray& grad_loss, NDArray& output,
                                          const int64_t ignored_index,
                                          const Stream& stream) {

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  size_t n_cols = pred->shape(pred->ndim() - 1);

  CPUStream cpu_stream(stream);

  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "SoftmaxCrossEntropySparseGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [pred, label, grad_loss, output, n_rows, n_cols, ignored_index]() {
        softmax_cross_entropy_sparse_gradient_cpu(
          pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(),
          grad_loss->data_ptr<spec_t>(), n_rows, n_cols,
          ignored_index, output->data_ptr<spec_t>());
        },"SoftmaxCrossEntropySparseGradient");
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output}, stream);
}

} // namespace impl
} // namespace hetu
