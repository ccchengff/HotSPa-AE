#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void nllloss_kernel(const spec_t* pred, const int64_t* label, 
                               size_t n_rows, size_t n_cols, spec_t* loss,
                               const OffsetCalculator* pred_offset_calculator,
                               const OffsetCalculator* label_offset_calculator,
                               const OffsetCalculator* loss_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  auto label_offset = label_offset_calculator->get(idx);
  int64_t id = label[label_offset];
  auto pred_offset = pred_offset_calculator->get(n_cols * idx + id);
  auto loss_offset = loss_offset_calculator->get(idx);
  spec_t zero = 0;
  loss[loss_offset] = (id < 0 || id >= n_cols) ? zero : - pred[pred_offset];
}

template <typename spec_t>
__global__ void
nllloss_gradient_kernel(const spec_t* pred, const int64_t* label, const spec_t* grad_loss,
                        size_t n_rows, size_t n_cols, spec_t* output,
                        const OffsetCalculator* label_offset_calculator,
                        const OffsetCalculator* grad_loss_offset_calculator,
                        const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  auto label_offset = label_offset_calculator->get(idx);
  int64_t id = label[label_offset];
  auto grad_loss_offset = grad_loss_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(n_cols * idx + id);
  spec_t zero = 0;
  output[out_offset] = (id < 0 || id >= n_cols) ? zero : - grad_loss[grad_loss_offset] * n_cols;
}

void NLLLossCuda(const NDArray& pred, const NDArray& label,
                 NDArray& loss, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_SHAPE(label, loss);
  HT_ASSERT(pred->ndim() == label->ndim() + 1);
  for (size_t i = 0; i < label->ndim(); i++)
    HT_ASSERT(pred->shape(i) == label->shape(i));

  size_t n_rows = 1, n_cols;
  for (size_t i = 0; i < label->ndim(); i++)
    n_rows *= label->shape(i);
  n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray pred_offset_calculator_arr, label_offset_calculator_arr,
          loss_offset_calculator_arr;
  OffsetCalculator *pred_offset_calculator, *label_offset_calculator,
                   *loss_offset_calculator;
  std::tie(pred_offset_calculator_arr, pred_offset_calculator) =
    AllocOffsetCalculator(pred, stream);
  std::tie(label_offset_calculator_arr, label_offset_calculator) =
    AllocOffsetCalculator(label, stream);
  std::tie(loss_offset_calculator_arr, loss_offset_calculator) = 
    AllocOffsetCalculator(loss, stream);
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "NLLLossCuda", [&]() {
      nllloss_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(), n_rows, n_cols,
        loss->data_ptr<spec_t>(), pred_offset_calculator, label_offset_calculator,
        loss_offset_calculator);
    });
  NDArray::MarkUsedBy({pred, label, loss, pred_offset_calculator_arr,
                      label_offset_calculator_arr, loss_offset_calculator_arr}, stream);
}

void NLLLossGradientCuda(const NDArray& pred, const NDArray& label,
                         const NDArray& grad_loss, NDArray& output,
                         const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, grad_loss);
  HT_ASSERT_SAME_DEVICE(pred, output);

  size_t n_rows = 1, n_cols;
  for (size_t i = 0; i < pred->ndim() - 1; ++i)
    n_rows *= pred->shape(i);
  n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray label_offset_calculator_arr, grad_loss_offset_calculator_arr,
          out_offset_calculator_arr;
  OffsetCalculator *label_offset_calculator, *grad_loss_offset_calculator,
                   *out_offset_calculator;
  std::tie(label_offset_calculator_arr, label_offset_calculator) =
    AllocOffsetCalculator(label, stream);
  std::tie(grad_loss_offset_calculator_arr, grad_loss_offset_calculator) =
    AllocOffsetCalculator(grad_loss, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "NLLLossGradientCuda", [&]() {
      NDArray::zeros_(output, stream.stream_index());
      threads.x = MIN(n_rows, 1024);
      blocks.x = DIVUP(n_rows, 1024);
      nllloss_gradient_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, n_cols, output->data_ptr<spec_t>(),
        label_offset_calculator, grad_loss_offset_calculator,
        out_offset_calculator);
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output, label_offset_calculator_arr,
                      grad_loss_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
