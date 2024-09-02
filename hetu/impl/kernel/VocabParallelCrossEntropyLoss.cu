#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

// vocab_parallel_logits: [batch_size * seq_len, vocab_size], splited by tp in vocab_size dimension
// labels: [batch_size, seq_len], duplicate
// predicted_logits_partial: [batch_size * seq_len]
// log_sum_exp_logits: [batch_size * seq_len]
template <typename spec_t>
__global__ void vocab_parallel_cross_entropy_kernel(const spec_t* vocab_parallel_logits, 
                                                    const int64_t* labels, 
                                                    size_t n_rows, size_t n_cols,
                                                    const int64_t vocab_start_index, 
                                                    const int64_t vocab_end_index,
                                                    const int64_t ignored_index,
                                                    spec_t* predicted_logits_partial,
                                                    spec_t* log_sum_exp_logits) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  if (labels[idx] == ignored_index) {
    predicted_logits_partial[idx] = 0;
    log_sum_exp_logits[idx] = 0;
    return;
  }
  if (labels[idx] < vocab_start_index || labels[idx] >= vocab_end_index) {
    predicted_logits_partial[idx] = 0;
  } else {
    predicted_logits_partial[idx] = vocab_parallel_logits[idx * n_cols + labels[idx] - vocab_start_index];
  }
}

// vocab_parallel_logits: [batch_size * seq_len, vocab_size], splited by tp in vocab_size dimension
// labels: [batch_size, seq_len], duplicate
// predicted_logits_partial: [batch_size * seq_len]
// log_sum_exp_logits: [batch_size * seq_len]
void VocabParallelCrossEntropyCuda(const NDArray& vocab_parallel_logits, const NDArray& labels,
                                   const int64_t vocab_start_index, const int64_t vocab_end_index,
                                   const int64_t ignored_index, NDArray& predicted_logits_partial, 
                                   NDArray& log_sum_exp_logits, const Stream& stream) {
  size_t n_rows = vocab_parallel_logits->shape(0); // batch_size * seq_len
  size_t n_cols = vocab_parallel_logits->shape(1); // vocab_size
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  threads.x = MIN(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_FLOATING_TYPES(
    vocab_parallel_logits->dtype(), spec_t, "VocabParallelCrossEntropyCuda", [&]() {
      vocab_parallel_cross_entropy_kernel<<<blocks, threads, 0, cuda_stream>>>(
        vocab_parallel_logits->data_ptr<spec_t>(), labels->data_ptr<int64_t>(), 
        n_rows, n_cols, vocab_start_index, vocab_end_index, ignored_index,
        predicted_logits_partial->data_ptr<spec_t>(), log_sum_exp_logits->data_ptr<spec_t>());
    });
  NDArray::MarkUsedBy({vocab_parallel_logits, labels, predicted_logits_partial, log_sum_exp_logits}, stream);
}


// softmax: [batch_size * seq_len, vocab_size], splited by tp in vocab_size dimension
// labels: [batch_size, seq_len], duplicate
// grad = (softmax(prediction) - labels) / N
template <typename spec_t>
__global__ void vocab_prarallel_cross_entropy_gradient_kernel(const spec_t* softmax, 
                                                              const int64_t* labels,
                                                              size_t n_rows, size_t n_cols,
                                                              const int64_t vocab_start_index,
                                                              const int64_t vocab_end_index,
                                                              const int64_t ignored_index,
                                                              const spec_t* grad_loss, spec_t* output) {
  auto idx = blockIdx.x;
  if (idx >= n_rows)
    return;
  size_t vocab_size_per_thread = DIVUP(n_cols, blockDim.x);
  size_t start_col_idx = vocab_size_per_thread * threadIdx.x;
  size_t end_col_idx = start_col_idx + vocab_size_per_thread > n_cols ? 
                   n_cols : start_col_idx + vocab_size_per_thread;
  size_t predict_idx = labels[idx] - vocab_start_index;
  for (size_t col_idx = start_col_idx; col_idx < end_col_idx; col_idx++) {
    if (col_idx == predict_idx) {
      output[idx * n_cols + col_idx] = (softmax[idx * n_cols + col_idx] - 1.0) * grad_loss[idx];
    } else {
      output[idx * n_cols + col_idx] = softmax[idx * n_cols + col_idx] * grad_loss[idx];
    }
  }
}

// softmax: [batch_size * seq_len, vocab_size], splited by tp in vocab_size dimension
// labels: [batch_size, seq_len], duplicate
void VocabParallelCrossEntropyGradientCuda(const NDArray& softmax, 
                                           const NDArray& labels,
                                           const int64_t vocab_start_index, 
                                           const int64_t vocab_end_index,
                                           const int64_t ignored_index,
                                           const NDArray& grad_loss, 
                                           NDArray& output, const Stream& stream) {
  size_t n_rows = softmax->shape(0); // batch_size * seq_len
  size_t n_cols = softmax->shape(1); // vocab_size
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  threads.x = MIN(MAX(32, n_cols), HT_DEFAULT_NUM_THREADS_PER_BLOCK); // each thread handle vocab_size/1024 columns
  blocks.x = n_rows; // each block handle one row
  HT_DISPATCH_FLOATING_TYPES(
    softmax->dtype(), spec_t, "VocabParallelCrossEntropyGradientCuda", [&]() {
      vocab_prarallel_cross_entropy_gradient_kernel<<<blocks, threads, 0, cuda_stream>>>(
        softmax->data_ptr<spec_t>(), labels->data_ptr<int64_t>(), 
        n_rows, n_cols, vocab_start_index, vocab_end_index, ignored_index,
        grad_loss->data_ptr<spec_t>(), output->data_ptr<spec_t>());
    });
  NDArray::MarkUsedBy({softmax, labels, grad_loss, output}, stream);
}

} // namespace impl
} // namespace hetu