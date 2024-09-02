#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include <chrono>

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void conv2d_naive_kernel(const spec_t* input, spec_t* output, const spec_t* weight,
                              const spec_t* bias, bool use_bias, size_t size,
                              const int inputChannels, const int outputChannels, 
                              const int inputWidth, const int inputHeight,
                              const int outputWidth, const int outputHeight,
                              const int kernelWidth, const int kernelHeight,
                              const int strideWidth, const int strideHeight,
                              const int padWidth, const int padHeight) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if (idx >= size)
    return;

  size_t tmp = idx;
  const int w = tmp % outputWidth;
  tmp = tmp / outputWidth;
  const int h = tmp % outputHeight;
  tmp = tmp / outputHeight;
  const int c = tmp % outputChannels;
  tmp = tmp / outputChannels;
  const int n = tmp;
  

  spec_t value = use_bias ? static_cast<spec_t>(bias[c]) : spec_t(0);
#if defined(_OPENMP)
#pragma unroll
#endif
  for (int kC = 0; kC < inputChannels; ++kC) {
    size_t offset0 = (n * inputChannels + kC) * inputHeight * inputWidth;
    int weightOffset = ((c * inputChannels + kC % inputChannels) * kernelHeight * kernelWidth);
#if defined(_OPENMP)
#pragma unroll
#endif
    for (int kH = 0; kH < kernelHeight; ++kH) {
#if defined(_OPENMP)
#pragma unroll
#endif
      for (int kW = 0; kW < kernelWidth; ++kW) {
        const int h_in = -padHeight + h * strideHeight + kH;
        const int w_in = -padWidth + w * strideWidth + kW;

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth)) {
          const size_t offset = offset0 + h_in * inputWidth + w_in;
          value += (static_cast<spec_t>(weight[weightOffset]) *
                    static_cast<spec_t>(input[offset]));
        }
        ++weightOffset;
      }
    }
  }
  output[idx] = value;
}

template <typename spec_t>
__global__ void conv2d_gradient_data_naive_kernel(const spec_t* grad_output, spec_t* grad_input, const spec_t* weight, 
                                                  size_t size, const int inputChannels, const int outputChannels,
                                                  const int inputWidth, const int inputHeight,
                                                  const int outputWidth, const int outputHeight,
                                                  const int kernelWidth, const int kernelHeight,
                                                  const int strideWidth, const int strideHeight,
                                                  const int padWidth, const int padHeight) {

  auto idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if (idx >= size)
    return;
  int indtmp1 = idx / inputWidth;
  const int w = idx % inputWidth;
  int indtmp2 = indtmp1 / inputHeight;
  const int h = indtmp1 % inputHeight;
  indtmp1 = indtmp2;
  const int n = indtmp1 / inputChannels;
  const int c = indtmp1 % inputChannels;

  spec_t value = 0;

#if defined(__OPENMP)
#pragma unroll
#endif
  for (int kc = 0; kc < outputChannels; ++kc) {
    int weightOffset = (kc * inputChannels + c) * kernelHeight * kernelWidth;
#if defined(__OPENMP)
#pragma unroll
#endif
    for (int kh = 0; kh < kernelHeight; ++kh) {
#if defined(__OPENMP)
#pragma unroll
#endif
      for (int kw = 0; kw < kernelWidth; ++kw) {
        int h_out = h + padHeight - kh;
        int w_out = w + padWidth - kw;
        if ((h_out % strideHeight == 0) && (w_out % strideWidth == 0)) {
          h_out = h_out / strideHeight;
          w_out = w_out / strideWidth;

          if ((h_out >= 0) && (h_out < outputHeight)
                && (w_out >= 0) && (w_out < outputWidth)) {

            const size_t offset = ((n * outputChannels + c) * outputHeight + h_out)
                  * outputWidth + w_out;
            value += (static_cast<spec_t>(weight[weightOffset]) *
                      static_cast<spec_t>(grad_output[offset]));
          }
        }
        weightOffset++;
      }
    }
  }
  grad_input[idx] = value;
}


template <typename spec_t>
__global__ void conv2d_gradient_filter_naive_kernel(const spec_t* grad_output, const spec_t* input, spec_t* grad_weight,
                                                    const int batchSize, const int inputChannels, const int kernelChannels,
                                                    const int inputWidth, const int inputHeight,
                                                    const int outputWidth, const int outputHeight,
                                                    const int kernelWidth, const int kernelHeight,
                                                    const int strideWidth, const int strideHeight,
                                                    const int padWidth, const int padHeight) {
  const int channelStride = kernelWidth * kernelHeight;

  int bidx = blockIdx.x;
  int kW = bidx % kernelWidth;
  int kH = (bidx / kernelWidth) % kernelHeight;
  int ch = (bidx / channelStride);
  int kC = ch % inputChannels;
  int kN = ch / inputChannels;

  spec_t grad = 0;

  const int laneId = threadIdx.x % 32;
  const int batch = threadIdx.x / 32;
  const int nwarps = blockDim.x / 32;
  const int imageElements = outputWidth * outputHeight;
  for (int batchIdx = batch; batchIdx < batchSize; batchIdx += nwarps){
    // Warp-stride loop over elements in a batch item
    for (size_t idx = laneId; idx < imageElements; idx += 32) {
      int go_w_offset = idx % outputWidth;
      int go_h_offset = (idx / outputWidth);

      int i_w_offset = (go_w_offset * strideWidth) + kW - padWidth;
      int i_h_offset = (go_h_offset * strideHeight) + kH - padHeight;

      if (i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < inputWidth && i_h_offset < inputHeight) {
        int inputOffset = ((batchIdx * inputChannels + kC) * inputHeight + i_h_offset) * inputWidth + i_w_offset;
        int outputOffset = ((batchIdx * kernelChannels + kN) * outputHeight ) * outputWidth + idx;
        grad += (static_cast<spec_t>(input[inputOffset]) *
                 static_cast<spec_t>(grad_output[outputOffset]));
      }
    }
  }
  __syncthreads();

  extern __shared__ char smem[];
  spec_t* buf = reinterpret_cast<spec_t*>(smem);
  hetu::cuda::BlockReduceSum(grad, buf);

  if (threadIdx.x == 0) {
    grad_weight[bidx] = grad;
  }
}

void Conv2dNaiveCuda(const NDArray& input_x, const NDArray& input_f, NDArray& output,
                     const int padding_h, const int padding_w, const int stride_h,
                     const int stride_w, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  size_t input_N = input_x->shape(0);
  size_t input_C = input_x->shape(1);
  size_t input_H = input_x->shape(2);
  size_t input_W = input_x->shape(3);

  size_t filter_N = input_f->shape(0);
  size_t filter_C = input_f->shape(1);
  size_t filter_H = input_f->shape(2);
  size_t filter_W = input_f->shape(3);

  size_t out_N = output->shape(0);
  size_t out_C = output->shape(1);
  size_t out_H = output->shape(2);
  size_t out_W = output->shape(3);

  size_t size = output->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dCuda", [&]() {
      conv2d_naive_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input_x->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
        input_f->data_ptr<spec_t>(), NULL, false, size,
        input_C, out_C, input_W, input_H,
        out_W, out_H, filter_W, filter_H,
        stride_w, stride_h, padding_w, padding_h);
    });
  NDArray::MarkUsedBy({input_x, input_f, output}, stream);
}

void Conv2dGradientofFilterNaiveCuda(const NDArray& input_x,
                                const NDArray& gradient_y, NDArray& gradient_f,
                                const int padding_h, const int padding_w,
                                const int stride_h, const int stride_w,
                                const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_f);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  // input
  size_t input_N = input_x->shape(0);
  size_t input_C = input_x->shape(1);
  size_t input_H = input_x->shape(2);
  size_t input_W = input_x->shape(3);
  // dy
  size_t dy_N = gradient_y->shape(0);
  size_t dy_C = gradient_y->shape(1);
  size_t dy_H = gradient_y->shape(2);
  size_t dy_W = gradient_y->shape(3);
  // dw
  size_t df_N = gradient_f->shape(0);
  size_t df_C = gradient_f->shape(1);
  size_t df_H = gradient_f->shape(2);
  size_t df_W = gradient_f->shape(3);

  size_t size = gradient_f->numel();
  dim3 blocks, threads;
  blocks.x = size;
  threads.x = MIN(input_N * 32, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dGradientofFilterCuda", [&]() {
      size_t mem = (blocks.x  / 32) * sizeof(spec_t);
      conv2d_gradient_filter_naive_kernel<spec_t><<<blocks, threads, mem, cuda_stream>>>(
        gradient_y->data_ptr<spec_t>(), input_x->data_ptr<spec_t>(), 
        gradient_f->data_ptr<spec_t>(), input_N,
        input_C, df_C, input_W, input_H, dy_W, dy_H,
        df_W, df_H, stride_w, stride_h, padding_w, padding_h); 
    });
  NDArray::MarkUsedBy({gradient_y, input_x, gradient_f}, stream);
}

void Conv2dGradientofDataNaiveCuda(const NDArray& input_f, const NDArray& gradient_y,
                                   NDArray& gradient_x, const int padding_h,
                                   const int padding_w, const int stride_h,
                                   const int stride_w, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_f);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_x);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());
  // filter
  size_t filter_N = input_f->shape(0);
  size_t filter_C = input_f->shape(1);
  size_t filter_H = input_f->shape(2);
  size_t filter_W = input_f->shape(3);
  // dy
  size_t dy_N = gradient_y->shape(0);
  size_t dy_C = gradient_y->shape(1);
  size_t dy_H = gradient_y->shape(2);
  size_t dy_W = gradient_y->shape(3);
  // dx
  size_t dx_N = gradient_x->shape(0);
  size_t dx_C = gradient_x->shape(1);
  size_t dx_H = gradient_x->shape(2);
  size_t dx_W = gradient_x->shape(3);

  size_t size = gradient_x->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_f->dtype(), spec_t, "Conv2dGradientofDataCuda", [&]() {
      conv2d_gradient_data_naive_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        gradient_y->data_ptr<spec_t>(), gradient_x->data_ptr<spec_t>(), 
        input_f->data_ptr<spec_t>(), size,
        dx_C, dy_C, dx_W, dx_H,
        dy_W, dy_H, filter_W, filter_H,
        stride_w, stride_h, padding_w, padding_h);      
    });
  NDArray::MarkUsedBy({gradient_y, gradient_x, input_f}, stream);
}

void Conv2dAddBiasNaiveCuda(const NDArray& input_x, const NDArray& input_f,
                            const NDArray& bias, NDArray& output,
                            const int padding_h, const int padding_w,
                            const int stride_h, const int stride_w,
                            const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, bias);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  size_t input_N = input_x->shape(0);
  size_t input_C = input_x->shape(1);
  size_t input_H = input_x->shape(2);
  size_t input_W = input_x->shape(3);

  size_t filter_N = input_f->shape(0);
  size_t filter_C = input_f->shape(1);
  size_t filter_H = input_f->shape(2);
  size_t filter_W = input_f->shape(3);

  size_t out_N = output->shape(0);
  size_t out_C = output->shape(1);
  size_t out_H = output->shape(2);
  size_t out_W = output->shape(3);

  size_t size = output->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dCuda", [&]() {
      conv2d_naive_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input_x->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
        input_f->data_ptr<spec_t>(), bias->data_ptr<spec_t>(), true, size,
        input_C, out_C, input_W, input_H,
        out_W, out_H, filter_W, filter_H,
        stride_w, stride_h, padding_w, padding_h);
    });
  NDArray::MarkUsedBy({input_x, output, input_f, bias}, stream);
}

} // namespace impl
} // namespace hetu
