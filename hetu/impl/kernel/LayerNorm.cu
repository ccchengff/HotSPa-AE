#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Binary.cuh"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include <chrono>

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void layer_norm_kernel(const spec_t* x, const spec_t* scale,
                                  const spec_t* bias, spec_t* y, spec_t* mean,
                                  spec_t* var, const float eps, const int last_dim,
                                  const OffsetCalculator* x_offset_calculator,
                                  const OffsetCalculator* scale_offset_calculator,
                                  const OffsetCalculator* bias_offset_calculator,
                                  const OffsetCalculator* y_offset_calculator,
                                  const OffsetCalculator* mean_offset_calculator,
                                  const OffsetCalculator* var_offset_calculator) {
  __shared__ spec_t var_share;
  __shared__ spec_t mean_share;
  __shared__ spec_t shared_var[32];
  __shared__ spec_t shared_mean[32];

  int begin = blockIdx.x * last_dim + threadIdx.x;
  int end = (blockIdx.x + 1) * last_dim;

  spec_t mean_thread = 0, var_thread = 0;
  for (int i = begin; i < end; i += blockDim.x) {
    auto x_offset = x_offset_calculator->get(i);
    mean_thread += x[x_offset];
    var_thread += (x[x_offset] * x[x_offset]);
  }

  hetu::cuda::BlockReduceSum(mean_thread, shared_mean);
  hetu::cuda::BlockReduceSum(var_thread, shared_var);
  if (threadIdx.x == 0) {
    auto mean_offset = mean_offset_calculator->get(blockIdx.x);
    mean[mean_offset] = mean_share = mean_thread / last_dim;
    var_share = var_thread / last_dim - mean_share * mean_share;
    if (double(var_share) < 0)
      var_share = 0;
    auto var_offset = var_offset_calculator->get(blockIdx.x);
    var[var_offset] = var_share;
  }
  __syncthreads();

  mean_thread = mean_share;
  var_thread = var_share;
  spec_t tmp = 1.0f / sqrtf(var_thread + eps);
  for (int i = begin, j = threadIdx.x; i < end;
       i += blockDim.x, j += blockDim.x) {
    auto x_offset = x_offset_calculator->get(i);
    auto scale_offset = scale_offset_calculator->get(j);
    auto bias_offset = bias_offset_calculator->get(j);
    auto y_offset = y_offset_calculator->get(i);
    y[y_offset] = (x[x_offset] - mean_thread) * tmp * scale[scale_offset] + bias[bias_offset];
  }
}

void LayerNormCuda(const NDArray& in_arr, const NDArray& ln_scale,
                   const NDArray& ln_bias, NDArray& mean_arr, NDArray& var_arr,
                   NDArray& out_arr, int64_t reduce_dims, 
                   float eps, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(in_arr);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_scale);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_bias);
  HT_ASSERT_SAME_DEVICE(in_arr, mean_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  int ndim = in_arr->ndim();
  int base_dim = 1, last_dim = 1;
  for (int i = 0; i < ndim - reduce_dims; ++i)
    base_dim *= in_arr->shape(i);
  for (int i = ndim - reduce_dims; i < ndim; ++i)
    last_dim *= in_arr->shape(i);
  dim3 blocks, threads;
  threads.x = (last_dim >= 1024 ? 1024 : 64);
  blocks.x = base_dim;
  NDArray in_offset_calculator_arr, scale_offset_calculator_arr,
          bias_offset_calculator_arr, mean_offset_calculator_arr,
          var_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *scale_offset_calculator,
                   *bias_offset_calculator, *mean_offset_calculator,
                   *var_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(in_arr, stream);
  std::tie(scale_offset_calculator_arr, scale_offset_calculator) =
    AllocOffsetCalculator(ln_scale, stream);
  std::tie(bias_offset_calculator_arr, bias_offset_calculator) = 
    AllocOffsetCalculator(ln_bias, stream);
  std::tie(mean_offset_calculator_arr, mean_offset_calculator) = 
    AllocOffsetCalculator(mean_arr, stream);
  std::tie(var_offset_calculator_arr, var_offset_calculator) = 
    AllocOffsetCalculator(var_arr, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(out_arr, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "LayerNormCuda", [&]() {
      layer_norm_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        in_arr->data_ptr<spec_t>(), ln_scale->data_ptr<spec_t>(),
        ln_bias->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(), eps,
        last_dim, in_offset_calculator, scale_offset_calculator,
        bias_offset_calculator, out_offset_calculator,
        mean_offset_calculator, var_offset_calculator);
    });
  NDArray::MarkUsedBy({in_arr, ln_scale, ln_bias, mean_arr, var_arr, out_arr,
                      in_offset_calculator_arr, scale_offset_calculator_arr,
                      bias_offset_calculator_arr, mean_offset_calculator_arr,
                      var_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

template <typename spec_t>
__global__ void calculate_gscale(const spec_t* grads, const spec_t* in_arr,
                                 const spec_t* mean_arr, const spec_t* var_arr,
                                 spec_t* grad_scale, float eps,
                                 int last_dim, size_t size,
                                 const OffsetCalculator* grads_offset_calculator,
                                 const OffsetCalculator* in_offset_calculator,
                                 const OffsetCalculator* mean_offset_calculator,
                                 const OffsetCalculator* var_offset_calculator,
                                 const OffsetCalculator* grad_scale_offset_calculator) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size)
    return;
  int mo_ind = ind / last_dim;
  auto var_offset = var_offset_calculator->get(mo_ind);
  auto mean_offset = mean_offset_calculator->get(mo_ind);
  auto in_offset = in_offset_calculator->get(ind);
  auto grads_offset = grads_offset_calculator->get(ind);
  auto grad_scale_offset = grad_scale_offset_calculator->get(ind);
  spec_t std = hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps);
  spec_t x_centered = in_arr[in_offset] - mean_arr[mean_offset];
  spec_t x_norm = x_centered / std;
  grad_scale[grad_scale_offset] = grads[grads_offset] * x_norm;
}

template <>
__global__ void calculate_gscale<float16>(const float16* grads, const float16* in_arr,
                                          const float16* mean_arr, const float16* var_arr,
                                          float16* grad_scale, float eps,
                                          int last_dim, size_t size,
                                          const OffsetCalculator* grads_offset_calculator,
                                          const OffsetCalculator* in_offset_calculator,
                                          const OffsetCalculator* mean_offset_calculator,
                                          const OffsetCalculator* var_offset_calculator,
                                          const OffsetCalculator* grad_scale_offset_calculator) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size)
    return;
  int mo_ind = ind / last_dim;
  auto var_offset = var_offset_calculator->get(mo_ind);
  auto mean_offset = mean_offset_calculator->get(mo_ind);
  auto in_offset = in_offset_calculator->get(ind);
  auto grads_offset = grads_offset_calculator->get(ind);
  auto grad_scale_offset = grad_scale_offset_calculator->get(ind);
  float16 std = hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps);
  float16 x_centered = in_arr[in_offset] - mean_arr[mean_offset];
  float16 x_norm = x_centered / std;
  grad_scale[grad_scale_offset] = grads[grads_offset] * x_norm;
}

template <typename spec_t>
__global__ void calculate_grad_kernel_layer(const spec_t* out_grads,
                                      const spec_t* in_arr,
                                      const spec_t* scale_arr,
                                      const spec_t* mean_arr,
                                      const spec_t* var_arr, 
                                      spec_t* ds, spec_t* db,
                                      spec_t* grad_arr,
                                      size_t lastdim, float eps, size_t size,
                                      const OffsetCalculator* out_grads_offset_calculator,
                                      const OffsetCalculator* in_offset_calculator,
                                      const OffsetCalculator* scale_offset_calculator,
                                      const OffsetCalculator* mean_offset_calculator,
                                      const OffsetCalculator* var_offset_calculator,
                                      const OffsetCalculator* grad_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / lastdim;
  auto out_grads_offset = out_grads_offset_calculator->get(idx);
  auto in_offset = in_offset_calculator->get(idx);
  auto scale_offset = scale_offset_calculator->get(idx % lastdim);
  auto mean_offset = mean_offset_calculator->get(mo_idx);
  auto var_offset = var_offset_calculator->get(mo_idx);
  auto grad_offset = grad_offset_calculator->get(idx);
  spec_t tmp = (db[mo_idx] * mean_arr[mean_offset] - ds[mo_idx]) * (in_arr[in_offset] - mean_arr[mean_offset]) /
                (var_arr[var_offset] + eps);
  grad_arr[grad_offset] = (scale_arr[scale_offset] * out_grads[out_grads_offset] + (tmp - db[mo_idx]) / (spec_t)lastdim) / 
    hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps);
}

template <>
__global__ void calculate_grad_kernel_layer<float16>(const float16* out_grads,
                                      const float16* in_arr,
                                      const float16* scale_arr,
                                      const float16* mean_arr,
                                      const float16* var_arr, 
                                      float16* ds, float16* db,
                                      float16* grad_arr,
                                      size_t lastdim, float eps, size_t size,
                                      const OffsetCalculator* out_grads_offset_calculator,
                                      const OffsetCalculator* in_offset_calculator,
                                      const OffsetCalculator* scale_offset_calculator,
                                      const OffsetCalculator* mean_offset_calculator,
                                      const OffsetCalculator* var_offset_calculator,
                                      const OffsetCalculator* grad_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / lastdim;
  auto out_grads_offset = out_grads_offset_calculator->get(idx);
  auto in_offset = in_offset_calculator->get(idx);
  auto scale_offset = scale_offset_calculator->get(idx % lastdim);
  auto mean_offset = mean_offset_calculator->get(mo_idx);
  auto var_offset = var_offset_calculator->get(mo_idx);
  auto grad_offset = grad_offset_calculator->get(idx);
  float16 tmp = (db[mo_idx] * mean_arr[mean_offset] - ds[mo_idx]) * (in_arr[in_offset] - mean_arr[mean_offset]) /
                (var_arr[var_offset] + eps);
  grad_arr[grad_offset] = (scale_arr[scale_offset] * out_grads[out_grads_offset] + (tmp - db[mo_idx]) / (float16)lastdim) / 
    hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps);
}

void LayerNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
                           const NDArray& ln_scale, NDArray& grad_arr,
                           NDArray& grad_scale, NDArray& grad_bias,
                           const NDArray& mean_arr, const NDArray& var_arr,
                           int64_t reduce_dims, float eps, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(out_grads);
  HT_ASSERT_SAME_DEVICE(out_grads, ln_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, mean_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_bias);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  int ndim = out_grads->ndim();
  // HT_ASSERT(ndim == 4);
  size_t total_elements = 1;

  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);

  HTAxes reduce_axes_before = {}, reduce_axes_after = {};
  for (int i = 0; i < ndim; ++i) {
    if (i < ndim - reduce_dims)
      reduce_axes_before.emplace_back(i);
    else
      reduce_axes_after.emplace_back(i);
  }

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int lastdim = 1;
  for (size_t i = 0; i < reduce_dims; ++i) {
    lastdim *= out_grads->shape(ndim - 1 - i);
  }

  size_t size = total_elements;
  if (size == 0)
    return;
  NDArray::sum(out_grads, reduce_axes_before, true, stream.stream_index(), grad_bias);
  NDArray gscale_ = NDArray::empty_like(in_arr);

  NDArray out_grad_offset_calculator_arr, in_offset_calculator_arr,
          mean_offset_calculator_arr, var_offset_calculator_arr,
          gscale_offset_calculator_arr, scale_offset_calculator_arr,
          grad_offset_calculator_arr;
  OffsetCalculator *out_grad_offset_calculator, *in_offset_calculator,
                   *mean_offset_calculator, *var_offset_calculator,
                   *gscale_offset_calculator, *scale_offset_calculator,
                   *grad_offset_calculator;
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) =
    AllocOffsetCalculator(out_grads, stream);
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(in_arr, stream);
  std::tie(mean_offset_calculator_arr, mean_offset_calculator) = 
    AllocOffsetCalculator(mean_arr, stream);
  std::tie(var_offset_calculator_arr, var_offset_calculator) = 
    AllocOffsetCalculator(var_arr, stream);
  std::tie(gscale_offset_calculator_arr, gscale_offset_calculator) = 
    AllocOffsetCalculator(gscale_, stream);
  std::tie(scale_offset_calculator_arr, scale_offset_calculator) = 
    AllocOffsetCalculator(ln_scale, stream);
  std::tie(grad_offset_calculator_arr, grad_offset_calculator) = 
    AllocOffsetCalculator(grad_arr, stream);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "CalculateGradCuda", [&]() {
      calculate_gscale<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        gscale_->data_ptr<spec_t>(), eps, lastdim, in_arr->numel(),
        out_grad_offset_calculator, in_offset_calculator,
        mean_offset_calculator, var_offset_calculator,
        gscale_offset_calculator);
      
      NDArray::sum(gscale_, reduce_axes_before, true, stream.stream_index(), grad_scale);
      NDArray scale_out_grads_ = NDArray::mul(out_grads, ln_scale, stream.stream_index());
      NDArray db_ = NDArray::sum(scale_out_grads_, reduce_axes_after, true, stream.stream_index());
      NDArray dy_mul_x_ = NDArray::mul(scale_out_grads_, in_arr, stream.stream_index());
      NDArray ds_ = NDArray::sum(dy_mul_x_, reduce_axes_after, true, stream.stream_index());

      calculate_grad_kernel_layer<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(), ln_scale->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        ds_->data_ptr<spec_t>(), db_->data_ptr<spec_t>(),
        grad_arr->data_ptr<spec_t>(), lastdim, eps, size,
        out_grad_offset_calculator, in_offset_calculator,
        scale_offset_calculator, mean_offset_calculator,
        var_offset_calculator, grad_offset_calculator);
    });
    
  NDArray::MarkUsedBy({out_grads, in_arr, ln_scale, grad_arr,
                       grad_scale, grad_bias, mean_arr, var_arr,
                       out_grad_offset_calculator_arr, in_offset_calculator_arr,
                       mean_offset_calculator_arr, var_offset_calculator_arr,
                       gscale_offset_calculator_arr, scale_offset_calculator_arr,
                       grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
