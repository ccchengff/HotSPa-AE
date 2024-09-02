#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Binary.cuh"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void instance_norm_kernel(const spec_t* x,  spec_t* y, spec_t* mean,
                                     spec_t* var, const float eps,
                                     const int last_dim,
                                     const OffsetCalculator* x_offset_calculator,
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
    auto y_offset = y_offset_calculator->get(i);
    y[y_offset] = (x[x_offset] - mean_thread) * tmp;
  }
}

void InstanceNormCuda(const NDArray& in_arr, NDArray& mean_arr,
                      NDArray& var_arr, NDArray& out_arr, float eps,
                      const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(in_arr);
  HT_ASSERT_SAME_DEVICE(in_arr, mean_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);   

  int ndim = in_arr->ndim();
  HT_ASSERT(ndim == 4);
  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);
  int base_dim = in_arr->shape(0) * in_arr->shape(1);

  auto device_id = in_arr->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  NDArray in_offset_calculator_arr, out_offset_calculator_arr,
          mean_offset_calculator_arr, var_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator,
                   *mean_offset_calculator, *var_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(in_arr, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) =
    AllocOffsetCalculator(out_arr, stream);
  std::tie(mean_offset_calculator_arr, mean_offset_calculator) = 
    AllocOffsetCalculator(mean_arr, stream);
  std::tie(var_offset_calculator_arr, var_offset_calculator) = 
    AllocOffsetCalculator(var_arr, stream);
  dim3 blocks, threads;
  threads.x = (last_2dim >= 1024 ? 1024 : 64);
  blocks.x = base_dim;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "InstanceNormCuda", [&]() {
      instance_norm_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        in_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(), 
        eps, last_2dim, in_offset_calculator, out_offset_calculator,
        mean_offset_calculator, var_offset_calculator);
    });
  NDArray::MarkUsedBy({in_arr, mean_arr, var_arr, out_arr,
                      in_offset_calculator_arr, out_offset_calculator_arr,
                      mean_offset_calculator_arr, var_offset_calculator_arr}, stream);
}

template <typename spec_t>
__global__ void calculate_grad_kernel(const spec_t* out_grads,
                                      const spec_t* in_arr,
                                      const spec_t* mean_arr,
                                      const spec_t* var_arr, 
                                      spec_t* ds, spec_t* dbias,
                                      spec_t* grad_arr,
                                      size_t last2dim, float eps, size_t size,
                                      const OffsetCalculator* out_grads_offset_calculator,
                                      const OffsetCalculator* in_offset_calculator,
                                      const OffsetCalculator* mean_offset_calculator,
                                      const OffsetCalculator* var_offset_calculator,
                                      const OffsetCalculator* grad_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last2dim;
  auto out_grads_offset = out_grads_offset_calculator->get(idx);
  auto in_offset = in_offset_calculator->get(idx);
  auto mean_offset = mean_offset_calculator->get(mo_idx);
  auto var_offset = var_offset_calculator->get(mo_idx);
  auto grad_offset = grad_offset_calculator->get(idx);
  spec_t tmp = (dbias[mo_idx] * mean_arr[mean_offset] - ds[mo_idx]) * (in_arr[in_offset] - mean_arr[mean_offset]) /
                (var_arr[var_offset] + eps);
  grad_arr[grad_offset] = out_grads[out_grads_offset] / hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps) +
    ((tmp - dbias[mo_idx]) / (spec_t)last2dim) / 
    hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps);
}

template <>
__global__ void calculate_grad_kernel<float16>(const float16* out_grads,
                                      const float16* in_arr,
                                      const float16* mean_arr,
                                      const float16* var_arr, 
                                      float16* ds, float16* dbias,
                                      float16* grad_arr,
                                      size_t last2dim, float eps, size_t size,
                                      const OffsetCalculator* out_grads_offset_calculator,
                                      const OffsetCalculator* in_offset_calculator,
                                      const OffsetCalculator* mean_offset_calculator,
                                      const OffsetCalculator* var_offset_calculator,
                                      const OffsetCalculator* grad_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last2dim;
  auto out_grads_offset = out_grads_offset_calculator->get(idx);
  auto in_offset = in_offset_calculator->get(idx);
  auto mean_offset = mean_offset_calculator->get(mo_idx);
  auto var_offset = var_offset_calculator->get(mo_idx);
  auto grad_offset = grad_offset_calculator->get(idx);
  float16 tmp = (dbias[mo_idx] * mean_arr[mean_offset] - ds[mo_idx]) * (in_arr[in_offset] - mean_arr[mean_offset]) /
                (var_arr[var_offset] + eps);
  grad_arr[grad_offset] = out_grads[out_grads_offset] / hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps) +
    ((tmp - dbias[mo_idx]) / (float16)last2dim) / 
    hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps);
}

template <>
__global__ void calculate_grad_kernel<bfloat16>(const bfloat16* out_grads,
                                      const bfloat16* in_arr,
                                      const bfloat16* mean_arr,
                                      const bfloat16* var_arr, 
                                      bfloat16* ds, bfloat16* dbias,
                                      bfloat16* grad_arr,
                                      size_t last2dim, float eps, size_t size,
                                      const OffsetCalculator* out_grads_offset_calculator,
                                      const OffsetCalculator* in_offset_calculator,
                                      const OffsetCalculator* mean_offset_calculator,
                                      const OffsetCalculator* var_offset_calculator,
                                      const OffsetCalculator* grad_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last2dim;
  auto out_grads_offset = out_grads_offset_calculator->get(idx);
  auto in_offset = in_offset_calculator->get(idx);
  auto mean_offset = mean_offset_calculator->get(mo_idx);
  auto var_offset = var_offset_calculator->get(mo_idx);
  auto grad_offset = grad_offset_calculator->get(idx);
  bfloat16 tmp = (dbias[mo_idx] * mean_arr[mean_offset] - ds[mo_idx]) * (in_arr[in_offset] - mean_arr[mean_offset]) /
                (var_arr[var_offset] + eps);
  grad_arr[grad_offset] = out_grads[out_grads_offset] / hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps) +
    ((tmp - dbias[mo_idx]) / (bfloat16)last2dim) / 
    hetu::cuda::cuda_sqrt(var_arr[var_offset] + eps);
}

void InstanceNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
                              NDArray& grad_arr, const NDArray& mean_arr,
                              const NDArray& var_arr, float eps,
                              const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(out_grads);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, mean_arr);   
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 

  int ndim = out_grads->ndim();
  HT_ASSERT(ndim == 4);
  size_t total_elements = 1;


  HT_ASSERT(ndim == 4);
  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int last2dim = out_grads->shape(ndim - 1) * out_grads->shape(ndim - 2);

  size_t size = total_elements;
  if (size == 0)
    return;
  
  auto device_id = out_grads->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  NDArray out_grad_offset_calculator_arr, in_offset_calculator_arr,
          mean_offset_calculator_arr, var_offset_calculator_arr,
          grad_offset_calculator_arr;
  OffsetCalculator *out_grad_offset_calculator, *in_offset_calculator,
                   *mean_offset_calculator, *var_offset_calculator,
                   *grad_offset_calculator;
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) =
    AllocOffsetCalculator(out_grads, stream);
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(in_arr, stream);
  std::tie(mean_offset_calculator_arr, mean_offset_calculator) = 
    AllocOffsetCalculator(mean_arr, stream);
  std::tie(var_offset_calculator_arr, var_offset_calculator) = 
    AllocOffsetCalculator(var_arr, stream);
  std::tie(grad_offset_calculator_arr, grad_offset_calculator) = 
    AllocOffsetCalculator(grad_arr, stream);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  NDArray dbias_arr = NDArray::sum(out_grads, {2, 3}, true, stream.stream_index());
  NDArray dy_mul_x_arr = NDArray::mul(out_grads, in_arr, stream.stream_index());
  NDArray dscale_arr = NDArray::sum(dy_mul_x_arr, {2, 3}, true, stream.stream_index());
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "CauculateGradCuda", [&]() {
      calculate_grad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        dscale_arr->data_ptr<spec_t>(), dbias_arr->data_ptr<spec_t>(),
        grad_arr->data_ptr<spec_t>(), last2dim, eps, size,
        out_grad_offset_calculator, in_offset_calculator,
        mean_offset_calculator, var_offset_calculator,
        grad_offset_calculator);
    });
  NDArray::MarkUsedBy({out_grads, in_arr, grad_arr, mean_arr, var_arr,
                       dbias_arr, dy_mul_x_arr, dscale_arr,
                       out_grad_offset_calculator_arr, in_offset_calculator_arr,
                       mean_offset_calculator_arr, var_offset_calculator_arr,
                       grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
