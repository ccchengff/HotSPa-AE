#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cuda_bf16.h>
#include <cuda_bf16.hpp>
#include "hetu/core/float16.h"
#include "hetu/core/bfloat16.h"
#include "hetu/impl/utils/numeric_utils.h"

namespace hetu {
namespace cuda {

template <typename T>
__forceinline__ __device__ T shfl_down_sync(const unsigned mask, T var, const unsigned int delta, const int width) {
  // HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type ";
  return var;
}

template <>
__forceinline__ __device__ hetu::bfloat16 shfl_down_sync<hetu::bfloat16>(const unsigned mask, hetu::bfloat16 var, const unsigned int delta, const int width) {
  #if(__CUDA_ARCH__ >= 800)
  return __shfl_down_sync(mask, __nv_bfloat16(var), delta, width);
  #else
  return __shfl_down_sync(mask, float(var), delta, width);
  #endif
}

template <>
__forceinline__ __device__ hetu::float16 shfl_down_sync<hetu::float16>(const unsigned mask, hetu::float16 var, const unsigned int delta, const int width) {
  return __shfl_down_sync(mask, __half(var), delta, width);
}

template <>
__forceinline__ __device__ float shfl_down_sync<float>(const unsigned mask, float var, const unsigned int delta, const int width) {
  return __shfl_down_sync(mask, float(var), delta, width);
}

template <>
__forceinline__ __device__ double shfl_down_sync<double>(const unsigned mask, double var, const unsigned int delta, const int width) {
  return __shfl_down_sync(mask, double(var), delta, width);
}

template <typename T>
__forceinline__ __device__ T cuda_max(T x, T y) {
  // HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type ";
  return x;
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_max<hetu::bfloat16>(hetu::bfloat16 x, hetu::bfloat16 y) {
  #if(__CUDA_ARCH__ >= 800)
  return __hmax(x, y);
  #else
  return static_cast<hetu::bfloat16>(max(float(x), float(y)));
  #endif
}

template <>
__forceinline__ __device__ hetu::float16 cuda_max<hetu::float16>(hetu::float16 x, hetu::float16 y) {
  #if(__CUDA_ARCH__ >= 800)
  return __hmax(x, y);
  #else
  return static_cast<hetu::float16>(max(float(x), float(y)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_max<float>(float x, float y) {
  return max(x, y);
}

template <>
__forceinline__ __device__ double cuda_max<double>(double x, double y) {
  return max(x, y);
}

template <typename T>
__forceinline__ __device__ T cuda_min(T x, T y) {
  // HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type ";
  return x;
}

template <>
__forceinline__ __device__ hetu::float16 cuda_min<hetu::float16>(hetu::float16 x, hetu::float16 y) {
  #if(__CUDA_ARCH__ >= 800)
  return __hmin(x, y);
  #else
  return static_cast<hetu::float16>(min(float(x), float(y)));
  #endif
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_min<hetu::bfloat16>(hetu::bfloat16 x, hetu::bfloat16 y) {
  #if(__CUDA_ARCH__ >= 800)
  return __hmin(x, y);
  #else
  return static_cast<hetu::bfloat16>(min(float(x), float(y)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_min<float>(float x, float y) {
  return min(x, y);
}

template <>
__forceinline__ __device__ double cuda_min<double>(double x, double y) {
  return min(x, y);
}

template <typename T>
__forceinline__ __device__ T cuda_abs(T x) {
  // HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type "
  //                    << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_abs<hetu::float16>(hetu::float16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return __habs(x);
  #else
  return static_cast<hetu::float16>(abs(float(x)));
  #endif
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_abs<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return __habs(x);
  #else
  return static_cast<hetu::bfloat16>(abs(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_abs<float>(float x) {
  return abs(x);
}

template <>
__forceinline__ __device__ double cuda_abs<double>(double x) {
  return abs(x);
}

template <typename T>
__forceinline__ __device__ T cuda_floor(T x) {
  HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_floor<hetu::float16>(hetu::float16 x) {
  return hfloor(x);
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_floor<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return hfloor(x);
  #else
  return static_cast<hetu::bfloat16>(floorf(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_floor<float>(float x) {
  return floorf(x);
}

template <>
__forceinline__ __device__ double cuda_floor<double>(double x) {
  return floor(x);
}

template <typename T>
__forceinline__ __device__ T cuda_ceil(T x) {
  HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_ceil<hetu::float16>(hetu::float16 x) {
  return hceil(x);
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_ceil<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return hceil(x);
  #else
  return static_cast<hetu::bfloat16>(floorf(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_ceil<float>(float x) {
  return ceilf(x);
}

template <>
__forceinline__ __device__ double cuda_ceil<double>(double x) {
  return ceil(x);
}

template <typename T>
__forceinline__ __device__ T cuda_round(T x) {
  HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_round<hetu::float16>(hetu::float16 x) {
  return static_cast<hetu::float16>(roundf(float(x)));
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_round<hetu::bfloat16>(hetu::bfloat16 x) {
  return static_cast<hetu::bfloat16>(roundf(float(x)));
}

template <>
__forceinline__ __device__ float cuda_round<float>(float x) {
  return roundf(x);
}

template <>
__forceinline__ __device__ double cuda_round<double>(double x) {
  return round(x);
}


template <typename T>
__forceinline__ __device__ T cuda_log(T x) {
  HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_log<hetu::float16>(hetu::float16 x) {
  return hlog(x);
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_log<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return hlog(x);
  #else
  return static_cast<hetu::bfloat16>(logf(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_log<float>(float x) {
  return logf(x);
}

template <>
__forceinline__ __device__ double cuda_log<double>(double x) {
  return log(x);
}

template <typename T>
__forceinline__ __device__ T cuda_exp(T x) {
}

template <>
__forceinline__ __device__ hetu::float16 cuda_exp<hetu::float16>(hetu::float16 x) {
  return hexp(x);
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_exp<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return hexp(x);
  #else
  return static_cast<hetu::bfloat16>(expf(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_exp<float>(float x) {
  return expf(x);
}

template <>
__forceinline__ __device__ double cuda_exp<double>(double x) {
  return exp(x);
}

template <typename T>
__forceinline__ __device__ T cuda_erf(T x) {
  HT_NOT_IMPLEMENTED << "cuda_erf is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_erf<hetu::float16>(hetu::float16 x) {
  // return herf(x);
  return static_cast<hetu::float16>(erff(float(x)));
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_erf<hetu::bfloat16>(hetu::bfloat16 x) {
  // #if(__CUDA_ARCH__ >= 800)
  // return herf(x);
  // #else
  return static_cast<hetu::bfloat16>(erff(float(x)));
  // #endif
}

template <>
__forceinline__ __device__ float cuda_erf<float>(float x) {
  return erff(x);
}

template <>
__forceinline__ __device__ double cuda_erf<double>(double x) {
  return erf(x);
}

template <typename T>
__forceinline__ __device__ T cuda_sqrt(T x) {
  HT_NOT_IMPLEMENTED << "cuda_sqrt is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_sqrt<hetu::float16>(hetu::float16 x) {
  return hsqrt(x);
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_sqrt<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return hsqrt(x);
  #else
  return static_cast<hetu::bfloat16>(sqrtf(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_sqrt<float>(float x) {
  return sqrtf(x);
}

template <>
__forceinline__ __device__ double cuda_sqrt<double>(double x) {
  return sqrt(x);
}

template <typename T>
__forceinline__ __device__ T cuda_rsqrt(T x) {
  HT_NOT_IMPLEMENTED << "cuda_rsqrt is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_rsqrt<hetu::float16>(hetu::float16 x) {
  return hrsqrt(x);
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_rsqrt<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return hrsqrt(x);
  #else
  return static_cast<hetu::bfloat16>(rsqrtf(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_rsqrt<float>(float x) {
  return rsqrtf(x);
}

template <>
__forceinline__ __device__ double cuda_rsqrt<double>(double x) {
  return rsqrt(x);
}

template <typename T>
__forceinline__ __device__ T cuda_sin(T x) {
  // HT_NOT_IMPLEMENTED << "cuda_sin is not implemented for type "
  //                    << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_sin<hetu::float16>(hetu::float16 x) {
  return hsin(x);
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_sin<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return hsin(x);
  #else
  return static_cast<hetu::bfloat16>(sinf(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_sin<float>(float x) {
  return sinf(x);
}

template <>
__forceinline__ __device__ double cuda_sin<double>(double x) {
  return sin(x);
}

template <typename T>
__forceinline__ __device__ T cuda_cos(T x) {
  // HT_NOT_IMPLEMENTED << "cuda_cos is not implemented for type "
  //                    << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_cos<hetu::float16>(hetu::float16 x) {
  return hcos(x);
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_cos<hetu::bfloat16>(hetu::bfloat16 x) {
  #if(__CUDA_ARCH__ >= 800)
  return hcos(x);
  #else
  return static_cast<hetu::bfloat16>(cosf(float(x)));
  #endif
}

template <>
__forceinline__ __device__ float cuda_cos<float>(float x) {
  return cosf(x);
}

template <>
__forceinline__ __device__ double cuda_cos<double>(double x) {
  return cos(x);
}

template <typename T>
__forceinline__ __device__ T cuda_tanh(T x) {
  HT_NOT_IMPLEMENTED << "cuda_tanh is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ hetu::float16 cuda_tanh<hetu::float16>(hetu::float16 x) {
  return static_cast<hetu::float16>(tanhf(float(x)));
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_tanh<hetu::bfloat16>(hetu::bfloat16 x) {
  return static_cast<hetu::bfloat16>(tanhf(float(x)));
}

template <>
__forceinline__ __device__ float cuda_tanh<float>(float x) {
  return tanhf(x);
}

template <>
__forceinline__ __device__ double cuda_tanh<double>(double x) {
  return tanh(x);
}

template <typename T>
__forceinline__ __device__ T cuda_pow(T x, T exponent) {
  // HT_NOT_IMPLEMENTED << "cuda_pow is not implemented for type "
  //                    << typeid(T).name();
}

__forceinline__ __device__ hetu::float16 cuda_pow(hetu::float16 x, double exponent) {
  return static_cast<hetu::float16>(powf(float(x), float(exponent)));
}

template <>
__forceinline__ __device__ hetu::float16 cuda_pow<hetu::float16>(hetu::float16 x, hetu::float16 exponent) {
  return static_cast<hetu::float16>(powf(float(x), float(exponent)));
}

__forceinline__ __device__ hetu::bfloat16 cuda_pow(hetu::bfloat16 x, double exponent) {
  return static_cast<hetu::bfloat16>(powf(float(x), float(exponent)));
}

template <>
__forceinline__ __device__ hetu::bfloat16 cuda_pow<hetu::bfloat16>(hetu::bfloat16 x, hetu::bfloat16 exponent) {
  return static_cast<hetu::bfloat16>(powf(float(x), float(exponent)));
}

__forceinline__ __device__ float cuda_pow(float x, double exponent) {
  return powf(x, float(exponent));
}

template <>
__forceinline__ __device__ float cuda_pow<float>(float x, float exponent) {
  return powf(x, exponent);
}

template <>
__forceinline__ __device__ double cuda_pow<double>(double x, double exponent) {
  return pow(x, exponent);
}

template <typename T>
__forceinline__ __device__ void AtomicAdd(T* address, T val) {
  HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ void AtomicAdd<hetu::float16>(hetu::float16* address, hetu::float16 val) {
  #if(__CUDA_ARCH__ >= 700)
  atomicAdd(reinterpret_cast<__half*>(address), __half(val));
  #else
  unsigned int * address_as_ui =
    (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  hetu::float16 hsum;
  do {
    assumed = old;
    hsum.val = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.val << 16) : (old & 0xffff0000) | hsum.val;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  #endif
}

template <>
__forceinline__ __device__ void AtomicAdd<hetu::bfloat16>(hetu::bfloat16* address, hetu::bfloat16 val) {
  #if(__CUDA_ARCH__ >= 800)
  atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), __nv_bfloat16(val));
  #else
  unsigned int * address_as_ui =
    (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  hetu::bfloat16 bsum;
  do {
    assumed = old;
    bsum.val = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    bsum = bsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (bsum.val << 16) : (old & 0xffff0000) | bsum.val;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  #endif
}

template <>
__forceinline__ __device__ void AtomicAdd<float>(float* address, float val) {
    atomicAdd(address, val);
}

template <>
__forceinline__ __device__ void AtomicAdd<double>(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    // return __longlong_as_double(old);
}

template <typename spec_t>
__forceinline__ __device__ spec_t WarpReduceSum(spec_t val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += shfl_down_sync(mask, val, k, warpSize);
  return val;
}

template <>
__forceinline__ __device__ bfloat16 WarpReduceSum<bfloat16>(bfloat16 val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  #if(__CUDA_ARCH__ >= 800)
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += shfl_down_sync(mask, val, k, warpSize);
  #else
  float val_f = float(val);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val_f += shfl_down_sync(mask, val_f, k, warpSize); 
  val = bfloat16(val_f); 
  #endif
  return val;
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceSum(spec_t& val, spec_t* shared) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);

  __syncthreads();
  if (tid == 0)
    shared[wid] = val;

  __syncthreads();
  spec_t zero = 0;
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[tid] : zero;

  if (wid == 0)
    val = WarpReduceSum(val);
  __syncthreads();
}

template <typename spec_t>
__forceinline__ __device__ void WarpReduceArgmax(spec_t& val) {
  spec_t tmp_val;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = shfl_down_sync(mask, val, k, warpSize);
    if (tmp_val > val) {
      val = tmp_val;
    }
  }
}

template <>
__forceinline__ __device__ void WarpReduceArgmax(bfloat16& val) {
  bfloat16 tmp_val;
  #if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800)
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = shfl_down_sync(mask, val, k, warpSize);
    if (tmp_val > val) {
      val = tmp_val;
    }
  }
  #endif
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceArgmax(spec_t& val,
                                                  spec_t* shared_value) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  WarpReduceArgmax(val);

  __syncthreads();
  if (tid == 0) {
    shared_value[wid] = val;
  }

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared_value[tid] : -SIZE_MAX;

  if (wid == 0)
    WarpReduceArgmax(val);
  __syncthreads();
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceArgmax(spec_t& val,
                                                  spec_t* shared_value,
                                                  spec_t* wrap_max) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  WarpReduceArgmax(val);

  __syncthreads();
  if (tid == 0) {
    shared_value[wid] = val;
  }

  __syncthreads();
  spec_t inf = numeric_limits<spec_t>::lowest();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared_value[tid] : inf;

  if (wid == 0)
    WarpReduceArgmax(val);
    if (threadIdx.x == 0)
      wrap_max[0] = val;
  __syncthreads();
}

template <typename spec_t>
__forceinline__ __device__ void WarpReduceArgmin(spec_t& val) {
  spec_t tmp_val;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = shfl_down_sync(mask, val, k, warpSize);
    if (tmp_val < val) {
      val = tmp_val;
    }
  }
}

template <>
__forceinline__ __device__ void WarpReduceArgmin(bfloat16& val) {
  bfloat16 tmp_val;
  #if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800)
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = shfl_down_sync(mask, val, k, warpSize);
    if (tmp_val < val) {
      val = tmp_val;
    }
  }
  #endif
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceArgmin(spec_t& val,
                                                  spec_t* shared_value) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  WarpReduceArgmin(val);

  __syncthreads();
  if (tid == 0) {
    shared_value[wid] = val;
  }

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared_value[tid] : SIZE_MAX;

  if (wid == 0)
    WarpReduceArgmin(val);
  __syncthreads();
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceArgmin(spec_t& val,
                                                  spec_t* shared_value,
                                                  spec_t* wrap_min) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  WarpReduceArgmin(val);

  __syncthreads();
  if (tid == 0) {
    shared_value[wid] = val;
  }

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared_value[tid] : SIZE_MAX;

  if (wid == 0)
    WarpReduceArgmin(val);
    if (threadIdx.x == 0)
      wrap_min[0] = val;
  __syncthreads();
}

__forceinline__ __device__ int64_t get_index(int64_t idx, int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
  int64_t i_idx = 0;
  int64_t t = idx;
  for (int i = 0; i < ndims; ++i) {
    int64_t ratio = t / c_shape[i];
    t -= ratio * c_shape[i];
    i_idx += ratio * stride[i];
  }
  return i_idx;
}

} // namespace cuda
} // namespace hetu