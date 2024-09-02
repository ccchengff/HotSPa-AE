#pragma once

#include "hetu/impl/utils/cuda_utils.h"
#include <curand.h>
#include <curand_kernel.h>

namespace hetu {
namespace impl {

template <typename T>
inline void curand_gen_uniform(curandGenerator_t gen, T* ptr, size_t size) {
  HT_NOT_IMPLEMENTED << "curand_gen_uniform is not implemented for type "
                     << typeid(T).name();
}

template <>
void curand_gen_uniform<float>(curandGenerator_t gen, float* ptr, size_t size);

template <>
void curand_gen_uniform<double>(curandGenerator_t gen, double* ptr,
                                size_t size);

} // namespace impl
} // namespace hetu

namespace {
inline const char* curandGetErrorString(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "CURAND_UNKNOWN_ERROR";
}
} // namespace

#define CURAND_CALL(f)                                                         \
  for (curandStatus_t status = (f); status != CURAND_STATUS_SUCCESS;           \
       status = CURAND_STATUS_SUCCESS)                                         \
  __HT_FATAL_SILENT(hetu::cuda::cuda_error)                                    \
    << "Curand call " << #f << " failed: " << curandGetErrorString(status)
