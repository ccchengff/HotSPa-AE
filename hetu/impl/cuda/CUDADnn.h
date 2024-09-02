#pragma once

#include "hetu/core/dtype.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <cudnn.h>

namespace hetu {
namespace impl {

inline cudnnDataType_t to_cudnn_DataType(DataType dtype) {
  switch (dtype) {
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
    case kBFloat16: return CUDNN_DATA_BFLOAT16;
#endif
    case kFloat16: return CUDNN_DATA_HALF;
    case kFloat32: return CUDNN_DATA_FLOAT;
    case kFloat64: return CUDNN_DATA_DOUBLE;
    default:
      HT_NOT_IMPLEMENTED << "Data type " << dtype
                         << " is not supported for cuDNN.";
      __builtin_unreachable();
  }
}

inline cudnnIndicesType_t to_cudnn_IndicidesType(DataType dtype) {
  switch (dtype) {
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
    case kBFloat16: return CUDNN_32BIT_INDICES;
#endif
    case kFloat16: return CUDNN_32BIT_INDICES;
    case kFloat32: return CUDNN_32BIT_INDICES;
    case kFloat64: return CUDNN_64BIT_INDICES;
    default:
      HT_NOT_IMPLEMENTED << "Data type " << dtype
                         << " is not supported for cuDNN.";
      __builtin_unreachable();
  }
}

cudnnHandle_t GetCudnnHandle(int32_t device_id);

} // namespace impl
} // namespace hetu

#define CUDNN_CALL(f)                                                          \
  for (cudnnStatus_t status = (f); status != CUDNN_STATUS_SUCCESS;             \
       status = CUDNN_STATUS_SUCCESS)                                          \
  __HT_FATAL_SILENT(hetu::cuda::cuda_error)                                    \
    << "Cudnn call " << #f << " failed: " << cudnnGetErrorString(status)
