#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/stream/CUDAStream.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {

static std::once_flag cublas_device_init_flags[HT_MAX_GPUS_COMPILE_TIME];
static cublasHandle_t cublas_handles[HT_MAX_GPUS_COMPILE_TIME];

static void InitCublas(int32_t device_id) {
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUBLAS_CALL(cublasCreate(&cublas_handles[device_id]));
  cudaStream_t cuda_stream = GetCUDAComputingStream(device_id).cuda_stream();
  if (cuda_stream)
    CUBLAS_CALL(cublasSetStream(cublas_handles[device_id], cuda_stream));
}

void InitCublasOnce(int32_t device_id) {
  std::call_once(cublas_device_init_flags[device_id], InitCublas, device_id);
}

} // namespace

cublasHandle_t GetCublasHandle(int32_t device_id) {
  InitCublasOnce(device_id);
  return cublas_handles[device_id];
}

template <>
void cublas_dot<float16>(cublasHandle_t cublas_handle, int32_t n, const float16* x,
                         int32_t incx, const float16* y, int32_t incy, float16* result) {
  CUBLAS_CALL(cublasDotEx(cublas_handle, n, x, CUDA_R_16F, incx, y, CUDA_R_16F, incy, result,
                          CUDA_R_16F, CUDA_R_32F));
}

template <>
void cublas_dot<bfloat16>(cublasHandle_t cublas_handle, int32_t n, const bfloat16* x,
                          int32_t incx, const bfloat16* y, int32_t incy, bfloat16* result) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  CUBLAS_CALL(cublasDotEx(cublas_handle, n, x, CUDA_R_16BF, incx, y, CUDA_R_16BF, incy, result,
                          CUDA_R_16BF, CUDA_R_32F));
#else
  HT_NOT_IMPLEMENTED << "bfloat16 is not supported on this device.";
#endif
}

template <>
void cublas_dot<float>(cublasHandle_t cublas_handle, int32_t n, const float* x,
                       int32_t incx, const float* y, int32_t incy, float* result) {
  CUBLAS_CALL(cublasSdot(cublas_handle, n, x, incx, y, incy, result));
}

template <>
void cublas_dot<double>(cublasHandle_t cublas_handle, int32_t n, const double* x,
                        int32_t incx, const double* y, int32_t incy, double* result) {
  CUBLAS_CALL(cublasDdot(cublas_handle, n, x, incx, y, incy, result));
}

template <>
void cublas_gemv<float16>(cublasHandle_t cublas_handle, cublasOperation_t trans,
                          int32_t m, int32_t n, const void* alpha, const float16* A,
                          int32_t lda, const float16* x, int32_t incx,
                          const void* beta, float16* y, int32_t incy) {
  if (trans != CUBLAS_OP_N) {
    cublas_gemm<float16>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, n, m, alpha,
                         x, incx, A, lda, beta, y, incy);
  } else {
    cublas_gemm<float16>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, m, n, alpha,
                         x, incx, A, lda, beta, y, incy);
  }
}

template <>
void cublas_gemv<bfloat16>(cublasHandle_t cublas_handle, cublasOperation_t trans,
                          int32_t m, int32_t n, const void* alpha, const bfloat16* A,
                          int32_t lda, const bfloat16* x, int32_t incx,
                          const void* beta, bfloat16* y, int32_t incy) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (trans != CUBLAS_OP_N) {
    cublas_gemm<bfloat16>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, n, m, alpha,
                         x, incx, A, lda, beta, y, incy);
  } else {
    cublas_gemm<bfloat16>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, m, n, alpha,
                         x, incx, A, lda, beta, y, incy);
  }
#else
  HT_NOT_IMPLEMENTED << "bfloat16 is not supported on this device.";
#endif
}

template <>
void cublas_gemv<float>(cublasHandle_t cublas_handle, cublasOperation_t trans,
                        int32_t m, int32_t n, const void* alpha, const float* A,
                        int32_t lda, const float* x, int32_t incx,
                        const void* beta, float* y, int32_t incy) {
  CUBLAS_CALL(cublasSgemv(cublas_handle, trans, m, n, static_cast<const float*>(alpha), A, lda, x, incx,
                          static_cast<const float*>(beta), y, incy));
}

template <>
void cublas_gemv<double>(cublasHandle_t cublas_handle, cublasOperation_t trans,
                        int32_t m, int32_t n, const void* alpha, const double* A,
                        int32_t lda, const double* x, int32_t incx,
                        const void* beta, double* y, int32_t incy) {
  CUBLAS_CALL(cublasDgemv(cublas_handle, trans, m, n, static_cast<const double*>(alpha), A, lda, x, incx,
                          static_cast<const double*>(beta), y, incy));
}


template <>
void cublas_gemm<float16>(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                        cublasOperation_t trans_b, int32_t m, int32_t n,
                        int32_t k, const void* alpha, const float16* A,
                        int32_t lda, const float16* B, int32_t ldb,
                        const void* beta, float16* C, int32_t ldc) {
  CUBLAS_CALL(cublasGemmEx(cublas_handle, trans_a, trans_b, m, n, k, alpha, A, CUDA_R_16F,
              lda, B, CUDA_R_16F, ldb, beta, C, CUDA_R_16F, ldc, CUDA_R_32F,
              CUBLAS_GEMM_DFALT_TENSOR_OP));
}

template <>
void cublas_gemm<bfloat16>(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                           cublasOperation_t trans_b, int32_t m, int32_t n,
                           int32_t k, const void* alpha, const bfloat16* A,
                           int32_t lda, const bfloat16* B, int32_t ldb,
                           const void* beta, bfloat16* C, int32_t ldc) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  CUBLAS_CALL(cublasGemmEx(cublas_handle, trans_a, trans_b, m, n, k, alpha, A, CUDA_R_16BF,
              lda, B, CUDA_R_16BF, ldb, beta, C, CUDA_R_16BF, ldc, CUDA_R_32F,
              CUBLAS_GEMM_DFALT_TENSOR_OP));
#else
  HT_NOT_IMPLEMENTED << "bfloat16 is not supported on this device.";
#endif
}

template <>
void cublas_gemm<float>(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                        cublasOperation_t trans_b, int32_t m, int32_t n,
                        int32_t k, const void* alpha, const float* A,
                        int32_t lda, const float* B, int32_t ldb,
                        const void* beta, float* C, int32_t ldc) {
  CUBLAS_CALL(cublasSgemm(cublas_handle, trans_a, trans_b, m, n, k, static_cast<const float*>(alpha), A,
                          lda, B, ldb, static_cast<const float*>(beta), C, ldc));
}

template <>
void cublas_gemm<double>(cublasHandle_t cublas_handle,
                         cublasOperation_t trans_a, cublasOperation_t trans_b,
                         int32_t m, int32_t n, int32_t k, const void* alpha,
                         const double* A, int32_t lda, const double* B,
                         int32_t ldb, const void* beta, double* C,
                         int32_t ldc) {
  CUBLAS_CALL(cublasDgemm(cublas_handle, trans_a, trans_b, m, n, k, static_cast<const double*>(alpha), A,
                          lda, B, ldb, static_cast<const double*>(beta), C, ldc));
}

template <>
void cublas_batch_gemm<float16>(cublasHandle_t cublas_handle,
                                cublasOperation_t trans_a,
                                cublasOperation_t trans_b, int32_t m, int32_t n,
                                int32_t k, const void* alpha, const float16* A,
                                int32_t lda, int32_t strideA, const float16* B,
                                int32_t ldb, int32_t strideB, const void* beta,
                                float16* C, int32_t ldc, int32_t strideC,
                                int32_t batch_count) {
  CUBLAS_CALL(cublasGemmStridedBatchedEx(cublas_handle, trans_a, trans_b, m, n, k, alpha, A, CUDA_R_16F, lda, strideA, B, CUDA_R_16F, ldb,
              strideB, beta, C, CUDA_R_16F, ldc, strideC, batch_count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <>
void cublas_batch_gemm<bfloat16>(cublasHandle_t cublas_handle,
                                 cublasOperation_t trans_a,
                                 cublasOperation_t trans_b, int32_t m, int32_t n,
                                 int32_t k, const void* alpha, const bfloat16* A,
                                 int32_t lda, int32_t strideA, const bfloat16* B,
                                 int32_t ldb, int32_t strideB, const void* beta,
                                 bfloat16* C, int32_t ldc, int32_t strideC,
                                 int32_t batch_count) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  CUBLAS_CALL(cublasGemmStridedBatchedEx(cublas_handle, trans_a, trans_b, m, n, k, alpha, A, CUDA_R_16BF, lda, strideA, B, CUDA_R_16BF, ldb,
              strideB, beta, C, CUDA_R_16BF, ldc, strideC, batch_count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
  HT_NOT_IMPLEMENTED << "bfloat16 is not supported on this device.";
#endif
}

template <>
void cublas_batch_gemm<float>(cublasHandle_t cublas_handle,
                              cublasOperation_t trans_a,
                              cublasOperation_t trans_b, int32_t m, int32_t n,
                              int32_t k, const void* alpha, const float* A,
                              int32_t lda, int32_t strideA, const float* B,
                              int32_t ldb, int32_t strideB, const void*beta,
                              float* C, int32_t ldc, int32_t strideC,
                              int32_t batch_count) {
  CUBLAS_CALL(cublasSgemmStridedBatched(
    cublas_handle, trans_a, trans_b, m, n, k, static_cast<const float*>(alpha), A, lda, strideA, B, ldb,
    strideB, static_cast<const float*>(beta), C, ldc, strideC, batch_count));
}

template <>
void cublas_batch_gemm<double>(cublasHandle_t cublas_handle,
                               cublasOperation_t trans_a,
                               cublasOperation_t trans_b, int32_t m, int32_t n,
                               int32_t k, const void* alpha, const double* A,
                               int32_t lda, int32_t strideA, const double* B,
                               int32_t ldb, int32_t strideB, const void* beta,
                               double* C, int32_t ldc, int32_t strideC,
                               int32_t batch_count) {
  CUBLAS_CALL(cublasDgemmStridedBatched(
    cublas_handle, trans_a, trans_b, m, n, k, static_cast<const double*>(alpha), A, lda, strideA, B, ldb,
    strideB, static_cast<const double*>(beta), C, ldc, strideC, batch_count));
}

} // namespace impl
} // namespace hetu
