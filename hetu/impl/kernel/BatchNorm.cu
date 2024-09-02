#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void BatchNormCuda(const NDArray& input_X, const NDArray& bn_scale,
                   const NDArray& bn_bias, NDArray& output_Y, double momentum,
                   double eps, NDArray& running_mean, NDArray& running_var,
                   NDArray& save_mean, NDArray& save_var,
                   const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_X);
  HT_ASSERT_SAME_DEVICE(input_X, bn_scale);
  HT_ASSERT_SAME_DEVICE(input_X, bn_bias);
  HT_ASSERT_SAME_DEVICE(input_X, output_Y);
  HT_ASSERT_SAME_DEVICE(input_X, running_mean);
  HT_ASSERT_SAME_DEVICE(input_X, running_var);
  HT_ASSERT_SAME_DEVICE(input_X, save_mean);
  HT_ASSERT_SAME_DEVICE(input_X, save_var);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype = to_cudnn_DataType(input_X->dtype());

  // input
  size_t input_N = input_X->shape(0);
  size_t input_C = input_X->shape(1);
  size_t input_H = input_X->shape(2);
  size_t input_W = input_X->shape(3);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_X->dtype(), spec_t, "BatchNormCuda", [&]() {
      // input descriptor
      cudnnTensorDescriptor_t input_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));
      // output descriptor
      cudnnTensorDescriptor_t output_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));
      // bn parameter descriptor
      cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
      CUDNN_CALL(
        cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc,
                                      CUDNN_BATCHNORM_SPATIAL)); // after conv

      spec_t alpha = 1.0;
      spec_t beta = 0.0;

      float alpha_f = 1.0f;
      float beta_f = 0.0f;
      if (input_X->dtype() == DataType::FLOAT16 || input_X->dtype() == DataType::BFLOAT16 ) {
        CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
          handle, CUDNN_BATCHNORM_SPATIAL, &alpha_f, &beta_f, input_desc, input_X->data_ptr<spec_t>(),
          output_desc, output_Y->data_ptr<spec_t>(), bnScaleBiasMeanVar_desc, bn_scale->data_ptr<float>(),
          bn_bias->data_ptr<float>(), momentum, running_mean->data_ptr<void>(), running_var->data_ptr<void>(), eps,
          save_mean->data_ptr<void>(), save_var->data_ptr<void>()));
      } else {
        CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
          handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, input_desc, input_X->data_ptr<spec_t>(),
          output_desc, output_Y->data_ptr<spec_t>(), bnScaleBiasMeanVar_desc, bn_scale->data_ptr<spec_t>(),
          bn_bias->data_ptr<spec_t>(), momentum, running_mean->data_ptr<void>(), running_var->data_ptr<void>(), eps,
          save_mean->data_ptr<void>(), save_var->data_ptr<void>()));
      }

      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));
    });
  
  NDArray::MarkUsedBy({input_X, bn_scale, bn_bias, output_Y, running_mean,
                       running_var, save_mean, save_var},
                      stream);
}

void BatchNormGradientCuda(const NDArray& gradient_Y, const NDArray& input_X,
                           const NDArray& bn_scale, NDArray& gradient_X,
                           NDArray& gradient_bn_scale,
                           NDArray& gradient_bn_bias, double eps,
                           NDArray& save_mean, NDArray& save_var,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(gradient_Y);
  HT_ASSERT_SAME_DEVICE(gradient_Y, input_X);
  HT_ASSERT_SAME_DEVICE(gradient_Y, bn_scale);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_X);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_bn_scale);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_bn_bias);
  HT_ASSERT_SAME_DEVICE(gradient_Y, save_mean);
  HT_ASSERT_SAME_DEVICE(gradient_Y, save_var);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype = to_cudnn_DataType(input_X->dtype());

  // input
  size_t input_N = input_X->shape(0);
  size_t input_C = input_X->shape(1);
  size_t input_H = input_X->shape(2);
  size_t input_W = input_X->shape(3);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_X->dtype(), spec_t, "BatchNormGradientCuda", [&]() {
      // input descriptor
      cudnnTensorDescriptor_t input_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));
      // output descriptor
      cudnnTensorDescriptor_t output_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));
      // bn parameter descriptor
      cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
      CUDNN_CALL(
        cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc,
                                      CUDNN_BATCHNORM_SPATIAL)); // after conv

      spec_t one = 1.0;
      spec_t zero = 0.0;

      float one_f = 1.0f;
      float zero_f = 0.0f;

      if (input_X->dtype() == DataType::FLOAT16 || input_X->dtype() == DataType::BFLOAT16) {  
        CUDNN_CALL(cudnnBatchNormalizationBackward(
          handle, CUDNN_BATCHNORM_SPATIAL, &one_f, &zero_f, &one_f, &zero_f,
          input_desc, input_X->data_ptr<spec_t>(), output_desc, gradient_Y->data_ptr<spec_t>(), input_desc,
          gradient_X->data_ptr<spec_t>(), bnScaleBiasMeanVar_desc, bn_scale->data_ptr<spec_t>(),
          gradient_bn_scale->data_ptr<float>(), gradient_bn_bias->data_ptr<float>(), eps, 
          save_mean->data_ptr<void>(), save_var->data_ptr<void>()));
      } else {
        CUDNN_CALL(cudnnBatchNormalizationBackward(
          handle, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &zero,
          input_desc, input_X->data_ptr<spec_t>(), output_desc, gradient_Y->data_ptr<spec_t>(), input_desc,
          gradient_X->data_ptr<spec_t>(), bnScaleBiasMeanVar_desc, bn_scale->data_ptr<spec_t>(),
          gradient_bn_scale->data_ptr<spec_t>(), gradient_bn_bias->data_ptr<spec_t>(), eps, 
          save_mean->data_ptr<void>(), save_var->data_ptr<void>()));
      }

      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));
    });
  
  NDArray::MarkUsedBy({gradient_Y, input_X, bn_scale, gradient_X,
                       gradient_bn_scale, gradient_bn_bias, save_mean,
                       save_var},
                      stream);
}

} // namespace impl
} // namespace hetu
