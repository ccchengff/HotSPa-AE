#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class BatchNormOpDef;
class BatchNormOp;
class BatchNormGradientOpDef;
class BatchNormGradientOp;
class BatchNormGradientofDataOpDef;
class BatchNormGradientofDataOp;
class BatchNormGradientofScaleOpDef;
class BatchNormGradientofScaleOp;
class BatchNormGradientofBiasOpDef;
class BatchNormGradientofBiasOp;

class BatchNormOpDef : public OperatorDef {
 private:
  friend class BatchNormOp;
  struct constrcutor_access_key {};

 public:
  BatchNormOpDef(const constrcutor_access_key&, Tensor input, Tensor bn_scale,
                 Tensor bn_bias, Tensor running_mean, Tensor running_var,
                 double momentum = 0.1, double eps = 1e-5,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchNormOp), {input, bn_scale, bn_bias, 
                running_mean, running_var}, op_meta),
    _momentum(momentum),
    _eps(eps) {
    DoInferMeta();
    DoDeduceStates();
  }

  double get_momentum() const {
    return _momentum;
  }

  double get_eps() const {
    return _eps;
  }

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;  

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _momentum;

  double _eps;
};

class BatchNormOp final : public OpWrapper<BatchNormOpDef> {
 public:
  BatchNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias,
              Tensor running_mean, Tensor running_var,
              double momentum = 0.1, double eps = 1e-5,
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchNormOpDef>(
      make_ptr<BatchNormOpDef>(BatchNormOpDef::constrcutor_access_key(), input,
                               bn_scale, bn_bias, running_mean, running_var, 
                               momentum, eps, op_meta)) {}
};

class BatchNormGradientOpDef : public OperatorDef {
 private:
  friend class BatchNormGradientOp;
  struct constrcutor_access_key {};

 public:
  BatchNormGradientOpDef(const constrcutor_access_key&, Tensor output_grad,
                         Tensor input, Tensor bn_scale,
                         Tensor save_mean, Tensor save_var, double eps,
                         const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchNormGradientOp), {output_grad, input, bn_scale, save_mean, save_var},
                op_meta),
    _eps(eps) {
    DoInferMeta();
    DoDeduceStates();
  }

  double get_eps() const {
    return _eps;
  }

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _eps;
};

class BatchNormGradientOp final : public OpWrapper<BatchNormGradientOpDef> {
 public:
  BatchNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                      Tensor save_mean, Tensor save_var, double eps,
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchNormGradientOpDef>(make_ptr<BatchNormGradientOpDef>(
      BatchNormGradientOpDef::constrcutor_access_key(), output_grad, input,
      bn_scale, save_mean, save_var, eps, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
