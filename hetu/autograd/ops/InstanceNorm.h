#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class InstanceNormOpDef;
class InstanceNormOp;
class InstanceNormGradientOpDef;
class InstanceNormGradientOp;

class InstanceNormOpDef : public OperatorDef {
 private:
  friend class InstanceNormOp;
  struct constrcutor_access_key {};

 public:
  InstanceNormOpDef(const constrcutor_access_key&, Tensor input,
                    double eps = 1e-7, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(InstanceNormOp), {input}, op_meta), _eps(eps) {
    DoInferMeta();
    DoDeduceStates();
  }

  double get_momentum() const {
    return _momentum;
  }

  double get_eps() const {
    return _eps;
  }

  HTShape get_shape() const {
    return _shape;
  }

  void set_shape(HTShape shape) {
    _shape = shape;
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

  HTShape _shape;
};

class InstanceNormOp final : public OpWrapper<InstanceNormOpDef> {
 public:
  InstanceNormOp(Tensor input, double eps = 1e-7,
                 const OpMeta& op_meta = OpMeta())
  : OpWrapper<InstanceNormOpDef>(make_ptr<InstanceNormOpDef>(
      InstanceNormOpDef::constrcutor_access_key(), input, eps, op_meta)) {}
};

class InstanceNormGradientOpDef : public OperatorDef {
 private:
  friend class InstanceNormGradientOp;
  struct constrcutor_access_key {};

 public:
  InstanceNormGradientOpDef(const constrcutor_access_key&, Tensor output_grad,
                            Tensor input, Tensor save_mean, Tensor save_var,
                            double eps = 1e-7,
                            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(InstanceNormGradientOp), {output_grad, input, save_mean, save_var}, op_meta),
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

class InstanceNormGradientOp final
: public OpWrapper<InstanceNormGradientOpDef> {
 public:
  InstanceNormGradientOp(Tensor output_grad, Tensor input,
                         Tensor save_mean, Tensor save_var,
                         double eps = 1e-7,
                         const OpMeta& op_meta = OpMeta())
  : OpWrapper<InstanceNormGradientOpDef>(make_ptr<InstanceNormGradientOpDef>(
      InstanceNormGradientOpDef::constrcutor_access_key(), output_grad, input,
      save_mean, save_var, eps, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
