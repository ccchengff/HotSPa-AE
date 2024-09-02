#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class LayerNormOpDef;
class LayerNormOp;
class LayerNormGradientOpDef;
class LayerNormGradientOp;
class LayerNormGradientofDataOpDef;
class LayerNormGradientofDataOp;
class LayerNormGradientofScaleOpDef;
class LayerNormGradientofScaleOp;
class LayerNormGradientofBiasOpDef;
class LayerNormGradientofBiasOp;

class LayerNormOpDef : public OperatorDef {
 private:
  friend class LayerNormOp;
  struct constrcutor_access_key {};

 public:
  LayerNormOpDef(const constrcutor_access_key&, Tensor input, Tensor ln_scale,
                 Tensor ln_bias, const HTShape& normalized_shape, double eps = 0.01,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LayerNormOp), {input, ln_scale, ln_bias}, op_meta),
    _normalized_shape(normalized_shape),
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

  HTShape normalized_shape() const {
    return _normalized_shape;
  }


 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _momentum;

  HTShape _normalized_shape;

  double _eps;

};

class LayerNormOp final : public OpWrapper<LayerNormOpDef> {
 public:
  LayerNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias,
              const HTShape& normalized_shape, double eps = 0.01,
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<LayerNormOpDef>(
      make_ptr<LayerNormOpDef>(LayerNormOpDef::constrcutor_access_key(), input,
                               bn_scale, bn_bias, normalized_shape, eps, op_meta)) {}
};

class LayerNormGradientOpDef : public OperatorDef {
 private:
  friend class LayerNormGradientOp;
  struct constrcutor_access_key {};

 public:
  LayerNormGradientOpDef(const constrcutor_access_key&, Tensor output_grad,
                         Tensor input, Tensor bn_scale, Tensor save_mean, Tensor save_var, 
                         const HTShape& normalized_shape, double eps,
                         const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LayerNormGradientOp), {output_grad, input, bn_scale, save_mean, save_var},
                op_meta),
    _normalized_shape(normalized_shape),
    _eps(eps) {
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
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

  HTShape _normalized_shape;

  double _eps;
};

class LayerNormGradientOp final : public OpWrapper<LayerNormGradientOpDef> {
 public:
  LayerNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                      Tensor save_mean, Tensor save_var,
                      const HTShape& normalized_shape, double eps,
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<LayerNormGradientOpDef>(make_ptr<LayerNormGradientOpDef>(
      LayerNormGradientOpDef::constrcutor_access_key(), output_grad, input,
      bn_scale, save_mean, save_var, normalized_shape, eps, op_meta)) {}
};


} // namespace autograd
} // namespace hetu
