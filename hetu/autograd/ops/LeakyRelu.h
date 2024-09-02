#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class LeakyReluOpDef;
class LeakyReluOp;
class LeakyReluGradientOpDef;
class LeakyReluGradientOp;

class LeakyReluOpDef : public OperatorDef {
 private:
  friend class LeakyReluOp;
  struct constrcutor_access_key {};

 public:
  LeakyReluOpDef(const constrcutor_access_key&, Tensor input, double alpha,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LeakyReluOp), {input}, op_meta), _alpha(alpha) {
    DoInferMeta();
    DoDeduceStates();
  }

  double get_alpha() const {
    return _alpha;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _alpha;
};

class LeakyReluOp final : public OpWrapper<LeakyReluOpDef> {
 public:
  LeakyReluOp(Tensor input, double alpha, const OpMeta& op_meta = OpMeta())
  : OpWrapper<LeakyReluOpDef>(make_ptr<LeakyReluOpDef>(
      LeakyReluOpDef::constrcutor_access_key(), input, alpha, op_meta)) {}
};

class LeakyReluGradientOpDef : public OperatorDef {
 private:
  friend class LeakyReluGradientOp;
  struct constrcutor_access_key {};

 public:
  LeakyReluGradientOpDef(const constrcutor_access_key&, Tensor input,
                         Tensor grad_output, double alpha,
                         const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LeakyReluGradientOp), {input, grad_output}, op_meta),
    _alpha(alpha) {
    DoInferMeta();
    DoDeduceStates();
  }

  double get_alpha() const {
    return _alpha;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _alpha;
};

class LeakyReluGradientOp final : public OpWrapper<LeakyReluGradientOpDef> {
 public:
  LeakyReluGradientOp(Tensor input, Tensor grad_output, double alpha,
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<LeakyReluGradientOpDef>(make_ptr<LeakyReluGradientOpDef>(
      LeakyReluGradientOpDef::constrcutor_access_key(), input, grad_output,
      alpha, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
