#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class Dropout2dOpDef;
class Dropout2dOp;
class Dropout2dGradientWithRecomputationOpDef;
class Dropout2dGradientWithRecomputationOp;

class Dropout2dOpDef : public OperatorDef {
 private:
  friend class Dropout2dOp;
  struct constrcutor_access_key {};

 public:
  Dropout2dOpDef(const constrcutor_access_key&, Tensor input, double keep_prob,
                 bool recompute = false, bool inplace = false,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(Dropout2dOp), {input}, op_meta),
    _keep_prob(keep_prob),
    _recompute(recompute || inplace),
    _inplace(inplace) {
    // TODO: support without recomputation
    HT_ASSERT(inplace) << "Currently we require Conv2D to be in place";
    AddOutput(input->meta());
    DoDeduceStates();
  }

  double keep_prob() const {
    return _keep_prob;
  };

  bool recompute() const {
    return _recompute;
  }

  bool inplace() const override {
    return _inplace;
  }

 protected:
  void DoDeduceStates() override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _keep_prob;
  bool _recompute;
  bool _inplace;
};

class Dropout2dOp final : public OpWrapper<Dropout2dOpDef> {
 public:
  Dropout2dOp(Tensor input, double keep_prob, bool recompute = false,
              bool inplace = false, const OpMeta& op_meta = OpMeta())
  : OpWrapper<Dropout2dOpDef>(
      make_ptr<Dropout2dOpDef>(Dropout2dOpDef::constrcutor_access_key(), input,
                               keep_prob, recompute, inplace, op_meta)) {}
};

class Dropout2dGradientWithRecomputationOpDef : public OperatorDef {
 private:
  friend class Dropout2dGradientWithRecomputationOp;
  struct constrcutor_access_key {};

 public:
  Dropout2dGradientWithRecomputationOpDef(const constrcutor_access_key&,
                                          Tensor grad_output,
                                          Dropout2dOp forward_op,
                                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(Dropout2dGradientWithRecomputationOp), {grad_output},
                op_meta),
    _forward_op(forward_op) {
    AddOutput(grad_output->meta());
    DoDeduceStates();
  }

  double keep_prob() const {
    return _forward_op->keep_prob();
  }

  bool inplace() const override {
    return true;
  }

 protected:
  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  Dropout2dOp _forward_op;
};

class Dropout2dGradientWithRecomputationOp final
: public OpWrapper<Dropout2dGradientWithRecomputationOpDef> {
 public:
  Dropout2dGradientWithRecomputationOp(Tensor grad_output,
                                       Dropout2dOp forward_op,
                                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<Dropout2dGradientWithRecomputationOpDef>(
      make_ptr<Dropout2dGradientWithRecomputationOpDef>(
        Dropout2dGradientWithRecomputationOpDef::constrcutor_access_key(),
        grad_output, forward_op, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
