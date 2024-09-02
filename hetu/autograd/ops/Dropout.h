#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class DropoutOpDef;
class DropoutOp;
class DropoutGradientOpDef;
class DropoutGradientOp;
class DropoutGradientWithRecomputationOpDef;
class DropoutGradientWithRecomputationOp;

class DropoutOpDef : public OperatorDef {
 private:
  friend class DropoutOp;
  struct constrcutor_access_key {};

 public:
  DropoutOpDef(const constrcutor_access_key&, Tensor input, double keep_prob,
               bool recompute = false, bool inplace = false,
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(DropoutOp), {input}, op_meta),
    _keep_prob(keep_prob),
    _recompute(recompute || inplace),
    _inplace(inplace) {
    DoInferMeta();
    DoDeduceStates();
  }

  double keep_prob() const {
    return _keep_prob;
  }

  bool recompute() const {
    return _recompute;
  }

  bool inplace() const override {
    return _inplace;
  }

 protected:
  void DoInferMeta() override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _keep_prob;
  bool _recompute;
  bool _inplace;
};

class DropoutOp final : public OpWrapper<DropoutOpDef> {
 public:
  DropoutOp(Tensor input, double keep_prob, bool recompute = false,
            bool inplace = false, const OpMeta& op_meta = OpMeta())
  : OpWrapper<DropoutOpDef>(
      make_ptr<DropoutOpDef>(DropoutOpDef::constrcutor_access_key(), input,
                             keep_prob, recompute, inplace, op_meta)) {}
};

class DropoutGradientOpDef : public OperatorDef {
 private:
  friend class DropoutGradientOp;
  struct constrcutor_access_key {};

 public:
  DropoutGradientOpDef(const constrcutor_access_key&, Tensor grad_output,
                       Tensor output, double keep_prob,
                       const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(DropoutGradientOp), {grad_output, output}, op_meta),
    _keep_prob(keep_prob) {
    DoInferMeta();
    DoDeduceStates();
  }

  double keep_prob() const {
    return _keep_prob;
  }

 protected:
  void DoInferMeta() override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _keep_prob;
};

class DropoutGradientOp final : public OpWrapper<DropoutGradientOpDef> {
 public:
  DropoutGradientOp(Tensor grad_output, Tensor output, double keep_prob,
                    const OpMeta& op_meta = OpMeta())
  : OpWrapper<DropoutGradientOpDef>(make_ptr<DropoutGradientOpDef>(
      DropoutGradientOpDef::constrcutor_access_key(), grad_output, output,
      keep_prob, op_meta)) {}
};

class DropoutGradientWithRecomputationOpDef : public OperatorDef {
 private:
  friend class DropoutGradientWithRecomputationOp;
  struct constrcutor_access_key {};

 public:
  DropoutGradientWithRecomputationOpDef(const constrcutor_access_key&,
                                        Tensor grad_output,
                                        OpId forward_op,
                                        double keep_prob,
                                        const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(DropoutGradientWithRecomputationOp), {grad_output},
                op_meta),
    _forward_op(forward_op),
    _keep_prob(keep_prob) {
    DoInferMeta();
    DoDeduceStates();
  }

  double keep_prob() const {
    return _keep_prob;
  }

 protected:
  void DoInferMeta() override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  OpId _forward_op;

  double _keep_prob;
};

class DropoutGradientWithRecomputationOp final
: public OpWrapper<DropoutGradientWithRecomputationOpDef> {
 public:
  DropoutGradientWithRecomputationOp(Tensor grad_output, OpId forward_op, double keep_prob,
                                     const OpMeta& op_meta = OpMeta())
  : OpWrapper<DropoutGradientWithRecomputationOpDef>(
      make_ptr<DropoutGradientWithRecomputationOpDef>(
        DropoutGradientWithRecomputationOpDef::constrcutor_access_key(),
        grad_output, forward_op, keep_prob, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
