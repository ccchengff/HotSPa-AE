#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SoftmaxCrossEntropyOpDef;
class SoftmaxCrossEntropyOp;
class SoftmaxCrossEntropyGradientOpDef;
class SoftmaxCrossEntropyGradientOp;

class SoftmaxCrossEntropyOpDef : public OperatorDef {
 private:
  friend class SoftmaxCrossEntropyOp;
  struct constrcutor_access_key {};

 public:
  SoftmaxCrossEntropyOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, ReductionType reduction = kMEAN,
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SoftmaxCrossEntropyOp), {preds, labels}, op_meta),
    _reduction(reduction) {
    DoInferMeta();
    DoDeduceStates();
  }

  ReductionType reduction() const {
    return _reduction;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  ReductionType _reduction;
};

class SoftmaxCrossEntropyOp final : public OpWrapper<SoftmaxCrossEntropyOpDef> {
 public:
  SoftmaxCrossEntropyOp(Tensor preds, Tensor labels,
                       ReductionType reduction = kMEAN,
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropyOpDef>(make_ptr<SoftmaxCrossEntropyOpDef>(
      SoftmaxCrossEntropyOpDef::constrcutor_access_key(), preds, labels,
      reduction, op_meta)) {}

  SoftmaxCrossEntropyOp(Tensor preds, Tensor labels,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropyOpDef>(make_ptr<SoftmaxCrossEntropyOpDef>(
      SoftmaxCrossEntropyOpDef::constrcutor_access_key(), preds, labels,
      Str2ReductionType(reduction), op_meta)) {}
};

class SoftmaxCrossEntropyGradientOpDef : public OperatorDef {
 private:
  friend class SoftmaxCrossEntropyGradientOp;
  struct constrcutor_access_key {};

 public:
  SoftmaxCrossEntropyGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output,
                                  ReductionType reduction = kMEAN,
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SoftmaxCrossEntropyGradientOp),
                {preds, labels, grad_output}, op_meta),
    _reduction(reduction) {
    DoInferMeta();
    DoDeduceStates();
  }

  ReductionType reduction() const {
    return _reduction;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  ReductionType _reduction;
};

class SoftmaxCrossEntropyGradientOp final
: public OpWrapper<SoftmaxCrossEntropyGradientOpDef> {
 public:
  SoftmaxCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               ReductionType reduction = kMEAN,
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropyGradientOpDef>(
      make_ptr<SoftmaxCrossEntropyGradientOpDef>(
        SoftmaxCrossEntropyGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduction, op_meta)) {}

  SoftmaxCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropyGradientOpDef>(
      make_ptr<SoftmaxCrossEntropyGradientOpDef>(
        SoftmaxCrossEntropyGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, Str2ReductionType(reduction), op_meta)) {}
};

} // namespace autograd
} // namespace hetu
