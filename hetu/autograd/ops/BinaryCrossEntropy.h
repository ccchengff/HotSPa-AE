#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class BinaryCrossEntropyOpDef;
class BinaryCrossEntropyOp;
class BinaryCrossEntropyGradientOpDef;
class BinaryCrossEntropyGradientOp;

class BinaryCrossEntropyOpDef : public OperatorDef {
 private:
  friend class BinaryCrossEntropyOp;
  struct constrcutor_access_key {};

 public:
  BinaryCrossEntropyOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, ReductionType reduction = kMEAN,
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BinaryCrossEntropyOp), {preds, labels}, op_meta),
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

class BinaryCrossEntropyOp final : public OpWrapper<BinaryCrossEntropyOpDef> {
 public:
  BinaryCrossEntropyOp(Tensor preds, Tensor labels,
                       ReductionType reduction = kMEAN,
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<BinaryCrossEntropyOpDef>(make_ptr<BinaryCrossEntropyOpDef>(
      BinaryCrossEntropyOpDef::constrcutor_access_key(), preds, labels,
      reduction, op_meta)) {}

  BinaryCrossEntropyOp(Tensor preds, Tensor labels,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<BinaryCrossEntropyOpDef>(make_ptr<BinaryCrossEntropyOpDef>(
      BinaryCrossEntropyOpDef::constrcutor_access_key(), preds, labels,
      Str2ReductionType(reduction), op_meta)) {}
};

class BinaryCrossEntropyGradientOpDef : public OperatorDef {
 private:
  friend class BinaryCrossEntropyGradientOp;
  struct constrcutor_access_key {};

 public:
  BinaryCrossEntropyGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output,
                                  ReductionType reduction = kMEAN,
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BinaryCrossEntropyGradientOp),
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

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  ReductionType _reduction;
};

class BinaryCrossEntropyGradientOp final
: public OpWrapper<BinaryCrossEntropyGradientOpDef> {
 public:
  BinaryCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               ReductionType reduction = kMEAN,
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<BinaryCrossEntropyGradientOpDef>(
      make_ptr<BinaryCrossEntropyGradientOpDef>(
        BinaryCrossEntropyGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduction, op_meta)) {}

  BinaryCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<BinaryCrossEntropyGradientOpDef>(
      make_ptr<BinaryCrossEntropyGradientOpDef>(
        BinaryCrossEntropyGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, Str2ReductionType(reduction), op_meta)) {}
};

} // namespace autograd
} // namespace hetu
