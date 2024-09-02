#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class MSELossOpDef;
class MSELossOp;
class MSELossGradientOpDef;
class MSELossGradientOp;

class MSELossOpDef : public OperatorDef {
 private:
  friend class MSELossOp;
  struct constrcutor_access_key {};

 public:
  MSELossOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, ReductionType reduction = kMEAN,
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MSELossOp), {preds, labels}, op_meta),
    _reduction(reduction) {
    DoInferMeta();
  }

  ReductionType reduction() const {
    return _reduction;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  ReductionType _reduction;
};

class MSELossOp final : public OpWrapper<MSELossOpDef> {
 public:
  MSELossOp(Tensor preds, Tensor labels,
                       ReductionType reduction = kMEAN,
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<MSELossOpDef>(make_ptr<MSELossOpDef>(
      MSELossOpDef::constrcutor_access_key(), preds, labels,
      reduction, op_meta)) {}

  MSELossOp(Tensor preds, Tensor labels,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<MSELossOpDef>(make_ptr<MSELossOpDef>(
      MSELossOpDef::constrcutor_access_key(), preds, labels,
      Str2ReductionType(reduction), op_meta)) {}
};

class MSELossGradientOpDef : public OperatorDef {
 private:
  friend class MSELossGradientOp;
  struct constrcutor_access_key {};

 public:
  MSELossGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output,
                                  ReductionType reduction = kMEAN,
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MSELossGradientOp),
                {preds, labels, grad_output}, op_meta),
    _reduction(reduction) {
    DoInferMeta();
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

class MSELossGradientOp final
: public OpWrapper<MSELossGradientOpDef> {
 public:
  MSELossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               ReductionType reduction = kMEAN,
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<MSELossGradientOpDef>(
      make_ptr<MSELossGradientOpDef>(
        MSELossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduction, op_meta)) {}

  MSELossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<MSELossGradientOpDef>(
      make_ptr<MSELossGradientOpDef>(
        MSELossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, Str2ReductionType(reduction), op_meta)) {}
};

} // namespace autograd
} // namespace hetu
