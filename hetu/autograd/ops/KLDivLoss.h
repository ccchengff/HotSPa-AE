#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class KLDivLossOpDef;
class KLDivLossOp;
class KLDivLossGradientOpDef;
class KLDivLossGradientOp;

class KLDivLossOpDef : public OperatorDef {
 private:
  friend class KLDivLossOp;
  struct constrcutor_access_key {};

 public:
  KLDivLossOpDef(const constrcutor_access_key&, Tensor preds,
                 Tensor labels, ReductionType reduction = kMEAN,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(KLDivLossOp), {preds, labels}, op_meta),
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

class KLDivLossOp final : public OpWrapper<KLDivLossOpDef> {
 public:
  KLDivLossOp(Tensor preds, Tensor labels,
                       ReductionType reduction = kMEAN,
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<KLDivLossOpDef>(make_ptr<KLDivLossOpDef>(
      KLDivLossOpDef::constrcutor_access_key(), preds, labels,
      reduction, op_meta)) {}

  KLDivLossOp(Tensor preds, Tensor labels,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<KLDivLossOpDef>(make_ptr<KLDivLossOpDef>(
      KLDivLossOpDef::constrcutor_access_key(), preds, labels,
      Str2ReductionType(reduction), op_meta)) {}
};

class KLDivLossGradientOpDef : public OperatorDef {
 private:
  friend class KLDivLossGradientOp;
  struct constrcutor_access_key {};

 public:
  KLDivLossGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output,
                                  ReductionType reduction = kMEAN,
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(KLDivLossGradientOp),
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

class KLDivLossGradientOp final
: public OpWrapper<KLDivLossGradientOpDef> {
 public:
  KLDivLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               ReductionType reduction = kMEAN,
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<KLDivLossGradientOpDef>(
      make_ptr<KLDivLossGradientOpDef>(
        KLDivLossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduction, op_meta)) {}

  KLDivLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<KLDivLossGradientOpDef>(
      make_ptr<KLDivLossGradientOpDef>(
        KLDivLossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, Str2ReductionType(reduction), op_meta)) {}
};

} // namespace autograd
} // namespace hetu
