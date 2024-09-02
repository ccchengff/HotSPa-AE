#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class NLLLossOpDef;
class NLLLossOp;
class NLLLossGradientOpDef;
class NLLLossGradientOp;

class NLLLossOpDef : public OperatorDef {
 private:
  friend class NLLLossOp;
  struct constrcutor_access_key {};

 public:
  NLLLossOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, ReductionType reduction = kMEAN,
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(NLLLossOp), {preds, labels}, op_meta),
    _reduction(reduction) {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type()
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
    AddOutput(preds->meta());
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

class NLLLossOp final : public OpWrapper<NLLLossOpDef> {
 public:
  NLLLossOp(Tensor preds, Tensor labels,
                       ReductionType reduction = kMEAN,
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<NLLLossOpDef>(make_ptr<NLLLossOpDef>(
      NLLLossOpDef::constrcutor_access_key(), preds, labels,
      reduction, op_meta)) {}

  NLLLossOp(Tensor preds, Tensor labels,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<NLLLossOpDef>(make_ptr<NLLLossOpDef>(
      NLLLossOpDef::constrcutor_access_key(), preds, labels,
      Str2ReductionType(reduction), op_meta)) {}
};

class NLLLossGradientOpDef : public OperatorDef {
 private:
  friend class NLLLossGradientOp;
  struct constrcutor_access_key {};

 public:
  NLLLossGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output,
                                  ReductionType reduction = kMEAN,
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(NLLLossGradientOp),
                {preds, labels, grad_output}, op_meta),
    _reduction(reduction) {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type()
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
    AddOutput(preds->meta());
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

class NLLLossGradientOp final
: public OpWrapper<NLLLossGradientOpDef> {
 public:
  NLLLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               ReductionType reduction = kMEAN,
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<NLLLossGradientOpDef>(
      make_ptr<NLLLossGradientOpDef>(
        NLLLossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduction, op_meta)) {}

  NLLLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<NLLLossGradientOpDef>(
      make_ptr<NLLLossGradientOpDef>(
        NLLLossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, Str2ReductionType(reduction), op_meta)) {}
};

} // namespace autograd
} // namespace hetu
