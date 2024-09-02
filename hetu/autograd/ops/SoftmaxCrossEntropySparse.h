#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SoftmaxCrossEntropySparseOpDef;
class SoftmaxCrossEntropySparseOp;
class SoftmaxCrossEntropySparseGradientOpDef;
class SoftmaxCrossEntropySparseGradientOp;

class SoftmaxCrossEntropySparseOpDef : public OperatorDef {
 private:
  friend class SoftmaxCrossEntropySparseOp;
  struct constrcutor_access_key {};

 public:
  SoftmaxCrossEntropySparseOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, const int64_t ignored_index = -1, 
                          ReductionType reduction = kMEAN,
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SoftmaxCrossEntropySparseOp), {preds, labels}, op_meta),
    _ignored_index(ignored_index),
    _reduction(reduction) {
    DoInferMeta();
  }

  ReductionType reduction() const {
    return _reduction;
  }

  int64_t ignored_index() const {
    return _ignored_index;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _ignored_index;

  ReductionType _reduction;
};

class SoftmaxCrossEntropySparseOp final : public OpWrapper<SoftmaxCrossEntropySparseOpDef> {
 public:
  SoftmaxCrossEntropySparseOp(Tensor preds, Tensor labels, const int64_t ignored_index = -1, 
                              ReductionType reduction = kMEAN,
                              const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropySparseOpDef>(make_ptr<SoftmaxCrossEntropySparseOpDef>(
      SoftmaxCrossEntropySparseOpDef::constrcutor_access_key(), preds, labels, ignored_index,
      reduction, op_meta)) {}

  SoftmaxCrossEntropySparseOp(Tensor preds, Tensor labels, const int64_t ignored_index = -1, 
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropySparseOpDef>(make_ptr<SoftmaxCrossEntropySparseOpDef>(
      SoftmaxCrossEntropySparseOpDef::constrcutor_access_key(), preds, labels, ignored_index,
      Str2ReductionType(reduction), op_meta)) {}
};

class SoftmaxCrossEntropySparseGradientOpDef : public OperatorDef {
 private:
  friend class SoftmaxCrossEntropySparseGradientOp;
  struct constrcutor_access_key {};

 public:
  SoftmaxCrossEntropySparseGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output, const int64_t ignored_index = -1, 
                                  ReductionType reduction = kMEAN,
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SoftmaxCrossEntropySparseGradientOp),
                {preds, labels, grad_output}, op_meta),
    _ignored_index(ignored_index),
    _reduction(reduction) {
    DoInferMeta();
  }

  ino64_t ignored_index() const {
    return _ignored_index;
  }

  ReductionType reduction() const {
    return _reduction;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _ignored_index; 

  ReductionType _reduction;
};

class SoftmaxCrossEntropySparseGradientOp final
: public OpWrapper<SoftmaxCrossEntropySparseGradientOpDef> {
 public:
  SoftmaxCrossEntropySparseGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               const int64_t ignored_index = -1, 
                               ReductionType reduction = kMEAN,
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropySparseGradientOpDef>(
      make_ptr<SoftmaxCrossEntropySparseGradientOpDef>(
        SoftmaxCrossEntropySparseGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, ignored_index, reduction, op_meta)) {}

  SoftmaxCrossEntropySparseGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               const int64_t ignored_index = -1, 
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropySparseGradientOpDef>(
      make_ptr<SoftmaxCrossEntropySparseGradientOpDef>(
        SoftmaxCrossEntropySparseGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, ignored_index, Str2ReductionType(reduction), op_meta)) {}
};

} // namespace autograd
} // namespace hetu
