#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class ScalarLikeOpDef;
class ScalarLikeOp;

class ScalarLikeOpDef : public OperatorDef {
 private:
  friend class ScalarLikeOp;
  struct constrcutor_access_key {};

 protected:
  ScalarLikeOpDef(OpType&& op_type, Tensor input, double scalar_value,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(std::move(op_type), {input}, op_meta),
    _scalar_value(scalar_value) {
    DoInferMeta();
    DoDeduceStates();
  }

 public:
  ScalarLikeOpDef(const constrcutor_access_key&, Tensor input,
                  double scalar_value, const OpMeta& op_meta = OpMeta())
  : ScalarLikeOpDef(quote(ScalarLikeOp), input, scalar_value, op_meta) {}

  inline bool scalar_value() const {
    return _scalar_value;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _scalar_value;
};

class ScalarLikeOp final : public OpWrapper<ScalarLikeOpDef> {
 public:
  ScalarLikeOp(Tensor input, double scalar_value,
               const OpMeta& op_meta = OpMeta())
  : OpWrapper<ScalarLikeOpDef>(
      make_ptr<ScalarLikeOpDef>(ScalarLikeOpDef::constrcutor_access_key(),
                                input, scalar_value, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
