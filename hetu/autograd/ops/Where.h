#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class WhereOpDef;
class WhereOp;

class WhereOpDef : public OperatorDef {
 private:
  friend class WhereOp;
  struct constrcutor_access_key {};

 public:
  WhereOpDef(const constrcutor_access_key&, Tensor cond, Tensor inputA,
             Tensor inputB, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(WhereOp), {cond, inputA, inputB}, op_meta) {
    DoInferMeta();
    DoDeduceStates();
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class WhereOp final : public OpWrapper<WhereOpDef> {
 public:
  WhereOp(Tensor cond, Tensor inputA, Tensor inputB,
          const OpMeta& op_meta = OpMeta())
  : OpWrapper<WhereOpDef>(make_ptr<WhereOpDef>(
      WhereOpDef::constrcutor_access_key(), cond, inputA, inputB, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
