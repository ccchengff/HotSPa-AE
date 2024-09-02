#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class BoolOpDef;
class BoolOp;

class BoolOpDef : public OperatorDef {
 private:
  friend class BoolOp;
  struct constrcutor_access_key {};

 public:
  BoolOpDef(const constrcutor_access_key&, Tensor input,
            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BoolOp), {input}, op_meta) {
    DoInferMeta();
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class BoolOp final : public OpWrapper<BoolOpDef> {
 public:
  BoolOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<BoolOpDef>(make_ptr<BoolOpDef>(
      BoolOpDef::constrcutor_access_key(), input, op_meta)) {}
};


} // namespace autograd
} // namespace hetu
