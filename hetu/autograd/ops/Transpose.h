#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class TransposeOpDef;
class TransposeOp;

class TransposeOpDef : public OperatorDef {
 private:
  friend class TransposeOp;
  struct constrcutor_access_key {};

 public:
  TransposeOpDef(const constrcutor_access_key&, Tensor input, HTShape perms,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(TransposeOp), {input}, op_meta), _perms(perms) {
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape get_perms() const {
    return _perms;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _perms;
};

class TransposeOp final : public OpWrapper<TransposeOpDef> {
 public:
  TransposeOp(Tensor input, HTShape perms, const OpMeta& op_meta = OpMeta())
  : OpWrapper<TransposeOpDef>(make_ptr<TransposeOpDef>(
      TransposeOpDef::constrcutor_access_key(), input, perms, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
