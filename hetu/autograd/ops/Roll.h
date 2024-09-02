#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class RollOpDef;
class RollOp;
class RollGradientOpDef;
class RollGradientOp;

class RollOpDef : public OperatorDef {
 private:
  friend class RollOp;
  struct constrcutor_access_key {};

 public:
  RollOpDef(const constrcutor_access_key&, Tensor input, HTShape shifts,
            HTAxes dims, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(RollOp), {input}, op_meta),
  _shifts(shifts),
  _dims(dims) {
    DoInferMeta();
  }

  inline HTShape shifts() const{
    return _shifts;
  }

  inline HTAxes dims() const{
    return _dims;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _shifts;

  HTAxes _dims;
};

class RollOp final : public OpWrapper<RollOpDef> {
 public:
  RollOp(Tensor input, HTShape shifts,
         HTAxes dims, const OpMeta& op_meta = OpMeta())
  : OpWrapper<RollOpDef>(make_ptr<RollOpDef>(
      RollOpDef::constrcutor_access_key(), input, shifts, dims, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
