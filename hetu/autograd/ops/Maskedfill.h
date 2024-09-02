#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class MaskedfillOpDef;
class MaskedfillOp;

class MaskedfillOpDef : public OperatorDef {
 private:
  friend class MaskedfillOp;
  struct constrcutor_access_key {};

 public:
  MaskedfillOpDef(const constrcutor_access_key&, Tensor input,
                  Tensor mask, double val, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MaskedfillOp), {input, mask}, op_meta),
  _val(val) {
    DoInferMeta();
  }

  inline double val() const {
    return _val;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _val;
};

class MaskedfillOp final : public OpWrapper<MaskedfillOpDef> {
 public:
  MaskedfillOp(Tensor input, Tensor mask, double val,
               const OpMeta& op_meta = OpMeta())
  : OpWrapper<MaskedfillOpDef>(make_ptr<MaskedfillOpDef>(
      MaskedfillOpDef::constrcutor_access_key(), input, mask, val, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
