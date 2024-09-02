#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class Conv2dReduceSumOpDef;
class Conv2dReduceSumOp;

class Conv2dReduceSumOpDef : public OperatorDef {
 private:
  friend class Conv2dReduceSumOp;
  struct constrcutor_access_key {};

 public:
  Conv2dReduceSumOpDef(const constrcutor_access_key&, Tensor input,
                       const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(Conv2dReduceSumOp), {input}, op_meta) {
    HTShape shape = {};
    if (input->has_shape() && input->shape().size() >= 2)
      shape = {input->shape(1)};
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
    // DoDeduceStates();
  }

 protected:
  void DoDeduceStates() override;
   
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class Conv2dReduceSumOp final : public OpWrapper<Conv2dReduceSumOpDef> {
 public:
  Conv2dReduceSumOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<Conv2dReduceSumOpDef>(make_ptr<Conv2dReduceSumOpDef>(
      Conv2dReduceSumOpDef::constrcutor_access_key(), input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
