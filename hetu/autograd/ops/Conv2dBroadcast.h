#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class Conv2dBroadcastOpDef;
class Conv2dBroadcastOp;

class Conv2dBroadcastOpDef : public OperatorDef {
 private:
  friend class Conv2dBroadcastOp;
  struct constrcutor_access_key {};

 public:
  Conv2dBroadcastOpDef(const constrcutor_access_key&, Tensor input,
                       Tensor output, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(Conv2dBroadcastOp), {input, output}, op_meta) {
    AddOutput(output->meta()); // output->meta() ?
    // DoDeduceStates();
  }

 protected:
  void DoDeduceStates() override;
  
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class Conv2dBroadcastOp final : public OpWrapper<Conv2dBroadcastOpDef> {
 public:
  Conv2dBroadcastOp(Tensor input, Tensor output,
                    const OpMeta& op_meta = OpMeta())
  : OpWrapper<Conv2dBroadcastOpDef>(make_ptr<Conv2dBroadcastOpDef>(
      Conv2dBroadcastOpDef::constrcutor_access_key(), input, output, op_meta)) {
  }
};

} // namespace autograd
} // namespace hetu
