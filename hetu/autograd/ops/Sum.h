#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class SumOpDef;
class SumOp;

class SumOpDef : public OperatorDef {
 private:
  friend class SumOp;
  struct constrcutor_access_key {};

 public:
  SumOpDef(const constrcutor_access_key&, TensorList inputs,
           const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SumOp), inputs, op_meta) {
    HT_ASSERT(_inputs.size() > 0) << "No inputs are provided";
    int len = inputs.size();
    HTShape output_shape = inputs[0]->shape();
    for (int i = 1; i < len; ++i) {
      output_shape = Broadcast(output_shape, inputs[i]->shape());
    }
    AddOutput(NDArrayMeta(output_shape, inputs[0]->dtype(), inputs[0]->device()));
    if (op_meta.is_deduce_states) {  
      DoDeduceStates();
    }    
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class SumOp final : public OpWrapper<SumOpDef> {
 public:
  SumOp(TensorList inputs, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SumOpDef>(make_ptr<SumOpDef>(SumOpDef::constrcutor_access_key(),
                                           inputs, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
