#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class RepeatOpDef;
class RepeatOp;
class RepeatGradientOpDef;
class RepeatGradientOp;

class RepeatOpDef : public OperatorDef {
 private:
  friend class RepeatOp;
  struct constrcutor_access_key {};

 public:
  RepeatOpDef(const constrcutor_access_key&, Tensor input, HTShape repeats,
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(RepeatOp), {input}, op_meta),
  _repeats(repeats){
    DoInferMeta();
  }

  inline HTShape repeats() const{
    return _repeats;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _repeats;
};

class RepeatOp final : public OpWrapper<RepeatOpDef> {
 public:
  RepeatOp(Tensor input, HTShape repeats, const OpMeta& op_meta = OpMeta())
  : OpWrapper<RepeatOpDef>(make_ptr<RepeatOpDef>(
      RepeatOpDef::constrcutor_access_key(), input, repeats, op_meta)) {}
};

class RepeatGradientOpDef : public OperatorDef {
 private:
  friend class RepeatGradientOp;
  struct constrcutor_access_key {};

 public:
  RepeatGradientOpDef(const constrcutor_access_key&,
                      Tensor grad_output, Tensor input,
                      const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(RepeatGradientOp), {grad_output, input},
                op_meta) {
    DoInferMeta();
  }


 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

};

class RepeatGradientOp final
: public OpWrapper<RepeatGradientOpDef> {
 public:
  RepeatGradientOp(Tensor grad_output, Tensor input,
                    const OpMeta& op_meta = OpMeta())
  : OpWrapper<RepeatGradientOpDef>(
      make_ptr<RepeatGradientOpDef>(
        RepeatGradientOpDef::constrcutor_access_key(), grad_output, 
        input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
