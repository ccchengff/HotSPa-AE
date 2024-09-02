#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class ArrayReshapeOpDef;
class ArrayReshapeOp;
class ArrayReshapeGradientOpDef;
class ArrayReshapeGradientOp;

class ArrayReshapeOpDef : public OperatorDef {
 private:
  friend class ArrayReshapeOp;
  struct constrcutor_access_key {};

 public:
  ArrayReshapeOpDef(const constrcutor_access_key&, Tensor input,
                    const HTShape& output_shape,
                    const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ArrayReshapeOp), {input}, op_meta),
    _output_shape(output_shape) {
    DoInferMeta();
    DoDeduceStates(); // TODO: local shape vs global shape?
  }

  HTShape get_output_shape() const {
    return _output_shape;
  }

  HTShape get_input_shape() const {
    return _input_shape;
  }

  void set_input_shape(HTShape input_shape) {
    _input_shape = input_shape;
  };

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _output_shape;

  HTShape _input_shape;
};

class ArrayReshapeOp final : public OpWrapper<ArrayReshapeOpDef> {
 public:
  ArrayReshapeOp(Tensor input, const HTShape& output_shape,
                 const OpMeta& op_meta = OpMeta())
  : OpWrapper<ArrayReshapeOpDef>(
      make_ptr<ArrayReshapeOpDef>(ArrayReshapeOpDef::constrcutor_access_key(),
                                  input, output_shape, op_meta)) {}
};

class ArrayReshapeGradientOpDef : public OperatorDef {
 private:
  friend class ArrayReshapeGradientOp;
  struct constrcutor_access_key {};

 public:
  ArrayReshapeGradientOpDef(const constrcutor_access_key&, Tensor grad_output,
                            Tensor ori_input, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ArrayReshapeGradientOp), {grad_output, ori_input}, op_meta) {
    DoInferMeta();
    DoDeduceStates();
  }

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

};

class ArrayReshapeGradientOp final
: public OpWrapper<ArrayReshapeGradientOpDef> {
 public:
  ArrayReshapeGradientOp(Tensor grad_output, Tensor ori_input,
                         const OpMeta& op_meta = OpMeta())
  : OpWrapper<ArrayReshapeGradientOpDef>(make_ptr<ArrayReshapeGradientOpDef>(
      ArrayReshapeGradientOpDef::constrcutor_access_key(), grad_output,
      ori_input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
