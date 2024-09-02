#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SliceOpDef;
class SliceOp;
class SliceGradientOpDef;
class SliceGradientOp;

class SliceGradientOpDef : public OperatorDef {
 private:
  friend class SliceGradientOp;
  struct constrcutor_access_key {};

 public:
  SliceGradientOpDef(const constrcutor_access_key&, Tensor grad_output,
                     Tensor ori_output, Tensor ori_input, const HTShape& begin_pos,
                     const HTShape& output_shape,
                     const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SliceGradientOp), {grad_output, ori_output, ori_input}, op_meta),
    _begin_pos(begin_pos),
    _output_shape(output_shape) {
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape get_begin_pos() const {
    return _begin_pos;
  }

  HTShape get_output_shape() const {
    return _output_shape;
  }

  HTShape get_ori_output_shape() const {
    return _ori_output_shape;
  }

  void set_output_shape(HTShape output_shape) {
    _output_shape = output_shape;
  }

  void set_ori_output_shape(HTShape output_shape) {
    _ori_output_shape = output_shape;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _begin_pos;

  HTShape _output_shape;

  HTShape _ori_output_shape;
};

class SliceGradientOp final : public OpWrapper<SliceGradientOpDef> {
 public:
  SliceGradientOp() : OpWrapper<SliceGradientOpDef>() {}
  SliceGradientOp(Tensor grad_output, Tensor ori_output, Tensor ori_input,
                  const HTShape& begin_pos, const HTShape& output_shape,
                  const OpMeta& op_meta = OpMeta())
  : OpWrapper<SliceGradientOpDef>(make_ptr<SliceGradientOpDef>(
      SliceGradientOpDef::constrcutor_access_key(), grad_output, ori_output, ori_input,
      begin_pos, output_shape, op_meta)) {}
};

class SliceOpDef : public OperatorDef {
 private:
  friend class SliceOp;
  struct constrcutor_access_key {};

 public:
  SliceOpDef(const constrcutor_access_key&, Tensor input,
             const HTShape& begin_pos, const HTShape& output_shape,
             const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SliceOp), {input}, op_meta),
    _begin_pos(begin_pos),
    _output_shape(output_shape),
    _ori_output_shape(output_shape) {
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape get_begin_pos() const {
    return _begin_pos;
  }

  HTShape get_output_shape() const {
    return _output_shape;
  }

  HTShape get_ori_output_shape() const {
    return _ori_output_shape;
  }

  HTShape get_grad_output_shape() const {
    return _grad_output_shape;
  }

  void set_output_shape(HTShape output_shape) {
    _output_shape = output_shape;
  }

  void set_ori_output_shape(HTShape output_shape) {
    _ori_output_shape = output_shape;
  }

  void set_grad_output_shape(HTShape output_shape) {
    _grad_output_shape = output_shape;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _begin_pos;

  HTShape _output_shape;

  HTShape _ori_output_shape;

  HTShape _grad_output_shape;
};

class SliceOp final : public OpWrapper<SliceOpDef> {
 public:
  SliceOp(Tensor input, const HTShape& begin_pos, const HTShape& output_shape,
          const OpMeta& op_meta = OpMeta())
  : OpWrapper<SliceOpDef>(
      make_ptr<SliceOpDef>(SliceOpDef::constrcutor_access_key(), input,
                           begin_pos, output_shape, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
