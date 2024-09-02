#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SplitOpDef;
class SplitOp;
class SplitGradientOpDef;
class SplitGradientOp;

class SplitGradientOpDef : public OperatorDef {
 private:
  friend class SplitGradientOp;
  struct constrcutor_access_key {};

 public:
  SplitGradientOpDef(const constrcutor_access_key&, Tensor grad_output,
                     Tensor ori_output, Tensor ori_input, const HTAxes& axes,
                     const HTShape& indices, const HTShape& splits,
                     const HTShape& begin_pos, const HTShape& output_shape,
                     const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SplitGradientOp), {grad_output, ori_output, ori_input}, op_meta),
    _axes(axes),
    _indices(indices),
    _splits(splits),
    _begin_pos(begin_pos),
    _output_shape(output_shape) {
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape get_axes() const {
    return _axes;
  }

  HTShape get_indices() const {
    return _indices;
  }

  HTShape get_splits() const {
    return _splits;
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

  void set_begin_pos(HTShape begin_pos) {
    _begin_pos = begin_pos;
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

  HTShape _axes;

  HTShape _indices;

  HTShape _splits;

  HTShape _begin_pos;

  HTShape _output_shape;

  HTShape _ori_output_shape;
};

class SplitGradientOp final : public OpWrapper<SplitGradientOpDef> {
 public:
  SplitGradientOp() : OpWrapper<SplitGradientOpDef>() {}
  SplitGradientOp(Tensor grad_output, Tensor ori_output, Tensor ori_input, const HTAxes& axes,
                  const HTShape& indices, const HTShape& splits,
                  const HTShape& begin_pos, const HTShape& output_shape,
                  const OpMeta& op_meta = OpMeta())
  : OpWrapper<SplitGradientOpDef>(make_ptr<SplitGradientOpDef>(
      SplitGradientOpDef::constrcutor_access_key(), grad_output, ori_output, ori_input,
      axes, indices, splits, begin_pos, output_shape, op_meta)) {}
};

class SplitOpDef : public OperatorDef {
 private:
  friend class SplitOp;
  struct constrcutor_access_key {};

 public:
  SplitOpDef(const constrcutor_access_key&, Tensor input, const HTAxes& axes,
             const HTShape& indices, const HTShape& splits,
             const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SplitOp), {input}, op_meta),
    _axes(axes),
    _indices(indices),
    _splits(splits) {
    DoInferMeta();
    if (op_meta.is_deduce_states) {
      DoDeduceStates();
    }
  }

  uint64_t op_indicator() const noexcept {
    return SPLIT_OP;
  }  

  HTShape get_axes() const {
    return _axes;
  }

  HTShape get_indices() const {
    return _indices;
  }

  HTShape get_splits() const {
    return _splits;
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

  HTShape get_grad_begin_pos() const {
    return _grad_begin_pos;
  }

  void set_begin_pos(HTShape begin_pos) {
    _begin_pos = begin_pos;
  }

  void set_output_shape(HTShape output_shape) {
    _output_shape = output_shape;
  }

  void set_ori_output_shape(HTShape output_shape) {
    _ori_output_shape = output_shape;
  }

  void set_grad_begin_pos(HTShape begin_pos) {
    _grad_begin_pos = begin_pos;
  }

  void set_grad_output_shape(HTShape output_shape) {
    _grad_output_shape = output_shape;
  }

  Operator grad;

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _axes;

  HTShape _indices;

  HTShape _splits;

  HTShape _begin_pos;

  HTShape _output_shape;

  HTShape _ori_output_shape;

  HTShape _grad_begin_pos;

  HTShape _grad_output_shape;
};

class SplitOp final : public OpWrapper<SplitOpDef> {
 public:
  SplitOp(Tensor input, const HTAxes& axes, const HTShape& indices,
          const HTShape& splits, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SplitOpDef>(
      make_ptr<SplitOpDef>(SplitOpDef::constrcutor_access_key(), input, axes,
                           indices, splits, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
