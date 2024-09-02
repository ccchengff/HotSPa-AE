#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class BroadcastOpDef;
class BroadcastOp;

class BroadcastOpDef : public OperatorDef {
 private:
  friend class BroadcastOp;
  struct constrcutor_access_key {};

 public:
  BroadcastOpDef(const constrcutor_access_key&, Tensor input, Tensor output,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastOp), {input, output}, op_meta), _mode(0) {
    DoInferMeta();
    // DoDeduceStates(); // not sure
  }

  BroadcastOpDef(const constrcutor_access_key&, Tensor input,
                 const HTShape& shape, const HTShape& add_axes = HTShape(),
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastOp), {input}, op_meta),
    _mode(1),
    _shape(shape),
    _add_axes(add_axes) {
    DoInferMeta();
    // DoDeduceStates(); // not sure
  }

  const HTShape& get_shape() const {
    return _shape;
  }

  const HTAxes& get_add_axes() const {
    return _add_axes;
  }

  void set_shape(const HTShape& shape) {
    _shape = shape;
  }

  void set_add_axes(const HTAxes& shape) {
    _add_axes = shape;
  }

  const HTAxes& get_grad_axes() const {
    return _grad_add_axes;
  }

  const HTKeepDims& get_grad_keep_dims() const {
    return _grad_keep_dims;
  }

  int mode() const {
    return _mode;
  }

  void set_grad_axes(const HTAxes& axes) {
    _grad_add_axes = axes;
  }

  void set_grad_keep_dims(const HTKeepDims& keep_dims) {
    _grad_keep_dims = keep_dims;
  }

 protected:
  void DoInferMeta() override;

  // void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int _mode; // 0 = Broadcast, 1 = BroadcastShape

  HTShape _shape;

  HTShape _add_axes;

  HTShape _grad_add_axes;

  HTKeepDims _grad_keep_dims;
};

class BroadcastOp final : public OpWrapper<BroadcastOpDef> {
 public:
  BroadcastOp(Tensor input, Tensor output, const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastOpDef>(make_ptr<BroadcastOpDef>(
      BroadcastOpDef::constrcutor_access_key(), input, output, op_meta)) {}

  BroadcastOp(Tensor input, const HTShape& shape,
              const HTShape& add_axes = HTShape(),
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastOpDef>(
      make_ptr<BroadcastOpDef>(BroadcastOpDef::constrcutor_access_key(), input,
                               shape, add_axes, op_meta)) {}
};

class BroadcastGradientOpDef : public OperatorDef {
 private:
  friend class BroadcastGradientOp;
  struct constrcutor_access_key {};

 public:
  BroadcastGradientOpDef(const constrcutor_access_key&, Tensor input,
                         Tensor ori_input, Tensor ori_output, const HTShape& axes,
                         const HTKeepDims& keepdims,
                         const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastGradientOp), {input, ori_input, ori_output}, op_meta),
    _axes(axes),
    _keepdims(keepdims) {
        // HT_LOG_INFO << ori_input->shape();
    DoInferMeta();
  }

  const HTShape& get_axes() const {
    return _axes;
  }

  const HTKeepDims& get_keepdims() const {
    return _keepdims;
  }

  void set_axes(const HTShape& axes) {
    _axes = axes;
  }

  void set_keepdims(const HTKeepDims& keepdims) {
    _keepdims = keepdims;
  }

  const HTShape& get_grad_axes() const {
    return _grad_add_axes;
  }

  const HTShape& get_grad_shape() const {
    return _grad_shape;
  }

  double get_grad_const() const {
    return _grad_const;
  }

  void set_grad_axes(const HTShape& axes) {
    _grad_add_axes = axes;
  }

  void set_grad_shape(const HTShape& shape) {
    _grad_shape = shape;
  }

  void set_grad_const(double constant) {
    _grad_const = constant;
  }

  Operator grad, grad_;

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _axes;

  HTKeepDims _keepdims;

  HTShape _grad_add_axes;

  HTShape _grad_shape;

  double _grad_const;
};

class BroadcastGradientOp final : public OpWrapper<BroadcastGradientOpDef> {
 public:
  BroadcastGradientOp(Tensor input, Tensor ori_input, Tensor ori_output, const HTShape& axes,
                      const HTKeepDims& keepdims,
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastGradientOpDef>(make_ptr<BroadcastGradientOpDef>(
      BroadcastGradientOpDef::constrcutor_access_key(), input, ori_input, ori_output, axes,
      keepdims, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
