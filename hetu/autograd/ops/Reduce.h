#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class ReduceOpDef;
class ReduceOp;
class ReduceGradientOpDef;
class ReduceGradientOp;

class ReduceOpDef : public OperatorDef {
 private:
  friend class ReduceOp;
  struct constrcutor_access_key {};

protected:
  ReduceOpDef(OpType&& op_type, Tensor input, ReductionType reduction = kMEAN, 
              const HTAxes& axes = {},
              const HTKeepDims& keepdims = {false},
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(std::move(op_type), {input}, op_meta),
    _axes(axes),
    _keepdims(keepdims),
    _reduction(reduction) {
    DoInferMeta();
    DoDeduceStates();
  }

 public:
  ReduceOpDef(const constrcutor_access_key&, Tensor input,
              ReductionType reduction, const HTAxes& axes = {},
              const HTKeepDims& keepdims = {false},
              const OpMeta& op_meta = OpMeta())
  : ReduceOpDef(quote(ReduceOp), input, reduction, axes, keepdims, op_meta) {}

  const HTAxes& get_axes() const {
    return _axes;
  }

  const HTKeepDims& get_keepdims() const {
    return _keepdims;
  }

  const std::string& mode() const {
    return _mode;
  }

  ReductionType reduction() const {
    return _reduction;
  }

  void set_axes(const HTAxes& axes) {
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

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  std::string _mode;

  HTAxes _axes;

  HTKeepDims _keepdims;

  HTShape _grad_add_axes;

  HTShape _grad_shape;

  double _grad_const;

  ReductionType _reduction;
};

class ReduceOp final : public OpWrapper<ReduceOpDef> {
 public:
  ReduceOp() : OpWrapper<ReduceOpDef>() {}
  ReduceOp(Tensor input, ReductionType reduction, const HTAxes& axes = {},
           const HTKeepDims& keepdims = {false},
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceOpDef>(
      make_ptr<ReduceOpDef>(ReduceOpDef::constrcutor_access_key(), input, reduction,
                            axes, keepdims, op_meta)) {}

  ReduceOp(Tensor input, const std::string& mode, const HTAxes& axes = {},
           const HTKeepDims& keepdims = {false},
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceOpDef>(
      make_ptr<ReduceOpDef>(ReduceOpDef::constrcutor_access_key(), input, Str2ReductionType(mode),
                            axes, keepdims, op_meta)) {}
};

class ReduceGradientOpDef : public OperatorDef {
 private:
  friend class ReduceGradientOp;
  struct constrcutor_access_key {};

 public:
  ReduceGradientOpDef(const constrcutor_access_key&, Tensor input,
                      Tensor ori_output, Tensor ori_input, const HTShape& shape,
                      ReductionType reduction = kMEAN,
                      const HTAxes add_axes = HTAxes(),
                      const HTKeepDims& keepdims = HTKeepDims(),
                      const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReduceGradientOp), {input, ori_output, ori_input}, op_meta),
    _shape(shape),
    _axes(add_axes),
    _keepdims(keepdims),
    _reduction(reduction) {
    DoInferMeta();
    DoDeduceStates();
  }

  const HTShape& get_shape() const {
    return _shape;
  }

  const HTAxes& get_axes() const {
    return _axes;
  }

  const HTAxes& get_add_axes() const {
    return _add_axes;
  }

  const HTKeepDims& get_keepdims() const {
    return _keepdims;
  }

  void set_shape(const HTShape& shape) {
    _shape = shape;
  }

  void set_axes(const HTAxes& shape) {
    _axes = shape;
  }

  void set_add_axes(const HTAxes& shape) {
    _add_axes = shape;
  }

  double get_const_value() const {
    return _constant;
  }

  std::string mode() const {
    return _mode;
  }

  ReductionType reduction() const {
    return _reduction;
  }

  void set_keepdims(const HTKeepDims& keepdims) {
    _keepdims = keepdims;
  }

  void set_const_value(double constant) {
    _constant = constant;
  }

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  std::string _mode;

  HTShape _shape;

  HTAxes _add_axes;

  HTAxes _axes;

  HTKeepDims _keepdims;

  double _constant;

  ReductionType _reduction;
};

class ReduceGradientOp final : public OpWrapper<ReduceGradientOpDef> {
 public:
  ReduceGradientOp(Tensor input, Tensor ori_output, Tensor ori_input, const HTShape& shape,
                   ReductionType reduction, const HTAxes add_axes = HTAxes(), const HTKeepDims& keepdims = HTKeepDims(),
                   const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceGradientOpDef>(make_ptr<ReduceGradientOpDef>(
      ReduceGradientOpDef::constrcutor_access_key(), input, ori_output, ori_input, shape,
      reduction, add_axes, keepdims, op_meta)) {}

  ReduceGradientOp(Tensor input, Tensor ori_output, Tensor ori_input, const HTShape& shape,
                   std::string mode, const HTAxes add_axes = HTAxes(), const HTKeepDims& keepdims = HTKeepDims(),
                   const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceGradientOpDef>(make_ptr<ReduceGradientOpDef>(
      ReduceGradientOpDef::constrcutor_access_key(), input, ori_output, ori_input, shape,
      Str2ReductionType(mode), add_axes, keepdims, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
