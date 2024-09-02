#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/ops/ReduceSum.h"

namespace hetu {
namespace autograd {

class BroadcastShapeOpDef;
class BroadcastShapeOp;

class BroadcastShapeOpDef : public OperatorDef {
 private:
  friend class BroadcastShapeOp;
  struct constrcutor_access_key {};

 public:
  BroadcastShapeOpDef(const constrcutor_access_key&, Tensor input,
                      const HTShape& shape, const HTAxes& add_axes = HTAxes(),
                      const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastShapeOp), {input}, op_meta),
    _shape(shape),
    _add_axes(add_axes) {
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

  const HTShape& get_shape() const {
    return _shape;
  }

  const HTShape& get_add_axes() const {
    return _add_axes;
  }

  void set_shape(const HTShape& shape) {
    _shape = shape;
  }

  void set_add_axes(const HTShape& shape) {
    _add_axes = shape;
  }

  const HTShape& get_grad_axes() const {
    return _grad_add_axes;
  }

  const HTKeepDims& get_grad_keep_dims() const {
    return _grad_keep_dims;
  }

  void set_grad_axes(const HTShape& axes) {
    _grad_add_axes = axes;
  }

  void set_grad_keep_dims(const HTKeepDims& keep_dims) {
    _grad_keep_dims = keep_dims;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _shape;

  HTShape _add_axes;

  HTShape _grad_add_axes;

  HTKeepDims _grad_keep_dims;
};

class BroadcastShapeOp final : public OpWrapper<BroadcastShapeOpDef> {
 public:
  BroadcastShapeOp() : OpWrapper<BroadcastShapeOpDef>() {}
  BroadcastShapeOp(Tensor input, const HTShape& shape,
                   const HTAxes& add_axes = HTAxes(),
                   const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastShapeOpDef>(make_ptr<BroadcastShapeOpDef>(
      BroadcastShapeOpDef::constrcutor_access_key(), input, shape, add_axes,
      op_meta)) {}
};

} // namespace autograd
} // namespace hetu