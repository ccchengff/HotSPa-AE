#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class MatMul2OpDef;
class MatMul2Op;
class MatMul2GradientOpDef;
class MatMul2GradientOp;

class MatMul2OpDef : public OperatorDef {
 private:
  friend class MatMul2Op;
  struct constrcutor_access_key {};

 public:
  MatMul2OpDef(const constrcutor_access_key&, Tensor a, Tensor b,
              bool trans_a = false, bool trans_b = false,
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MatMul2Op), {a, b}, op_meta),
    _trans_a(trans_a),
    _trans_b(trans_b) {
    DoInferMeta();
  }

  inline bool trans_a() const {
    return _trans_a;
  }

  inline bool trans_b() const {
    return _trans_b;
  }

  void set_grad_axes(HTAxes axe, int idx) {
    _grad_add_axes[idx] = axe;
  };

  void set_grad_keep_dims(HTKeepDims keep_dims, int idx) {
    _grad_keep_dims[idx] = keep_dims;
  };

  HTAxes grad_axes(int idx) const {
    return _grad_add_axes[idx];
  };

  HTKeepDims grad_keep_dims(int idx) const{
    return _grad_keep_dims[idx];
  };

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _trans_a;
  bool _trans_b;

  HTAxes _grad_add_axes[2];

  HTKeepDims _grad_keep_dims[2];
};

class MatMul2Op final : public OpWrapper<MatMul2OpDef> {
 public:
  MatMul2Op(Tensor a, Tensor b, bool trans_a = false, bool trans_b = false,
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<MatMul2OpDef>(make_ptr<MatMul2OpDef>(
      MatMul2OpDef::constrcutor_access_key(), a, b, trans_a, trans_b, op_meta)) {
  }
};

class MatMul2GradientOpDef : public OperatorDef {
 private:
  friend class MatMul2GradientOp;
  struct constrcutor_access_key {};

 public:
  MatMul2GradientOpDef(const constrcutor_access_key&, Tensor a, Tensor b, int index, Tensor dst, Tensor output,
                       bool trans_a = false, bool trans_b = false, 
                       const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MatMul2GradientOp), {a, b, dst, output}, op_meta),
    _trans_a(trans_a),
    _trans_b(trans_b),
    _index(index) {
    DoInferMeta();
  }

  inline bool trans_a() const {
    return _trans_a;
  }

  inline bool trans_b() const {
    return _trans_b;
  }

  void set_axes(HTAxes axe) {
    _add_axes = axe;
  }

  void set_keep_dims(HTKeepDims keep_dims) {
    _keep_dims = keep_dims;
  }

  HTAxes axes() const {
    return _add_axes;
  }

  HTKeepDims keep_dims() const{
    return _keep_dims;
  }

  int index() const {
    return _index;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _trans_a;
  bool _trans_b;
  int _index;

  HTAxes _add_axes;

  HTKeepDims _keep_dims;

};

class MatMul2GradientOp final : public OpWrapper<MatMul2GradientOpDef> {
 public:
  MatMul2GradientOp(Tensor a, Tensor b, int index, Tensor dst, Tensor output, bool trans_a = false, bool trans_b = false, 
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<MatMul2GradientOpDef>(make_ptr<MatMul2GradientOpDef>(
      MatMul2GradientOpDef::constrcutor_access_key(), a, b, index, dst, output, trans_a, trans_b,  op_meta)) {
  }
};


} // namespace autograd
} // namespace hetu
