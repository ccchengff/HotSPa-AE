#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class GatherOpDef;
class GatherOp;
class GatherGradientOpDef;
class GatherGradientOp;

class GatherOpDef : public OperatorDef {
 private:
  friend class GatherOp;
  struct constrcutor_access_key {};

 public:
  GatherOpDef(const constrcutor_access_key&, Tensor input, int64_t dim, 
              Tensor id, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(GatherOp), {input, id}, op_meta), _dim(dim) {
    DoInferMeta();
  }

  int64_t get_dim() {
    return _dim;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _dim;
};

class GatherOp final : public OpWrapper<GatherOpDef> {
 public:
  GatherOp(Tensor input, int64_t dim, Tensor id, const OpMeta& op_meta = OpMeta())
  : OpWrapper<GatherOpDef>(make_ptr<GatherOpDef>(
      GatherOpDef::constrcutor_access_key(), input, dim, id, op_meta)) {}
};

class GatherGradientOpDef : public OperatorDef {
 private:
  friend class GatherGradientOp;
  struct constrcutor_access_key {};

 public:
  GatherGradientOpDef(const constrcutor_access_key&,
                      Tensor grad_output, int64_t dim, Tensor id, Tensor input,
                      const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(GatherGradientOp), {grad_output, id, input}, op_meta),
  _dim(dim) {
    DoInferMeta();
  }

  int64_t get_dim() {
    return _dim;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _dim;
};

class GatherGradientOp final
: public OpWrapper<GatherGradientOpDef> {
 public:
  GatherGradientOp(Tensor grad_output, int64_t dim, Tensor id, Tensor input,
                   const OpMeta& op_meta = OpMeta())
  : OpWrapper<GatherGradientOpDef>(
      make_ptr<GatherGradientOpDef>(
        GatherGradientOpDef::constrcutor_access_key(), grad_output, dim, id, input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
