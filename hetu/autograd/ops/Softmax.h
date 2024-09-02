#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SoftmaxOpDef;
class SoftmaxOp;
class SoftmaxGradientOpDef;
class SoftmaxGradientOp;

class SoftmaxOpDef : public OperatorDef {
 private:
  friend class SoftmaxOp;
  struct constrcutor_access_key {};

 public:
  SoftmaxOpDef(const constrcutor_access_key&, Tensor input, int64_t dim,
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SoftmaxOp), {input}, op_meta),
  _dim(dim) {
    DoInferMeta();
    DoDeduceStates();
  }

  int64_t get_dim() {
    return _dim;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _dim;
};

class SoftmaxOp final : public OpWrapper<SoftmaxOpDef> {
 public:
  SoftmaxOp(Tensor input, int64_t dim, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxOpDef>(make_ptr<SoftmaxOpDef>(
      SoftmaxOpDef::constrcutor_access_key(), input, dim, op_meta)) {}
};

class SoftmaxGradientOpDef : public OperatorDef {
 private:
  friend class SoftmaxGradientOp;
  struct constrcutor_access_key {};

 public:
  SoftmaxGradientOpDef(const constrcutor_access_key&, Tensor input,
                       Tensor grad_output, int64_t dim, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SoftmaxGradientOp), {input, grad_output}, op_meta),
  _dim(dim) {
    DoInferMeta();
    DoDeduceStates();
  }

  int64_t get_dim() {
    return _dim;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _dim;
};

class SoftmaxGradientOp final : public OpWrapper<SoftmaxGradientOpDef> {
 public:
  SoftmaxGradientOp(Tensor input, Tensor grad_output, int64_t dim,
                    const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxGradientOpDef>(make_ptr<SoftmaxGradientOpDef>(
      SoftmaxGradientOpDef::constrcutor_access_key(), input, grad_output, dim,
      op_meta)) {}
};

} // namespace autograd
} // namespace hetu
