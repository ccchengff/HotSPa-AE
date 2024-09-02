#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class ReluOpDef;
class ReluOp;
class ReluGradientOpDef;
class ReluGradientOp;

class ReluOpDef : public OperatorDef {
 private:
  friend class ReluOp;
  struct constrcutor_access_key {};

 public:
  ReluOpDef(const constrcutor_access_key&, Tensor input,
            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReluOp), {input}, op_meta) {
    DoInferMeta();
    DoDeduceStates();
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class ReluOp final : public OpWrapper<ReluOpDef> {
 public:
  ReluOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReluOpDef>(make_ptr<ReluOpDef>(
      ReluOpDef::constrcutor_access_key(), input, op_meta)) {}
};

class ReluGradientOpDef : public OperatorDef {
 private:
  friend class ReluGradientOp;
  struct constrcutor_access_key {};

 public:
  ReluGradientOpDef(const constrcutor_access_key&, Tensor input,
                    Tensor grad_output, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReluGradientOp), {input, grad_output}, op_meta) {
    DoInferMeta();
    DoDeduceStates();
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class ReluGradientOp final : public OpWrapper<ReluGradientOpDef> {
 public:
  ReluGradientOp(Tensor input, Tensor grad_output,
                 const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReluGradientOpDef>(
      make_ptr<ReluGradientOpDef>(ReluGradientOpDef::constrcutor_access_key(),
                                  input, grad_output, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
