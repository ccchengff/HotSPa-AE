#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class TanhOpDef;
class TanhOp;
class TanhGradientOpDef;
class TanhGradientOp;

class TanhOpDef : public OperatorDef {
 private:
  friend class TanhOp;
  struct constrcutor_access_key {};

 public:
  TanhOpDef(const constrcutor_access_key&, Tensor input,
            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(TanhOp), {input}, op_meta) {
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

class TanhOp final : public OpWrapper<TanhOpDef> {
 public:
  TanhOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<TanhOpDef>(make_ptr<TanhOpDef>(
      TanhOpDef::constrcutor_access_key(), input, op_meta)) {}
};

class TanhGradientOpDef : public OperatorDef {
 private:
  friend class TanhGradientOp;
  struct constrcutor_access_key {};

 public:
  TanhGradientOpDef(const constrcutor_access_key&, Tensor input,
                    Tensor grad_output, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(TanhGradientOp), {input, grad_output}, op_meta) {
    DoInferMeta();
    DoDeduceStates();
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class TanhGradientOp final : public OpWrapper<TanhGradientOpDef> {
 public:
  TanhGradientOp(Tensor input, Tensor grad_output,
                 const OpMeta& op_meta = OpMeta())
  : OpWrapper<TanhGradientOpDef>(
      make_ptr<TanhGradientOpDef>(TanhGradientOpDef::constrcutor_access_key(),
                                  input, grad_output, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
