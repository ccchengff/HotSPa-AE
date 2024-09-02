#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SqrtOpDef;
class SqrtOp;
class ReciprocalSqrtOpDef;
class ReciprocalSqrtOp;

class SqrtOpDef : public OperatorDef {
 private:
  friend class SqrtOp;
  struct constrcutor_access_key {};

 public:
  SqrtOpDef(const constrcutor_access_key&, Tensor input,
            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SqrtOp), {input}, op_meta) {
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

class SqrtOp final : public OpWrapper<SqrtOpDef> {
 public:
  SqrtOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SqrtOpDef>(make_ptr<SqrtOpDef>(
      SqrtOpDef::constrcutor_access_key(), input, op_meta)) {}
};

class ReciprocalSqrtOpDef : public OperatorDef {
 private:
  friend class ReciprocalSqrtOp;
  struct constrcutor_access_key {};

 public:
  ReciprocalSqrtOpDef(const constrcutor_access_key&, Tensor grad_output,
                      const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReciprocalSqrtOp), {grad_output}, op_meta) {
    DoInferMeta();
    DoDeduceStates();
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class ReciprocalSqrtOp final : public OpWrapper<ReciprocalSqrtOpDef> {
 public:
  ReciprocalSqrtOp(Tensor grad_output, const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReciprocalSqrtOpDef>(make_ptr<ReciprocalSqrtOpDef>(
      ReciprocalSqrtOpDef::constrcutor_access_key(), grad_output, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
