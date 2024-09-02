#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SinOpDef;
class SinOp;
class CosOpDef;
class CosOp;

class SinOpDef : public OperatorDef {
 private:
  friend class SinOp;
  struct constrcutor_access_key {};

 public:
  SinOpDef(const constrcutor_access_key&, Tensor input,
            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SinOp), {input}, op_meta) {
    DoInferMeta();
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class SinOp final : public OpWrapper<SinOpDef> {
 public:
  SinOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SinOpDef>(make_ptr<SinOpDef>(
      SinOpDef::constrcutor_access_key(), input, op_meta)) {}
};

class CosOpDef : public OperatorDef {
 private:
  friend class CosOp;
  struct constrcutor_access_key {};

 public:
  CosOpDef(const constrcutor_access_key&, Tensor input,
           const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(CosOp), {input}, op_meta) {
    DoInferMeta();
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class CosOp final : public OpWrapper<CosOpDef> {
 public:
  CosOp(Tensor input,
        const OpMeta& op_meta = OpMeta())
  : OpWrapper<CosOpDef>(
      make_ptr<CosOpDef>(CosOpDef::constrcutor_access_key(),
                                  input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
