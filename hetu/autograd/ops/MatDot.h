#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class MatDotOpDef;
class MatDotOp;

class MatDotOpDef : public OperatorDef {
 private:
  friend class MatDotOp;
  struct constrcutor_access_key {};

 public:
  MatDotOpDef(const constrcutor_access_key&, Tensor a, Tensor b,
              int64_t axes = 0, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MatDotOp), {a, b}, op_meta) {
    DoInferMeta();
    DoDeduceStates();
  }

  int64_t get_axes() const {
    return _axes;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _axes;
};

class MatDotOp final : public OpWrapper<MatDotOpDef> {
 public:
  MatDotOp(Tensor a, Tensor b, int64_t axes = 0,
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<MatDotOpDef>(make_ptr<MatDotOpDef>(
      MatDotOpDef::constrcutor_access_key(), a, b, axes, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
