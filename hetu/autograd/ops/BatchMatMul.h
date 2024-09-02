#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class BatchMatMulOpDef;
class BatchMatMulOp;

class BatchMatMulOpDef : public OperatorDef {
 private:
  friend class BatchMatMulOp;
  struct constrcutor_access_key {};

 public:
  BatchMatMulOpDef(const constrcutor_access_key&, Tensor a, Tensor b,
                   bool trans_a = false, bool trans_b = false,
                   const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchMatMulOp), {a, b}, op_meta),
    _trans_a(trans_a),
    _trans_b(trans_b) {
    DoInferMeta();
    DoDeduceStates();
  }

  inline bool trans_a() const {
    return _trans_a;
  }

  inline bool trans_b() const {
    return _trans_b;
  }

 protected:  
  void DoInferMeta() override;

  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _trans_a;
  bool _trans_b;
};

class BatchMatMulOp final : public OpWrapper<BatchMatMulOpDef> {
 public:
  BatchMatMulOp(Tensor a, Tensor b, bool trans_a = false, bool trans_b = false,
                const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchMatMulOpDef>(
      make_ptr<BatchMatMulOpDef>(BatchMatMulOpDef::constrcutor_access_key(), a,
                                 b, trans_a, trans_b, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
