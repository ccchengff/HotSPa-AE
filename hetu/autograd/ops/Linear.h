#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class LinearOpDef;
class LinearOp;

class LinearOpDef : public OperatorDef {
 private:
  friend class LinearOp;
  struct constrcutor_access_key {};

 public:
  LinearOpDef(const constrcutor_access_key&, Tensor a, Tensor b, Tensor bias,
              bool trans_a = false, bool trans_b = true,
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LinearOp), {a, b, bias}, op_meta),
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

class LinearOp final : public OpWrapper<LinearOpDef> {
 public:
  LinearOp(Tensor a, Tensor b, Tensor bias, bool trans_a = false,
           bool trans_b = true, const OpMeta& op_meta = OpMeta())
  : OpWrapper<LinearOpDef>(
      make_ptr<LinearOpDef>(LinearOpDef::constrcutor_access_key(), a, b, bias,
                            trans_a, trans_b, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
