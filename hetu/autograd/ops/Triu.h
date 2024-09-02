#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class TriuTrilOpDef;
class TriuTrilOp;

class TriuTrilOpDef : public OperatorDef {
 private:
  friend class TriuTrilOp;
  struct constrcutor_access_key {};

 public:
  TriuTrilOpDef(const constrcutor_access_key&, Tensor input,
                bool lower = false, int64_t diagonal = 0,
                const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(TriuTrilOp), {input}, op_meta),
  _lower(lower),
  _diagonal(diagonal) {
    DoInferMeta();
  }

  inline bool lower() const {
    return _lower;
  }

  inline int64_t diagonal() const {
    return _diagonal;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _lower;

  int64_t _diagonal;
};

class TriuTrilOp final : public OpWrapper<TriuTrilOpDef> {
 public:
  TriuTrilOp(Tensor input,  bool lower = false, int64_t diagonal = 0,  
             const OpMeta& op_meta = OpMeta())
  : OpWrapper<TriuTrilOpDef>(make_ptr<TriuTrilOpDef>(
      TriuTrilOpDef::constrcutor_access_key(), input, lower, diagonal, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
