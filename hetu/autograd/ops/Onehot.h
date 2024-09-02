#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class OnehotOpDef;
class OnehotOp;

class OnehotOpDef : public OperatorDef {
 private:
  friend class OnehotOp;
  struct constrcutor_access_key {};

 public:
  OnehotOpDef(const constrcutor_access_key&, Tensor input, size_t num_classes,
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(OnehotOp), {input}, op_meta), _classes(num_classes) {
    DoInferMeta();
    DoDeduceStates();
  }

  size_t num_classes() const {
    return _classes;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _classes;
};

class OnehotOp final : public OpWrapper<OnehotOpDef> {
 public:
  OnehotOp(Tensor input, size_t num_classes, const OpMeta& op_meta = OpMeta())
  : OpWrapper<OnehotOpDef>(make_ptr<OnehotOpDef>(
      OnehotOpDef::constrcutor_access_key(), input, num_classes, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
