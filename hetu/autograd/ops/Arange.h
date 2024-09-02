#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class ArangeOpDef;
class ArangeOp;

class ArangeOpDef : public OperatorDef {
 private:
  friend class ArangeOp;
  struct constrcutor_access_key {};

 public:
  ArangeOpDef(const constrcutor_access_key&, double start,
              double end, double step, 
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ArangeOp), {}, op_meta),
  _start(start),
  _end(end),
  _step(step) {
    DoInferMeta();
  }

  inline double start() const {
    return _start;
  }

  inline double end() const {
    return _end;
  }

  inline double step() const {
    return _step;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _start;

  double _end;

  double _step;

};

class ArangeOp final : public OpWrapper<ArangeOpDef> {
 public:
  ArangeOp(double start, double end, double step,
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<ArangeOpDef>(make_ptr<ArangeOpDef>(
      ArangeOpDef::constrcutor_access_key(), start, end, step, op_meta)) {}
};


} // namespace autograd
} // namespace hetu
