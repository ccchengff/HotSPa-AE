#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/autograd/ops/Group.h"

namespace hetu {
namespace autograd {

class OptimizerUpdateOpDef;
class OptimizerUpdateOp;
class SGDUpdateOpDef;
class SGDUpdateOp;
class MomemtumUpdateOpDef;
class MomemtumUpdateOp;

class OptimizerUpdateOpDef : public OperatorDef {
 protected:
  OptimizerUpdateOpDef(OpType&& op_type, const TensorList& inputs,
                       float learning_rate, const OpMeta& op_meta = OpMeta())
  : OperatorDef(std::move(op_type), inputs, op_meta),
    _learning_rate(learning_rate) {
    auto& var_op = _inputs[0]->producer();
    HT_ASSERT(is_variable_op(var_op))
      << "Failed to construct the \"" << type() << "\" operation "
      << "(with name \"" << name() << "\"): "
      << "The first input is expected to be a variable. "
      << "Got \"" << var_op->type() << "\" "
      << "(with name \"" << var_op->name() << "\").";
    HT_ASSERT(reinterpret_cast<const VariableOp&>(var_op)->trainable())
      << "Failed to construct the \"" << type() << "\" operation "
      << "(with name \"" << name() << "\"): "
      << "The first input (with name \"" << var_op->name() << "\") "
      << "is not trainable.";
  }

 public:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group);

  float learning_rate() const {
    return _learning_rate;
  }

  uint64_t op_indicator() const noexcept {
    return OPTIMIZER_UPDATE_OP;
  }

 protected:
  float _learning_rate;
};

class SGDUpdateOpDef : public OptimizerUpdateOpDef {
 private:
  friend class SGDUpdateOp;
  struct constrcutor_access_key {};

 public:
  SGDUpdateOpDef(const constrcutor_access_key&, const Tensor& param,
                 const Tensor& grad, float learning_rate,
                 const OpMeta& op_meta = OpMeta())
  : OptimizerUpdateOpDef(quote(SGDUpdateOp), {param, grad}, learning_rate,
                         op_meta) {}

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;
};

class SGDUpdateOp final : public OpWrapper<SGDUpdateOpDef> {
 public:
  SGDUpdateOp(Tensor param, Tensor grad, float learning_rate,
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<SGDUpdateOpDef>(
      make_ptr<SGDUpdateOpDef>(SGDUpdateOpDef::constrcutor_access_key(), param,
                               grad, learning_rate, op_meta)) {}
};

class MomemtumUpdateOpDef : public OptimizerUpdateOpDef {
 private:
  friend class MomemtumUpdateOp;
  struct constrcutor_access_key {};

 public:
  MomemtumUpdateOpDef(const constrcutor_access_key&, Tensor param, Tensor grad,
                      Tensor velocity, float learning_rate, float momentum,
                      bool nesterov, const OpMeta& op_meta = OpMeta())
  : OptimizerUpdateOpDef(quote(MomemtumUpdateOp), {param, grad, velocity},
                         learning_rate, op_meta),
    _momentum(momentum),
    _nesterov(nesterov) {}

  inline float momentum() const {
    return _momentum;
  }

  inline bool nesterov() const {
    return _nesterov;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  float _momentum;
  bool _nesterov;
};

class MomemtumUpdateOp final : public OpWrapper<MomemtumUpdateOpDef> {
 public:
  MomemtumUpdateOp(Tensor param, Tensor grad, Tensor velocity,
                   float learning_rate, float momentum, bool nesterov,
                   const OpMeta& op_meta = OpMeta())
  : OpWrapper<MomemtumUpdateOpDef>(make_ptr<MomemtumUpdateOpDef>(
      MomemtumUpdateOpDef::constrcutor_access_key(), param, grad, velocity,
      learning_rate, momentum, nesterov, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
