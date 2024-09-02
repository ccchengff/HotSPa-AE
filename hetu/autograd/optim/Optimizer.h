#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/ops/Variable.h"

namespace hetu {
namespace autograd {

using GradAndVar = std::pair<Tensor, Tensor>;
using GradAndVarList = std::vector<GradAndVar>;

// Question: Should we decouple the define-and-run and define-by-run modes?
class Optimizer {
 public:
  Optimizer(float learning_rate) : _learning_rate(learning_rate) {}

  Optimizer(const TensorList& vars, float learning_rate)
  : _vars(vars), _learning_rate(learning_rate) {
    HT_VALUE_ERROR_IF(_vars.empty()) << "No variables are provided";
    for (auto& var : _vars) {
      HT_ASSERT(var->is_trainable())
        << "Tensor " << var << " is not a trainable variable";
    }
  }

  Tensor Minimize(const Tensor& loss, const TensorList& var_list = {},
                  const Tensor& grad_loss = {}, const OpName& name = "");

  virtual GradAndVarList ComputeGradients(const Tensor& loss,
                                          const TensorList& var_list = {},
                                          const Tensor& grad_loss = {});

  virtual Tensor ApplyGradients(const GradAndVarList& grads_and_vars,
                                const OpName& name = "");

  void ZeroGrad();

  void Step();

  float learning_rate() const {
    return _learning_rate;
  }

 protected:
  virtual Tensor ApplyDense(const GradAndVar& grad_and_var) = 0;

  virtual Tensor MakeStates(const Tensor& variable, const OpName& state_name);

  TensorList _vars;
  float _learning_rate;
};

class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer(float learning_rate, float momentum = 0.9f,
               bool nesterov = false)
  : Optimizer(learning_rate) {
    _init(momentum, nesterov);
  }

  SGDOptimizer(const TensorList& vars, float learning_rate,
               float momentum = 0.9f, bool nesterov = false)
  : Optimizer(vars, learning_rate) {
    _init(momentum, nesterov);
  }

  Tensor ApplyDense(const GradAndVar& grad_and_var);

  float momentum() const {
    return _momentum;
  }

  bool nesterov() const {
    return _nesterov;
  }

 protected:
  void _init(float momentum, bool nesterov) {
    HT_ASSERT(momentum >= 0 && momentum <= 1)
      << "Invalid momemtum: " << momentum;
    _momentum = momentum;
    _nesterov = nesterov;
  }

  float _momentum;
  bool _nesterov;
};

} // namespace autograd
} // namespace hetu
