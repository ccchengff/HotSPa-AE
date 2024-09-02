#pragma once

#include "hetu/graph/headers.h"
#include "hetu/graph/ops/variable.h"

namespace hetu {
namespace graph {

class Optimizer {
 public:
  Optimizer() {};

  Optimizer(float learning_rate) : _learning_rate(learning_rate) {}

  Optimizer(TensorList params, float learning_rate)
  : _params(std::move(params)), _learning_rate(learning_rate) {
    HT_VALUE_ERROR_IF(_params.empty()) << "No parameters are provided";
    for (auto& param : _params) {
      HT_VALUE_ERROR_IF(!param->is_parameter())
        << "Tensor " << param << " is not a parameter";
    }
  }

  Tensor Minimize(const Tensor& loss, const TensorList& var_list = {},
                  const Tensor& grad_loss = {}, const OpName& name = "");

  virtual GradAndVarList ComputeGradients(const Tensor& loss,
                                          const TensorList& var_list = {},
                                          const Tensor& grad_loss = {});

  virtual Tensor ApplyGradients(const GradAndVarList& grads_and_vars,
                                const OpName& name = OpName(),
                                const Tensor& infinite_count = Tensor());

  float learning_rate() const {
    return _learning_rate;
  }

 protected:
  virtual Tensor ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count = Tensor()) { return Tensor(); }

  virtual Tensor MakeStates(const Tensor& variable, const Tensor& grad, const OpName& state_name);

  TensorList _params;
  float _learning_rate;
};

class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer(): Optimizer() {};

  SGDOptimizer(float learning_rate, float momentum = 0.9f,
               bool nesterov = false)
  : Optimizer(learning_rate) {
    _init(momentum, nesterov);
  }

  SGDOptimizer(TensorList params, float learning_rate, float momentum = 0.9f,
               bool nesterov = false)
  : Optimizer(std::move(params), learning_rate) {
    _init(momentum, nesterov);
  }

  Tensor ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count = Tensor());

  float momentum() const {
    return _momentum;
  }

  bool nesterov() const {
    return _nesterov;
  }

 protected:
  void _init(float momentum, bool nesterov) {
    HT_VALUE_ERROR_IF(momentum < 0 || momentum > 1)
      << "Invalid momemtum: " << momentum;
    _momentum = momentum;
    _nesterov = nesterov;
  }

  float _momentum;
  bool _nesterov;
};

class AdamOptimizer : public Optimizer {
 public:
  AdamOptimizer(): Optimizer() {};

  AdamOptimizer(float learning_rate, float beta1 = 0.9,
                float beta2 = 0.999, float eps = 1e-8,
                float weight_decay = 0)
  : Optimizer(learning_rate) {
    _init(beta1, beta2, eps, weight_decay);
  }

  AdamOptimizer(TensorList params, float learning_rate, float beta1 = 0.9,
                float beta2 = 0.999, float eps = 1e-8,
                float weight_decay = 0)
  : Optimizer(std::move(params), learning_rate) {
    _init(beta1, beta2, eps, weight_decay);
  }

  Tensor ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count = Tensor());

  float beta1() const {
    return _beta1;
  }

  float beta2() const {
    return _beta2;
  }

  float eps() const {
    return _eps;
  }

  float weight_decay() const {
    return _weight_decay;
  }

 protected:
  void _init(float beta1, float beta2, float eps,
             float weight_decay) {
    HT_VALUE_ERROR_IF(beta1 < 0 || beta1 > 1)
      << "Invalid beta1: " << beta1;
    HT_VALUE_ERROR_IF(beta2 < 0 || beta1 > 2)
      << "Invalid beta2: " << beta2;
    _beta1 = beta1;
    _beta2 = beta2;
    _eps = eps;
    _weight_decay = weight_decay;
  }

  float _beta1;
  float _beta2;
  float _eps;
  float _weight_decay;
};

} // namespace graph
} // namespace hetu
