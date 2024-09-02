#pragma once

#include "hetu/graph/headers.h"
#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/optim/optimizer.h"

namespace hetu {
namespace graph {

struct MultiDeviceTensor{
  Tensor _origin;
  std::unordered_map<Device, Tensor> multitensors;
  MultiDeviceTensor(Tensor origin) 
  : _origin(origin) {
  }

  Tensor fetch(const Device& dev) {
    auto it = multitensors.find(dev);
    if (it != multitensors.end())
      return it->second;
    else {
      multitensors.insert(std::pair<Device, Tensor>(dev, 
                          MakeDataTransferOp(_origin->dtype(), _origin, dev)));
      return multitensors.find(dev)->second;
    }
  }
};

class GradScaler {
public:
  GradScaler(float init_scale = 65536.0f, double growth_factor = 2.0,
             double backoff_factor = 0.5, int growth_interval = 2000,
             bool enabled = true) : _init_scale(init_scale), 
             _growth_factor(growth_factor), _backoff_factor(backoff_factor), 
             _growth_interval(growth_interval), _enabled(enabled) {
    HT_ASSERT(growth_factor > 1.0) 
    << "growth_factor must > 1.0.";
    HT_ASSERT(backoff_factor < 1.0)
    << "backoff_factor must < 1.0.";
  }

  TensorList scale(TensorList outputs);

  Tensor scale(Tensor output);

  Tensor minimize(SGDOptimizer op, const Tensor& loss, const TensorList& var_list = {},
                  const Tensor& grad_loss = {});
  
  Tensor update(float new_scale = 0.f);

protected:
  void _Init(const Device& dev);
  Tensor unscale(SGDOptimizer op, const Tensor& loss, const TensorList& var_list,
                 const Tensor& grad_loss);

private:
  float _init_scale;
  double _growth_factor;
  double _backoff_factor;
  int _growth_interval;
  bool _enabled;
  Tensor _scale;
  Tensor _growth_tracker;
  TensorList _check_grads;
  Tensor _infinite_count;
};

} // namespace graph
} // namespace hetu
