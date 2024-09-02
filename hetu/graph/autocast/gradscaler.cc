#include "hetu/graph/autocast/gradscaler.h"
#include "hetu/graph/ops/op_headers.h"

namespace hetu {
namespace graph {

void GradScaler::_Init(const Device& dev) {
  _scale = MakeVariableOp(NDArray::full({1}, _init_scale, dev == kUndeterminedDevice ? kCUDA : dev, DataType::FLOAT32));
  _growth_tracker = MakeVariableOp(NDArray::full({1}, 0, dev == kUndeterminedDevice ? kCUDA : dev, DataType::INT32));
  HT_LOG_INFO << dev << " " << _scale->device() << " " << _scale->dtype();
}

Tensor GradScaler::scale(Tensor output) {
  if (!_enabled)
    return output;
  if (_scale == Tensor()) {
    _Init(output->device());
  }
  HT_ASSERT(_scale != Tensor());
  return MakeMulElewiseOp(output, MakeDataTransferOp(output->dtype(), _scale, output->device()));
}

TensorList GradScaler::scale(TensorList outputs) {
  TensorList outputlist = {};
  if (_scale == Tensor()) {
    _Init(outputs[0]->device());
  }
  MultiDeviceTensor scales(_scale);
  for (auto& output: outputs) {
    HT_ASSERT(output->device().is_cuda())
    << "Cpu is not supported.";
    outputlist.push_back(MakeMulElewiseOp(output, scales.fetch(output->device())));
  }
  return outputlist;
}

Tensor GradScaler::minimize(SGDOptimizer op, const Tensor& loss, const TensorList& var_list,
                            const Tensor& grad_loss) {
  if (!_enabled)
    return op.Minimize(loss, var_list, grad_loss, "scaler");
  if (_scale == Tensor()) {
    HT_LOG_INFO << loss->device();
    _Init(loss->device());
  }
  HT_ASSERT(_scale != Tensor())
  << "scale should not be empty";
  HT_ASSERT(_growth_tracker != Tensor())
  << "growth_tracker should not be empty";  
  return unscale(op, loss, var_list, grad_loss);
}

Tensor GradScaler::unscale(SGDOptimizer op, const Tensor& loss, const TensorList& var_list,
                           const Tensor& grad_loss) {
  HT_ASSERT(_scale != Tensor())
  << "scale should not be empty";
  HT_LOG_INFO << _scale->device();
  Tensor inv_scale =MakeReciprocalOp(_scale);
  Tensor found_inf = MakeVariableOp(NDArray::full({1}, 0.0, _scale->device(), DataType::FLOAT32));
  Tensor grad_loss_ = grad_loss;
  if (grad_loss_ == Tensor()) {
    if (_scale->dtype() != loss->dtype() || _scale->device() != loss->device())
      grad_loss_ = MakeBroadcastOp(MakeDataTransferOp(loss->dtype(), _scale, loss->device()), loss);
    else
      grad_loss_ = MakeBroadcastOp(_scale, loss);
  } 
  GradAndVarList grads_and_vars = op.ComputeGradients(loss, var_list, grad_loss_);
  GradAndVarList filtered_grads_and_vars;
  filtered_grads_and_vars.reserve(grads_and_vars.size());
  std::copy_if(grads_and_vars.begin(), grads_and_vars.end(),
               std::back_inserter(filtered_grads_and_vars),
               [](const GradAndVar& n) { return n.first.is_defined(); });
  _check_grads = {};
  for (auto& grad_and_var: filtered_grads_and_vars) {
    auto checked = MakeCheckFiniteOp(grad_and_var.first);
    _check_grads.push_back(checked);
  }
  _infinite_count = MakeSumOp(_check_grads);
  // sum_checked = NDArray::to(sum_checked, kCPU);
  // _infinite_count = sum_checked->data_ptr<float>()[0];
  // HT_LOG_INFO << _infinite_count;
  // if (_infinite_count) {
  //   HT_LOG_INFO << "INF";
  //   return Tensor();
  // }
  std::transform(filtered_grads_and_vars.begin(),
                 filtered_grads_and_vars.end(),
                 filtered_grads_and_vars.begin(),
                 [&inv_scale] (GradAndVar& n) {
                   n.first = MakeMulElewiseOp(n.first, inv_scale);
                   return n;
                 });
  return op.ApplyGradients(filtered_grads_and_vars, "scaler", _infinite_count);
}

Tensor GradScaler::update(float new_scale) {
  if (!_enabled)
    return _scale;
  if (new_scale > 0) {
    _scale = MakeScalarsLikeOp(_scale, new_scale);
  }
  else {
    auto update = MakeUpdateScaleOp(_scale, _growth_tracker, _infinite_count,
                                    _growth_factor, _backoff_factor,
                                    _growth_interval);
    _scale = update[0];
    _growth_tracker = update[1];
  }
  return _scale;
}

} // namespace graph
} // namespace hetu