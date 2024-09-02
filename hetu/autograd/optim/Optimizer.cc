#include "hetu/autograd/optim/Optimizer.h"
#include "hetu/autograd/autograd.h"
#include "hetu/autograd/topo.h"
#include "hetu/autograd/ops/Optimizer.h"
#include "hetu/autograd/ops/Group.h"

namespace hetu {
namespace autograd {

Tensor Optimizer::Minimize(const Tensor& loss, const TensorList& var_list,
                           const Tensor& grad_loss, const OpName& name) {
  GradAndVarList grads_and_vars = ComputeGradients(loss, var_list, grad_loss);
  GradAndVarList filtered_grads_and_vars;
  filtered_grads_and_vars.reserve(grads_and_vars.size());
  std::copy_if(grads_and_vars.begin(), grads_and_vars.end(),
               std::back_inserter(filtered_grads_and_vars),
               [](const GradAndVar& n) { return n.first.is_defined(); });
  return ApplyGradients(filtered_grads_and_vars, name);
}

Tensor Optimizer::ApplyGradients(const GradAndVarList& grads_and_vars,
                                 const OpName& name) {
  TensorList update_edges;
  update_edges.reserve(grads_and_vars.size());
  std::transform(
    grads_and_vars.begin(), grads_and_vars.end(),
    std::back_inserter(update_edges),
    [&](const GradAndVar& grad_and_var) { return ApplyDense(grad_and_var); });
  return GroupOp(update_edges, OpMeta().set_name(name))->out_dep_linker();
}

void Optimizer::ZeroGrad() {
  for (auto& var : _vars)
    var->ZeroGrad();
}

void Optimizer::Step() {
  TensorList update_edges;
  update_edges.reserve(_vars.size());
  for (size_t i = 0; i < _vars.size(); i++) {
    auto& var = _vars[i];
    auto& grad = var->Gradient();
    if (!grad.is_defined())
      continue;
    auto update_edge = ApplyDense({grad, var});
    update_edges.push_back(update_edge);
  }
  auto update_op = GroupOp(update_edges)->out_dep_linker();
  update_op->GetOrCompute();
}

Tensor Optimizer::MakeStates(const Tensor& variable, const OpName& state_name) {
  auto var = reinterpret_cast<const VariableOp&>(variable->producer());
  return VariableOp(variable->shape(), ZerosInitializer(), variable->dtype(),
                    false,
                    OpMeta()
                      .set_device_group(var->device_group())
                      .set_name(var->name() + "_" + state_name))
    ->output(0);
}

GradAndVarList Optimizer::ComputeGradients(const Tensor& loss,
                                           const TensorList& var_list,
                                           const Tensor& grad_loss) {
  TensorList vars = var_list;
  if (vars.empty()) {
    auto topo_order = TopoSort(loss);
    vars.reserve(topo_order.size());
    for (auto& op : topo_order)
      if (is_trainable_op(op))
        vars.push_back(op->output(0));
  } else {
    for (const auto& var : vars)
      HT_ASSERT(is_trainable_op(var->producer()))
        << "Operator \"" << var->producer()->name() << "\" is not trainable";
  }
  TensorList grads = Gradients(loss, vars, grad_loss);
  HT_ASSERT_EQ(grads.size(), vars.size())
    << "Only " << grads.size() << " gradients are returned for " << vars.size()
    << " variables.";
  GradAndVarList grads_and_vars;
  grads_and_vars.reserve(grads.size());
  for (size_t i = 0; i < grads.size(); i++)
    grads_and_vars.emplace_back(grads[i], vars[i]);
  return grads_and_vars;
}

Tensor SGDOptimizer::ApplyDense(const GradAndVar& grad_and_var) {
  Tensor grad = grad_and_var.first;
  Tensor var = grad_and_var.second;
  auto update_op_meta = OpMeta()
                          .set_device_group(var->producer()->device_group())
                          .set_name("Update_" + var->name());
  if (momentum() == 0) {
    return SGDUpdateOp(var, grad, learning_rate(), update_op_meta)
      ->out_dep_linker();
  } else {
    return MomemtumUpdateOp(var, grad, MakeStates(var, "velocity"),
                            learning_rate(), momentum(), nesterov(),
                            update_op_meta)
      ->out_dep_linker();
  }
}

} // namespace autograd
} // namespace hetu
