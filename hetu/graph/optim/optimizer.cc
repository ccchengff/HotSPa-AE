#include "hetu/graph/optim/optimizer.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/ones_like.h"
#include "hetu/graph/ops/optimizer_update.h"

namespace hetu {
namespace graph {

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
                                 const OpName& name, const Tensor& infinite_count) {
  TensorList updated_params;
  updated_params.reserve(grads_and_vars.size());
  std::transform(
    grads_and_vars.begin(), grads_and_vars.end(),
    std::back_inserter(updated_params),
    [&](const GradAndVar& grad_and_var) { return ApplyDense(grad_and_var, infinite_count); });
  return MakeGroupOp(OpMeta().set_extra_deps(updated_params)
                      .set_name(name).set_is_deduce_states(false)); // group op needn't deduce states
}

// the distributed states for adam mean/variance was dummy, just do allgather for dp groups in later execution
Tensor Optimizer::MakeStates(const Tensor& variable, const Tensor& grad, const OpName& state_name) {
  const auto& producer = variable->producer();
  HT_VALUE_ERROR_IF(!producer->is_parameter());
  // special case: Varibale States should be set distributed_states as ds_grad (whether zero or not)
  // const DistributedStates& ds_variable = variable->get_distributed_states(); 
  // const DistributedStates& ds_grad = grad->get_distributed_states();
  const auto& variable_ds_hierarchy = variable->ds_hierarchy(); 
  const auto& grad_ds_hierarchy = grad->ds_hierarchy(); 
  // HT_ASSERT (ds_variable.is_valid() && ds_grad.is_valid()) 
  //   << "Diastributed States for varibale " << variable << " must be valid!";  
  // HT_LOG_INFO << variable->name() + "_" + state_name << " directly use grad " << grad << ": " << grad_ds_hierarchy.get(1).ds_union_info();
  Tensor states = MakeParallelVariableOp(ZerosInitializer(), grad->global_shape(),
                                         grad_ds_hierarchy, {0}, grad->dtype(), false,
                                         OpMeta()
                                          .set_device_group_hierarchy(producer->device_group_hierarchy())
                                          .set_eager_device(producer->eager_device())
                                          .set_name(variable->name() + "_" + state_name));  
  Graph::MarkAsOptimizerVariable(states);
  return std::move(states);
}

GradAndVarList Optimizer::ComputeGradients(const Tensor& loss,
                                           const TensorList& var_list,
                                           const Tensor& grad_loss) {
  TensorList vars = var_list;
  if (vars.empty()) {
    auto topo_order = Graph::TopoSort(loss);
    for (auto& op_ref : topo_order)
      if (op_ref.get()->is_parameter())
        vars.push_back(op_ref.get()->output(0));
  }
  TensorList grads = Graph::Gradients(loss, vars, grad_loss);
  GradAndVarList grads_and_vars;
  grads_and_vars.reserve(grads.size());
  for (size_t i = 0; i < grads.size(); i++)
    grads_and_vars.emplace_back(grads[i], vars[i]);
  return grads_and_vars;
}

Tensor SGDOptimizer::ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count) {
  const Tensor& grad = grad_and_var.first;
  const Tensor& var = grad_and_var.second;
  auto update_op_meta = OpMeta()
                          .set_device_group_hierarchy(var->producer()->device_group_hierarchy())
                          .set_name("Update_" + var->name())
                          .set_is_deduce_states(false);
  if (momentum() == 0) {
    if (infinite_count != Tensor())
      return MakeSGDUpdateWithGradScalerOp(var, grad, infinite_count, learning_rate(), update_op_meta);
    return MakeSGDUpdateOp(var, grad, learning_rate(), update_op_meta);
  } else {
    return MakeMomentumUpdateOp(var, grad, MakeStates(var, grad, "velocity"),
                                learning_rate(), momentum(), nesterov(),
                                update_op_meta);
  }
}

Tensor AdamOptimizer::ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count) {
  const Tensor& grad = grad_and_var.first;
  const Tensor& var = grad_and_var.second;
  auto update_op_meta = OpMeta()
                          .set_device_group_hierarchy(var->producer()->device_group_hierarchy())
                          .set_name("Update_" + var->name())
                          .set_is_deduce_states(false); // update op needn't deduce states
  HTShape step_shape = {1};
  Tensor step = MakeVariableOp(OnesInitializer(), step_shape, kInt64,
                                false, var->ds_hierarchy(), 
                                OpMeta()
                                  .set_device_group_hierarchy(var->producer()->device_group_hierarchy())
                                  .set_eager_device(kCPU)
                                  .set_name(var->name() + "_step")
                                  .set_is_step(true));
  // variable: dup in dp group, grad: reduce-scatter in dp group, mean & variance: same as grad
  return MakeAdamOp(var, grad, MakeStates(var, grad, "mean"),
                    MakeStates(var, grad, "variance"),
                    learning_rate(), step, beta1(), beta2(),
                    eps(), weight_decay(), update_op_meta);
}

} // namespace graph
} // namespace hetu