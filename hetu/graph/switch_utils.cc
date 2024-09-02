#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/ops/variable.h"

namespace hetu {
namespace graph {

// currenty deprecated
// not support hetero APIs
/*
// temp utils of changing ds
static DistributedStates dup2split(const DistributedStates& ds, int32_t split_dim) {
  const auto& states = ds.get_states();
  std::unordered_map<int32_t, int32_t> new_states;
  for (const auto& kv : states) {
    if (kv.first == -2) {
      HT_ASSERT(kv.second == 1)
        << "partial shouldn't exist";
    } else if (kv.first == -1) {
      HT_ASSERT(kv.second >= 2 && kv.second % 2 == 0)
        << "dup should >= 2 and could be divided by 2";
      new_states[-1] = kv.second / 2;
    } else if (kv.first == split_dim) {
      new_states[kv.first] = kv.second * 2;
    } else {
      HT_ASSERT(kv.second == 1) 
        << "there shouldn't be another split dim";
    }
  }
  HT_LOG_DEBUG << "dup2split " << states << " to " << new_states;
  if (ds.get_placement_group().empty()) {
    return DistributedStates(ds.get_device_num(), new_states, ds.get_order());
  }
  return DistributedStates(ds.get_placement_group(), new_states, ds.get_order());
}

static DistributedStates split2dup(const DistributedStates& ds, int32_t split_dim) {
  const auto& states = ds.get_states();
  std::unordered_map<int32_t, int32_t> new_states;
  for (const auto& kv : states) {
    if (kv.first == -2) {
      HT_ASSERT(kv.second == 1)
        << "partial shouldn't exist";
    } else if (kv.first == -1) {
      new_states[-1] = kv.second * 2;
    } else if (kv.first == split_dim) {
      HT_ASSERT(kv.second >= 2 && kv.second % 2 == 0)
        << "split should >= 2 and could be divided by 2";
      new_states[kv.first] = kv.second / 2;
    } else {
      HT_ASSERT(kv.second == 1)
        << "there shouldn't be another split dim";
    }
  }
  HT_LOG_DEBUG << "split2dup " << states << " to " << new_states;
  if (ds.get_placement_group().empty()) {
    return DistributedStates(ds.get_device_num(), new_states, ds.get_order());
  }
  return DistributedStates(ds.get_placement_group(), new_states, ds.get_order());
}

static DistributedStates split2split(const DistributedStates& ds, int32_t split_dim_before, int32_t split_dim_after) {
  const auto& states = ds.get_states();
  std::unordered_map<int32_t, int32_t> new_states;
  for (const auto& kv : states) {
    if (kv.first == -2) {
      HT_ASSERT(kv.second == 1)
        << "partial shouldn't exist";
    } else if (kv.first == -1) {
      HT_ASSERT(kv.second == 1)
        << "dup shouldn't exist";
    } else if (kv.first == split_dim_before) {
      HT_ASSERT(kv.second >= 2 && kv.second % 2 == 0)
        << "before split should >= 2 and could be divided by 2";
      new_states[kv.first] = kv.second / 2;
    } else if (kv.first == split_dim_after) {
      new_states[kv.first] = kv.second * 2;
    } else {
      HT_ASSERT(kv.second == 1)
        << "there shouldn't be another split dim";
    }
  }
  HT_LOG_DEBUG << "split2split " << states << " to " << new_states;
  if (ds.get_placement_group().empty()) {
    return DistributedStates(ds.get_device_num(), new_states, ds.get_order());
  }
  return DistributedStates(ds.get_placement_group(), new_states, ds.get_order());
}

// dp /= 2 and tp *= 2
// Note only work when tp is already >= 2
// need to revamp that
void DefineAndRunGraph::dp2tp(Operator& op) {
  if (is_variable_op(op) && op->_body->type() == "ParallelVariableOp") {
    auto& variable = op->output(0);
    HT_ASSERT(variable->has_distributed_states())
      << "variable " << variable << " doesn't have distributed_states";
    const auto& ds = variable->get_distributed_states();
    // pure dup
    if (ds.check_pure_duplicate()) {
      // HT_LOG_INFO << "variable " << variable << " is pure dup, do not change";
      return;
    }
    // split 0, row parallel
    if (ds.get_dim(0) >= 2) {
      // HT_LOG_INFO << "variable " << variable << " is splited at dim 0, split it more";
      auto new_ds = dup2split(ds, 0);
      (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(new_ds);
      variable->set_distributed_states(new_ds);
      return;
    }
    // split 1, col parallel
    if (ds.get_dim(1) >= 2) {
      // HT_LOG_INFO << "variable " << variable << " is splited at dim 1, split it more";
      auto new_ds = dup2split(ds, 1);
      (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(new_ds);
      variable->set_distributed_states(new_ds);
      return;
    }
    HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
  }
  if (is_placeholder_op(op)) {
    auto& placeholder = op->output(0);
    HT_ASSERT(placeholder->has_distributed_states())
      << "placeholder " << placeholder << " doesn't have distributed_states";
    const auto& ds = placeholder->get_distributed_states();
    // split 0 means dp for placeholder
    // input related
    if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
      // HT_LOG_INFO << "placeholder " << placeholder << " is splited at dim 0, split it less";
      placeholder->set_distributed_states(split2dup(ds, 0));
      return;
    } 
    // mask related
    if (ds.get_dim(0) >= 2 && ds.get_dim(1) >= 2) {
      // HT_LOG_INFO << "placeholder " << placeholder << " is splited at dim 0 and dim 1, split dim 0 less and dim 1 more";
      placeholder->set_distributed_states(split2split(ds, 0, 1));
      return;
    }
    HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
  }
  if (is_comm_op(op)) {
    auto& comm_output = op->output(0);
    HT_ASSERT(comm_output->has_distributed_states())
      << "comm_output " << comm_output << " doesn't have distributed_states";
    const auto& ds = comm_output->get_distributed_states();
    // gradient allreduce
    if (op->name().find("comm_op_after_") != std::string::npos) {
      // pure dup
      if (ds.check_pure_duplicate()) {
        return;
      }
      // col parallel gradient allreduce
      if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
        // "partial_grad_sum" is wte_table
        if (op->name().find("partial_grad_sum") != std::string::npos || op->name().find("weight") != std::string::npos || op->name().find("bias") != std::string::npos) {
          auto new_ds = dup2split(ds, 0);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        } else {
          auto new_ds = split2dup(ds, 0);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        }
      } 
      // row parallel gradient allreduce
      if (ds.get_dim(-1) >= 2 && ds.get_dim(1) >= 2) {
        if (op->name().find("weight") != std::string::npos || op->name().find("bias") != std::string::npos) {
          auto new_ds = dup2split(ds, 1);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        } else {
          auto new_ds = split2dup(ds, 1);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        }
      }
      HT_LOG_WARN << "op " << op << " ds states = " << ds.get_states();
      HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
    } else {
      // comm in row parallel & last col parallel & grad comm in row parallel
      if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
        // HT_LOG_INFO << "comm_output " << comm_output << " is splited at dim 0, split it less";
        auto new_ds = split2dup(ds, 0);
        (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
        comm_output->set_distributed_states(new_ds);
        return;
      } 
      // grad comm in last col parallel
      if (ds.get_dim(0) >= 2 && ds.get_dim(1) >= 2) {
        // HT_LOG_INFO << "comm_output " << comm_output << " is splited at dim 0 and dim 1, split dim 0 less and dim 1 more";
        auto new_ds = split2split(ds, 0, 1);
        (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
        comm_output->set_distributed_states(new_ds);
        return;
      } 
      HT_LOG_WARN << "op " << op << " ds states = " << ds.get_states();
      HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
    }
  }
  HT_LOG_INFO << "op " << op << " is not a variable/placeholder nor a comm";
  HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
}

// dp *= 2 and tp /= 2
// Note only work when dp is already >= 2
// need to revamp that
void DefineAndRunGraph::tp2dp(Operator& op) {
  if (is_variable_op(op) && op->_body->type() == "ParallelVariableOp") {
    auto& variable = op->output(0);
    HT_ASSERT(variable->has_distributed_states())
      << "variable " << variable << " doesn't have distributed_states";
    const auto& ds = variable->get_distributed_states();
    // pure dup
    if (ds.check_pure_duplicate()) {
      // HT_LOG_INFO << "variable " << variable << " is pure dup, do not change";
      return;
    }
    // split 0, row parallel
    if (ds.get_dim(0) >= 2) {
      auto new_ds = split2dup(ds, 0);
      (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(new_ds);
      variable->set_distributed_states(new_ds);
      return;
    }
    // split 1, col parallel
    if (ds.get_dim(1) >= 2) {
      auto new_ds = split2dup(ds, 1);
      (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(new_ds);
      variable->set_distributed_states(new_ds);
      return;
    }
    HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
  }
  if (is_placeholder_op(op)) {
    auto& placeholder = op->output(0);
    HT_ASSERT(placeholder->has_distributed_states())
      << "placeholder " << placeholder << " doesn't have distributed_states";
    const auto& ds = placeholder->get_distributed_states();
    // split 0 means dp for placeholder
    // input related
    if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
      placeholder->set_distributed_states(dup2split(ds, 0));
      return;
    } 
    // mask related
    if (ds.get_dim(0) >= 2 && ds.get_dim(1) >= 2) {
      placeholder->set_distributed_states(split2split(ds, 1, 0));
      return;
    }
    HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
  }
  if (is_comm_op(op)) {
    auto& comm_output = op->output(0);
    HT_ASSERT(comm_output->has_distributed_states())
      << "comm_output " << comm_output << " doesn't have distributed_states";
    const auto& ds = comm_output->get_distributed_states();
    // gradient allreduce
    if (op->name().find("comm_op_after_") != std::string::npos) {
      // pure dup
      if (ds.check_pure_duplicate()) {
        return;
      }
      // col parallel gradient allreduce
      if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
        // "partial_grad_sum" is wte_table
        if (op->name().find("partial_grad_sum") != std::string::npos || op->name().find("weight") != std::string::npos || op->name().find("bias") != std::string::npos) {
          auto new_ds = split2dup(ds, 0);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        } else {
          auto new_ds = dup2split(ds, 0);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        }
      } 
      // row parallel gradient allreduce
      if (ds.get_dim(-1) >= 2 && ds.get_dim(1) >= 2) {
        if (op->name().find("weight") != std::string::npos || op->name().find("bias") != std::string::npos) {
          auto new_ds = split2dup(ds, 1);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        } else {
          auto new_ds = dup2split(ds, 1);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        }
      }
      HT_LOG_WARN << "op " << op << " ds states = " << ds.get_states();
      HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
    } else {
      // comm in row parallel & last col parallel & grad comm in row parallel
      if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
        auto new_ds = dup2split(ds, 0);
        (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
        comm_output->set_distributed_states(new_ds);
        return;
      } 
      // grad comm in last col parallel
      if (ds.get_dim(0) >= 2 && ds.get_dim(1) >= 2) {
        auto new_ds = split2split(ds, 1, 0);
        (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
        comm_output->set_distributed_states(new_ds);
        return;
      } 
      HT_LOG_WARN << "op " << op << " ds states = " << ds.get_states();
      HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
    }
  }
  HT_LOG_INFO << "op " << op << " is not a variable/placeholder nor a comm";
  HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
}

void DefineAndRunGraph::SetVariableDistributedStates(Operator& op, int32_t dp, int32_t tp) {
  // *先split后dup，靠近的机器会在一个dup组中，切换起来通信开销期望上会更小
  DistributedStates dup_ds(4, {{-1, 4}}, {-1});
  DistributedStates col_ds(4, {{-1, dp}, {0, tp}}, {0, -1});
  DistributedStates row_ds(4, {{-1, dp}, {1, tp}}, {1, -1});
  // DistributedStates dup_ds(8, {{-1, 8}}, {-1});
  // DistributedStates col_ds(8, {{-1, dp}, {0, tp}}, {0, -1});
  // DistributedStates row_ds(8, {{-1, dp}, {1, tp}}, {1, -1});
  if (op->name().find("wte") != std::string::npos
      || op->name().find("col") != std::string::npos) {
    (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(col_ds);
    op->output(0)->set_distributed_states(col_ds);
    return;
  }
  if (op->name().find("wpe") != std::string::npos
      || op->name().find("ln") != std::string::npos
      || (op->name().find("row") != std::string::npos && op->name().find("bias") != std::string::npos)) {
    op->output(0)->set_distributed_states(dup_ds);
    return;
  }
  if (op->name().find("row") != std::string::npos && op->name().find("weight") != std::string::npos) {
    (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(row_ds);
    op->output(0)->set_distributed_states(row_ds);
    return;
  }
  HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
}
*/

void DefineAndRunGraph::InstantiateTestCase(const OpRefList& topo,
                                            Tensor2ShapeMap& shape_plan) {
  /*
  auto local_device = hetu::impl::comm::GetLocalDevice();
  static size_t instantiate_test_case = 1;

  auto exec_graph_num = _exec_graph_plan_pool.size();
  Tensor2ShapeMap exec_shape_plan;
  Op2OpMap op_to_exec_op_mapping;
  Tensor2TensorMap tensor_to_exec_tensor_mapping;
  auto exec_graph = Graph::_make_new_graph<ExecutableGraph>(name() + "_executable_test_case_" + std::to_string(exec_graph_num));
  
  exec_shape_plan.reserve(shape_plan.size());
  op_to_exec_op_mapping.reserve(_init_capacity);
  tensor_to_exec_tensor_mapping.reserve(_init_capacity);

  Graph::push_graph_ctx(exec_graph->id());

  auto get_exec_input = [&](const Tensor& input) -> Tensor {
    auto it = tensor_to_exec_tensor_mapping.find(input->id());
    HT_RUNTIME_ERROR_IF(it == tensor_to_exec_tensor_mapping.end())
      << "Cannot find the executable version of Tensor " << input;
    return it->second;
  };

  auto handle_exec_output = [&](Tensor& tensor, Tensor& exec_tensor) -> void {
    // assign tensor mapping
    tensor_to_exec_tensor_mapping[tensor->id()] = exec_tensor;
    // assign shape
    auto plan_it = shape_plan.find(tensor->id());
    // The shape plan will be expanded step by step
    if (plan_it != shape_plan.end()) {
      exec_tensor->set_shape(plan_it->second);
    } else {
      shape_plan[tensor->id()] = exec_tensor->shape();
    }
    exec_shape_plan[exec_tensor->id()] = exec_tensor->shape();
  };

  HT_LOG_DEBUG << local_device << ": Instantiating a " << type() << " graph with topo " << topo;
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();

    // 一个只有variable的exec图
    if (!is_variable_op(op))
      continue;

    // 前处理
    // 1、获取exec op的inputs
    TensorList exec_inputs, exec_in_deps;
    std::tie(exec_inputs, exec_in_deps) = Operator::transform_each_input_tensor(op, get_exec_input);

    // 前处理
    // 2、获取exec op的OpMeta，修正device_group
    // Test Case: 手动让device_group发生一下变化
    OpMeta exec_op_meta = OpMeta().set(op->op_meta()).set_extra_deps(std::move(exec_in_deps));
    if (instantiate_test_case == 1) {
      HT_ASSERT(!exec_op_meta.is_deduce_states) 
        << "varaiable op is not supposed to deduce states";
      // exec_op_meta.set_device_group(hetu::impl::comm::GetGlobalDeviceGroup());
    }

    auto& exec_op = Graph::MakeOp(
      op->_body, 
      std::move(exec_inputs),
      exec_op_meta,
      *exec_graph);

    // 后处理
    // 1、建立op和exec_op的映射
    // 2、修正op和tensor的distributed_states
    // 3、标记parameter
    op_to_exec_op_mapping[op->id()] = exec_op;
    // Test Case: 手动让distributed_states发生一下变化
    if (instantiate_test_case == 1) {
      // 重新设置variable的distributed_states与输出的shape
      SetVariableDistributedStates(exec_op, 4, 1);
      // SetVariableDistributedStates(exec_op, 8, 1);
      RuntimeContext runtime_ctx{};
      shape_plan[op->output(0)->id()] = exec_op->InferShape({}, runtime_ctx)[0];
    }
    if (_parameter_ops.find(op->id()) != _parameter_ops.end()) {
      Graph::MarkAsParameter(exec_op);
    }

    // 后处理
    // 4、建立tensor和exec_tensor的映射
    // 5、修正tensor的shape
    // 6、扩展define图的当前shape_plan和exec图的唯一exec_shape_plan
    Operator::for_each_output_tensor_pair(op, exec_op, handle_exec_output);
    // Test Case Log
    HT_LOG_DEBUG << local_device << ": exec op " << exec_op << " output ds states = " << exec_op->output(0)->get_distributed_states().get_states()
      << " output shape = " << exec_op->output(0)->shape();
  }

  // assign initial shape plan
  exec_graph->AddShapePlan(std::move(exec_shape_plan));
  
  // wrap up all of this as an exec graph plan
  _exec_graph_plan_pool.emplace_back(std::move(exec_graph), 
                                     std::move(op_to_exec_op_mapping),
                                     std::move(tensor_to_exec_tensor_mapping));

  Graph::pop_graph_ctx();
  HT_LOG_DEBUG << local_device << ": instantiate test case " << instantiate_test_case << " finished";
  instantiate_test_case++;
  */
}

} // namespace graph
} // namespace hetu
