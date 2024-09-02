#include "hetu/graph/ops/variable.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

NDArrayList VariableOpImpl::DoAllocOutputs(Operator& op,
                                           const NDArrayList& inputs,
                                           RuntimeContext& runtime_ctx) const {
  if (_init != nullptr) {
    Graph::AllocVariableData(op->output(0), *_init);
  } else {
    if (_copy_provided_data || dtype() != _provided_data->dtype() ||
        op->instantiation_ctx().placement != _provided_data->device()) {
      Graph::AllocVariableData(op->output(0),
                               ProvidedInitializer(_provided_data));
    } else {
      Graph::RegisterVariableData(op->output(0), _provided_data);
    }
  }
  return {Graph::GetVariableData(op->output(0))};
}

NDArrayList ParallelVariableOpImpl::DoAllocOutputs(Operator& op,
                                                   const NDArrayList& inputs,
                                                   RuntimeContext& runtime_ctx) const {
  auto ds_union = _ds_hierarchy.get(op->graph().CUR_STRATEGY_ID);
  HT_ASSERT(ds_union.hetero_dim() == -1 || ds_union.hetero_dim() == 0 || ds_union.hetero_dim() == NULL_HETERO_DIM)
    << "ParallelVariableOp " << op << " can only hetero on dup (normal dp param) or split0 (Adam mean or var)"
    << ", but found " << ds_union.hetero_dim();
  auto ds = ds_union.get(op->inferred_local_placement_group_idx());
  auto local_idx = _local_idx.empty() ? -1 : _local_idx[op->graph().CUR_STRATEGY_ID];
  HT_ASSERT(_init == nullptr || local_idx != -1)
    << "ParallelVariableOp: when use initializer, local_idx "
    << "must be assigned when local_device is in pipeline device_group!";

  if (_init != nullptr) {
    int32_t dup_group_idx = ds.get_dup_group_index(local_idx);
    // support 100 different duplicate group to set different seed
    uint64_t seed = 2023 + op->id() * 100 + dup_group_idx;
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": " << op << " inits by initializer.";
    // TODO: reset variable data also need parallel version
    Graph::AllocVariableData(op->output(0), *_init, seed, _global_shape);
  } else {
    auto& provided_data = _multi_provided_data.empty() ? 
      _provided_data : _multi_provided_data[op->graph().CUR_STRATEGY_ID]; 
    if (_copy_provided_data || dtype() != provided_data->dtype() ||
        op->instantiation_ctx().placement != provided_data->device()) {
      HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": " << op << " inits by provided data.";
      Graph::AllocVariableData(op->output(0),
                               ProvidedInitializer(provided_data));
    } else {
      Graph::RegisterVariableData(op->output(0), provided_data);
    }
  }
  return {Graph::GetVariableData(op->output(0))};
}

// in_degree=0, should set distributed states manually
Tensor MakeVariableOp(const Initializer& init, HTShape shape, 
                      DataType dtype, bool requires_grad, 
                      const DistributedStatesHierarchy& ds_hierarchy, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           init, std::move(shape), dtype, requires_grad), TensorList(), 
                           std::move(op_meta.set_is_deduce_states(false)))->output(0);
  if (ds_hierarchy.size() == 0) {
    return out;
  }
  // assign multi ds for variable
  auto& graph = out->graph();
  graph.CREATE_STRATEGY = true;
  graph.CREATE_HETERO = true;
  graph.USE_HETERO_ID = true;
  HT_ASSERT(graph.NUM_STRATEGY == ds_hierarchy.size())
    << "NUM_STRATEGY of graph should equal to size of the ds hierarchy";
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    for (size_t cur_hetero_id = 0; cur_hetero_id < ds_hierarchy.get(cur_strategy_id).size(); cur_hetero_id++) {
      const auto& ds = ds_hierarchy.get(cur_strategy_id).get(cur_hetero_id);
      HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
        << "DistributedStates for VariableOp must be valid! got: " 
        << ds.ds_info();
      graph.CUR_HETERO_ID = cur_hetero_id;
      out->set_distributed_states(ds);
    }
    out->cur_ds_union().set_hetero_dim(ds_hierarchy.get(cur_strategy_id).hetero_dim());
    out->cur_ds_union().set_split_pattern(ds_hierarchy.get(cur_strategy_id).split_pattern());
  }
  graph.CUR_STRATEGY_ID = 0;
  graph.CUR_HETERO_ID = 0;
  graph.USE_HETERO_ID = false;
  graph.CREATE_HETERO = false;
  graph.CREATE_STRATEGY = false;
  return out;
}

// in_degree=0, should set distributed states manually
Tensor MakeVariableOp(NDArray provided_data, bool copy_provided_data, 
                      DataType dtype, bool requires_grad, 
                      const DistributedStatesHierarchy& ds_hierarchy, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           std::move(provided_data), copy_provided_data, dtype,
                           requires_grad), TensorList(), 
                           std::move(op_meta.set_is_deduce_states(false)))->output(0);
  if (ds_hierarchy.size() == 0) {
    return out;
  }
  // assign multi ds for variable
  auto& graph = out->graph();
  graph.CREATE_STRATEGY = true;
  graph.CREATE_HETERO = true;
  graph.USE_HETERO_ID = true;
  HT_ASSERT(graph.NUM_STRATEGY == ds_hierarchy.size())
    << "NUM_STRATEGY of graph should equal to size of the ds hierarchy";
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    for (size_t cur_hetero_id = 0; cur_hetero_id < ds_hierarchy.get(cur_strategy_id).size(); cur_hetero_id++) {
      const auto& ds = ds_hierarchy.get(cur_strategy_id).get(cur_hetero_id);
      HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
        << "DistributedStates for VariableOp must be valid! got: " 
        << ds.ds_info();
      graph.CUR_HETERO_ID = cur_hetero_id;
      out->set_distributed_states(ds);
    }
    out->cur_ds_union().set_hetero_dim(ds_hierarchy.get(cur_strategy_id).hetero_dim());
    out->cur_ds_union().set_split_pattern(ds_hierarchy.get(cur_strategy_id).split_pattern());
  }
  graph.CUR_STRATEGY_ID = 0;
  graph.CUR_HETERO_ID = 0;
  graph.USE_HETERO_ID = false;
  graph.CREATE_HETERO = false;
  graph.CREATE_STRATEGY = false;
  return out;
}

Tensor MakeParameterOp(const Initializer& init, HTShape shape, 
                       DataType dtype, bool requires_grad, 
                       const DistributedStatesHierarchy& ds_hierarchy, OpMeta op_meta) {
  auto out = MakeVariableOp(init, std::move(shape), dtype, 
                            requires_grad, ds_hierarchy, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParameterOp(NDArray provided_data, bool copy_provided_data, 
                       DataType dtype, bool requires_grad, 
                       const DistributedStatesHierarchy& ds_hierarchy, OpMeta op_meta) {
  auto out = MakeVariableOp(std::move(provided_data), copy_provided_data, 
                            dtype, requires_grad, ds_hierarchy, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParallelVariableOp(const Initializer& init, HTShape global_shape, 
                              const DistributedStatesHierarchy& ds_hierarchy, std::vector<int64_t> local_idx,
                              DataType dtype, bool requires_grad, OpMeta op_meta) {
  // init local_idx vector
  if (local_idx.size() == 1) {
    local_idx.resize(ds_hierarchy.size(), local_idx[0]);
  } else {
    HT_ASSERT(local_idx.size() == ds_hierarchy.size()) 
      << "ParallelVariableOp: local_idx size must equal to ds_hierarchy!";
  }
  auto out = Graph::MakeOp(std::make_shared<ParallelVariableOpImpl>(
                           init, std::move(global_shape), ds_hierarchy, 
                           std::move(local_idx), dtype, requires_grad),
                           TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);
  // assign multi ds for variable
  auto& graph = out->graph();
  graph.CREATE_STRATEGY = true;
  graph.CREATE_HETERO = true;
  graph.USE_HETERO_ID = true;
  HT_ASSERT(graph.NUM_STRATEGY == ds_hierarchy.size())
    << "NUM_STRATEGY of graph should equal to size of the ds hierarchy";
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    for (size_t cur_hetero_id = 0; cur_hetero_id < ds_hierarchy.get(cur_strategy_id).size(); cur_hetero_id++) {
      const auto& ds = ds_hierarchy.get(cur_strategy_id).get(cur_hetero_id);
      HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
        << "DistributedStates for ParallelVariableOp must be valid! got: " 
        << ds.ds_info();
      graph.CUR_HETERO_ID = cur_hetero_id;
      out->set_distributed_states(ds);
    }
    out->cur_ds_union().set_hetero_dim(ds_hierarchy.get(cur_strategy_id).hetero_dim());
    out->cur_ds_union().set_split_pattern(ds_hierarchy.get(cur_strategy_id).split_pattern());
  }
  graph.CUR_STRATEGY_ID = 0;
  graph.CUR_HETERO_ID = 0;
  graph.USE_HETERO_ID = false;
  graph.CREATE_HETERO = false;
  graph.CREATE_STRATEGY = false;
  return out;
}

Tensor MakeParallelVariableOp(NDArray provided_data, 
                              const DistributedStatesHierarchy& ds_hierarchy, 
                              bool copy_provided_data, DataType dtype, 
                              bool requires_grad, OpMeta op_meta) {
  // auto placement_group = op_meta.device_group;
  auto out = Graph::MakeOp(std::make_shared<ParallelVariableOpImpl>(
                           provided_data, copy_provided_data, 
                           ds_hierarchy, dtype, requires_grad),
                           TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);
  // assign multi ds for variable
  auto& graph = out->graph();
  graph.CREATE_STRATEGY = true;
  graph.CREATE_HETERO = true;
  graph.USE_HETERO_ID = true;
  HT_ASSERT(graph.NUM_STRATEGY == ds_hierarchy.size())
    << "NUM_STRATEGY of graph should equal to size of the ds hierarchy";
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    for (size_t cur_hetero_id = 0; cur_hetero_id < ds_hierarchy.get(cur_strategy_id).size(); cur_hetero_id++) {
      const auto& ds = ds_hierarchy.get(cur_strategy_id).get(cur_hetero_id);
      HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
        << "DistributedStates for ParallelVariableOp must be valid! got: " 
        << ds.ds_info();
      graph.CUR_HETERO_ID = cur_hetero_id;
      out->set_distributed_states(ds);
    }
    out->cur_ds_union().set_hetero_dim(ds_hierarchy.get(cur_strategy_id).hetero_dim());
    out->cur_ds_union().set_split_pattern(ds_hierarchy.get(cur_strategy_id).split_pattern());
  }
  graph.CUR_STRATEGY_ID = 0;
  graph.CUR_HETERO_ID = 0;
  graph.USE_HETERO_ID = false;
  graph.CREATE_HETERO = false;
  graph.CREATE_STRATEGY = false;
  return out;  
}

Tensor MakeParallelVariableOp(NDArrayList multi_provided_data, 
                              DistributedStatesHierarchy ds_hierarchy, 
                              bool copy_provided_data, DataType dtype, 
                              bool requires_grad, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<ParallelVariableOpImpl>(
                           std::move(multi_provided_data), copy_provided_data, 
                           std::move(ds_hierarchy), dtype, requires_grad),
                           TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);
  // assign multi ds for variable
  auto& graph = out->graph();
  graph.CREATE_STRATEGY = true;
  graph.CREATE_HETERO = true;
  graph.USE_HETERO_ID = true;
  HT_ASSERT(graph.NUM_STRATEGY == ds_hierarchy.size())
    << "NUM_STRATEGY of graph should equal to size of the ds hierarchy";
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    for (size_t cur_hetero_id = 0; cur_hetero_id < ds_hierarchy.get(cur_strategy_id).size(); cur_hetero_id++) {
      const auto& ds = ds_hierarchy.get(cur_strategy_id).get(cur_hetero_id);
      HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
        << "DistributedStates for ParallelVariableOp must be valid! got: " 
        << ds.ds_info();
      graph.CUR_HETERO_ID = cur_hetero_id;
      out->set_distributed_states(ds);
    }
    out->cur_ds_union().set_hetero_dim(ds_hierarchy.get(cur_strategy_id).hetero_dim());
    out->cur_ds_union().set_split_pattern(ds_hierarchy.get(cur_strategy_id).split_pattern());
  }
  graph.CUR_STRATEGY_ID = 0;
  graph.CUR_HETERO_ID = 0;
  graph.USE_HETERO_ID = false;
  graph.CREATE_HETERO = false;
  graph.CREATE_STRATEGY = false;
  return out;  
}

Tensor MakeParallelParameterOp(const Initializer& init, HTShape global_shape, 
                               const DistributedStatesHierarchy& ds_hierarchy, std::vector<int64_t> local_idx,
                               DataType dtype, bool requires_grad, OpMeta op_meta) {
  auto out = MakeParallelVariableOp(init, std::move(global_shape), ds_hierarchy, std::move(local_idx), 
                                    dtype, requires_grad, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;                                    
}

Tensor MakeParallelParameterOp(NDArray provided_data, 
                               const DistributedStatesHierarchy& ds_hierarchy, 
                               bool copy_provided_data, DataType dtype, 
                               bool requires_grad, OpMeta op_meta) {    
  auto out = MakeParallelVariableOp(std::move(provided_data), ds_hierarchy, copy_provided_data, 
                                    dtype, requires_grad, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParallelParameterOp(NDArrayList multi_provided_data, 
                               DistributedStatesHierarchy ds_hierarchy, 
                               bool copy_provided_data, DataType dtype, 
                               bool requires_grad, OpMeta op_meta) {    
  auto out = MakeParallelVariableOp(std::move(multi_provided_data), std::move(ds_hierarchy), 
                                    copy_provided_data, dtype, requires_grad, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

} // namespace graph
} // namespace hetu
