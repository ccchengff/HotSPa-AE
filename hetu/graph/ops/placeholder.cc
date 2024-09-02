#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

// in_degree=0, should set distributed states manually
Tensor MakePlaceholderOp(NDArrayMeta data_meta, const DistributedStatesHierarchy& ds_hierarchy, OpMeta op_meta) {
  Tensor output = Graph::MakeOp(
    std::make_shared<PlaceholderOpImpl>(std::move(data_meta)),
    TensorList(), std::move(op_meta.set_is_deduce_states(false)))
    ->output(0);
  if (ds_hierarchy.size() != 0) {
    auto& graph = output->graph();
    graph.CREATE_STRATEGY = true;
    graph.CREATE_HETERO = true;
    graph.USE_HETERO_ID = true;
    HT_ASSERT(graph.NUM_STRATEGY == ds_hierarchy.size())
      << "NUM_STRATEGY of graph should equal to size of the ds hierarchy";
    for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
      graph.CUR_STRATEGY_ID = cur_strategy_id;
      for (size_t cur_hetero_id = 0; cur_hetero_id < ds_hierarchy.get(cur_strategy_id).size(); cur_hetero_id++) {
        auto& ds = ds_hierarchy.get(cur_strategy_id).get(cur_hetero_id);
        HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
          << "DistributedStates for PlaceholderOp must be valid! got: " 
          << ds.ds_info();    
        graph.CUR_HETERO_ID = cur_hetero_id;
        output->set_distributed_states(ds);
      }
      output->cur_ds_union().set_hetero_dim(ds_hierarchy.get(cur_strategy_id).hetero_dim());
    }
    graph.CUR_STRATEGY_ID = 0;
    graph.CUR_HETERO_ID = 0;
    graph.USE_HETERO_ID = false;
    graph.CREATE_HETERO = false;
    graph.CREATE_STRATEGY = false;
  }
  return output;
}

Tensor MakeParallelPlaceholderOp(NDArrayMeta data_meta, const DistributedStatesHierarchy& ds_hierarchy, OpMeta op_meta) {
  HTShape global_shape = data_meta.shape;
  HTShape local_shape(global_shape.size());
  for (size_t d = 0; d < global_shape.size(); d++) {
    local_shape[d] = global_shape[d] / ds_hierarchy.get_default_ds().get_dim(d);
  }
  Tensor output = Graph::MakeOp(
    std::make_shared<PlaceholderOpImpl>(std::move(data_meta.set_shape(local_shape))),
    TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);

  // assign multi ds for placeholder
  auto& graph = output->graph();
  graph.CREATE_STRATEGY = true;
  graph.CREATE_HETERO = true;
  graph.USE_HETERO_ID = true;
  HT_ASSERT(graph.NUM_STRATEGY == ds_hierarchy.size())
    << "NUM_STRATEGY of graph should equal to size of the ds hierarchy";
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    for (size_t cur_hetero_id = 0; cur_hetero_id < ds_hierarchy.get(cur_strategy_id).size(); cur_hetero_id++) {
      auto& ds = ds_hierarchy.get(cur_strategy_id).get(cur_hetero_id);
      HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
        << "DistributedStates for PlaceholderOp must be valid! got: " 
        << ds.ds_info();    
      graph.CUR_HETERO_ID = cur_hetero_id;
      output->set_distributed_states(ds);
    }
    output->cur_ds_union().set_hetero_dim(ds_hierarchy.get(cur_strategy_id).hetero_dim());
  }
  graph.CUR_STRATEGY_ID = 0;
  graph.CUR_HETERO_ID = 0;
  graph.USE_HETERO_ID = false;
  graph.CREATE_HETERO = false;
  graph.CREATE_STRATEGY = false;
  return output;
}

} // namespace graph
} // namespace hetu
