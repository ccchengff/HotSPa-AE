#pragma once

#include "hetu/graph/graph.h"

namespace hetu {
namespace graph {

class EagerGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;

  EagerGraph(GraphName name, size_t init_capacity)
  : Graph(name, init_capacity), _runtime_ctxs(init_capacity) {
    _op_to_num_destructed_outputs.reserve(init_capacity);
  }

 public:
  EagerGraph(const constrcutor_access_key&, GraphName name,
             size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : EagerGraph(name, init_capacity) {}

  NDArrayList Run(const TensorList& fetches, const FeedDict& feed_dict = {});

  GraphType type() const {
    return GraphType::EAGER;
  }

 protected:
  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void ResetVariableDataInner(const Tensor& tensor,
                              const Initializer& init) override;

  NDArray& GetVariableDataInner(const Tensor& tensor) override;

  NDArray& AllocVariableDataInner(
    const Tensor& tensor,
    const Initializer& init = VoidifiedInitializer(),
    uint64_t seed = 0, const HTShape& global_shape = HTShape()) override;

  void RegisterVariableDataInner(
    const Tensor& tensor, NDArray data,
    const Initializer& init = VoidifiedInitializer()) override;

  NDArray GetOrCompute(Tensor& tensor);
  
  void RemoveTensor(const Tensor& tensor);

  void RemoveOp(Operator& op) override {
    _runtime_ctxs.remove(op->id());
    _op_to_num_destructed_outputs.erase(op->id());
    Graph::RemoveOp(op);
  }
  
  void Clear() override {
    _runtime_ctxs.clear();
    Graph::Clear();
  }

  RuntimeContext _runtime_ctxs;
  std::unordered_map<OpId, size_t> _op_to_num_destructed_outputs;
};

} // namespace graph
} // namespace hetu
