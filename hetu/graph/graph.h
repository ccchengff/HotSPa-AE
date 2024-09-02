#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/init/initializer.h"
#include <mutex>
#include <stack>

#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

enum class GraphType : int8_t {
  EAGER = 0,
  DEFINE_BY_RUN,
  DEFINE_AND_RUN,
  EXECUTABLE,
  NUM_GRAPH_TYPES
};

enum class RunLevel : int8_t {
  UPDATE = 0,
  GRAD,
  ALLOC,
  TOPO
};

std::string GraphType2Str(GraphType);
std::ostream& operator<<(std::ostream&, GraphType);

class Graph {
 protected:
  friend class OpDef;
  friend class TensorDef;
  friend class SwitchExecGraph;
  
  struct constrcutor_access_key {};

  Graph(GraphName name, size_t init_capacity)
  : _id{_next_graph_id()}, _name(name) {
    _op_indexing.reserve(init_capacity);
    _parameter_ops.reserve(init_capacity);
    _source_ops.reserve(init_capacity);
    _sink_ops.reserve(init_capacity);
    _preserved_data.reserve(init_capacity);
  }

 public:
  static constexpr size_t DEFAULT_GRAPH_INITIAL_CAPACITY = 4096;
  bool CREATE_STRATEGY = false;
  size_t NUM_STRATEGY = 1;
  size_t CUR_STRATEGY_ID = 0;
  bool CREATE_HETERO = false;
  bool USE_HETERO_ID = false;
  size_t CUR_HETERO_ID = 0;
  size_t SUGGESTED_HETERO_ID = 0;

  // disable copy constructor and move constructor
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = delete;
  Graph& operator=(Graph&&) = delete;

  virtual NDArrayList Run(const TensorList& fetches,
                          const FeedDict& feed_dict = {}) = 0;

  virtual NDArrayList Run(const Tensor& loss, const TensorList& fetches, 
                          const FeedDict& feed_dict = {}, const int num_micro_batches = 1,
                          const int cur_strategy_id = 0, RunLevel run_level = RunLevel::UPDATE, const double grad_scale = 1) {}                          

  GraphId id() const noexcept {
    return _id;
  }

  const GraphName& name() const noexcept {
    return _name;
  }

  OpRefList topo_order() {
    // TODO: make it a const function
    OpRefList sink_ops;
    sink_ops.reserve(_sink_ops.size());
    for (auto op_id : _sink_ops)
      sink_ops.push_back(std::ref(_op_indexing[op_id]));
    return Graph::TopoSort(sink_ops, num_ops());
  }

  TensorCRefList params() const {
    TensorCRefList params;
    for (auto& op_id : _parameter_ops) {
      auto it = _op_indexing.find(op_id);
      HT_ASSERT(it != _op_indexing.end());
      Operator::for_each_output_tensor(it->second, 
        [&](const Tensor& tensor) { params.emplace_back(tensor); });
    }
    return params;
  }

  TensorCRefList params_and_opt_vars() const {
    TensorCRefList params_and_opt_vars;
    for (auto& op_id : _parameter_ops) {
      auto it = _op_indexing.find(op_id);
      HT_ASSERT(it != _op_indexing.end());
      Operator::for_each_output_tensor(it->second, 
        [&](const Tensor& tensor) { params_and_opt_vars.emplace_back(tensor); });
    }
    for (auto& op_id : _optimizer_variable_ops) {
      auto it = _op_indexing.find(op_id);
      HT_ASSERT(it != _op_indexing.end());
      Operator::for_each_output_tensor(it->second, 
        [&](const Tensor& tensor) { params_and_opt_vars.emplace_back(tensor); });
    }
    return params_and_opt_vars;
  }

  std::unordered_set<TensorId> param_ids() const {
    std::unordered_set<TensorId> param_ids;
    for (auto& op_id : _parameter_ops) {
      auto it = _op_indexing.find(op_id);
      HT_ASSERT(it != _op_indexing.end());
      Operator::for_each_output_tensor(it->second, 
        [&](const Tensor& tensor) { param_ids.insert(tensor->id()); });
    }
    return param_ids;
  }

  virtual GraphType type() const = 0;

  uint32_t num_ops() const {
    return _op_indexing.size();
  }

  uint32_t get_op_type_cnt(const OpType& op_type) const {
    auto it = _op_type_cnts.find(op_type);
    if (it != _op_type_cnts.end()) {
      return it->second;
    } else {
      return 0;
    }
  }

  const Operator& GetOp(OpId op_id) const {
    return _op_indexing.at(op_id);
  }

  Operator& GetOp(OpId op_id) {
    return _op_indexing[op_id];
  }

 protected:
  virtual Operator& MakeOpInner(std::shared_ptr<OpInterface> body,
                                TensorList inputs, OpMeta op_meta) = 0;

  Operator& MakeAndAddOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                         OpMeta op_meta) {
    Operator op(OpIdentifier{id(), next_op_id()}, std::move(body),
                std::move(inputs), std::move(op_meta));
    AddOp(op);
    return _op_indexing[op->id()];
  }

  virtual void AddOp(Operator& op) {
    ++_op_type_cnts[op->type()];
    _op_indexing[op->id()] = op;
    if (op->in_degrees() == 0) {
      _source_ops.insert(op->id());
    } else {
      // Note: Since we cannot retrive `Operator` inside the constructor of
      // `OpDef`, we defer the assignment of consumers here.
      Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
        tensor->AddConsumer(_op_indexing[op->id()]);
        _sink_ops.erase(tensor->producer_id());
        _op_out_degrees[tensor->producer_id()]++;
      });
    }
    _sink_ops.insert(op->id());
    _op_out_degrees[op->id()] = 0;
  }

  virtual void MarkOpAsParameter(OpId id) {
    auto it = _op_indexing.find(id);
    HT_VALUE_ERROR_IF(it == _op_indexing.end())
      << "Operator with id " << id << " is not in graph " << name();
    HT_VALUE_ERROR_IF(!is_variable_op(it->second))
      << "Cannot mark a non-variable op " << it->second << " as a parameter";
    _parameter_ops.insert(id);
  }

  virtual void MarkOpAsOptimizerVariable(OpId id) {
    auto it = _op_indexing.find(id);
    HT_VALUE_ERROR_IF(it == _op_indexing.end())
      << "Operator with id " << id << " is not in graph " << name();
    HT_VALUE_ERROR_IF(!is_variable_op(it->second))
      << "Cannot mark a non-variable op " << it->second << " as a parameter";
    _optimizer_variable_ops.insert(id);
  }

  virtual void RemoveOp(Operator& op) {
    _parameter_ops.erase(op->id());
    _optimizer_variable_ops.erase(op->id());
    _source_ops.erase(op->id());
    _sink_ops.erase(op->id());
    _op_out_degrees.erase(op->id());
    Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
      tensor->DelConsumer(op);
      if ((--_op_out_degrees[tensor->producer_id()]) == 0)
        _sink_ops.insert(tensor->producer_id());
    });
    Operator::for_each_output_tensor(
      op, [&](Tensor& tensor) { _preserved_data.erase(tensor->id()); });
    _op_indexing.erase(op->id());
  }

  virtual void ResetVariableDataInner(const Tensor& tensor,
                                      const Initializer& init) {
    HT_RUNTIME_ERROR << "NotImplementedError: Cannot reset variable data in graph " << name()
                     << " with type " << type();
    __builtin_unreachable();
  }

  virtual NDArray& GetVariableDataInner(const Tensor& tensor) {
    HT_RUNTIME_ERROR << "NotImplementedError: Cannot get variable data from graph " << name()
                     << " with type " << type();
    __builtin_unreachable();
  }

  virtual NDArray GetDetachedVariableDataInner(const Tensor& tensor) {
    HT_RUNTIME_ERROR << "NotImplementedError: Cannot get detached variable data from graph " << name()
                     << " with type " << type();
    __builtin_unreachable();
  }

  virtual DeviceGroupUnion GetVariableDeviceGroupUnionInner(const Tensor& tensor) {
    HT_RUNTIME_ERROR << "NotImplementedError: Cannot get variable device group from graph " << name()
                     << " with type " << type();
    __builtin_unreachable();
  }

  virtual NDArray&
  AllocVariableDataInner(const Tensor& tensor,
                         const Initializer& init = VoidifiedInitializer(),
                         uint64_t seed = 0, const HTShape& global_shape = HTShape()) {
    HT_RUNTIME_ERROR << "NotImplementedError: Cannot allocate variable data in graph " << name()
                     << " with type " << type();
    __builtin_unreachable();
  }  

  virtual void
  RegisterVariableDataInner(const Tensor& tensor, NDArray data,
                            const Initializer& init = VoidifiedInitializer()) {
    HT_RUNTIME_ERROR << "NotImplementedError: Cannot register variable data for graph " << name()
                     << " with type " << type();
    __builtin_unreachable();
  }

  virtual NDArray GetOrCompute(Tensor& tensor) {
    auto it = _preserved_data.find(tensor->id());
    if (it != _preserved_data.end()) {
      return it->second;
    } else {
      return Run(TensorList({tensor})).front();
    }
  }

  virtual void Clear() {
    _op_indexing.clear();
    _preserved_data.clear();
    _parameter_ops.clear();
    _optimizer_variable_ops.clear();
    _source_ops.clear();
    _sink_ops.clear();
  }

  void _check_all_inputs_in_graph(const TensorList& inputs,
                                  const TensorList& extra_in_deps) {
    auto check_fn = [&](const Tensor& input) {
      HT_RUNTIME_ERROR_IF(input->graph_id() != id())
        << "Graph " << id() << " cannot accept tensor " << input->name()
        << " since it belongs to another graph " << input->graph_id();
    };
    for (const auto& input : inputs)
      check_fn(input);
    for (const auto& in_dep : extra_in_deps)
      check_fn(in_dep);
  }

  const GraphId _id;
  const GraphName _name;
  std::unordered_map<OpType, uint32_t> _op_type_cnts;

  std::unordered_map<OpId, Operator> _op_indexing;
  std::unordered_set<OpId> _parameter_ops;
  std::unordered_set<OpId> _optimizer_variable_ops;
  std::unordered_set<OpId> _source_ops;
  std::unordered_set<OpId> _sink_ops;
  
  std::unordered_map<OpId, uint32_t> _op_out_degrees;
  Tensor2NDArrayMap _preserved_data;
  RunLevel _run_level;

 protected:
  OpId next_op_id() {
    return _next_op_id++;
  }

  TensorId next_tensor_id() {
    return _next_tensor_id++;
  }

  std::atomic<GraphId> _next_op_id{0};
  std::atomic<GraphId> _next_tensor_id{0};

 private:
  static GraphId _next_graph_id();

  /******************************************************
   * Static helper functions
   ******************************************************/
 public:
  static Operator& MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                          OpMeta op_meta);

  static Operator& MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                          OpMeta op_meta, Graph& graph);
  
  static inline Graph& GetGraph(const Operator& op) {
    return GetGraph(op->graph_id());
  }

  static inline Graph& GetGraph(const Tensor& tensor) {
    return GetGraph(tensor->graph_id());
  }

  static inline Graph& GetGraph(GraphId graph_id) {
    HT_VALUE_ERROR_IF(graph_id >= Graph::_global_graphs.size())
      << "Graph with id " << graph_id << " does not exist";
    return *Graph::_global_graphs[graph_id];
  }

  static inline Graph& GetGraph(const GraphName& graph_name) {
    auto it = Graph::_name_to_graphs.find(graph_name);
    HT_VALUE_ERROR_IF(it == Graph::_name_to_graphs.end())
      << "Graph with name \"" << graph_name << "\" does not exist";
    return *(it->second);
  }

  static void DeleteGraph(GraphId graph_id) {
    HT_VALUE_ERROR_IF(graph_id >= Graph::_global_graphs.size())
      << "Graph with id " << graph_id << " does not exist";
    const auto& graph_name = Graph::_global_graphs[graph_id]->name();
    auto it = Graph::_name_to_graphs.find(graph_name);
    HT_VALUE_ERROR_IF(it == Graph::_name_to_graphs.end())
      << "Graph with name \"" << graph_name << "\" does not exist";
    Graph::_global_graphs[graph_id]->Clear();
    Graph::_global_graphs[graph_id] = nullptr;
    Graph::_name_to_graphs.erase(it);
  }
 
  static void DeleteGraph(const GraphName& graph_name) {
    auto it = Graph::_name_to_graphs.find(graph_name);
    HT_VALUE_ERROR_IF(it == Graph::_name_to_graphs.end())
      << "Graph with name \"" << graph_name << "\" does not exist";
    auto graph_id = it->second->id();
    HT_VALUE_ERROR_IF(graph_id >= Graph::_global_graphs.size())
      << "Graph with id " << graph_id << " does not exist";
    Graph::_global_graphs[graph_id]->Clear();
    Graph::_global_graphs[graph_id] = nullptr;
    Graph::_name_to_graphs.erase(it);
  }

  static Graph& get_default_eager_graph() {
    Graph::InitOnce();
    return *Graph::_default_eager_graph;
  }

  static Graph& get_default_define_by_run_graph() {
    Graph::InitOnce();
    return *Graph::_default_define_by_run_graph;
  }

  static Graph& get_default_define_and_run_graph() {
    Graph::InitOnce();
    return *Graph::_default_define_and_run_graph;
  }

  static void push_graph_ctx(GraphId id) {
    HT_VALUE_ERROR_IF(id >= Graph::_global_graphs.size())
      << "Graph with id " << id << " does not exist";
    Graph::_cur_graph_ctx.push(id);
  }

  static GraphId cur_graph_ctx() {
    return Graph::_cur_graph_ctx.top();
  }

  static void pop_graph_ctx() {
    Graph::_cur_graph_ctx.pop();
  }

  static void MarkAsParameter(const Operator& op) {
    Graph::GetGraph(op->graph_id()).MarkOpAsParameter(op->id());
  }
  
  static void MarkAsParameter(const Tensor& tensor) {
    Graph::GetGraph(tensor->graph_id())
      .MarkOpAsParameter(tensor->producer_id());
  }

  static void MarkAsOptimizerVariable(const Operator& op) {
    Graph::GetGraph(op->graph_id()).MarkOpAsOptimizerVariable(op->id());
  }

  static void MarkAsOptimizerVariable(const Tensor& tensor) {
    Graph::GetGraph(tensor->graph_id())
      .MarkOpAsOptimizerVariable(tensor->producer_id());
  }

  static inline OpCRefList GetProducerOpCRefList(const TensorList& tensors) {
    OpCRefList ret;
    ret.reserve(tensors.size());
    for (const auto& tensor : tensors)
      ret.push_back(std::cref(tensor->producer()));
    return ret;
  }

  static inline OpCRefList GetProducerOpCRefList(const TensorRefList& tensors) {
    OpCRefList ret;
    ret.reserve(tensors.size());
    for (const auto& tensor_ref : tensors)
      ret.push_back(std::cref(tensor_ref.get()->producer()));
    return ret;
  }

  static OpRefList TopoSort(const OpRefList& ops, int32_t num_ops_hint = -1,
                            std::function<bool(const Operator&)> stop_at = {});

  static OpRefList TopoSort(const TensorList& tensors,
                            int32_t num_ops_hint = -1,
                            std::function<bool(const Operator&)> stop_at = {}) {
    OpRefList ops;
    ops.reserve(tensors.size());
    for (const auto& tensor : tensors)
      ops.push_back(std::ref(tensor->producer()));
    return Graph::TopoSort(ops, num_ops_hint, stop_at);
  }

  static OpRefList TopoSort(const Tensor& tensor, int32_t num_ops_hint = -1,
                            std::function<bool(const Operator&)> stop_at = {}) {
    OpRefList ops;
    ops.push_back(std::ref(tensor->producer()));
    return Graph::TopoSort(ops, num_ops_hint, stop_at);
  }

  static std::tuple<OpRefList, OpRefList> disentangle_forward_and_backward_ops_by_loss(
    const OpRefList& topo, const TensorList& losses);  

  static std::tuple<OpRefList, OpRefList> disentangle_forward_and_backward_ops(
    const OpRefList& topo);    

  // static TensorRefList DependentVariables(const TensorList& tensors,
  //                                         int32_t num_ops_hint = -1) {
  //   auto var_op_refs = TopoSort(tensors, num_ops_hint, [](const Operator& op) {
  //     return is_variable(op);
  //   });
  //   TensorRefList var_refs;
  //   var_refs.reserve(var_op_refs.size());
  //   for (auto& var_op_ref : var_op_refs)
  //     var_refs.push_back(std::ref(var_op_ref->output(0)));
  //   return var_refs;
  // }

  // static TensorRefList DependentVariables(const TensorCRefList& tensors,
  //                                         int32_t num_ops_hint = -1) {
  //   auto var_op_refs = TopoSort(tensors, num_ops_hint, [](const Operator& op) {
  //     return is_variable(op);
  //   });
  //   TensorRefList var_refs;
  //   var_refs.reserve(var_op_refs.size());
  //   for (auto& var_op_ref : var_op_refs)
  //     var_refs.push_back(std::ref(var_op_ref->output(0)));
  //   return var_refs;
  // }

  // static TensorRefList DependentVariables(const Tensor& tensor,
  //                                         int32_t num_ops_hint = -1) {
  //   return Graph::DependentVariables(TensorCRefList{std::cref(tensor)},
  //                                    num_ops_hint);
  // }

  static TensorList Gradients(const TensorList& ys, const TensorList& xs,
                              const TensorList& grad_ys = TensorList(),
                              int32_t num_ops_hint = -1);

  static TensorList Gradients(const Tensor& y, const TensorList& xs,
                              const Tensor& grad_y = Tensor(),
                              int32_t num_ops_hint = -1) {
    return Gradients(TensorList{y}, xs, TensorList{grad_y}, num_ops_hint);
  }

  static void ResetVariableData(const Tensor& tensor, const Initializer& init) {
    HT_VALUE_ERROR_IF(!tensor->is_variable())
      << "'ResetVariableData' does not support non-variable tensor: " << tensor;
    Graph::GetGraph(tensor).ResetVariableDataInner(tensor, init);
  }

  static NDArray& GetVariableData(const Tensor& tensor) {
    HT_VALUE_ERROR_IF(!tensor->is_variable())
      << "'GetVariableData' does not support non-variable tensor: " << tensor;
    return Graph::GetGraph(tensor).GetVariableDataInner(tensor);
  }

  static NDArray GetDetachedVariableData(const Tensor& tensor) {
    HT_VALUE_ERROR_IF(!tensor->is_variable())
      << "'GetDetachedVariableData' does not support non-variable tensor: " << tensor;
    return Graph::GetGraph(tensor).GetDetachedVariableDataInner(tensor);
  }

  static DeviceGroupUnion GetVariableDeviceGroupUnion(const Tensor& tensor) {
    HT_VALUE_ERROR_IF(!tensor->is_variable())
      << "'GetDetachedVariableData' does not support non-variable tensor: " << tensor;
    return Graph::GetGraph(tensor).GetVariableDeviceGroupUnionInner(tensor);
  }

  static NDArray&
  AllocVariableData(const Tensor& tensor,
                    const Initializer& init = VoidifiedInitializer(),
                    uint64_t seed = 0, const HTShape& global_shape = HTShape()) {
    HT_VALUE_ERROR_IF(!tensor->is_variable())
      << "'AllocVariableData' does not support non-variable tensor: " << tensor;
    return Graph::GetGraph(tensor).AllocVariableDataInner(tensor, init, seed, global_shape);
  }

  static void
  RegisterVariableData(const Tensor& tensor, NDArray data,
                       const Initializer& init = VoidifiedInitializer()) {
    HT_VALUE_ERROR_IF(!tensor->is_variable())
      << "'RegisterVariableData' does not support non-variable tensor: "
      << tensor;
    return Graph::GetGraph(tensor).RegisterVariableDataInner(tensor, data,
                                                             init);
  }

  static void 
  ReplaceInput(Operator& op, size_t input_index, Tensor& new_input, bool ignore_shape=false) {
    auto& old_input = op->_inputs[input_index];
    // stride may not be equal
    // since we need to replace the uncontiguous tensor
    if (!ignore_shape) {
      HT_ASSERT(old_input->shape() == new_input->shape())
        << "ReplaceInput can only used when the new tensor shape is equal to the old one"
        << ", but the new tensor " << new_input << " shape is " << new_input->shape()
        << " and the old tensor " << old_input << " shape is " << old_input->shape();
    }
    if (old_input->symbolic()) {
      new_input->copy_symbolic_shape(old_input->symbolic_shape());
    }
    old_input->DelConsumer(op);
    op->_inputs[input_index] = new_input;
    new_input->AddConsumer(op);
  }

  static void 
  ReplaceOutput(Operator& op, size_t output_index, Tensor& new_output) {
    auto& old_output = op->_outputs[output_index];
    // stride may not be equal
    // since we need to replace the uncontiguous tensor
    HT_ASSERT(old_output->shape() == new_output->shape())
      << "ReplaceOutput can only used when the new tensor shape is equal to the old one"
      << ", but the new tensor " << new_output << " shape is " << new_output->shape()
      << " and the old tensor " << old_output << " shape is " << old_output->shape();
    HT_ASSERT(old_output->num_consumers() == 0)
      << "ReplaceOutput can only used when the old tensor has no consumers (guarantee the safeness of topo)";
    if (old_output->symbolic()) {
      new_output->copy_symbolic_shape(old_output->symbolic_shape());
    }
    op->_outputs[output_index] = new_output;
    new_output->set_producer_id(op->id());
  }

  static void 
  AddInDeps(Operator& op, const TensorList& in_deps) {
    if (in_deps.empty()) {
      return;
    }
    if (in_deps.size() == 1) {
      op->_extra_in_dep_linkers.push_back(in_deps.front());
    } else {
      op->_extra_in_dep_linkers.push_back(
        MakeGroupOp(OpMeta()
                      .set_extra_deps(in_deps)
                      .set_name(op->name() + "_extra_in_dep")));
    }
    op->_extra_in_dep_linkers.back()->AddConsumer(op);
  }

  template <class T, class... Args>
  static T& make_new_graph(Args&&... args) {
    return *Graph::_make_new_graph<T>(std::forward<Args>(args)...);
  }

 protected:
  static void InitOnce() {
    std::call_once(Graph::_init_flag, Graph::Init);
  }

  static void Init();

  template <class T, class... Args>
  static std::shared_ptr<T>& _make_new_graph(Args&&... args) {
    static_assert(std::is_base_of<Graph, T>::value,
                  "Template class is not derived from Graph");
    auto graph = std::make_shared<T>(Graph::constrcutor_access_key(),
                                     std::forward<Args>(args)...);
    HT_LOG_DEBUG << "device = " << hetu::impl::comm::GetLocalDevice() << ": make a new graph named " 
      << graph->name() << ", whose id is " << graph->id();
    HT_VALUE_ERROR_IF(Graph::_global_graphs.size() != graph->id())
      << "Graph must be initialized using the `_make_new_graph` function";
    HT_VALUE_ERROR_IF(Graph::_name_to_graphs.find(graph->name()) !=
                      Graph::_name_to_graphs.end())
      << "Graph with name \"" << graph->name() << "\" already exists";
    Graph::_global_graphs.push_back(graph);
    Graph::_name_to_graphs[graph->name()] = graph;
    return reinterpret_cast<std::shared_ptr<T>&>(Graph::_global_graphs.back());
  }

  static size_t get_tensor_referrence_count(const Tensor& tensor) {
    return tensor.get_referrence_count();
  }

  static std::once_flag _init_flag;
  static std::vector<std::shared_ptr<Graph>> _global_graphs;
  static std::unordered_map<GraphName, std::shared_ptr<Graph>> _name_to_graphs;
  static std::shared_ptr<Graph> _default_eager_graph;
  static std::shared_ptr<Graph> _default_define_by_run_graph;
  static std::shared_ptr<Graph> _default_define_and_run_graph;
  static thread_local std::stack<GraphId> _cur_graph_ctx;
};

inline OpRefList Graph::TopoSort(const OpRefList& ops, int32_t num_ops_hint,
                                 std::function<bool(const Operator&)> stop_at) {
  std::unordered_map<OpId, int32_t> in_degrees;
  std::unordered_set<OpId> visited;
  std::unordered_map<OpId, OpIdSet> extra_edges;
  OpRefList inplace_ops;
  OpRefDeque traverse_queue;
  OpRefDeque topo_queue;
  OpRefList ret;
  if (num_ops_hint != -1) {
    in_degrees.reserve(num_ops_hint);
    visited.reserve(num_ops_hint);
    ret.reserve(num_ops_hint);
  }

  for (auto& op_ref : ops) {
    if (visited.find(op_ref.get()->id()) == visited.end()) {
      traverse_queue.push_back(op_ref);
      visited.insert(op_ref.get()->id());
    }
  }

  // traverse all ops that are connected with the target ops
  // and enqueue source ops into the topo queue
  while (!traverse_queue.empty()) {
    auto& op_ref = traverse_queue.front();
    traverse_queue.pop_front();
    auto& op = op_ref.get();
    auto op_in_degrees = (stop_at && stop_at(op)) ? 0 : op->in_degrees();
    if (op_in_degrees != op->in_degrees()) {
      HT_LOG_DEBUG << "make topo: " << op << " is forced to be stopped";
    }
    in_degrees[op->id()] = op_in_degrees;
    if (op_in_degrees == 0) {
      topo_queue.push_back(op_ref);
    } else {
      Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
        if (visited.find(tensor->producer_id()) == visited.end()) {
          traverse_queue.push_back(std::ref(tensor->producer()));
          visited.insert(tensor->producer_id());
        }
      });
    }
  }

  // iteratively find the topo order
  while (!topo_queue.empty()) {
    auto& op_ref = topo_queue.front();
    topo_queue.pop_front();
    ret.push_back(op_ref);
    Operator::for_each_output_tensor(op_ref.get(), [&](Tensor& tensor) {
      // Place inplace ops after other ops
      // Step 1. Find all inplace ops.
      int64_t num_inplace_consumers = 0;
      int64_t num_consumers = tensor->num_consumers();
      OpIdSet inplace_ids;
      Tensor::for_each_consumer(tensor, [&](Operator& consumer_op) {
        if (in_degrees.find(consumer_op->id()) == in_degrees.end())
          return;
        // NOTE: Only inplace ops that do operation on `tensor` directly
        // are considered.
        // For example, for `where` op, its inplace tensor is the 1st
        // input tensor, not the 2nd input tensor. So for the 2nd input,
        // we should not consider `where` as its inplace op.
        if (is_inplace_op(std::ref(consumer_op)) &&
            tensor->id() == consumer_op->inplace_tensor_id()) {
          inplace_ids.insert(consumer_op->id());
          inplace_ops.push_back(std::ref(consumer_op));
          num_inplace_consumers++;
          return;
        }
      });
      bool has_extra_edge = num_inplace_consumers > 0 &&
                            num_inplace_consumers < num_consumers;
      // Step 2. If there are both inplace and other ops,
      // add extra edges from others to inplace ops and add in_degrees.
      if (has_extra_edge) {
        Tensor::for_each_consumer(tensor, [&](Operator& consumer_op) {
          if (in_degrees.find(consumer_op->id()) == in_degrees.end() ||
              is_inplace_op(std::ref(consumer_op)))
            return;
          // NOTE: Check if the extra edge already exists.
          for (auto inplace_id : inplace_ids) {
            if (extra_edges[consumer_op->id()].insert(inplace_id).second)
              in_degrees[inplace_id]++;
          }
        });
      }
      // Step 3. Check extra_edges and decrease in_degrees of inplace ops.
      Tensor::for_each_consumer(tensor, [&](Operator& consumer_op) {
        if (in_degrees.find(consumer_op->id()) == in_degrees.end())
          return;
        if ((--in_degrees[consumer_op->id()]) == 0) {
          topo_queue.push_back(std::ref(consumer_op));
          if (has_extra_edge && !is_inplace_op(std::ref(consumer_op))) {
            for (auto inplace_id : extra_edges[consumer_op->id()]) {
              if (--in_degrees[inplace_id] == 0) {
                auto inplace_op = std::find_if(inplace_ops.begin(), inplace_ops.end(),
                  [&](const OpRef& op_ref) { return op_ref.get()->id() == inplace_id; });
                if (inplace_op != inplace_ops.end())
                  topo_queue.push_back(*inplace_op);
              }
            }
          }
        }
      });
    });
  }

  visited.clear();
  for (size_t i = 0; i < ret.size(); i++) {
    // BatchISendIRecvOp must be directly after nearest SplitOp
    if (is_batched_isend_irecv_op(ret[i])) {
      Operator& batched_isend_irecv_op = ret[i].get();
      // input must be split_op
      if (batched_isend_irecv_op->num_inputs() > 0) {
        for (size_t j = i - 1; i >= 2 && j >= 1; j--) {
          if (is_slice_op(ret[j])) {
            bool is_matched = false;
            for (auto& consumer : ret[j].get()->output(0)->consumers()) {
              if (consumer.get()->id() == batched_isend_irecv_op->id()) {
                is_matched = true;
                break;
              }
            }
            if (is_matched) {
              // move batched_isend_irecv_op (ret[i]) after split_op (ret[j])
              for (size_t k = i; k > j + 1; k--) {
                ret[k] = ret[k - 1];
              } 
              ret[j + 1] = batched_isend_irecv_op;
              break;
            }
          }
        }
      }
    }
    /*
    // Question: is it necessary?
    // ensure update ops are executed later
    if (is_optimizer_update_op(ret[i])) {
      if (visited.find(ret[i].get()->id()) != visited.end())
        continue;
      visited.insert(ret[i].get()->id());
      TensorId updated_var_id = ret[i].get()->input(0)->id();
      for (size_t j = ret.size() - 1; j > i; j--) {
        if (is_optimizer_update_op(ret[j]))
          continue;
        bool non_conflicting = Operator::all_input_tensors_of(
          ret[j].get(),
          [&](const Tensor& tensor) { return tensor->id() != updated_var_id; });
        if (non_conflicting)
          continue;
        // insert ret[i] after ret[j]
        auto& update_op_ref = ret[i];
        for (size_t k = i; k < j; k++)
          ret[k] = ret[k + 1];
        ret[j] = update_op_ref;
        i--;
        break;
      }
    }
    */
  }

  return ret;
}

inline std::tuple<OpRefList, OpRefList> Graph::disentangle_forward_and_backward_ops(
  const OpRefList& topo) {
  OpRefList fw_ops;
  OpRefList bw_ops;
  bool is_bw = false;
  for (auto& op_ref : topo) {
    if (is_bw) {
      bw_ops.push_back(op_ref);
    } else {
      bool is_grad = Operator::all_output_tensors_of(op_ref.get(), 
        [&](const Tensor& tensor) { return tensor->is_grad();});
      if (!is_grad) {
        fw_ops.push_back(op_ref);
      } else {
        is_bw = true;
        bw_ops.push_back(op_ref);
      }
    }
  }
  return {fw_ops, bw_ops};
}

inline std::tuple<OpRefList, OpRefList> Graph::disentangle_forward_and_backward_ops_by_loss(
  const OpRefList& topo, const TensorList& losses) {
  // traverse forward nodes (including losses)
  OpCRefDeque traverse_queue;
  for (const Tensor& loss : losses)
    traverse_queue.push_back(loss->producer());
  std::set<OpId> fw_set;
  while (!traverse_queue.empty()) {
    const Operator& op = traverse_queue.front().get();
    traverse_queue.pop_front();
    fw_set.insert(op->id());
    Operator::for_each_input_tensor(op, [&](const Tensor& tensor) {
      if (fw_set.find(tensor->producer()->id()) == fw_set.end()) {
        traverse_queue.push_back(tensor->producer());
      }
    });
  }

  // get the forward ops
  OpRefList fw_ops;
  fw_ops.reserve(fw_set.size());
  std::copy_if(topo.begin(), topo.end(), std::back_inserter(fw_ops),
               [&fw_set](const OpRef& op_ref) {
                 return fw_set.find(op_ref.get()->id()) != fw_set.end();
               });

  // get the backward ops
  OpRefList bw_ops;
  bw_ops.reserve(topo.size() - fw_ops.size());
  std::copy_if(topo.begin(), topo.end(), std::back_inserter(bw_ops),
               [&fw_set](const OpRef& op_ref) {
                 return fw_set.find(op_ref.get()->id()) == fw_set.end();
               });

  return {fw_ops, bw_ops};
}

std::ostream& operator<<(std::ostream&, const Graph&);

// variable related APIs that need to used in python

inline void ResetVariableData(const Tensor& tensor, const NDArray& provided_data) {
  Graph::ResetVariableData(tensor, ProvidedInitializer(provided_data));
}

inline NDArray GetDetachedVariableData(const Tensor& tensor) {
  return Graph::GetDetachedVariableData(tensor);
}

inline DeviceGroupUnion GetVariableDeviceGroupUnion(const Tensor& tensor) {
  return Graph::GetVariableDeviceGroupUnion(tensor);
}

} // namespace graph
} // namespace hetu