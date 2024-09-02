#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/profiler.h"
#include "hetu/graph/init/initializer.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/ops/group.h"

namespace hetu {
namespace graph {

struct ExecutePlan {
  OpRefList local_placeholder_variable_ops;
  OpRefList local_fw_topo;
  OpRefList local_bw_topo;
  OpRefList local_topo;
  TensorIdSet dtype_transfer_tensor;
  TensorIdSet shared_weight_tensor;
  OpIdSet shared_weight_p2p;
  OpIdSet shared_weight_grad_p2p;
  TensorIdSet accumulated_tensor;
  OpIdSet accumulated_ops;

  void update(OpRefList& _local_placeholder_variable_ops, 
              OpRefList& _local_fw_topo, OpRefList& _local_bw_topo, 
              OpRefList& _local_topo, TensorIdSet& _dtype_transfer_tensor, 
              TensorIdSet& _shared_weight_tensor, OpIdSet& _shared_weight_p2p, 
              OpIdSet& _shared_weight_grad_p2p, TensorIdSet& _accumulated_tensor, 
              OpIdSet& _accumulated_ops) {
    local_placeholder_variable_ops = _local_placeholder_variable_ops;
    local_fw_topo = _local_fw_topo;
    local_bw_topo = _local_bw_topo;
    local_topo = _local_topo;
    dtype_transfer_tensor = _dtype_transfer_tensor;
    shared_weight_tensor = _shared_weight_tensor;
    shared_weight_p2p = _shared_weight_p2p;
    shared_weight_grad_p2p = _shared_weight_grad_p2p;
    accumulated_tensor = _accumulated_tensor;
    accumulated_ops = _accumulated_ops;
  }
};

class ExecutableGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;
  friend class DefineAndRunGraph;
  friend class SwitchExecGraph;

  ExecutableGraph(GraphName name, size_t init_capacity)
  : Graph(name, init_capacity) {}

 public:
  ExecutableGraph(const constrcutor_access_key&, GraphName name,
                  size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : ExecutableGraph(name, init_capacity) {}

  // bool MapOpsToParallelDevices(const DeviceGroup& placement_group);

  bool Instantiate(const TensorList& fetches, const Device& placement);

  NDArrayList Run(const TensorList& fetches, 
                  const FeedDict& feed_dict = {});

  NDArrayList Run(const Tensor& loss, const TensorList& fetches, 
                  const FeedDict& feed_dict = {}, const int num_micro_batches = 1,
                  const int cur_strategy_id = 0, RunLevel run_level = RunLevel::UPDATE, const double grad_scale = 1);

  GraphType type() const {
    return GraphType::EXECUTABLE;
  }

  void SetPipeline(const Device2PipelineMap& pipeline_map) {
    _pipeline_map = pipeline_map;
  }

  void SetUsedRanks(const std::vector<int>& used_ranks) {
    _used_ranks = used_ranks;
  }

  bool NeedRank(int rank) {
    return std::find(_used_ranks.begin(), _used_ranks.end(), rank) != _used_ranks.end();
  }

  void SetRunLevel(RunLevel run_level) {
    _run_level = run_level;
  }

  void SetShapePlan(size_t num) {
    HT_ASSERT(num < _shape_plan_pool.size())
      << "plan number shouldn't exceed the size of the plan pool";
    _active_shape_plan = num;
  }

  void AddShapePlan(const Tensor2ShapeMap& shape_plan) {
    _shape_plan_pool.emplace_back(shape_plan);
  }

  void AddShapePlan(Tensor2ShapeMap&& shape_plan) {
    _shape_plan_pool.emplace_back(std::move(shape_plan));
  }

  // 目前主要功能是
  // 1、记录exec graph相较define graph新插入的tensor
  // 2、记录新插入tensor的shape到当前的shape plan
  void RecordExecTensor(const Tensor& tensor) {
    const auto& shape = tensor->shape();
    // need to record the shape for all shape plans in the shape plan pool
    // so we leverage _record_execute_tensor and do it lazily
    _record_exec_tensors.emplace_back(tensor);
    if (!_shape_plan_pool.empty()) {
      auto& shape_plan = _shape_plan_pool.at(_active_shape_plan);
      auto it = shape_plan.find(tensor->id());
      if (it != shape_plan.end()) {
        // already existed, then must be equal
        HT_ASSERT(it->second.size() == shape.size())
          << "Tensor " << tensor << " is already existed in shape plan but is unequal";
        for (size_t i = 0; i < shape.size(); i++) { 
          HT_ASSERT(it->second[i] == shape[i])
            << "Tensor " << tensor << " is already existed in shape plan but is unequal";
        }
        return;
      }
      shape_plan.insert(std::make_pair(tensor->id(), shape));
    }
  }

  // 与RecordExecTensor功能相反
  // 暂时用不到
  /*
  void EraseExecTensor(const Tensor& tensor) {
    auto& shape_plan = _shape_plan_pool.at(_active_shape_plan);
    const auto& shape = tensor->shape();
    _erase_execute_tensor.emplace_back(tensor);
    auto it = shape_plan.find(tensor->id());
    HT_ASSERT(it != shape_plan.end())
      << "Tensor " << tensor << " should exist in shape plan";
    shape_plan.erase(it);
  }
  */

  const HTShape& GetTensorShape(const Tensor& tensor) const {
    const auto& shape_plan = _shape_plan_pool.at(_active_shape_plan);
    auto it = shape_plan.find(tensor->id());
    HT_ASSERT(it != shape_plan.end())
      << "Tensor " << tensor << " is not existed in current shape plan";
    return it->second;
  }

 protected:
  DeviceGroup GetPrevStage();

  DeviceGroup GetNextStage();

  std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>>
  GenerateGpipeSchedule(size_t num_stages, size_t num_micro_batches, bool is_inference);

  std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>>
  GeneratePipedreamFlushSchedule(size_t num_stages, size_t num_micro_batches, bool is_inference);

  void ComputeFunc(size_t& micro_batch_id, const OpRefList& topo, RuntimeContext& runtime_ctx, 
                  Tensor2NDArrayMap& tensor2data, Tensor2IntMap& tensor2degrees, 
                  Tensor2NDArrayMap& grad_accumulation, bool grad_accumulation_finished,
                  const FeedDict& feed_dict, const TensorList& fetches,
                  const std::unordered_map<TensorId, size_t>& fetch_indices, 
                  bool& is_continuous_p2p);

  void SubstituteCommOp(const OpRefList& topo_order);

  void InsertContiguousOp(const OpRefList& topo_order);

  // deprecated
  /*
  void CrossSend(std::unordered_map<int32_t, int32_t> split_cur_state, 
                 std::unordered_map<int32_t, int32_t> split_target_state,
                 int32_t depth, bool need_split, int32_t& device_index, 
                 Operator& comm_op, TensorList& send_datas, 
                 std::vector<int32_t>& dsts, int32_t& used_device_index);

  Tensor CrossReceive(int32_t depth, int32_t& device_index, Operator& comm_op, 
                      TensorList& recv_datas, std::vector<int32_t>& srcs,
                      Tensor& self_send_data, int32_t& used_device_index);
  */

  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void ResetVariableDataInner(const Tensor& tensor,
                              const Initializer& init) override;

  NDArray& GetVariableDataInner(const Tensor& tensor) override;

  NDArray GetDetachedVariableDataInner(const Tensor& tensor) override;

  NDArray& AllocVariableDataInner(
    const Tensor& tensor,
    const Initializer& init = VoidifiedInitializer(),
    uint64_t seed = 0, const HTShape& global_shape = HTShape()) override;

  void RegisterVariableDataInner(
    const Tensor& tensor, NDArray data,
    const Initializer& init = VoidifiedInitializer()) override;

  void AllocRuntimeBuffer(std::vector<RuntimeContext>& runtime_ctx_list);

  void GetExecEnvs();

  // plan相关
  ExecutePlan _execute_plan;
  std::vector<Tensor2ShapeMap> _shape_plan_pool;
  size_t _active_shape_plan;
  std::vector<Tensor> _record_exec_tensors;

  // run相关
  std::unordered_map<TensorId, std::unique_ptr<Initializer>> _add_on_inits;
  Device2PipelineMap _pipeline_map;
  std::vector<int> _used_ranks;
  int _num_micro_batches;
  std::vector<std::unique_ptr<Event>> _p2p_events;

  // switch相关
  std::shared_ptr<ParamBuffer> _origin_param_buffer;
  std::shared_ptr<ParamBuffer> _transfer_param_buffer;
  std::shared_ptr<ParamBuffer> _origin_param_and_optimizer_buffer; // deprecated
  std::shared_ptr<ParamBuckets> _origin_param_and_optimizer_buckets; 
  std::shared_ptr<ParamBuffer> _current_grad_buffer; // deprecated
  std::shared_ptr<ParamBuffer> _accumulate_grad_buffer;
  Tensor2TensorMap _transfer_map; // origin param到transfer param的映射
  Tensor2TensorMap _grad_map; // origin param到未substitue comm op前的grad的映射
  Tensor2TensorMap _grad_grad_map; // 未substitue comm op前的grad到substitue comm op后的grad的映射
  Tensor2TensorMap _reversed_grad_grad_map; // substitue comm op后的grad到未substitue comm op前的grad的映射
  bool _use_current_grad_buffer{false};
  bool _use_origin_param_and_optimizer_buffer{false};
  bool _use_origin_param_and_optimizer_buckets{true};
  double _grad_scale; 

  // 记录上一个图的param切换完的event
  std::unordered_map<TensorId, std::unique_ptr<Event>> _switch_param_events;
  // 记录上一个图的grad切换完的event
  std::unordered_map<TensorId, std::unique_ptr<Event>> _switch_grad_events; // 注意这里的TensorId是未substitue comm op前的grad
  // 记录当前图的param不再用的event
  // 即意味着可以开始切换param了
  std::unordered_map<TensorId, std::unique_ptr<Event>> _run_param_events; 
  // 记录当前图的grad计算完的event
  // 即意味着可以开始切换grad了
  std::unordered_map<TensorId, std::unique_ptr<Event>> _run_grad_events; // 注意这里的TensorId是未substitue comm op后的grad

  // profile相关
  int32_t _straggler_flag;
  std::string _straggler_log_file_path;
  MEMORY_PROFILE_LEVEL _memory_profile_level;
  std::string _memory_log_file_path;
  std::vector<std::shared_ptr<MicroBatchMemoryInfo>> _all_micro_batches_memory_info;
};

} // namespace graph
} // namespace hetu
