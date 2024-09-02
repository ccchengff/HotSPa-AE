#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/autograd/operator.h"
#include "hetu/autograd/autograd.h"
#include "hetu/autograd/topo.h"
#include "hetu/autograd/runtime_context.h"
#include "hetu/execution/run_metadata.h"
#include "hetu/autograd/ops/Communicate.h"

namespace hetu {
namespace execution {

using namespace hetu::autograd;

// Define-And-Run Executor for static graphs
class DARExecutionContext;
class DARExecutor;
class DARSubExecutor;

using TensorIdList2SubExecutorMap =
  std::map<TensorIdList, std::shared_ptr<DARSubExecutor>>;
using OpIdList2SubExecutorMap =
  std::map<OpIdList, std::shared_ptr<DARSubExecutor>>;
using FeedDict = Tensor2NDArrayMap;

class DARExecutionContext {
 public:
  friend class DARExecutor;
  friend class DARSubExecutor;

  DARExecutionContext() = default;
  ~DARExecutionContext() = default;

  const Device& local_device() const {
    return _local_device;
  }

  const DeviceGroup& device_group() const {
    return _device_group;
  }

  const OpList& global_topo_order() const {
    return _global_topo_order;
  }

  const OpList& local_topo_order() const {
    return _local_topo_order;
  }

  int64_t seed() const {
    return _seed;
  }

  bool is_pipeline_parallel() const {
    return _global_topo_order.size() != _local_topo_order.size();
  }

 protected:
  Device _local_device;
  DeviceGroup _device_group;
  OpList _global_topo_order;
  OpList _local_topo_order;
  TensorList _losses;
  bool _training;
  uint64_t _seed{0};
};

class DARExecutor {
 public:
  DARExecutor(const Device& local_device, const DeviceGroup& device_group = {},
              const TensorList& losses = {});

  NDArrayList Run(const TensorList& fetches, const FeedDict& feed_dict = {},
                  const int num_micro_batches = 1);

  const Device& local_device() const {
    return _exec_ctx->local_device();
  }

  const DeviceGroup& device_group() const {
    return _exec_ctx->device_group();
  }

  const OpList& global_topo_order() const {
    return _exec_ctx->global_topo_order();
  }

  const OpList& local_topo_order() const {
    return _exec_ctx->local_topo_order();
  }

  uint64_t seed() const {
    return _exec_ctx->seed();
  }

 protected:
  bool PlaceDevices(const OpList& topo_order);
  void SubstituteCommOp(const OpList& local_topo_order);
  void CrossSend(std::unordered_map<int32_t, int32_t> split_cur_state, 
                 std::unordered_map<int32_t, int32_t> split_target_state,
                 int32_t depth, bool need_split, int32_t& device_index, 
                 CommOp& op, TensorList& send_datas, std::vector<int32_t>& dsts, 
                 int32_t& used_device_index);
  Tensor CrossReceive(int32_t depth, int32_t& device_index, CommOp& op, 
                      TensorList& recv_datas, std::vector<int32_t>& srcs,
                      Tensor& self_send_data, int32_t& used_device_index);
  std::shared_ptr<DARSubExecutor>
  GetOrCreateSubExecutor(const TensorList& fetches);

  std::shared_ptr<DARExecutionContext> _exec_ctx;
  TensorIdList2SubExecutorMap _fetches_to_sub_executors;
  OpIdList2SubExecutorMap _topo_to_sub_executors;
};

class DARSubExecutor {
 public:
  DARSubExecutor(std::shared_ptr<DARExecutionContext> exec_ctx,
                 const OpList& topo_order);

  virtual NDArrayList Run(const TensorList& fetches = {},
                          const FeedDict& feed_dict = {},
                          const int num_micro_batches = 1) = 0;

 protected:
  std::shared_ptr<DARExecutionContext> _exec_ctx;
  OpList _topo_order;
  std::unordered_map<OpId, int32_t> _edge_out_degrees;
  Device _device;
  OpList _variable_ops;
  OpList _placeholder_ops;
  OpList _data_loader_ops;
  OpList _computing_ops;
};

class DefaultDARSubExecutor : public DARSubExecutor {
 public:
  DefaultDARSubExecutor(std::shared_ptr<DARExecutionContext> exec_ctx,
                        const OpList& topo_order)
  : DARSubExecutor(exec_ctx, topo_order) {}

  virtual NDArrayList Run(const TensorList& fetches = {},
                          const FeedDict& feed_dict = {},
                          const int num_micro_batches = 1);
};

class PipelineDARSubExecutor : public DARSubExecutor {
 public:
  PipelineDARSubExecutor(std::shared_ptr<DARExecutionContext> exec_ctx,
                         const OpList& topo_order, const OpList& fw_topo_order,
                         const OpList& bw_topo_order);

  virtual NDArrayList Run(const TensorList& fetches = {},
                          const FeedDict& feed_dict = {},
                          const int num_micro_batches = 1);

  void compute_fn(OpList& compute_ops, Tensor2NDArrayMap& edge2arr,
                  std::unordered_map<TensorId, int>& edge2degrees,
                  std::unordered_map<TensorId, NDArray>& grad_accumulation,
                  bool grad_accumulation_finished,
                  std::unordered_set<OpId>& fetch_ids,
                  RuntimeContext& runtime_ctx);

  std::unordered_map<int, std::vector<std::pair<bool, int>>>
  generate_pipedream_flush_schedule(int num_stages, int num_micro_batches);

  std::unordered_map<int, std::vector<std::pair<bool, int>>>
  generate_gpipe_schedule(int num_stages, int num_micro_batches);

 protected:
  OpList _fw_topo_order;
  OpList _bw_topo_order;
  OpList _fw_computing_ops;
  OpList _bw_computing_ops;
  OpList _gradient_ops;
  OpList _gradient_consumer_ops;
};

} // namespace execution
} // namespace hetu
