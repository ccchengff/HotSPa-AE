#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/autograd/operator.h"
#include "hetu/autograd/autograd.h"
#include "hetu/autograd/topo.h"
#include "hetu/autograd/runtime_context.h"

namespace hetu {
namespace execution {

using namespace hetu::autograd;

// Define-By-Run Executor for dynamic graphs
class DBRExecutionContext;
class DBRExecutor;
class DBRSubExecutor;

class DBRExecutionContext {
 public:
  friend class DBRExecutor;
  friend class DBRSubExecutor;

  DBRExecutionContext() = default;
  ~DBRExecutionContext() = default;

  const Device& local_device() const {
    return _local_device;
  }

  const DeviceGroup& device_group() const {
    return _device_group;
  }

  bool parallel() const {
    return _device_group.num_devices() > 1;
  }

  int64_t seed() const {
    return _seed;
  }

 protected:
  Device _local_device;
  DeviceGroup _device_group;
  bool _training;
  uint64_t _seed{0};
};

class DBRExecutor {
 public:
  DBRExecutor();

  void Run(const TensorList& fetches);

  const Device& local_device() const {
    return _exec_ctx->local_device();
  }

  const DeviceGroup& device_group() const {
    return _exec_ctx->device_group();
  }

  uint64_t seed() const {
    return _exec_ctx->seed();
  }

 protected:
  OpList PlaceDevices(const OpList& topo_order);

  std::shared_ptr<DBRExecutionContext> _exec_ctx;
  std::map<OpIdList, std::shared_ptr<DBRSubExecutor>> _topo_to_sub_executors;
};

class DBRSubExecutor {
 public:
  DBRSubExecutor(std::shared_ptr<DBRExecutionContext> exec_ctx,
                 const OpList& topo_order);

  virtual void Run(const TensorList& fetches) = 0;

 protected:
  std::shared_ptr<DBRExecutionContext> _exec_ctx;
  OpList _topo_order;
  std::unordered_map<OpId, int32_t> _node_in_degrees;
  std::unordered_map<TensorId, int32_t> _edge_out_degrees;
  Device _device;
};

class DefaultDBRSubExecutor : public DBRSubExecutor {
 public:
  DefaultDBRSubExecutor(std::shared_ptr<DBRExecutionContext> exec_ctx,
                        const OpList& topo_order)
  : DBRSubExecutor(exec_ctx, topo_order) {}

  virtual void Run(const TensorList& fetches);
};

DBRExecutor& GetOrCreateDBRExecutor();

} // namespace execution
} // namespace hetu
