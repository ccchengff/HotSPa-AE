#include "hetu/execution/dbr_executor.h"
#include "hetu/execution/device_placer.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace execution {

using hetu::operator<<;

DBRExecutor& GetOrCreateDBRExecutor() {
  static DBRExecutor exec;
  return exec;
}

DBRExecutor::DBRExecutor() {
  _exec_ctx = std::make_shared<DBRExecutionContext>();
  if (hetu::impl::comm::IsGlobalDeviceGroupReady()) {
    // If we have set up the global device mapping
    // before the construction of Executor,
    // we can determine the device information.
    _exec_ctx->_local_device = hetu::impl::comm::GetLocalDevice();
    _exec_ctx->_device_group = hetu::impl::comm::GetGlobalDeviceGroup();
  } else {
    // If not, set device as undertermined and modify it later in `Run`
    _exec_ctx->_local_device = Device(kUndeterminedDevice);
  }
}

void DBRExecutor::Run(const TensorList& fetches) {
  if (_exec_ctx->_local_device.is_undetermined()) {
    if (hetu::impl::comm::IsGlobalDeviceGroupReady()) {
      _exec_ctx->_local_device = hetu::impl::comm::GetLocalDevice();
      _exec_ctx->_device_group = hetu::impl::comm::GetGlobalDeviceGroup();
    } else {
      // TODO: check cuda available
      _exec_ctx->_local_device = Device(kCUDA);
      _exec_ctx->_device_group = DeviceGroup({_exec_ctx->_local_device});
    }
  }

  auto topo_order = TopoSort(fetches);
  // TODO: cache and re-use sub-executors

  auto updated_topo = PlaceDevices(topo_order);
  auto& global_topo = updated_topo.empty() ? topo_order : updated_topo;

  auto sub_exec =
    std::make_shared<DefaultDBRSubExecutor>(_exec_ctx, global_topo);
  sub_exec->Run(fetches);
}

OpList DBRExecutor::PlaceDevices(const OpList& topo_order) {
  HT_LOG_TRACE << "Device placement for topo: " << topo_order;
  OpList unplaced_nodes = get_unplaced_nodes(topo_order);
  if (unplaced_nodes.size() != topo_order.size())
    HT_LOG_TRACE << "Unplaced ops: " << unplaced_nodes;

  // Question: If all ops have been placed, then there must be
  // a cached sub_executor, so the following condition cannot be true?
  if (unplaced_nodes.empty()) {
    return OpList();
  }

  // TODO: return the inserted ops during mapping and placement
  // so that we need not to call the TopoSort again
  if (!_exec_ctx->parallel()) {
    // Question: Can we check the validity in the construction of DeviceGroup?
    // If yes, then the following checks are unnecessary.
    for (const auto& op : topo_order) {
      const auto& op_devices = op->device_group();
      HT_RUNTIME_ERROR_IF(op_devices.num_devices() > 1)
        << "Please set up the device mapping before parallel training";
      HT_RUNTIME_ERROR_IF(!op_devices.empty() &&
                          op_devices.get(0) != _exec_ctx->_local_device)
        << "Expected to execute locally with device "
        << _exec_ctx->_local_device << " but got " << op_devices.get(0);
    }
    PlaceToLocalDevice(unplaced_nodes, _exec_ctx->_local_device);
    return TopoSort(topo_order);
  } else {
    MapOpsToParallelDevices(unplaced_nodes, _exec_ctx->_device_group);
    OpList updated_topo_order =
      TopoSort(ExtendSubgraphWithCommunicationNodes(topo_order));
    OpList local_topo_order =
      get_local_nodes(updated_topo_order, _exec_ctx->_local_device, false);
    OpList local_unplaced_nodes = get_unplaced_nodes(local_topo_order);
    PlaceToLocalDevice(local_unplaced_nodes, _exec_ctx->_local_device);
    return TopoSort(local_topo_order);
  }
}

DBRSubExecutor::DBRSubExecutor(std::shared_ptr<DBRExecutionContext> exec_ctx,
                               const OpList& topo_order)
: _exec_ctx(exec_ctx), _topo_order(topo_order) {}

void DefaultDBRSubExecutor::Run(const TensorList& fetches) {
  auto target_ops = TopoSort(fetches, false, true);
  OpList to_sync_ops;
  to_sync_ops.reserve(target_ops.size());
  Tensor2NDArrayMap edge2arr;
  
  RuntimeContext runtime_ctx;
  for (auto& op : target_ops) {
    if (!op.is_defined())
      continue; // should not happen
    HT_LOG_TRACE << "Executing op \"" << op->name() << "\"...";

    if (is_variable_op(op)) {
      VariableOp& var = reinterpret_cast<VariableOp&>(op);
      edge2arr[op->output(0)->id()] = var->data();
      op->MarkAsComputed({var->data()});
      continue;
    }
    if (op->is_computed())
      continue; // Question: Is it necessary?

    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& in_edge : op->inputs()) {
      auto it = edge2arr.find(in_edge->id());
      HT_ASSERT((it != edge2arr.end() && it->second.is_defined()) ||
                in_edge->is_computed())
        << "Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << in_edge;
      if (it == edge2arr.end()) {
        input_vals.push_back(in_edge->GetOrCompute());
      } else {
        input_vals.push_back(it->second);
        // TODO: erase vals that will never be used later
      }
    }
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx);
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& out_edge = op->output(i);
      // TODO: do not insert vals that will never be used later
      edge2arr[out_edge->id()] = output_vals[i];
    }
    to_sync_ops.push_back(op);
  }

  for (auto& op : to_sync_ops) {
    op->Sync();
    NDArrayList output_vals(op->num_outputs());
    for (size_t i = 0; i < op->num_outputs(); i++) {
      auto it = edge2arr.find(op->output(i)->id());
      if (it != edge2arr.end())
        output_vals[i] = it->second;
    }
    op->MarkAsComputed(output_vals);
  }
}

} // namespace execution
} // namespace hetu
