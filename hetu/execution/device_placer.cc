#include "hetu/execution/device_placer.h"
#include "hetu/autograd/ops/Group.h"

namespace hetu {
namespace execution {

void MapOpsToParallelDevices(const OpList& topo_order,
                             const DeviceGroup& device_group) {
  bool pure_dp = topo_order.end() ==
    std::find_if_not(topo_order.begin(), topo_order.end(),
                     [](const Operator& op) {
                       return op->device_group().empty();
                     });
  if (pure_dp) { // pure_dp or pure_tp
    // pure data parallel
    HT_ASSERT(device_group.num_devices() > 1)
      << "Invalid device group for data parallel: " << device_group;
    for (auto& op : topo_order) {
      op->MapToParallelDevices(device_group);
    }
  } else { // (dp, tp, pp)
    for (auto& op : topo_order) {
      if (!op->device_group().empty()) {
        op->MapToParallelDevices(op->device_group());
        // HT_LOG_INFO << op->name() << ": map to device group " << op->device_group();
      } else {
        DeviceGroup inferred;
        if (is_group_op(op)) {
          std::vector<Device> devices;
          auto& group_op = reinterpret_cast<const GroupOp&>(op);
          for (auto& input : group_op->get_in_dep_linkers())
            for (auto& device : input->producer()->placement_group().devices())
              devices.push_back(device);
          inferred = DeviceGroup(devices);
        } else {
          HT_ASSERT(op->num_inputs() > 0)
            << "Currently we cannot infer the devices "
            << "for operators with zero in-degree. : " << op;
          inferred = op->input(0)->producer()->placement_group();
        }
        op->MapToParallelDevices(inferred);
        // HT_LOG_INFO << op->name() << ": map to device group " << op->device_group() << ", placement group " << op->placement_group();        
      }
    }
  }
}

void PlaceToLocalDevice(const OpList& topo_order, const Device& device) {
  for (auto& op : topo_order) {
    Device placement = is_device_to_host_op(op) ? Device(kCPU) : device;
    StreamIndex stream_id = op->stream_index();
    if (stream_id == kUndeterminedStream) {
      if (is_host_to_device_op(op)) {
        stream_id = kH2DStream;
      } else if (is_device_to_host_op(op)) {
        stream_id = kD2HStream;
      } else if (is_peer_to_peer_send_op(op) || is_peer_to_peer_recv_op(op)) {
        stream_id = kP2PStream;
      } else if (is_all_to_all_op(op)) {
        stream_id = kCollectiveStream;
      } else {
        stream_id = kComputingStream;
      }
    }
    bool ok = op->PlaceToLocalDevice(placement, stream_id);
    if (!ok && placement.is_cuda()) {
      HT_LOG_WARN << "Failed to place the \"" << op->type() << "\" operation "
                  << "(with name \"" << op->name() << "\") to " << placement
                  << ". "
                  << "Will try to place it on the host device.";
      placement = Device(kCPU);
      ok = op->PlaceToLocalDevice(placement, stream_id);
    }
    HT_ASSERT(ok) << "Failed to place the \"" << op->type() << "\" operation "
                  << "(with name \"" << op->name() << "\") to " << placement
                  << ".";
  }
}

} // namespace execution
} // namespace hetu
