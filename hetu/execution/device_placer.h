#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace execution {

using namespace hetu::autograd;

void MapOpsToParallelDevices(const OpList& topo_order,
                             const DeviceGroup& device_group);

void PlaceToLocalDevice(const OpList& topo_order, const Device& device);

inline OpList get_local_nodes(const OpList& nodes, const Device& local_device,
                              bool all_device_groups_mapped = true) {
  OpList local_nodes;
  local_nodes.reserve(nodes.size());
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto& op = *it;
    if (all_device_groups_mapped) {
      HT_ASSERT(!op->placement_group().empty())
        << "Operator \"" << op->name() << "\" is not properly mapped";
    } else if (op->placement_group().empty()) {
      // This op is not going to be executed, so we skip it.
      continue;
    }
    if (op->placement_group().contains(local_device))
      local_nodes.push_back(op);
  }
  return local_nodes;
}

inline OpList get_unplaced_nodes(const OpList& nodes) {
  OpList unplaced_nodes;
  unplaced_nodes.reserve(nodes.size());
  std::copy_if(
    nodes.begin(), nodes.end(), std::back_inserter(unplaced_nodes), 
    [](const Operator& op) { return op->placement().is_undetermined(); });
  return unplaced_nodes;
}

} // namespace execution
} // namespace hetu
