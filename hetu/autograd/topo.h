#pragma once

#include "hetu/autograd/operator.h"
#include <tuple>
#include <set>

namespace hetu {
namespace autograd {

OpList TopoSort(const OpList& nodes, bool connect_p2p = true,
                bool skip_computed = false);

inline OpList TopoSort(const TensorList& edges, bool connect_p2p = true, 
                       bool skip_computed = false) {
  OpList nodes;
  nodes.reserve(edges.size());
  for (const auto& edge : edges)
    nodes.push_back(edge->producer());
  return TopoSort(nodes, connect_p2p, skip_computed);
}

inline OpList TopoSort(const Operator& node, bool connect_p2p = true,
                       bool skip_computed = false) {
  return TopoSort(OpList({node}), connect_p2p, skip_computed);
}

inline OpList TopoSort(const Tensor& edge, bool connect_p2p = true,
                       bool skip_computed = false) {
  return TopoSort(edge->producer(), connect_p2p, skip_computed);
}

OpList ExtendSubgraphWithCommunicationNodes(const OpList& nodes);

std::tuple<OpList, OpList> disentangle_forward_and_backward_nodes(
  const OpList& nodes, const TensorList& losses, bool connect_p2p);

inline std::tuple<OpList, OpList>
disentangle_forward_and_backward_nodes(const OpList& nodes, const Tensor& loss,
                                       bool connect_p2p) {
  return disentangle_forward_and_backward_nodes(nodes, TensorList({loss}),
                                                connect_p2p);
}

} // namespace autograd
} // namespace hetu
