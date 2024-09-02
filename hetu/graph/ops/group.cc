#include "hetu/graph/ops/group.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

namespace {
inline void _MergeDeps(TensorList&& deps, OpMeta& op_meta) {
  if (op_meta.extra_deps.empty()) {
    op_meta.extra_deps = std::move(deps);
  } else {
    op_meta.extra_deps.reserve(deps.size() + op_meta.extra_deps.size());
    std::move(deps.begin(), deps.end(), std::back_inserter(op_meta.extra_deps));
  }
}
} // namespace

Tensor MakeGroupOp(OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<GroupOpImpl>(), TensorList(),
                       std::move(op_meta))
    ->out_dep_linker();
}

Tensor MakeGroupOp(TensorList deps, OpMeta op_meta) {
  _MergeDeps(std::move(deps), op_meta);
  return Graph::MakeOp(std::make_shared<GroupOpImpl>(), TensorList(),
                       std::move(op_meta))
    ->out_dep_linker();
}

} // namespace graph
} // namespace hetu
