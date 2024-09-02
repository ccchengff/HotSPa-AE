#include "hetu/graph/ops/zeros_like.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

Tensor MakeZerosLikeOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<ZerosLikeOpImpl>(), {std::move(input)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
